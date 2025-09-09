#!/usr/bin/env python3
"""
    jlai.interp.lens.modal.remote
"""

import os
import asyncio
from time import time
from pathlib import Path
from typing import Optional

import modal
from .common import app

gpu_type       = os.getenv("JLAI_GPU_TYPE", "A10G")
max_containers = int(os.getenv("JLAI_MAX_CONTAINERS", "1"))

MODEL_CACHE_VOLUME = modal.Volume.from_name("lens-cache", create_if_missing=True)
MODEL_CACHE_PATH   = Path("/models")
VOLUMES            = {MODEL_CACHE_PATH: MODEL_CACHE_VOLUME}

# --
# Define container

_image = (
    modal.Image.from_registry(f"nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("libnuma-dev")
    .pip_install(
        "numpy",
        "rich",
        "transformer-lens",
        "torch",
        "transformers",
        "huggingface_hub",
        "hf-transfer",
        "hf-xet",
    )
    .env(
        {
            "HF_HOME"                   : str(MODEL_CACHE_PATH), # [TODO] does this work with transformer-lens?
            "HF_HUB_ENABLE_HF_TRANSFER" : "1",
        }
    )
)

with _image.imports():
    import torch
    import numpy as np
    import transformer_lens

# --
# Run

@app.cls(
    image                  = _image,
    gpu                    = gpu_type,
    max_containers         = max_containers,
    scaledown_window       = 2 * 60,
    retries                = 3,
    secrets                = [modal.Secret.from_name("huggingface-secret")],
    volumes                = VOLUMES,
    # enable_memory_snapshot = True,
    # experimental_options   = {"enable_gpu_snapshot": True}
)
@modal.concurrent(max_inputs=32)
class LensInference:
    model_str     : str = modal.parameter(default="Qwen/Qwen3-0.6B")
    padding_side  : str = modal.parameter(default="right")

    @modal.enter()
    def _on_enter(self):
        self.model = transformer_lens.HookedTransformer.from_pretrained(self.model_str)
        self.model.tokenizer.pad_token    = self.model.tokenizer.eos_token
        self.model.tokenizer.padding_side = self.padding_side
        print(f"Model {self.model_str} loaded successfully")
        
        self.semaphore = asyncio.Semaphore(1)

    @modal.method()
    def messages2tokens(self, messages):
        return [self._n_tokens(self._prep(msg)) for msg in messages]

    @modal.method()
    def hook_names(self):
        return list(self.model.hook_dict.keys())

    def _n_tokens(self, x):
        if isinstance(x, list):
            return [self._n_tokens(xx) for xx in x]
        
        return len(self.model.tokenizer.encode(x))
    
    def _prep(self, messages):
        return self.model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    @modal.method()
    async def forward(self, messages, return_logits=True, **kwargs) -> list[dict]:
        
        t0 = time()
        async with self.semaphore: # force max 1 concurrent forward call, even if more are queued
            t1 = time()
            print(f"LensInference.forward: start (waited {t1 - t0}s)")
            
            n_messages = len(messages)
            inputs_str = self._prep(messages)
            n_tokens   = self._n_tokens(inputs_str)
            
            with torch.no_grad():
                self.model.reset_hooks()
                logits, activations = self.model.run_with_cache(
                    inputs_str, 
                    padding_side        = self.model.tokenizer.padding_side, 
                    return_cache_object = False,
                    **kwargs
                )
            
            _cache = {
                **{k:v.to('cpu') for k, v in activations.items()},
            }
            if return_logits:
                _cache["logits"] = logits.to('cpu')
            
            out = []
            for i in range(n_messages):
                if self.model.tokenizer.padding_side == 'right':
                    tmp = {k: _cache[k][i,:n_tokens[i]] for k in _cache.keys()}
                elif self.model.tokenizer.padding_side == 'left':
                    tmp = {k: _cache[k][i,-n_tokens[i]:] for k in _cache.keys()}
                
                # <<
                # convert to numpy
                tmp = {k: v.numpy() for k, v in tmp.items()}
                # >>
                
                out.append(tmp)
            
            print(f"LensInference.forward: done (ran for {time() - t1}s)")
            return out