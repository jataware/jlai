#!/usr/bin/env python
"""
    jlai.interp.lens.modal.local
"""

import asyncio
import numpy as np
from tqdm import tqdm
import binpacking

import modal

from .common import app
from .remote import LensInference

class LensClient:
    def __init__(self, model_str='Qwen/Qwen3-0.6B', devel=False):
        if devel:
            _model_cls = LensInference
        else:
            _model_cls = modal.Cls.from_name(app.name, 'LensInference')

        self.model_str    = model_str
        self.model        = _model_cls(model_str=self.model_str)

    def n_tokens(self, messages):
        return self.model.messages2tokens.remote(messages)
    
    def hook_names(self):
        return self.model.hook_names.remote()

    def forward(self, messages : list):
        return self.model.forward.remote.aio(messages=messages)

    async def aforward(self, **kwargs):
        return await self.model.forward.remote.aio(**kwargs)

    def _compute_batches(self, n_tokens, tokens_per_batch, verbose=True):
        bins = binpacking.to_constant_volume(
            dict(zip(range(len(n_tokens)), n_tokens)),
            tokens_per_batch
        )
        bins = [list(b.keys()) for b in bins]
        
        if verbose:
            for idx, members in enumerate(bins):
                print(f'batch={idx} | n_tokens={sum(n_tokens[m] for m in members)} | members={members}')
        
        return bins
    
    async def abatched_forward(self, messages, tokens_per_batch=1024, max_concurrent=1500, **kwargs):
        n_tokens = self.model.messages2tokens.remote(messages)
        if max(n_tokens) > tokens_per_batch:
            print(f"WARNING: tokens_per_batch={tokens_per_batch} | max(n_tokens)={max(n_tokens)}")
        
        batch_idxs = self._compute_batches(n_tokens, tokens_per_batch)
        
        # --
        # Process all batch_idxs in parallel across different machines
        # [TODO] use map instead? but it's more annoying to pass args, etc
        
        sem = asyncio.Semaphore(max_concurrent)
        async def _process_batch(idxs):
            async with sem:
                batch_res = await self.aforward(messages=[messages[i] for i in idxs], **kwargs)
                return [(idx, res) for idx, res in zip(idxs, batch_res)]

        tasks = [_process_batch(idxs) for idxs in batch_idxs]
        pbar  = tqdm(total=len(messages))
        for task in asyncio.as_completed(tasks):
            for idx, res in (await task):
                yield idx, res
                pbar.update(1)