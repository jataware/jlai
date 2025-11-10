#!/usr/bin/env python3
"""
    jlai.infer.sgl.remote
"""

import os
from pathlib import Path
from typing import Optional

import modal
from .common import app

gpu_type       = os.getenv("JLAI_GPU_TYPE", "A10G")
max_containers = int(os.getenv("JLAI_MAX_CONTAINERS", "1"))

MODEL_CACHE_VOLUME = modal.Volume.from_name("sgl-cache", create_if_missing=True)
MODEL_CACHE_PATH   = Path("/models")
VOLUMES            = {MODEL_CACHE_PATH: MODEL_CACHE_VOLUME}

# --
# Define container

_image = (
    modal.Image.from_registry(f"nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("libnuma-dev")
    .pip_install( # TODO - pin versions
        "huggingface_hub",
        "sglang[all]",
        "hf-xet",
        "openai",
    )
    .env(
        {
            "HF_HOME"                   : str(MODEL_CACHE_PATH),
            "HF_HUB_ENABLE_HF_TRANSFER" : "1",
        }
    )
)

with _image.imports():
    import openai
    from sglang.test.doc_patch import launch_server_cmd
    from sglang.utils import wait_for_server, terminate_process

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
class SGLInference:
    model_str   : str = modal.parameter(default="qwen/qwen2.5-0.5b-instruct")

    @modal.enter()
    def _on_enter(self):
        if 'gpt-oss' in self.model_str:
            self.server_process, self.port = launch_server_cmd(
                f"python3 -m sglang.launch_server"
                f" --model-path {self.model_str}"
                f" --tool-call-parser gpt-oss"
                f" --reasoning-parser gpt-oss"
                f" --host 0.0.0.0"
            )
        
        else:
            self.server_process, self.port = launch_server_cmd(
                f"python3 -m sglang.launch_server --model-path {self.model_str} --host 0.0.0.0"
            )

        wait_for_server(f"http://localhost:{self.port}")
        print(f"Server started on http://localhost:{self.port}")

    @modal.method()
    async def completion(self, body: dict):
        client = openai.AsyncOpenAI(base_url=f"http://127.0.0.1:{self.port}/v1", api_key="None")
        return await client.chat.completions.create(model=self.model_str, **body)

    @modal.exit()
    def _on_exit(self):
        terminate_process(self.server_process)
