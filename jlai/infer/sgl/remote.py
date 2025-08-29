import os
import time
from pathlib import Path
from typing import Optional
from uuid import uuid4

import modal
from .common import app

gpu_type       = os.getenv("JLAI_GPU_TYPE", "A10G")
max_containers = int(os.getenv("JLAI_MAX_CONTAINERS", "1"))

GPU_TYPE      = os.environ.get("GPU_TYPE", "l40s")
GPU_COUNT     = os.environ.get("GPU_COUNT", 1)
GPU_CONFIG    = f"{GPU_TYPE}:{GPU_COUNT}"
SGL_LOG_LEVEL = "info"  # try "debug" or "info" if you have issues
MINUTES       = 60  # seconds

MODEL_PATH          = "qwen/qwen2.5-0.5b-instruct"
MODEL_REVISION      = "a7a06a1cc11b4514ce9edcde0e3ca1d16e5ff2fc"

MODEL_CACHE_VOLUME = modal.Volume.from_name("sgl-cache", create_if_missing=True)
MODEL_CACHE_PATH   = Path("/models")

VOLUMES = {MODEL_CACHE_PATH: MODEL_CACHE_VOLUME}

# --
# Define container

def _download_model():
    from huggingface_hub import snapshot_download
    snapshot_download(
        MODEL_PATH,
        local_dir=str(MODEL_CACHE_PATH / MODEL_PATH),
        ignore_patterns=["*.pt", "*.bin"],
    )

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
    # .run_function(
    #     _download_model, volumes=VOLUMES   
    # )
)

with _image.imports():
    import openai

# --
# Run

@app.cls(
    image            = _image,
    gpu              = gpu_type,
    max_containers   = max_containers,
    retries          = 3,
    secrets          = [modal.Secret.from_name("huggingface-secret")],
    volumes          = VOLUMES,
    # enable_memory_snapshot=True,
    # experimental_options={"enable_gpu_snapshot": True}
)
@modal.concurrent(max_inputs=10)
class SGLInference:
    @modal.enter()
    def _on_enter(self):
        from sglang.test.doc_patch import launch_server_cmd
        from sglang.utils import wait_for_server

        self.server_process, self.port = launch_server_cmd(
            f"python3 -m sglang.launch_server --model-path {MODEL_PATH} --host 0.0.0.0"
        )

        wait_for_server(f"http://localhost:{self.port}")
        print(f"Server started on http://localhost:{self.port}")

    @modal.method()
    async def completion(self, messages: list[dict], completion_kwargs: Optional[dict] = None):
        completion_kwargs = completion_kwargs or {}
        
        return (
            openai.Client(base_url=f"http://127.0.0.1:{self.port}/v1", api_key="None")
            .chat
            .completions
            .create(
                model       = MODEL_PATH,
                messages    = messages,
                **completion_kwargs,
            )
        )

    @modal.exit()
    def _on_exit(self):
        from sglang.utils import terminate_process
        terminate_process(self.server_process)
