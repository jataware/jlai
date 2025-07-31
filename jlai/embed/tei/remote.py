"""
    jlai.embed.tei.remote
"""
import os
import asyncio
import subprocess

import modal

from .common import app

MODEL_CACHE_VOLUME = modal.Volume.from_name("embedding-model-cache", create_if_missing=True)
MODEL_CACHE_DIR    = "/model"

gpu_type       = os.getenv("JLAI_GPU_TYPE", "A10G")
max_containers = int(os.getenv("JLAI_MAX_CONTAINERS", "1"))

tei_image = (
    modal.Image.from_registry(
        "ghcr.io/huggingface/text-embeddings-inference:86-1.7",
        add_python="3.12",
    )
    .dockerfile_commands("ENTRYPOINT []")
    .pip_install("httpx", "numpy")
)

with tei_image.imports():
    import os
    import socket
    import numpy as np
    from time import sleep
    from httpx import AsyncClient
    

def _start_hf_tei_server(batch_size: int, model_id: str) -> subprocess.Popen:
    process = subprocess.Popen(["text-embeddings-router"] + [
        "--hf-token",              os.environ["HF_TOKEN"],
        "--model-id",              model_id,
        "--port",                  "8000",
        "--max-client-batch-size", str(batch_size),
        "--max-batch-tokens",      str(batch_size * 512),
        "--huggingface-hub-cache", MODEL_CACHE_DIR,
    ])
    
    while True:
        try:
            socket.create_connection(("127.0.0.1", 8000), timeout=1).close()
            print("Webserver ready!")
            return process
        except (socket.timeout, ConnectionRefusedError):
            retcode = process.poll()
            if retcode is not None:
                raise RuntimeError(f"launcher exited unexpectedly with code {retcode}")

@app.cls(
    image            = tei_image,
    gpu              = gpu_type,
    max_containers   = max_containers,
    retries          = 3,
    secrets          = [modal.Secret.from_name("huggingface-secret")],
    volumes          = {
        MODEL_CACHE_DIR: MODEL_CACHE_VOLUME,
    },
)
@modal.concurrent(max_inputs=10)
class TextEmbeddingsInference:
    
    batch_size : int = modal.parameter(default=512)
    model_id   : str = modal.parameter(default="Qwen/Qwen3-Embedding-0.6B")
        
    @modal.enter()
    def open_connection(self):
        self.process = _start_hf_tei_server(
            batch_size = self.batch_size, 
            model_id   = self.model_id,
        )
        self.client  = AsyncClient(base_url="http://127.0.0.1:8000", timeout=30)

    @modal.exit()
    def terminate_connection(self):
        self.process.terminate()

    async def _embed(self, chunk_batch):
        res = await self.client.post("/embed", json={"inputs": chunk_batch})
        return res.json()

    @modal.method()
    async def embed(self, input_strs: list[str]):
        """Embeds a list of strings"""
        
        # make sure batch size isn't too big
        coros = [
            self._embed(input_strs[i:i+self.batch_size]) for i in range(0, len(input_strs), self.batch_size)
        ]

        out = await asyncio.gather(*coros)
        return np.row_stack(out, dtype=np.float32)
