import asyncio
import subprocess
import numpy as np
from time import time

import modal

app = modal.App("example-embeddings")
MODEL_CACHE_VOLUME = modal.Volume.from_name("embedding-model-cache", create_if_missing=True)
MODEL_CACHE_DIR    = "/model"

NUM_GPUS        = 10
MODEL_ID        = "Qwen/Qwen3-Embedding-0.6B"
BATCH_SIZE      = 512

# >>
# TEI Image

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
    

def start_hf_tei_server() -> subprocess.Popen:
    process = subprocess.Popen(["text-embeddings-router"] + [
        "--hf-token",              os.environ["HF_TOKEN"],
        "--model-id",              MODEL_ID,
        "--port",                  "8000",
        "--max-client-batch-size", str(BATCH_SIZE),
        "--max-batch-tokens",      str(BATCH_SIZE * 512),
        "--huggingface-hub-cache", MODEL_CACHE_DIR,
    ])
    print('... sleeping ...')
    sleep(5)
    
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
    gpu              = "A10G",
    max_containers   = NUM_GPUS,
    retries          = 3,
    secrets          = [modal.Secret.from_name("huggingface-secret")],
    volumes          = {
        MODEL_CACHE_DIR: MODEL_CACHE_VOLUME,
    },
)
@modal.concurrent(max_inputs=10)
class TextEmbeddingsInference:
    @modal.enter()
    def open_connection(self):
        self.process = start_hf_tei_server()
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
            self._embed(input_strs[i:i+BATCH_SIZE]) for i in range(0, len(input_strs), BATCH_SIZE)
        ]

        out = await asyncio.gather(*coros)
        return np.row_stack(out, dtype=np.float32)

# <<

# >>
# Client Image

def embed_dataset(input_strs):
    input_str_chunks = [input_strs[i:i+BATCH_SIZE] for i in range(0, len(input_strs), BATCH_SIZE)]
    
    t = time()
    model = TextEmbeddingsInference()
    
    embeddings = []
    for resp in model.embed.map(
        input_str_chunks,
        order_outputs          = True,
        return_exceptions      = True,
    ):
        if isinstance(resp, Exception):
            print('!' * 100)
            print(f"Exception: {resp}")
            print('!' * 100)
            continue

        embeddings.append(resp)

    print('embed_dataset - elapsed: ', time() - t)
    out = np.row_stack(embeddings)
    print(out.shape, out.nbytes / 1024 / 1024, 'MB')
    return out

# <<

# >>
# Local

@app.local_entrypoint()
def full_job():
    
    input_strs = [
        "The fox jumped over the lazy dog.",
        "The bear went to the market.",
        "The cat sat on the mat.",
        "The dog chased the cat.",
        "The mouse ran away from the cat.",
        "The bird flew over the tree.",
        "The fish swam in the sea.",
        "The elephant went to the forest.",
    ] * 10000
    
    t = time()
    results = embed_dataset(input_strs=input_strs)
    elapsed = time() - t
    
    print(f"full_job - elapsed: {elapsed}")
    print(f"full_job - throughput: {len(input_strs) / elapsed} records/s")
    
    print('saving results...')
    np.save('results.npy', results)
    print('done')

# <<