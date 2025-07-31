"""
    jlai.embed.tei
    
    LOCAL ENTRYPOINT
"""
import json
from time import time
import numpy as np
from rich import print as rprint

from .common import app
from .remote import TextEmbeddingsInference

def embed_dataset(input_strs, batch_size=512):
    input_str_chunks = [
        input_strs[i:i+batch_size] for i in range(0, len(input_strs), batch_size)
    ]
    
    model = TextEmbeddingsInference()
    
    embeddings = []
    for resp in model.embed.map(input_str_chunks, order_outputs=True):
        if isinstance(resp, Exception):
            # [BKJ] i haven't actually hit an error yet, AFAICT
            print('!' * 100)
            print(f"Exception: {resp}")
            print('!' * 100)
            continue

        embeddings.append(resp)

    return np.row_stack(embeddings)


def __get_test_data(n_replicates: int = 10):
    return [
        "The fox jumped over the lazy dog.",
        "The bear went to the market.",
        "The cat sat on the mat.",
        "The dog chased the cat.",
        "The mouse ran away from the cat.",
        "The bird flew over the tree.",
        "The fish swam in the sea.",
        "The elephant went to the forest.",
    ] * n_replicates


@app.local_entrypoint()
def main(inpath: str = None, outpath: str = None, test: bool = False):
    
    if test:
        assert (inpath is None) and (outpath is None)
        input_strs = __get_test_data()
    else:
        assert (inpath is not None) and (outpath is not None)
        input_strs = [json.loads(line)['text'] for line in open(inpath)]
    
    t       = time()
    results = embed_dataset(input_strs=input_strs)
    elapsed = time() - t
    
    rprint({
        'results':    results.shape,
        'elapsed':    elapsed,
        'throughput': len(input_strs) / elapsed,
    })
    
    np.save(outpath, results)
    
