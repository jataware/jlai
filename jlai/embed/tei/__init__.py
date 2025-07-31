"""
    jlai.embed.tei
    
    LOCAL ENTRYPOINT
    
    [TODO] - empty strings - causes an error
    [TODO] - long strings?  what happens?
"""

import json
from time import time
import numpy as np
from rich import print as rprint

from .common import app
from .remote import TextEmbeddingsInference

def embed_dataset(input_strs, batch_size=512):
    mask = np.array([len(s) > 0 for s in input_strs])
    
    input_str_chunks = [
        input_strs[i:i+batch_size] for i in range(0, len(input_strs), batch_size)
    ]
    
    model = TextEmbeddingsInference()
    
    embeddings = []
    for (chunk_embs, success) in model.embed.map(input_str_chunks, order_outputs=True):
        if isinstance(chunk_embs, Exception):
            # [BKJ] i haven't actually hit an error yet, AFAICT
            print('!' * 100)
            print(f"Exception: {chunk_embs}")
            print('!' * 100)
            continue

        embeddings.append(chunk_embs)

    _out      = np.vstack(embeddings)
    out       = np.zeros((mask.shape[0], _out.shape[1]), dtype=np.float32)
    out[mask] = _out
    return out


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
        input_strs = [json.loads(line)['text'].strip() for line in open(inpath)]
    
    t       = time()
    results = embed_dataset(input_strs=input_strs)
    elapsed = time() - t
    
    rprint({
        'results':    results.shape,
        'elapsed':    elapsed,
        'throughput': len(input_strs) / elapsed,
    })
    
    np.save(outpath, results)
    
