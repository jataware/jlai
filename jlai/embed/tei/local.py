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

import modal

from .common import app
from .remote import TextEmbeddingsInference

def _get_chunks(x, bs):
    return [x[i:i+bs] for i in range(0, len(x), bs)]

def embed_dataset(input_strs, batch_size=512, mode='deploy'):
    if mode == 'run':
        model = TextEmbeddingsInference()
    elif mode == 'deploy':
        model = modal.Cls.from_name(app.name, 'TextEmbeddingsInference')()
    
    mask        = np.array([len(s) > 0 for s in input_strs])
    _input_strs = [x for i, x in enumerate(input_strs) if mask[i]]
    
    embeddings = []
    for (chunk_embs, success) in model.embed.map(
        _get_chunks(_input_strs, batch_size),
        order_outputs=True
    ):
        embeddings.append(chunk_embs)

    # fix bad embeddings
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
        assert (inpath is not None) and (outpath is not None), "inpath and outpath must be provided"
        input_strs = [json.loads(line)['text'].strip() for line in open(inpath)]
    
    t       = time()
    results = embed_dataset(input_strs=input_strs, mode='dev')
    elapsed = time() - t
    
    rprint({
        'results':    results.shape,
        'elapsed':    elapsed,
        'throughput': len(input_strs) / elapsed,
    })
    
    if not test:
        np.save(outpath, results)

