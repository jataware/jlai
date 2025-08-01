import json
import numpy as np
from time import time
from rich import print as rprint

from .common import app
from .local import embed_dataset

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

