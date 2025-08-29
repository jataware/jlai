import json
import openai
import numpy as np
from time import time
from rich import print as rprint

from .common import app

from .remote import SGLInference

def __get_test_data():
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ]


@app.local_entrypoint()
def main():
    msgs = __get_test_data()
    
    t = time()
    
    inputs = (
        [msgs, {"temperature": 0, "max_tokens": 64, "top_logprobs" : 100}],
    )
    
    outputs = []
    for output in SGLInference().completion.starmap(inputs, order_outputs=True):
        outputs.append(output)
    
    rprint(outputs)
    elapsed = time() - t
    
    # rprint(results)
    # rprint({
    #     'results':    results.shape,
    #     'elapsed':    elapsed,
    #     'throughput': len(msgs) / elapsed,
    # })