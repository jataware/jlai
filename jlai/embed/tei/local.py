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

def embed_dataset(input_strs, batch_size=512, model_id='Qwen/Qwen3-Embedding-0.6B',mode='deploy'):
    assert mode in ['run', 'deploy']
    if mode == 'run':
        _model_cls = TextEmbeddingsInference
    elif mode == 'deploy':
        _model_cls = modal.Cls.from_name(app.name, 'TextEmbeddingsInference')
    
    model = _model_cls(model_id=model_id)
    
    mask        = np.array([len(s) > 0 for s in input_strs])
    _input_strs = [x for i, x in enumerate(input_strs) if mask[i]]
    
    embeddings = []
    chunks     = _get_chunks(_input_strs, batch_size)
    for (chunk_embs, success) in model.embed.map(
        chunks,
        order_outputs=True
    ):
        if mode == 'deploy':
            rprint(f'[yellow]embed_dataset: {len(embeddings) / len(chunks):.2f}[/yellow]', end='\r')
        
        embeddings.append(chunk_embs)

    # fix bad embeddings
    _out      = np.vstack(embeddings)
    out       = np.zeros((mask.shape[0], _out.shape[1]), dtype=np.float32)
    out[mask] = _out
    return out

