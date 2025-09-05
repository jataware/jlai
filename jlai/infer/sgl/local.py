#!/usr/bin/env python
"""
    jlai.infer.sgl.local
"""

import modal

from .common import app
from .remote import SGLInference

def run_inference(model_str, bodies, mode='deploy', **kwargs):
    assert mode in ['run', 'deploy']
    if mode == 'run':
        _model_cls = SGLInference
    elif mode == 'deploy':
        _model_cls = modal.Cls.from_name(app.name, 'SGLInference')
    
    if isinstance(bodies, dict):
        bodies = [bodies]
    
    model = _model_cls(model_str=model_str)
    
    outputs = []
    for output in model.completion.map(
        bodies,
        order_outputs=True
    ):
        outputs.append(output)
    
    return outputs