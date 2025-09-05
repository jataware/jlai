#!/usr/bin/env python
"""
    jlai.infer.sgl.local
"""

import modal

from .common import app
from .remote import SGLInference

class SGLClient:
    def __init__(self, model_str, devel=False):
        if devel:
            _model_cls = SGLInference
        else:
            _model_cls = modal.Cls.from_name(app.name, 'SGLInference')

        self.model_str = model_str
        self.model     = _model_cls(model_str=self.model_str)

    def completion(self, body):
        return self.model.completion.remote(body=body)

    async def acompletion(self, body):
        return await self.model.completion.remote.aio(body=body)
    
    def batch_completion(self, bodies):
        for output in self.model.completion.map(bodies, order_outputs=True):
            yield output
    
    async def batch_acompletion(self, bodies):
        async for output in self.model.completion.map.aio(bodies, order_outputs=True):
            yield output