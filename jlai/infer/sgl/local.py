#!/usr/bin/env python
"""
    jlai.infer.sgl.local
"""

import modal
from .common import app

def run_inference(model_str, msgs, **kwargs):
    if isinstance(msgs[0], dict):
        msgs = [msgs]
    
    _model_cls = modal.Cls.from_name(app.name, 'SGLInference')
    model      = _model_cls(model_str=model_str)

    return model.completion.starmap([[msg, kwargs] for msg in msgs])

if __name__ == "__main__":
    msgs = [
        [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "What is the capital of France?"
            }
        ],
        [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "What is the capital of Uruguay?"
            }
        ]
    ]
    
    output_gen = run_inference(
        model_str    = "qwen/qwen2.5-0.5b-instruct",
        msgs         = msgs,
        temperature  = 1.0,
        max_tokens   = 64,
        logprobs     = True,
        top_logprobs = 512,
    )
    
    outputs = []
    for output in output_gen:
        print(output.choices[0].message.content)
        print(len(output.choices[0].logprobs.content[0].top_logprobs))
        print('-' * 100)
        outputs.append(output)

    breakpoint()

    outputs[0]