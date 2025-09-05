from .common import app
from .local import run_inference

@app.local_entrypoint()
def main():
    all_messages = [
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
    
    bodies = [
        {
            "messages"     : messages,
            "temperature"  : 1.0,
            "max_tokens"   : 64,
            "logprobs"     : True,
            "top_logprobs" : 512,
        } for messages in all_messages
    ]
    
    outputs = run_inference(
        model_str    = "qwen/qwen2.5-0.5b-instruct",
        bodies       = bodies,
        mode         = "run",
    )

    print(outputs)