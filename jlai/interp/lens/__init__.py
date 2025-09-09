import re
from rich import print as rprint

from .common import app
from .local import LensClient


@app.local_entrypoint()
async def main():
    messages = [
        [
            {'role': 'user', 'content': 'What is the capital of France?'}
        ],
        [
            {'role': 'user', 'content': 'What is the capital of the most populous state of Germany?'}
        ],
        [
            {'role': 'user', 'content': 'What is the capital of France?'}
        ],
        [
            {'role': 'user', 'content': 'What is the capital of the most populous state of Germany?'}
        ],
        [
            {'role': 'user', 'content': 'What is the capital of France?'}
        ],
        [
            {'role': 'user', 'content': 'What is the capital of the most populous state of Germany?'}
        ],        
    ]
    
    
    client = LensClient(
        model_str    = "Qwen/Qwen3-0.6B",
        padding_side = "right",
        devel        = True,
    )

    def names_filter(layer_name):
        pattern = re.compile(r'blocks\.(\d+)\.hook_resid_post')
        match   = pattern.match(layer_name)
        if match:
            return int(match.group(1)) >= 25
        
        return False
    
    hook_names = client.model.hook_names.remote()
    hook_names = set([name for name in hook_names if names_filter(name)])
    rprint(hook_names)

    async for result in client.abatched_forward(messages, tokens_per_batch=51, return_logits=False, names_filter=hook_names):
        print(result)