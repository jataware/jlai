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
    
    def _names_filter(layer_name):
        import re
        pattern = re.compile(r'blocks\.(\d+)\.hook_resid_post')
        match   = pattern.match(layer_name)
        if match:
            return int(match.group(1)) >= 25
        
        return False
    
    client = LensClient(
        model_str    = "Qwen/Qwen3-0.6B",
        padding_side = "right",
        devel        = True,
    )

    async for result in client.abatched_forward(messages, tokens_per_batch=51, names_filter = _names_filter):
        print(result)