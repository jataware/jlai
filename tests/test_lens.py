import asyncio
from .local import LensClient

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
    )

    async for result in client.abatched_forward(messages, tokens_per_batch=51):
        print(result)

if __name__ == "__main__":
    asyncio.run(main())
