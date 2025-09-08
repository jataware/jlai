from jlai.infer.sgl import SGLClient

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

client = SGLClient(model_str="qwen/qwen3-8b")

for output in client.batch_completion(bodies):
    print(output)
