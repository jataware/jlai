from jlai.infer.sgl import SGLClient
from rich import print as rprint

all_messages = [
    [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "What is the square root of 1234323?"
        }
    ],
    [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "What is 23452345 * 12341234?"
        }
    ]
]

bodies = [
    {
        "messages"     : messages,
        "tools"        : [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "A calculator tool",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string", "description": "The expression to evaluate"},
                        },
                        "required": ["expression"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
        # "temperature"  : 1.0,
        # "max_tokens"   : 64,
        # "logprobs"     : True,
        # "top_logprobs" : 512,
    } for messages in all_messages
]

client = SGLClient(model_str="openai/gpt-oss-20b")

for output in client.batch_completion(bodies):
    rprint(output)
