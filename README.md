# A very simple unsloth + vllm wrapper to allow for easy generation locally.

## Getting Started:

1. Installation:
```bash
pip3 install vllm_sdk
```

2. Running:
```python
from vllm_sdk import get_vllm_generator

client = get_vllm_generator(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    max_seq_length = 4096,
    gpu_memory_utilization=0.5,
    load_in_4bit = True,
)

messages = [
    {
        "role" : "user",
        "content" : "Hello, could you tell me how to become a better human?",
    },
]

responses = client.generate(messages, num_generations=1)

print(responses[0]) #this will print the model output.
```

### (C) - Nikolai Rozanov, 2025 - Present