# SimpleVLLM - A very simple unsloth + vllm wrapper to allow for easy generation locally.

> "Does what is says on the tin." - good friend.

## Getting Started:

1. Installation:
```bash
pip3 install simple_vllm
```
**Note: this will take time, as you are install vllm and unsloth.**

2. Running:
```python
from simple_vllm import get_vllm_generator

# Create the client.
llm = get_vllm_generator(model_name="Qwen/Qwen2.5-3B-Instruct")

# Get the response as list of str
responses = llm("Hello, could you tell me how to become a better human")

print(responses[0])
```

---
## More Advanced examples and parameters:
```python
from simple_vllm import get_vllm_generator

# Create the client.
llm = get_vllm_generator(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    max_seq_length = 4096,
    gpu_memory_utilization=0.5,
    load_in_4bit = True,
)

# Create a messages object in OpenAI format.
messages = [
    {
        "role" : "user",
        "content" : "Hello, could you tell me how to become a better human?",
    },
]

# Get the response as list of str
responses = llm(
    messages, 
    num_generations=1,
    max_completion_tokens=100,
    temperature=0.0,
    # other params are also possible have a look inside: simple_vllm/llm.py
    )

# print response.
print(responses[0])

# generate again just using a string input this time, without having to reload the model.
responses = llm("tell me something about life.", max_completion_tokens=10)

print(responses[0])
```

---
### (C) - Nikolai Rozanov, 2025 - Present