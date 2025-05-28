# SimpleVLLM - A very simple unsloth + vllm wrapper to allow for easy generation locally.

> "Does what is says on the tin." - By a good friend.

## Getting Started:

1. Installation:
```bash
pip3 install easy_vllm
```
**Note: this will take time, as you are install vllm and unsloth.**

2. Running:
```python
from simple_vllm import get_vllm_generator

# Create the client.
client = get_vllm_generator(model_name="Qwen/Qwen2.5-3B-Instruct")

# Get the response as list of str
responses = client.generate("Hello, could you tell me how to become a better human")

print(responses[0])
```

---
## More Advanced examples and parameters:
```python
from simple_vllm import get_vllm_generator

# Create the client.
client = get_vllm_generator(
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
responses = client.generate(
    messages, 
    num_generations=1,
    max_completion_tokens=100,
    temperature=0.0,
    # other params are also possible have a look inside: simple_vllm/llm.py
    )

# print response.
print(responses[0]) #this will print the model output.
```

---
### (C) - Nikolai Rozanov, 2025 - Present