from simple_vllm import get_vllm_generator

if __name__== "__main__":
    # Create the client.
    llm = get_vllm_generator(model_name="Qwen/Qwen2.5-3B-Instruct")

    # Get the response as list of str
    responses = llm("Hello, could you tell me how to become a better human")

    print(responses[0])

# from simple_vllm import get_vllm_generator
# client = get_vllm_generator(model_name="Qwen/Qwen2.5-3B-Instruct")
# responses = client.generate("Hello, could you tell me how to become a better human")
# print(responses[0])