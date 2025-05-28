from simple_vllm import get_vllm_generator


if __name__=="__main__":
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
        )

    # print response.
    print(responses[0]) #this will print the model output.