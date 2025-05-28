# from typing import List, Dict, Any
# from dataclasses import dataclass
# from datetime import datetime

# @dataclass
# class Choice:
#     """Represents a single completion choice in the response."""
#     text: str
#     index: int
#     logprobs: Dict[str, Any] = None
#     finish_reason: str = None

# @dataclass
# class Usage:
#     """Represents token usage information."""
#     prompt_tokens: int
#     completion_tokens: int
#     total_tokens: int

# @dataclass
# class ResponseObject:
#     """Response object that mimics the OpenAI API response structure."""
#     id: str
#     object: str = "text_completion"
#     created: int = None
#     model: str = None
#     choices: List[Choice] = None
#     usage: Usage = None

#     def __init__(self, responses: List[str], model: str = "vllm"):
#         self.id = f"cmpl-{datetime.now().strftime('%Y%m%d%H%M%S')}"
#         self.created = int(datetime.now().timestamp())
#         self.model = model
#         self.choices = [
#             Choice(text=resp, index=idx)
#             for idx, resp in enumerate(responses)
#         ]
#         # Placeholder usage stats - you might want to get real values
#         self.usage = Usage(
#             prompt_tokens=0,
#             completion_tokens=0,
#             total_tokens=0
#         )

#     def __getitem__(self, key):
#         return self.choices[key].text

#     def __len__(self):
#         return len(self.choices)


def get_vllm_generator(        
        model_name = None ,
        max_seq_length = 1024,
        load_in_4bit = True, # False for LoRA 16bit
        load_in_8bit = False,    # A bit more accurate, uses 2x memory
        full_finetuning = False, # We have full finetuning now!
        fast_inference = True, # Enable vLLM fast inference
        max_lora_rank = 64,
        gpu_memory_utilization = 0.5, # Reduce if out of memory
        random_state=1234,
        **kwargs,
    ):
    """Gets a generator function to call vllm."""
    from unsloth import FastLanguageModel
    from vllm import SamplingParams

    if not model_name:
        raise ValueError("You need to specify a model name.")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name ,
        max_seq_length = max_seq_length,
        load_in_4bit = load_in_4bit, # False for LoRA 16bit
        load_in_8bit = False,    # A bit more accurate, uses 2x memory
        full_finetuning = False, # We have full finetuning now!
        fast_inference = True, # Enable vLLM fast inference
        max_lora_rank = max_lora_rank,
        gpu_memory_utilization = gpu_memory_utilization, # Reduce if out of memory
        **kwargs,
    )

    def generate(
            messages, 
            num_generations=1, 
            temperature=0.2, 
            max_completion_tokens=1024,
            top_p=0.95,
            lora_name=None,
            verbose=False,
            return_raw=False,
            **kwargs,
        ):
        assert max_completion_tokens < max_seq_length, f"max_completion_tokens exceeds the limit of the max_seq_length{max_seq_length} for the vllm model. Set a different param."

        text = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
        if verbose:
            print("\n\n----\nModel Input after tokenizer:")
            print(text)

        sampling_params = SamplingParams(
            n=num_generations,
            temperature = temperature,
            top_p = top_p,
            max_tokens = max_completion_tokens, #1024
            **kwargs,
        )
        if verbose:
            print("Generating Model output.")
        if lora_name:
            output = model.fast_generate(
                text,
                sampling_params = sampling_params,
                lora_request = model.load_lora(lora_name),
            )[0].outputs
        else:
            output = model.fast_generate(
                text,
                sampling_params = sampling_params,
            )[0].outputs            
        
        if verbose:
            print("Done.")
            print("Output is:\n\n----")
            print(output)
        
        # final_output = ResponseObject([x.text for x in output], model_name)
        final_output = [x.text for x in output]
        
        if return_raw:
            return final_output, output
        else:
            return final_output

    return generate