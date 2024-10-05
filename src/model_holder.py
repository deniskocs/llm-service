import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelHolder:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", device_map="auto",
                                                          torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

    def generate_text(self, history) -> str:
        print("\n--------------------------\n")
        for element in history:
            print(f"role: {element['role']}\n")
            print(f"content:\n{element['content']}\n")

        encodeds = self.tokenizer.apply_chat_template(history, return_tensors="pt")
        print("\n--------------------------\n")
        print(f"Number of received tokens: {encodeds.size()[1]}")

        model_inputs = encodeds.to('mps')

        start_time = time.time()
        generated_ids = self.model.generate(
            model_inputs,
            max_new_tokens=8192,
            # do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        end_time = time.time()

        input = self.tokenizer.decode(model_inputs[0], skip_special_tokens=True)
        prompt_len = len(input)
        output_tokens_len = len(generated_ids[0])
        decoded = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        output = decoded[prompt_len + 1:]

        generation_time = end_time - start_time
        tokens_per_second = output_tokens_len / generation_time
        print(f"\nTokens per second: {tokens_per_second}, {output_tokens_len}")

        print("\n------ Output ------\n")
        print(output)
        print("\n\n\n\n\n")
        return output

