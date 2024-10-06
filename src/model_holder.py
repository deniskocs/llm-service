import time
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# pylint: disable=too-few-public-methods
class ModelHolder:
    device: torch.device

    def __init__(self, model: str = "mistralai/Mistral-7B-Instruct-v0.3") -> None:
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model.to(self.device)

    def generate_text(self, history: List[Dict[str, str]]) -> str:
        print("\n--------------------------\n")
        for element in history:
            print(f"role: {element['role']}\n")
            print(f"content:\n{element['content']}\n")

        model_inputs = self.tokenizer.apply_chat_template(history, return_tensors="pt").to(self.device)

        print("\n--------------------------\n")
        print(f"Number of received tokens: {model_inputs.size()[1]}")

        start_time = time.time()
        generated_ids = self.model.generate(
            model_inputs,
            max_new_tokens=8192,
            # do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        end_time = time.time()

        decoded_input = self.tokenizer.decode(model_inputs[0], skip_special_tokens=True)
        prompt_len = len(decoded_input)
        output_tokens_len = len(generated_ids[0])
        decoded: str = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        output: str = decoded[prompt_len + 1:]

        generation_time = end_time - start_time
        tokens_per_second = output_tokens_len / generation_time
        print(f"\nTokens per second: {tokens_per_second}, {output_tokens_len}")

        print("\n------ Output ------\n")
        print(output)
        print("\n\n\n\n\n")
        return output
