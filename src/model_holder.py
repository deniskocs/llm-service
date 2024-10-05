import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel

class ModelHolder:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", device_map="auto", torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        # adapter_model_name = "/pretrained/general_purpose/adapter_model"
        adapter_model_name = "/pretrained/general_purpose"

        self.model.load_adapter(adapter_model_name)

    def generate_text(self, history) -> str:

        print("\n--------------------------\n")
        for element in history:
            print(f"role: {element['role']}\n")
            print(f"content:\n{element['content']}\n")

        encodeds = self.tokenizer.apply_chat_template(history, return_tensors="pt")
        print("\n--------------------------\n")
        print(f"Number of received tokens: {encodeds.size()[1]}")

        model_inputs = encodeds.to('cuda')

        generated_ids = self.model.generate(
            model_inputs,
            max_new_tokens=4096,
            # do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        input = self.tokenizer.decode(model_inputs[0], skip_special_tokens=True)
        prompt_len = len(input)
        decoded = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        output = decoded[prompt_len + 1:]

        print("\n------ Output ------\n")
        print(output)
        print("\n\n\n\n\n")
        return output

model_holder = ModelHolder()