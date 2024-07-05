from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LanguageModel:
    def __init__(self, model_id, use_quantization=False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            quantization_config=self._get_quantization_config() if use_quantization else None,
            device_map="auto"
        )

    def _get_quantization_config(self):
        from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

    def generate_response(self, prompt, max_new_tokens=256):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0])

    def format_prompt(self, query, context_items):
        context = "- " + "\n- ".join([item["chunk_text"] for item in context_items])
        base_prompt = f"""Based on the following context items, please answer the query.
        Give yourself room to think by extracting relevant passages from the context before answering the query.
        Don't return the thinking, only return the answer.
        Make sure your answers are as explanatory as possible.

        Context items:
        {context}

        User query: {query}
        Answer:"""
        return base_prompt