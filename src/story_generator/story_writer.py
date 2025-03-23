import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class StoryGenerator:
    def __init__(self, model_name):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model_name = model_name
        self._tokenizer = None
        self._model = None

    def load_model(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self._model_name)
        self._model.to(self._device)

    def generate_story(self, caption_str):
        message = [{"role": "user", "content": caption_str}]
        input_text = self._tokenizer.apply_chat_template(message, add_generation_prompt=True)
        inputs = self._tokenizer.encode(input_text, return_tensors="pt").to(self._device)

        with torch.no_grad():
            outputs = self._model.generate(inputs, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True)
            decoded_output = self._tokenizer.decode(outputs[0])
        return decoded_output
