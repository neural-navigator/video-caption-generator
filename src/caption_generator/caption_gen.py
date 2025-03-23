import os
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from PIL import Image
from transformers.image_utils import load_image
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from transformers import AutoProcessor, AutoModelForVision2Seq
from utils.clean_up import cleanup

class CaptionGenerator:
    def __init__(self, model_name):
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model_name = model_name
        self._processor = None
        self._model = None

    def load_model(self):
        self._processor = AutoProcessor.from_pretrained(self._model_name)
        self._model = AutoModelForVision2Seq.from_pretrained(self._model_name,
                                                             torch_dtype=torch.bfloat16,
                                                             _attn_implementation="flash_attention_2").to(self._device)

    def generate_caption(self, frame_path, chunk_size=2):
        """This method will generate caption for a given """
        self.load_model()
        images_files = [os.path.join(frame_path, i) for i in os.listdir(frame_path)]
        image_data = [load_image(i) for i in images_files]
        image_batches = [image_data[i:i + chunk_size] for i in range(0, len(image_data), chunk_size)]
        captions = []
        for image_batch in image_batches:
            message = [
                {"role": "user",
                 "content": ([{"type": "image"}] * len(image_batch)) + [{
                     "type": "text",
                     "text": """Generate a concise, engaging caption for the provided image. 
                     The caption should be vivid, descriptive, and suitable for narration in a video. 
                     Focus on capturing the key elements, mood, or action in the image, keeping it 
                     under 30 words. """
                 }]
                 }
                ]

            prompt = self._processor.apply_chat_template(message, add_generation_prompt=True)
            inputs = self._processor(text=prompt, images=image_batch, return_tensors="pt")
            inputs = inputs.to(self._device)
            # print(self._processor.decode([49154, 0, 2]))
            with torch.no_grad():
                generated_ids = self._model.generate(**inputs, max_new_tokens=500)

                generated_texts = self._processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )
                captions.append(generated_texts[0].split("\nAssistant:")[-1])

        return captions
