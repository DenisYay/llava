import runhouse as rh
from transformers import pipeline
from PIL import Image
import requests
import torch


class LlavaModel(rh.Module):
    def __init__(self, model_id="llava-hf/llava-1.5-7b-hf", **model_kwargs):
        super().__init__()
        self.model_id, self.model_kwargs = model_id, model_kwargs
        self.model = None

    def load_model(self):
        self.model = pipeline("image-to-text",
                              model=self.model_id,
                              device_map="auto",
                              torch_dtype=torch.bfloat16,
                              model_kwargs=self.model_kwargs)

    def predict(self, img_path, prompt, **inf_kwargs):
        if not self.model:
            self.load_model()
        with torch.no_grad():
            image = Image.open(requests.get(img_path, stream=True).raw)
            return self.model(image, prompt=prompt, generate_kwargs=inf_kwargs)[0]["generated_text"]


if __name__ == "__main__":
    gpu = rh.cluster(name="rh-a10x", instance_type="A10G:1")
    remote_llava_model = LlavaModel(load_in_4bit=True).get_or_to(system=gpu,
                                                                 env=rh.env(["transformers==4.36.0"],
                                                                            working_dir="local:./"),
                                                                 name="llava-model")
    ans = remote_llava_model.predict(img_path="https://upcdn.io/kW15bGw/raw/uploads/2023/09/22/file-387X.png",
                                     prompt="USER: <image>\nHow would I make this dish? Step by step please."
                                            "\nASSISTANT:",
                                     max_new_tokens=200)
    print(ans)
