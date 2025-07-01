# from PIL import Image
# import requests

import os
from dotenv import load_dotenv

# from transformers import CLIPProcessor, CLIPModel

MODEL_NAME = "openai/clip-vit-base-patch32"  # OpenAI CLIP model

# MODEL_DIR = "../.data/hf/models/"
# model_path = os.path.join(MODEL_DIR, MODEL_NAME)

# Load environment variables from .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# Show loaded HuggingFace cache environment variables
print("HF_HOME:", os.environ.get("HF_HOME"))
print("HF_DATASETS_CACHE:", os.environ.get("HF_DATASETS_CACHE"))
print("TRANSFORMERS_CACHE:", os.environ.get("TRANSFORMERS_CACHE"))

# model = CLIPModel.from_pretrained(MODEL_NAME, local_files_only=True, torch_dtype="auto", device_map="auto")
# processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

# outputs = model(**inputs)
# logits_per_image = outputs.logits_per_image # this is the image-text similarity score
# probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

# print("Logits per image:", logits_per_image)
# print("Probabilities:", probs)



import torch
from transformers import pipeline

# NOTE: If you see a torch.load security error, you must:
# 1. Upgrade torch to >=2.6, or
# 2. Convert your model weights to .safetensors format, or
# 3. Remove 'local_files_only=True' to load from the HuggingFace Hub (requires internet).
clip = pipeline(
   task="zero-shot-image-classification",
   model=MODEL_NAME,
   torch_dtype=torch.bfloat16,
   device=0,
   # local_files_only=True,  # Remove or comment out this line if you want to load from the Hub
)
labels = ["a photo of a cat", "a photo of a dog", "a photo of a car"]
result = clip("http://images.cocodataset.org/val2017/000000039769.jpg", candidate_labels=labels)

print("Result:", result)