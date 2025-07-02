import os
import json

import torch
from transformers import pipeline


from dotenv import load_dotenv


MODEL_NAME = "openai/clip-vit-base-patch32"  # OpenAI CLIP model

# MODEL_DIR = "../.data/hf/models/"
# model_path = os.path.join(MODEL_DIR, MODEL_NAME)

# Load environment variables from .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# Show loaded HuggingFace cache environment variables
print("HF_HOME:", os.environ.get("HF_HOME"))
print("HF_DATASETS_CACHE:", os.environ.get("HF_DATASETS_CACHE"))



# NOTE: If you see a torch.load security error, you must:
# 1. Upgrade torch to >=2.6, or
# 2. Convert your model weights to .safetensors format, or
# 3. Remove 'local_files_only=True' to load from the HuggingFace Hub (requires internet).
clip = pipeline(
   task="zero-shot-image-classification",
   model=MODEL_NAME,
   torch_dtype=torch.bfloat16,
   device=0,
   use_fast=False,
   # local_files_only=True,  # Remove or comment out this line if you want to load from the Hub
)

labels = ["a photo of a cat", "a photo of 2 cats", "two cats sleeping with 2 remote controllers", "there are 2 tv remote controllers", "a photo of a dog", "a photo of a car"]
result = clip("http://images.cocodataset.org/val2017/000000039769.jpg", candidate_labels=labels)
print(json.dumps(result, indent=2, ensure_ascii=False))


labels = ["a photo of a cat", "a photo of a dog", "a photo of a car"]
result = clip("http://images.cocodataset.org/val2017/000000039769.jpg", candidate_labels=labels)
print(json.dumps(result, indent=2, ensure_ascii=False))


labels = ["cat", "pink sofa", "tv remote controller", "dog", "car"]
result = clip("http://images.cocodataset.org/val2017/000000039769.jpg", candidate_labels=labels)
print(json.dumps(result, indent=2, ensure_ascii=False))

"""
Device set to use cuda:0
[
  {
    "score": 1.0,
    "label": "two cats sleeping with 2 remote controllers"
  },
  {
    "score": 0.00102996826171875,
    "label": "there are 2 tv remote controllers"
  },
  {
    "score": 0.000553131103515625,
    "label": "a photo of 2 cats"
  },
  {
    "score": 2.753734588623047e-05,
    "label": "a photo of a cat"
  },
  {
    "score": 1.4435499906539917e-07,
    "label": "a photo of a car"
  },
  {
    "score": 8.754432201385498e-08,
    "label": "a photo of a dog"
  }
]
[
  {
    "score": 0.9921875,
    "label": "a photo of a cat"
  },
  {
    "score": 0.005218505859375,
    "label": "a photo of a car"
  },
  {
    "score": 0.0024566650390625,
    "label": "a photo of a dog"
  }
]
[
  {
    "score": 0.62109375,
    "label": "cat"
  },
  {
    "score": 0.294921875,
    "label": "pink sofa"
  },
  {
    "score": 0.07421875,
    "label": "tv remote controller"
  },
  {
    "score": 0.0047607421875,
    "label": "car"
  },
  {
    "score": 0.0037078857421875,
    "label": "dog"
  }
]
"""


print("------------- testing CLIPModel from HuggingFace Hub -------------")

from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained(MODEL_NAME, torch_dtype="auto", device_map="auto")
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

# Check embedding dimensions from model config
print("Model config embedding dimensions:")
print(f"Text embedding dim: {model.config.text_config.hidden_size}")
print(f"Vision embedding dim: {model.config.vision_config.hidden_size}")
print(f"Projection dim: {model.config.projection_dim}")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

# Move inputs to the same device as the model
inputs = {k: v.to(model.device) for k, v in inputs.items()}

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

print("Logits per image:", logits_per_image)
print("Probabilities:", probs)

# Check actual embedding dimensions from outputs
print("\nActual embedding dimensions from outputs:")
print(f"Text embeddings shape: {outputs.text_embeds.shape}")
print(f"Image embeddings shape: {outputs.image_embeds.shape}")
print(f"Logits per image shape: {outputs.logits_per_image.shape}")





"""

------------- testing CLIPModel from HuggingFace Hub -------------
\Model config embedding dimensions:
Text embedding dim: 512
Vision embedding dim: 768
Projection dim: 512
Logits per image: tensor([[24.5701, 19.3049]], device='cuda:0', grad_fn=<TBackward0>)
Probabilities: tensor([[0.9949, 0.0051]], device='cuda:0', grad_fn=<SoftmaxBackward0>)

Actual embedding dimensions from outputs:
Text embeddings shape: torch.Size([2, 512])
Image embeddings shape: torch.Size([1, 512])
Logits per image shape: torch.Size([1, 2])


Your model config output shows:

Text embedding dimension: 512
Vision embedding dimension: 768 (before projection)
Projection dimension: 512 (final embedding size for both image and text after projection)
The actual output shapes confirm:

Text embeddings: 2 (labels) × 512 (projection dim)
Image embeddings: 1 × 512 (projection dim)
Logits per image: 1 × 2 (image vs. each label)
This means both image and text are projected to the same 512-dimensional space for similarity comparison.
"""