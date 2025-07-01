import torch, requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# --- 2-A  pick a demo picture ---
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png"
img = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# --- 2-B  load processor + model ---
MODEL_ID = "google/vit-base-patch16-224"         


processor = AutoImageProcessor.from_pretrained(MODEL_ID)  # ViTImageProcessor in ≥4.30 :contentReference[oaicite:2]{index=2}
vit        = AutoModel.from_pretrained(MODEL_ID,
                                       torch_dtype="auto",
                                       device_map="auto")                 # CPU-or-CUDA auto-placement

# --- 2-C  forward pass ---
inputs  = processor(images=img, return_tensors="pt").to(vit.device)
with torch.no_grad():
    out = vit(**inputs)
cls_emb = out.pooler_output         # shape (1, 768) for this backbone :contentReference[oaicite:3]{index=3}
print("CLS embedding size:", cls_emb.shape)
