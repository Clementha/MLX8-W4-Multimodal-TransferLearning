# scripts/img_embedder.py
"""
Module for computing image embeddings using Hugging Face vision models.
Supports both CLIP (vision‐only) and ViT.
"""
import torch
from torch import Tensor
from typing import List
from PIL import Image
from transformers import (
  CLIPImageProcessor,
  CLIPVisionModelWithProjection,
  ViTImageProcessor,
  ViTModel,
)

class ImgEmbedder:
  """
  Wraps a HF vision model to produce a (B, D_img) embedding tensor.
  """
  def __init__(
    self,
    model_name: str,
    choice: str = "ViT"
  ) -> None:
    """
    Args:
      model_name: HF repo name (e.g. "openai/clip-vit-base-patch32" or "google/vit-base-patch16-224-in21k").
      choice:     "CLIP" or "ViT".
    """
    self.choice = choice.upper()
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ImgEmbedder] loading {self.choice} vision model on {self.device}")

    if self.choice == "CLIP":
      # vision‐only CLIP (with projection head)
      self.processor = CLIPImageProcessor.from_pretrained(model_name)
      self.model     = CLIPVisionModelWithProjection.from_pretrained(model_name)
    else:
      # standard ViT
      self.processor = ViTImageProcessor.from_pretrained(model_name)
      self.model     = ViTModel.from_pretrained(model_name)

    self.model.to(self.device).eval()
    # Freeze weights
    for p in self.model.parameters():
      p.requires_grad_(False)

  def embed_batch(self, images: List[Image.Image]) -> Tensor:
    """
    Preprocess a batch of PIL images and return embeddings.
    """
    # 1) preprocess
    inputs = self.processor(images=images, return_tensors="pt").to(self.device)

    with torch.no_grad():
      if self.choice == "CLIP":
        # CLIPVisionModelWithProjection.forward returns .image_embeds
        raw    = self.model(**inputs)
        embeds = raw.image_embeds     # shape (B, projection_dim)
      else:
        # ViTModel returns pooler_output
        raw    = self.model(**inputs)
        embeds = raw.pooler_output    # shape (B, hidden_size)

    return embeds.cpu()

