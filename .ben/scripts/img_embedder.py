
# scripts/img_embedder.py
"""
Module for computing image embeddings using Hugging Face vision models.
"""
import torch
from torch import Tensor
from typing import List
from transformers import AutoFeatureExtractor, AutoModel
from PIL import Image


class ImgEmbedder:
    """
    Wraps a Hugging Face vision model and feature extractor to compute embeddings.
    """
    def __init__(
        self,
        model_name: str,
        device: torch.device = None
    ) -> None:
        """
        Initialize the embedder.

        Args:
            model_name: HF model identifier.
            device: torch.device (defaults to GPU if available).
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        model_mb = total_params * 4 / (1024 ** 2)
        print(f"Loaded model '{model_name}' with {total_params:,} params ({model_mb:.2f} MB)")

    def embed_batch(self, images: List[Image.Image]) -> Tensor:
        """
        Compute embeddings for a batch of PIL images.

        Returns:
            Tensor of shape (batch_size, hidden_size).
        """
        inputs = self.extractor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Return the CLS token embeddings only. CLS token is the first token in the sequence, so we take [:, 0, :]
        # This is a common practice to get a fixed-size representation of the input image
        return outputs.last_hidden_state[:, 0, :].cpu()

