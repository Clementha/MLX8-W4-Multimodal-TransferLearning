"""
img_embedder.py

Module for loading arbitrary Hugging Face vision models and extracting embeddings from images.
"""
import argparse
from pathlib import Path
from typing import List, Tuple

import torch
from torch import Tensor
from PIL import Image
import pandas as pd
from transformers import AutoFeatureExtractor, AutoModel


class Embedder:
    """
    Wraps a Hugging Face vision model and feature extractor to compute image embeddings.
    """

    def __init__(self, model_name: str, device: torch.device = None):
        """
        Initialize the embedder.

        Args:
            model_name: HF model identifier (e.g. 'google/vit-base-patch16-224-in21k').
            device: torch.device (defaults to GPU if available).
        """
        # Choose device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load feature extractor and model
        self.extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)

        # Print parameter counts
        total_params = sum(p.numel() for p in self.model.parameters())
        size_mb = total_params * 4 / (1024 ** 2)
        print(f"Loaded model '{model_name}' with {total_params:,} params ({size_mb:.2f} MB)")

    def embed_images(self, images: List[Image.Image]) -> Tensor:
        """
        Compute embeddings for a list of PIL images in a batch.

        Args:
            images: List of PIL.Image instances.

        Returns:
            Tensor of shape (batch_size, hidden_size).
        """
        # Preprocess images
        inputs = self.extractor(images=images, return_tensors="pt")
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Use CLS token representation
        # outputs.last_hidden_state shape: (B, seq_len, hidden_size)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        return cls_embeddings.cpu()


def main(metadata_path: Path, image_dir: Path, model_name: str):
    """
    Run embedding on first N images from metadata.
    """
    # Load metadata
    df = pd.read_parquet(metadata_path)

    # Take first 8 filenames
    filenames = df["filename"].tolist()[:8]
    images = [Image.open(image_dir / fn).convert("RGB") for fn in filenames]

    # Create embedder
    embedder = Embedder(model_name=model_name)

    # Compute embeddings
    embeddings = embedder.embed_images(images)
    print(f"Embeddings shape: {embeddings.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute image embeddings.")
    parser.add_argument(
        "--metadata", type=Path, default=Path("data/flickr30k/metadata.parquet"),
        help="Path to metadata.parquet"
    )
    parser.add_argument(
        "--imagedir", type=Path, default=Path("data/flickr30k/images"),
        help="Directory containing image files"
    )
    parser.add_argument(
        "--model", type=str, default="google/vit-base-patch16-224-in21k",
        help="Hugging Face vision model name"
    )
    args = parser.parse_args()
    main(args.metadata, args.imagedir, args.model)
