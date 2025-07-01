# scripts/text_embedder.py
"""
Module to fetch pretrained token embeddings for captions using BERT.
"""
import torch
from torch import Tensor
from typing import List
from transformers import AutoTokenizer, AutoModel


class TextEmbedder:
    """
    Provides token embeddings from a pretrained BERT embedding layer.
    """
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        device: torch.device = None
    ) -> None:
        """
        Initialize the text embedder.

        Args:
            model_name: HF model identifier for tokenizer & embeddings.
            device: torch.device to load embeddings onto (CPU/GPU).
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load tokenizer & model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        bert = AutoModel.from_pretrained(model_name)
        # grab just the embedding layer
        self.embed_layer = bert.get_input_embeddings().to(self.device)
        print(f"Loaded text embedder '{model_name}' with vocab size {self.embed_layer.num_embeddings} and hidden size {self.embed_layer.embedding_dim}")

    def tokenize(self, captions: List[str], max_length: int = 32) -> dict:
        """
        Tokenize a list of captions.

        Returns dict with 'input_ids' and 'attention_mask' tensors.
        """
        return self.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

    def embed_tokens(self, input_ids: Tensor) -> Tensor:
        """
        Look up embeddings for token IDs.

        Args:
            input_ids: Tensor of shape (B, seq_len)
        Returns:
            Tensor (B, seq_len, hidden_size)
        """
        # Keep on cpu for now to save gpu memory 
        input_ids = input_ids.to(self.device)
        return self.embed_layer(input_ids).cpu()
