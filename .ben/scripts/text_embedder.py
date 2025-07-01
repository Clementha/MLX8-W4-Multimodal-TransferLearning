import torch
from torch import Tensor
from typing import List
from transformers import AutoTokenizer, AutoModel

class TextEmbedder:
    """
    Provides token embeddings from a pretrained HF text model.
    """
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: torch.device = None
    ) -> None:
        # 1) Decide device once
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 2) Load & store the tokenizer (for collate_fn)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # 3) Load the full model, grab its embedding layer, move to self.device
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.embed_layer = model.get_input_embeddings().to(self.device)

        # 4) Discard the rest of the model immediately
        del model

        print(
            f"Loaded text embedder '{model_name}' "
            f"(vocab={self.embed_layer.num_embeddings}, "
            f"dim={self.embed_layer.embedding_dim}) on {self.device}"
        )

    def tokenize(self, captions: List[str], max_length: int = 32) -> dict:
        """
        Turn list of captions into input_ids & attention_mask.
        """
        return self.tokenizer(
            captions,
            padding="max_length", # add padding tokens up to max_length
            truncation=True, # Cut off seq's longer than max_length
            max_length=max_length, # Maximum length of the sequence
            return_tensors="pt",
            add_special_tokens=True  # Add special tokens like [CLS] and [SEP]
        )

    def embed_tokens(self, input_ids: Tensor) -> Tensor:
        """
        Look up embeddings for token IDs.
        """
        # move IDs → same device as embedding matrix
        input_ids = input_ids.to(self.device)
        # fetch embeddings and move them back to CPU
        return self.embed_layer(input_ids).cpu()
