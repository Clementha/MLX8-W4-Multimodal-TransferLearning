"""
Collate function for batching Flickr30k samples: tokenises captions on the fly.

Configure MAX_SEQ_LEN here; set BATCH_SIZE in your DataLoader.
"""
from typing import List, Tuple
from PIL import Image
import torch
from torch import Tensor
from text_embedder import TextEmbedder

# Maximum number of tokens per caption (including special tokens)
MAX_SEQ_LEN = 50



def collate_fn(
    batch: List[Tuple[Image.Image, str, str]],
    tokenizer,           # now passed in from main process
    max_seq_len: int
) -> Tuple[List[Image.Image], Tensor, Tensor, List[str]]:
    images, filenames, captions = zip(*batch)
    """
    Custom collate_fn to batch images and captions.

    Args:
        batch: List of tuples (PIL.Image, filename, caption_str).
        tokenizer: TextEmbedder instance for tokenising captions.
        max_seq_len: Maximum sequence length for tokenisation.
    Returns:
        images: List[PIL.Image] (length = batch size)
        input_ids: LongTensor of shape (batch_size, MAX_SEQ_LEN)
        attention_mask: LongTensor (batch_size, MAX_SEQ_LEN)
        filenames: List[str] for reference
    """

    # tokeniser is already in CPU or GPU, loaded once in main
    tokens = tokenizer(
        list(captions),
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt"
    )

    return list(images), tokens["input_ids"], tokens["attention_mask"], list(filenames), list(captions)
