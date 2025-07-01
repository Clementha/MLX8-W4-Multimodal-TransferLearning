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

# Instantiate your text embedder (provides tokenizer)
text_embedder = TextEmbedder(model_name="bert-base-uncased")


def collate_fn(batch: List[Tuple[Image.Image, str, str]]) -> Tuple[List[Image.Image], Tensor, Tensor, List[str]]:
    """
    Custom collate_fn to batch images and captions.

    Args:
        batch: List of tuples (PIL.Image, filename, caption_str).
    Returns:
        images: List[PIL.Image] (length = batch size)
        input_ids: LongTensor of shape (batch_size, MAX_SEQ_LEN)
        attention_mask: LongTensor (batch_size, MAX_SEQ_LEN)
        filenames: List[str] for reference
    """
    # 1) Unzip batch
    images, filenames, captions = zip(*batch)

    # 2) Tokenise all captions at once
    tokens = text_embedder.tokenize(
        list(captions),
        max_length=MAX_SEQ_LEN
    )

    # 3) Extract tensors
    input_ids      = tokens["input_ids"]      # shape (B, MAX_SEQ_LEN)
    attention_mask = tokens["attention_mask"] # same shape, used to 0 out padding tokens so model doesn't attend to them

    return list(images), input_ids, attention_mask, list(filenames)

# ── Usage example in your training script:
#
# from torch.utils.data import DataLoader
# from scripts.dataset import Flickr30kDataset
# from scripts.collate import collate_fn
#
# BATCH_SIZE = 32  # choose based on GPU memory
# dataset = Flickr30kDataset(metadata_path, image_dir)
# loader  = DataLoader(
#     dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     num_workers=4,
#     collate_fn=collate_fn
# )
