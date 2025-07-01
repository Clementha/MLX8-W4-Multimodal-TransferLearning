"""
Quick sanity‑check script to verify that:
• Flickr30kDataset loads images & captions
• collate_fn tokenises captions correctly
• ImgEmbedder produces image CLS embeddings
• TextEmbedder produces token embeddings

Run from project root (activate venv first):

python scripts/test_pipeline.py \
  --metadata data/flickr30k/metadata.parquet \
  --imagedir data/flickr30k/images 
"""

import argparse
import os
from pathlib import Path
from pprint import pprint

import torch
from torch.utils.data import DataLoader

from dataset import Flickr30kDataset
from img_embedder import ImgEmbedder
from text_embedder import TextEmbedder
from collate import collate_fn

# Config 
BATCH_SIZE = 4
MAX_SEQ_LEN = 50 #TODO DO NOT MODIFY HERE, must modify in collate.py. Figure out how to put everything in a config file instead.
VISION_MODEL = "google/vit-base-patch16-224-in21k"
METADATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "flickr30k", "metadata.parquet")
IMAGEDIR_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "flickr30k", "images")

def main():
    #   Init dataset & loader                         
    ds = Flickr30kDataset(METADATA_PATH, IMAGEDIR_PATH)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False, #TODO
        num_workers=0,#TODO
        collate_fn=collate_fn,
    )

    #   Initialise embedders                         
    img_embedder  = ImgEmbedder(VISION_MODEL)
    text_embedder = TextEmbedder()  # default bert‑base‑uncased #TODO: allow choice of test model like we do vision model

    #   Pull one batch          
    images, input_ids, attn_mask, fnames = next(iter(loader))

    print(f"\nBatch images: {len(images)}  » first fname: {fnames[0]}")
    print("Token IDs shape:", input_ids.shape)
    print("Attention mask shape:", attn_mask.shape)

    #   Embed                                 
    img_embeds  = img_embedder.embed_batch(images)                 # (B, D_img)
    txt_embeds  = text_embedder.embed_tokens(input_ids)            # (B, L, D_txt)

    print("Image CLS embeddings shape:", img_embeds.shape)
    print("Text token embeddings shape:", txt_embeds.shape)

    #   Show a miniature example         
    print("\nSample caption token decode:")
    tokenizer = text_embedder.tokenizer
    print(tokenizer.decode(input_ids[0], skip_special_tokens=False))
    print("First token vector (norm):", txt_embeds[0, 0].norm().item()) #print first token vector normalised

    print("\nEverything looks OK — ready for training.")


if __name__ == "__main__":
    main()
