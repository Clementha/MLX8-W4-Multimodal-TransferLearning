# scripts/train.py
import os
from functools import partial

import torch
from torch.utils.data import DataLoader

from dataset       import Flickr30kDataset
from img_embedder  import ImgEmbedder
from text_embedder import TextEmbedder
from collate       import collate_fn

# ==== Config ====
BATCH_SIZE   = 8
NUM_WORKERS  = 4
MAX_SEQ_LEN  = 50
NUM_EPOCHS   = 10

VISION_MODEL = "google/vit-base-patch16-224-in21k"
TEXT_MODEL   = "Qwen/Qwen3-Embedding-0.6B"

BASE_DIR     = os.path.dirname(__file__)
METADATA     = os.path.join(BASE_DIR, "..", "data", "flickr30k", "metadata.parquet")
IMAGEDIR     = os.path.join(BASE_DIR, "..", "data", "flickr30k", "images")
# ================

def main():
    # 1) Dataset & DataLoader
    dataset = Flickr30kDataset(METADATA, IMAGEDIR)
    text_embedder = TextEmbedder(TEXT_MODEL)
    img_embedder  = ImgEmbedder(VISION_MODEL)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=partial(
            collate_fn,
            tokenizer=text_embedder.tokenizer,
            max_seq_len=MAX_SEQ_LEN
        )
    )

    # 2) Device & (future) decoder + optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # TODO: instantiate your decoder model here, e.g.
    # decoder = MyCaptionDecoder(...).to(device)
    # optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-4)

    # 3) Training loop skeleton
    for epoch in range(NUM_EPOCHS):
        for images, input_ids, attention_mask, filenames, captions in loader:
            # 3a) Embed
            img_embeddings = img_embedder.embed_batch(images).to(device)       # (B, D_img)
            txt_embeddings = text_embedder.embed_tokens(input_ids).to(device)  # (B, L, D_txt)

            # 3b) TODO: forward pass through your decoder, compute loss
            # outputs = decoder(img_embeddings, txt_embeddings, attention_mask)
            # loss = criterion(outputs, target_ids)

            # 3c) TODO: backward + step
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

        # TODO: optional: print epoch metrics, save checkpoints, etc.

if __name__ == "__main__":
    main()
