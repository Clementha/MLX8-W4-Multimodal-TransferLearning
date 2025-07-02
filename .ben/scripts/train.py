# scripts/train.py
import os
from functools import partial
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset       import Flickr30kDataset
from img_embedder  import ImgEmbedder
from text_embedder import TextEmbedder
from collate       import collate_fn
from decoder       import DecoderModel  

# ==== Config ====
BATCH_SIZE   = 8
NUM_WORKERS  = 8
MAX_SEQ_LEN  = 50
NUM_EPOCHS   = 10
LEARNING_RATE = 0.0001
NUM_HEADS    = 8
NUM_LAYERS   = 6

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

    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,       # magic speed boost (supposedly)
        collate_fn=partial(
            collate_fn,
            tokenizer=text_embedder.tokenizer,
            max_seq_len=MAX_SEQ_LEN,
        )
    )

    # 2) Device & (future) decoder + optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Instantiate the decoder model, get embeddings dims first
    txt_hidden_dims = text_embedder.embed_layer.embedding_dim   # e.g. 1024
    vocab_size = text_embedder.tokenizer.vocab_size        # e.g. 151669 tokens
    img_dim = img_embedder.model.config.hidden_size # e.g. 768 for ViT base

    # Create the decoder model with the image and text embedding dimensions
    model = DecoderModel(
        image_dim=img_dim,  
        hidden_dim=txt_hidden_dims,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        vocab_size=vocab_size,
        max_seq_len=MAX_SEQ_LEN
    ).to(device)

    # 3) Optimizer and loss function
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Get pod token ID's to ignore in loss
    pad_id = text_embedder.tokenizer.pad_token_id
    # set up loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

    # 3) Training loop skeleton
    for epoch in range(NUM_EPOCHS):
        model.train()  # set model to training mode
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS},", leave=True)
        for batch_idx, (images, input_ids, attention_mask, filenames, captions) in enumerate(loop):
            # 3a) Embed
            img_embeddings = img_embedder.embed_batch(images).to(device, non_blocking=True)       # (B, D_img)
            txt_embeddings = text_embedder.embed_tokens(input_ids).to(device, non_blocking=True)  # (B, T, D_txt)

            # 3b) TODO: forward pass through your decoder, compute loss
            logits = model(img_embeddings, txt_embeddings, attention_mask.to(device)) # outputs (B, T, V)

            # 3c) flatten outputs for los function
            B, T, V = logits.shape
            logits = logits.reshape(B * T, V)               # (B*T, V)
            targets = input_ids.to(device).view(B * T)   # (B*T,)
            # Get logits and target_ids

       
            # 3d) loss + backward
            loss = loss_fn(logits, targets)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            # e) update progress bar
            loop.set_postfix(loss=loss.item())
            if batch_idx % 4 == 0:
                tqdm.write(f"[Epoch {epoch+1}] Batch {batch_idx:03d} — loss: {loss.item():.4f}")

if __name__ == "__main__":
    main()
