# scripts/train.py
import os
from functools import partial
from tqdm import tqdm
from datetime import datetime


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import wandb
from dotenv import load_dotenv

from dataset       import Flickr30kDataset
from img_embedder  import ImgEmbedder
from text_embedder import TextEmbedder
from collate       import collate_fn
from decoder       import DecoderModel  

# ==== Config ====
BATCH_SIZE   = 4
NUM_WORKERS  = 8
MAX_SEQ_LEN  = 50
NUM_EPOCHS   = 5
LEARNING_RATE = 0.0001
NUM_HEADS    = 8
NUM_LAYERS   = 6
VAL_SPLIT    = 0.15


VISION_MODEL = "google/vit-base-patch16-224-in21k"
TEXT_MODEL   = "Qwen/Qwen3-Embedding-0.6B"

BASE_DIR     = os.path.dirname(__file__)
METADATA     = os.path.join(BASE_DIR, "..", "data", "flickr30k", "metadata.parquet")
IMAGEDIR     = os.path.join(BASE_DIR, "..", "data", "flickr30k", "images")
# ================

def main():
    # Set up Weights & Biases logging
    # load API key and force WandB to pick it up
    load_dotenv(dotenv_path=os.path.join(BASE_DIR, "..", ".env"))
    os.environ["WANDB_API_KEY"] = os.environ.get("WANDB_API_KEY", "")
    wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True, anonymous="never")

    # Set up WandB logging directory
    wandb_dir = os.path.join(BASE_DIR, "..", "wandb_logs") # set wandb logs dir
    # pass config
    run = wandb.init(
        project="flickr30k-captioning",
        dir=wandb_dir,
        config={
            "batch_size": BATCH_SIZE,
            "lr":         LEARNING_RATE,
            "epochs":     NUM_EPOCHS,
            "max_seq_len":MAX_SEQ_LEN,
            "num_heads":  NUM_HEADS,
            "num_layers": NUM_LAYERS,
        }
    )
    config = run.config
    print("🌐 Your run is at ---->:", run.url)

    # 1) Dataset & DataLoader
    # Set up train and val datasets
    full_ds = Flickr30kDataset(METADATA, IMAGEDIR)
    n_val = int(len(full_ds) * VAL_SPLIT)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])
    
    # Define our models
    text_embedder = TextEmbedder(TEXT_MODEL)
    img_embedder  = ImgEmbedder(VISION_MODEL)

    # Set up train adn validation DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=partial(
            collate_fn,
            tokenizer=text_embedder.tokenizer,
            max_seq_len=config.max_seq_len,
        )
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=partial(
            collate_fn,
            tokenizer=text_embedder.tokenizer,
            max_seq_len=config.max_seq_len,
        )
    )
    # for periodic validation
    val_iter = iter(val_loader)  

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
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        vocab_size=vocab_size,
        max_seq_len=config.max_seq_len
    ).to(device)

    # 3) Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    # Reduce LR by 1/2 if val_loss plateaus for 2 epochs:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    # Get pad token ID's to ignore in loss
    pad_id = text_embedder.tokenizer.pad_token_id
    # set up loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

    # model checkpointing setup ———
    start_time   = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir     = os.path.join(BASE_DIR, "..", "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True) 
    best_val_acc   = 0.0
    last_ckpt_path = None # to track prev ckpt path for deletion if needed

    # 3) Training loop skeleton 
    global_step = 0 # for val tracking
    for epoch in range(1, config.epochs + 1):
        model.train()  # set model to training mode
        running_loss = 0.0
        running_tokens = 0
        running_correct = 0

        train_iter = tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=True)
        for batch_idx, (images, input_ids, attention_mask, filenames, captions) in enumerate(train_iter):
            # if batch_idx >= 10: # use to debug end of epoch behaviour (set val split low as well)
            #     break
            global_step += 1

            # 3o) Move data to device
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)

            # 3a) Embed
            img_embeddings = img_embedder.embed_batch(images).to(device, non_blocking=True)       # (B, D_img)
            txt_embeddings = text_embedder.embed_tokens(input_ids).to(device, non_blocking=True)

            # 3b) Forward pass through decoder, compute loss
            logits = model(img_embeddings, txt_embeddings, attention_mask) # outputs (B, T, V)

            # 3c) flatten outputs for los function
            B, T, V = logits.shape
            flat_logits = logits.reshape(B * T, V)             # (B*T, V)
            targets = input_ids.to(device).view(B * T)  # (B*T,)

            # 3d) loss + backpropagation
            loss = loss_fn(flat_logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 3e) Update metrics
            # token-level accuracy
            preds    = logits.argmax(-1)            # (B, T)
            correct  = ((preds == input_ids) & (input_ids != pad_id)).sum().item()
            total    = (input_ids != pad_id).sum().item()

            # log train metrics to WandB every batch
            wandb.log(
                {
                    "train/step_loss": loss.item(),
                    "train/step_acc":  correct / total,
                },
                step=global_step,
                commit=False,
            )

            # aggregate stats
            running_loss   += loss.item() * B
            running_tokens += total
            running_correct+= correct

            # Log train metrics every 4 batches
            train_iter.set_postfix(loss=loss.item())
            if batch_idx % 20 == 0:
                tqdm.write(f"[Epoch {epoch}] Batch {batch_idx:03d} » loss {loss.item():.4f}, acc {(correct/total):.4f}")

            # Log validation metrics every 20 batches
            if global_step % 20 == 0:
                model.eval()  # turn off dropout, etc.

                # 1) grab the next validation batch (and rewind if needed)
                try:
                    imgs_v, ids_v, mask_v, _, _ = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    imgs_v, ids_v, mask_v, _, _ = next(val_iter)

                # 2) move EVERYTHING to the same device
                #    — images need to be embedded, so leave them as PIL→Tensor in embedder
                #    — ids and mask are already Tensor from collate_fn
                ids_v = ids_v.to(device,   non_blocking=True)
                mask_v = mask_v.to(device, non_blocking=True)
                imgs_v = img_embedder.embed_batch(imgs_v).to(device, non_blocking=True)

                # 3) embed the validation captions
                with torch.no_grad():
                    # move text embeddings to GPU as well
                    txt_v = text_embedder.embed_tokens(ids_v).to(device, non_blocking=True)
                    logits_v = model(imgs_v, txt_v, mask_v)

                    # 4) flatten for loss
                    Bv, Tv, Vv = logits_v.shape
                    flat_logits_v = logits_v.reshape(Bv * Tv, Vv)   # (Bv*Tv, Vv)
                    flat_targets_v = ids_v.view(Bv * Tv)            # (Bv*Tv,)

                    # 5) compute loss & accuracy
                    loss_v = loss_fn(flat_logits_v, flat_targets_v)
                    preds_v = logits_v.argmax(-1)                   # (Bv, Tv)
                    # only count non-pad tokens
                    non_pad = ids_v != pad_id                        # (Bv, Tv) bool
                    correct_v = (preds_v.eq(ids_v) & non_pad).sum().item()
                    total_v   = non_pad.sum().item()
                    acc_v     = correct_v / total_v if total_v > 0 else 0.0

                # 6) log to wandb (or print), then back to train mode
                wandb.log({
                    "val/step_loss": loss_v.item(),
                    "val/step_acc":  acc_v,
                    "step":          global_step,
                })
                # Print in terminal
                tqdm.write(
                    f"[Epoch {epoch}] Step {global_step:04d} » "
                    f"val_loss {loss_v.item():.4f}, val_acc {acc_v:.4f}"
                )
                # back to train mode
                model.train()



        # Epoch metrics & scheduler
        train_loss = running_loss   / len(train_ds)
        train_acc  = running_correct / running_tokens

        # full-validation at epoch end
        val_loss = 0.0
        val_corr = 0
        val_tok  = 0
        model.eval()
        with torch.no_grad():
            for imgs_v, ids_v, mask_v, _, _ in val_loader:
                imgs_v = img_embedder.embed_batch(imgs_v).to(device, non_blocking=True)
                ids_v  = ids_v.to(device,   non_blocking=True)
                mask_v = mask_v.to(device, non_blocking=True)
                txt_v  = text_embedder.embed_tokens(ids_v).to(device, non_blocking=True)

                logits_v = model(imgs_v, txt_v, mask_v)
                Bv, Tv, Vv = logits_v.shape

                # flatten for loss with reshape
                flat_logits_v = logits_v.reshape(Bv * Tv, Vv)
                flat_targets_v = ids_v.reshape(Bv * Tv).to(device)
                l = loss_fn(flat_logits_v, flat_targets_v)
                val_loss += l.item() * Bv

                preds_v   = logits_v.argmax(-1)
                val_corr += ((preds_v == ids_v) & (ids_v != pad_id)).sum().item()
                val_tok  += (ids_v != pad_id).sum().item()

        val_loss = val_loss / len(val_ds)
        val_acc  = val_corr  / val_tok

        scheduler.step(val_loss)

        print(f"Epoch {epoch:02d} » "
              f"train_loss {train_loss:.4f}, train_acc {train_acc:.4f}, "
              f"val_loss {val_loss:.4f},   val_acc {val_acc:.4f}, "
              f"lr {optimizer.param_groups[0]['lr']:.2e}")

        wandb.log({
            "epoch":     epoch,
            "train/loss": train_loss,
            "train/acc":  train_acc,
            "val/loss":   val_loss,
            "val/acc":    val_acc,
            "lr":         optimizer.param_groups[0]['lr']
        })
        # Save checkpoint if validation accuracy improved
        if val_acc > best_val_acc:
            # remove previous ckpt if it exists
            if last_ckpt_path is not None and os.path.exists(last_ckpt_path):
                os.remove(last_ckpt_path)

            best_val_acc = val_acc
            ckpt_name    = f"ckpt_{start_time}_epoch{epoch:02d}_acc{val_acc:.4f}.pt"
            ckpt_path    = os.path.join(ckpt_dir, ckpt_name)
            torch.save(model.state_dict(), ckpt_path)

            last_ckpt_path = ckpt_path

    # Finish training
    print("Training complete!")
    wandb.finish()


if __name__ == "__main__":
    main()
