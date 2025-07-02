"""Multimodal Transfer‑Learning

A SINGLE‑FILE reference implementation that pairs a *frozen* vision encoder
(Google ViT‑B/16 or OpenAI CLIP ViT‑B/32) with Qwen‑3‑0.6B.  The bridge
adapter is a two‑layer MLP; the top‑K Qwen decoder blocks may optionally be
unfrozen.  Training, evaluation, W&B logging and colourful console output are
all handled inside one class for ease of experimentation.

Author: Charles Cai (github.com/charles‑cai)
"""
from __future__ import annotations

import argparse
import os
import random
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import wandb
from dotenv import load_dotenv
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    ViTModel, #AutoModel,
    ViTImageProcessorFast,  #AutoImageProcessor,
    CLIPProcessor,
    CLIPModel,
    AutoTokenizer,
    AutoModelForCausalLM,
)
import evaluate

import logging
import colorlog

###############################################################################
# 1.  Environment variables — loaded once, accessible via `self.xxx`           #
###############################################################################
load_dotenv()

# Vision encoder choice & dims ------------------------------------------------
VISION_ENCODER: str = os.getenv("VISION_ENCODER", "vit").lower()  # "vit" | "clip"
ENCODER_ID: str = {
    "vit": "google/vit-base-patch16-224",
    "clip": "openai/clip-vit-base-patch32",
}[VISION_ENCODER]
IMG_EMB_DIM: int = 768 if VISION_ENCODER == "vit" else 512

# Training hyper‑parameters ----------------------------------------------------
TOP_K: int = int(os.getenv("TOP_K", 3))            # 0 = fully frozen
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", 2))  # global batch size
EPOCHS: int = int(os.getenv("EPOCHS", 3))
LR_ADAPTER: float = float(os.getenv("LR_ADAPTER", 1e-4))
LR_QWEN: float = float(os.getenv("LR_QWEN", 2e-5))
SEED: int = int(os.getenv("SEED", 42))

# Paths & logging -------------------------------------------------------------
OUTPUT_DIR_MODELS: Path = Path(os.getenv("OUTPUT_DIR_MODELS", "../.data/models"))
OUTPUT_DIR_MODELS.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR_DATASETS: Path = Path(os.getenv("OUTPUT_DIR_DATASETS", "../.data/datasets"))
OUTPUT_DIR_DATASETS.mkdir(parents=True, exist_ok=True)

TRAINING_DATASET: str = os.getenv("TRAINING_DATASET", "flickr30k")

WANDB_ENTITY: str = os.getenv("WANDB_ENTITY", "charles-cai")
WANDB_PROJECT: str = os.getenv("WANDB_PROJECT", "mlx8-w4-multimodal-transferlearning")

# ---------------------------------------------------------------------------- #

###############################################################################
# 2.  Coloured logging helper                                                  #
###############################################################################

colour_handler = colorlog.StreamHandler()
colour_handler.setFormatter(
    colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s | %(message)s",
        log_colors={
            "DEBUG": "white",
            "INFO": "white",
            "SUCCESS": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bold",
        },
    )
)
logging.basicConfig(level=logging.INFO, handlers=[colour_handler])
log = logging.getLogger("MMTL")

###############################################################################
# 3.  Core model components                                                   #
###############################################################################

class ImageAdapter(nn.Module):
    """Two‑layer MLP that maps CLS / image_embeds → Qwen hidden dim (4 096)."""

    def __init__(self, in_dim: int, out_dim: int = 4096, hidden: int = 1024, n_tokens: int = 16):
        super().__init__()
        self.n_tokens = n_tokens
        self.out_dim = out_dim
        self.mapper = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, n_tokens * out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim)  → (B, n_tokens, out_dim)
        mapped = self.mapper(x).view(x.size(0), self.n_tokens, self.out_dim)
        return mapped

###############################################################################
# 4.  Main trainer class                                                      #
###############################################################################

class MultimodalTransferLearning:
    def __init__(self, prepare_data_only=False):
        self._set_seed(SEED)
        if not prepare_data_only:
            self._init_wandb()
        self._init_vision()
        if not prepare_data_only:
            self._init_qwen()
            self.bridge = ImageAdapter(IMG_EMB_DIM).to(self.device)
        self._prepare_data()
        if not prepare_data_only:
            self._init_optimisers()
            log.info("🚀 Initialisation complete. Starting training…")
        else:
            log.info("🚀 Data preparation complete.")

    # --------------------------------------------------------------------- #
    def _set_seed(self, seed: int):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        log.info(f"🔒 Set random seed to {seed} for reproducibility.")

    # --------------------------------------------------------------------- #
    def _init_wandb(self):
        TRAINING_DATASET = os.getenv("TRAINING_DATASET", "flickr30k")
        run_name = f"{TRAINING_DATASET}-{VISION_ENCODER}-top{TOP_K}"
        config = {
            "TRAINING_DATASET": TRAINING_DATASET,
            "VISION_ENCODER": VISION_ENCODER,
            "ENCODER_ID": ENCODER_ID,
            "IMG_EMB_DIM": IMG_EMB_DIM,
            "TOP_K": TOP_K,
            "BATCH_SIZE": BATCH_SIZE,
            "EPOCHS": EPOCHS,
            "LR_ADAPTER": LR_ADAPTER,
            "LR_QWEN": LR_QWEN,
            "SEED": SEED,
            "OUTPUT_DIR_MODELS": str(OUTPUT_DIR_MODELS),
            "WANDB_ENTITY": WANDB_ENTITY,
            "WANDB_PROJECT": WANDB_PROJECT,
        }
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=run_name,
            reinit=True,
            config=config
        )
        log.info(f"🎨 W&B initialised → {WANDB_PROJECT}/{run_name}")

    # --------------------------------------------------------------------- #
    def _init_vision(self):
        if VISION_ENCODER == "vit":
            self.processor = ViTImageProcessorFast.from_pretrained(ENCODER_ID) #AutoImageProcessor
            self.vision_encoder = ViTModel.from_pretrained(ENCODER_ID, torch_dtype="auto")
        else:
            self.processor = CLIPProcessor.from_pretrained(ENCODER_ID)
            self.vision_encoder = CLIPModel.from_pretrained(ENCODER_ID, torch_dtype="auto")
        self.vision_encoder.eval().requires_grad_(False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vision_encoder.to(self.device)
        log.info(f"🖼️  Loaded {ENCODER_ID} (frozen) → {self.device} .")

    # --------------------------------------------------------------------- #
    def _init_qwen(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
        self.qwen = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B-Base", torch_dtype="auto", #attn_implementation="flash_attention_2"
        ).to(self.device)
        self.qwen.eval().requires_grad_(False)
        if TOP_K > 0:
            # Hard fail if GPU likely insufficient — user responsibility.
            log.warning(f"🔓 Unfreezing last {TOP_K} Qwen blocks.")
            for p in self.qwen.model.layers[-TOP_K:].parameters():
                p.requires_grad = True
                self.qwen.train()  # enable dropout only if some params trainable
        else:
            log.info("🧊 Keeping Qwen fully frozen.")

    # --------------------------------------------------------------------- #
    def _prepare_data(self):
        # Check if fully processed data already exists
        processed_dir = OUTPUT_DIR_DATASETS / TRAINING_DATASET / "_processed"
        dataset_path = processed_dir / "dataset_dict"
        images_cache_path = processed_dir / "images_cache.pt"
        
        if (dataset_path.exists() and images_cache_path.exists()):
            log.info(f"📁 Loading cached dataset and images from {processed_dir}")
            try:
                from datasets import load_from_disk
                self.dataset = load_from_disk(str(dataset_path))
                self.images_cache = torch.load(images_cache_path, map_location='cpu')
                log.info(
                    f"📚 Loaded cached dataset: "
                    f"{len(self.dataset['train'])} train / {len(self.dataset['eval'])} eval / {len(self.dataset['test'])} test"
                )
                log.info(f"🖼️ Loaded {len(self.images_cache)} cached processed images")
                self._setup_data_loaders()
                return
            except Exception as e:
                log.warning(f"⚠️ Failed to load cached data: {e}. Reprocessing...")
        
        log.warning("⏳ Loading raw dataset from HuggingFace - this may take several minutes...")
        raw = load_dataset("lmms-lab/flickr30k", split="test", token=os.getenv("HF_TOKEN"))
        
        # First, process all unique images and cache them
        log.warning("⏳ Processing unique images - this may take time...")
        self._process_and_cache_images(raw, processed_dir)
        
        log.warning("⏳ Flattening dataset (creating image-caption pairs)...")
        def flatten_generator():
            for i, example in enumerate(tqdm(raw, desc="Flattening dataset")):
                # Use image index as key instead of storing image data
                for caption in example["caption"]:
                    yield {"image_idx": i, "caption": caption}

        flat_dataset = Dataset.from_generator(flatten_generator)
        
        # split 80/10/10 with fixed seed
        log.info("Splitting dataset 80/10/10...")
        train_test = flat_dataset.train_test_split(test_size=0.2, seed=SEED)
        eval_test = train_test["test"].train_test_split(test_size=0.5, seed=SEED)
        self.dataset = DatasetDict({
            "train": train_test["train"],
            "eval": eval_test["train"],
            "test": eval_test["test"],
        })
        
        log.info(
            f"📚 Dataset split: "
            f"{len(self.dataset['train'])} train / {len(self.dataset['eval'])} eval / {len(self.dataset['test'])} test"
        )
        
        # Save processed dataset to disk for future use
        log.info(f"💾 Caching processed dataset to {processed_dir}")
        processed_dir.mkdir(parents=True, exist_ok=True)
        self.dataset.save_to_disk(str(dataset_path))
        
        self._setup_data_loaders()
    
    def _process_and_cache_images(self, raw_dataset, processed_dir):
        """Process all unique images once and cache the results"""
        processed_dir.mkdir(parents=True, exist_ok=True)
        images_cache_path = processed_dir / "images_cache.pt"
        
        # Check if images are already cached
        if images_cache_path.exists():
            try:
                self.images_cache = torch.load(images_cache_path, map_location='cpu')
                expected_count = len(raw_dataset)
                if len(self.images_cache) == expected_count:
                    log.info(f"🖼️ Using existing cached images ({len(self.images_cache)} images)")
                    return
                else:
                    log.warning(f"⚠️ Cached images count mismatch. Expected {expected_count}, got {len(self.images_cache)}. Reprocessing...")
            except Exception as e:
                log.warning(f"⚠️ Failed to load cached images: {e}. Reprocessing...")
        
        log.info("🔄 Processing and caching images...")
        self.images_cache = {}
        
        # Move processor to GPU if possible and supported
        processor_device = self.device if self._can_use_gpu_for_processing() else "cpu"
        if processor_device == "cuda":
            log.info("🚀 Using GPU acceleration for image processing")
        
        # Process images in batches to optimize GPU usage
        batch_size = 32 if processor_device == "cuda" else 8
        
        for i in tqdm(range(0, len(raw_dataset), batch_size), desc="Processing image batches"):
            batch_end = min(i + batch_size, len(raw_dataset))
            batch_images = [raw_dataset[j]["image"] for j in range(i, batch_end)]
            
            try:
                if VISION_ENCODER == "vit":
                    batch_pixels = self.processor(images=batch_images, return_tensors="pt").pixel_values
                else:
                    batch_pixels = self.processor(images=batch_images, return_tensors="pt").pixel_values
                
                # Move to specified device
                if processor_device == "cuda":
                    batch_pixels = batch_pixels.to(self.device)
                
                # Store each processed image
                for j, pixel_tensor in enumerate(batch_pixels):
                    self.images_cache[i + j] = pixel_tensor.cpu()  # Always store on CPU to save GPU memory
                    
            except Exception as e:
                log.warning(f"⚠️ Batch processing failed at batch {i}, falling back to individual processing: {e}")
                # Fallback to individual processing
                for j in range(i, batch_end):
                    try:
                        image = raw_dataset[j]["image"]
                        if VISION_ENCODER == "vit":
                            pixel = self.processor(images=image, return_tensors="pt").pixel_values[0]
                        else:
                            pixel = self.processor(images=image, return_tensors="pt").pixel_values[0]
                        self.images_cache[j] = pixel.cpu()
                    except Exception as e2:
                        log.error(f"❌ Failed to process image {j}: {e2}")
                        # Create a dummy tensor as fallback
                        dummy_size = (3, 224, 224)  # Standard size
                        self.images_cache[j] = torch.zeros(dummy_size)
        
        log.info(f"💾 Saving {len(self.images_cache)} processed images to cache")
        torch.save(self.images_cache, images_cache_path)
        log.info(f"✅ Images cached to {images_cache_path}")
    
    def _can_use_gpu_for_processing(self):
        """Check if GPU can be used for image processing"""
        if not torch.cuda.is_available():
            return False
        
        # Test if processor can work with CUDA tensors
        try:
            # Create a small test image
            import PIL.Image
            import numpy as np
            test_img = PIL.Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            
            if VISION_ENCODER == "vit":
                test_result = self.processor(images=test_img, return_tensors="pt").pixel_values
            else:
                test_result = self.processor(images=test_img, return_tensors="pt").pixel_values
            
            # Try moving to GPU
            test_result.to(self.device)
            return True
        except Exception:
            return False
    
    def _setup_data_loaders(self):
        """Setup data loaders with preprocessing - separated to avoid code duplication"""
        log.info("⏳ Setting up data preprocessing and loaders...")
        
        def preprocess(example):
            # Get preprocessed image from cache using image_idx
            image_idx = example["image_idx"]
            pixel = self.images_cache[image_idx]
            example["pixel"] = pixel
            
            # Tokenize caption
            example["input_ids"] = self.tokenizer(
                example["caption"], truncation=True, return_tensors="pt"
            ).input_ids[0]
            return example

        # Process datasets - now much faster since images are pre-processed
        log.info("Setting up train set...")
        self.dataset["train"] = self.dataset["train"].map(
            preprocess, 
            remove_columns=["image_idx", "caption"],
            batch_size=1000,  # Can use larger batches now
            desc="Setting up train"
        )
        
        log.info("Setting up eval set...")
        self.dataset["eval"] = self.dataset["eval"].map(
            preprocess, 
            remove_columns=["image_idx", "caption"],
            batch_size=1000,
            desc="Setting up eval"
        )
        
        log.info("Setting up test set...")
        self.dataset["test"] = self.dataset["test"].map(
            preprocess, 
            remove_columns=["image_idx", "caption"],
            batch_size=1000,
            desc="Setting up test"
        )
        
        self.dataset.set_format(type="torch")
        self.train_loader = DataLoader(self.dataset["train"], batch_size=BATCH_SIZE, shuffle=True)
        self.eval_loader = DataLoader(self.dataset["eval"], batch_size=BATCH_SIZE)
        self.test_loader = DataLoader(self.dataset["test"], batch_size=BATCH_SIZE)
        
        log.info("✅ Data loaders ready!")

    # --------------------------------------------------------------------- #
    def _init_optimisers(self):
        # Separate param groups so the bridge learns faster
        params_bridge = [p for p in self.bridge.parameters() if p.requires_grad]
        params_qwen = [p for p in self.qwen.parameters() if p.requires_grad]
        
        # Create optimizer param groups only for parameters that require gradients
        param_groups = [{"params": params_bridge, "lr": LR_ADAPTER}]
        if params_qwen:  # Only add Qwen params if there are any trainable ones
            param_groups.append({"params": params_qwen, "lr": LR_QWEN})
            
        self.optimizer = optim.AdamW(param_groups)
        log.info(f"🛠️  Optimiser ready (AdamW). Bridge params: {len(params_bridge)}, Qwen params: {len(params_qwen)}")

    ############################################################################
    # 5.  Training & evaluation                                                #
    ############################################################################

    # SElf-attention flow across visual and text tokens
    # Visual tokens: [v1, v2, ..., v16] (from bridge)
    # Text tokens:   [t1, t2, ..., tn]  (caption embeddings)

    # Attention matrix (simplified):
    #        v1  v2  ... v16  t1  t2  ... tn
    # v1    [ ✓   ✗   ✗   ✗   ✗   ✗   ✗   ✗ ]  # can only see itself
    # v2    [ ✓   ✓   ✗   ✗   ✗   ✗   ✗   ✗ ]  # can see v1, v2
    # ...
    # t1    [ ✓   ✓   ✓   ✓   ✓   ✗   ✗   ✗ ]  # can see all visual + t1
    # t2    [ ✓   ✓   ✓   ✓   ✓   ✓   ✗   ✗ ]  # can see all visual + t1,t2

    # Loss gradient flow
    # Loss = CrossEntropy(predicted_text_tokens, actual_text_tokens)
    # ↓ (backprop)
    # Qwen layers (last TOP_K only)
    # ↓ (backprop)  
    # Bridge MLP (always trainable)
    # ↓ (no backprop - frozen)
    # Vision Encoder (frozen)

    def _forward_step(self, batch):
        pixel, caption_ids = batch["pixel"].to(self.device), batch["input_ids"].to(self.device)
        with torch.no_grad():
            if VISION_ENCODER == "vit":
                img_emb = self.vision_encoder(pixel)[0][:, 0]  # CLS
            else:
                img_emb = self.vision_encoder(pixel, output_hidden_states=False).image_embeds
        
        # Bridge: img_emb (B, 768/512) -> vis_tokens (B, 16, 4096)
        vis_tokens = self.bridge(img_emb)
        
        # Text embeddings: caption_ids (B, seq_len) -> text_emb (B, seq_len, 4096)
        text_emb = self.qwen.get_input_embeddings()(caption_ids)
        
        # Concatenate: inputs (B, 16+seq_len, 4096)
        inputs = torch.cat([vis_tokens, text_emb], dim=1)
        
        # Create labels for loss calculation:
        # - Visual tokens get -100 (ignored in CrossEntropyLoss)
        # - Text tokens get their actual token IDs (used for loss)
        batch_size, vis_seq_len = vis_tokens.shape[:2]
        vis_labels = torch.full((batch_size, vis_seq_len), -100, device=self.device)
        labels = torch.cat([vis_labels, caption_ids], dim=1)
        
        # Forward pass through Qwen:
        # 1. Self-attention across all tokens (visual + text)
        # 2. Generate logits for next token prediction
        # 3. Compute CrossEntropyLoss only where labels != -100
        outputs = self.qwen(inputs_embeds=inputs, labels=labels)
        
        # Loss calculation (internally in Qwen):
        # logits: (B, 16+seq_len, vocab_size)
        # For each text position i: loss += CrossEntropy(logits[i], labels[i])
        # Visual positions are ignored due to labels[vis_positions] = -100
        return outputs.loss

    # --------------------------------------------------------------------- #
    def _compute_metrics(self, preds: List[str], refs: List[str]) -> Dict[str, float]:
        if not preds or not refs:
            return dict(BLEU4=0.0, CIDEr=0.0, R1=0.0, R5=0.0, P1=0.0, P5=0.0)
            
        try:
            # Load BLEU metric from evaluate
            bleu_metric = evaluate.load("bleu")
            # Convert single strings to lists of references as expected by evaluate
            bleu_refs = [[ref] for ref in refs]
            bleu_result = bleu_metric.compute(predictions=preds, references=bleu_refs)
            bleu = bleu_result["bleu"]
        except Exception as e:
            print(f"BLEU computation failed: {e}")
            bleu = 0.0
            
        try:
            # Load CIDEr metric from evaluate (if available)
            # Note: CIDEr might not be available in all evaluate versions
            # Using a fallback approach
            cider = 0.0  # Placeholder - CIDEr is complex to implement from scratch
        except:
            cider = 0.0
            
        # Simplified retrieval metrics (dummy implementation)
        # In practice, you'd use proper sentence embeddings
        return dict(BLEU4=bleu, CIDEr=cider, R1=0.0, R5=0.0, P1=0.0, P5=0.0)

    # --------------------------------------------------------------------- #
    def train(self):
        for epoch in range(EPOCHS):
            # Set training mode only for trainable components
            if TOP_K > 0:
                self.qwen.train()
            self.bridge.train()
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
            for batch in pbar:
                loss = self._forward_step(batch)
                loss.backward()
                self.optimizer.step(); self.optimizer.zero_grad()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            self._evaluate(epoch)
            self._save_checkpoint(epoch)

        log.info("🎉 Training complete. Running final test…")
        self._evaluate(EPOCHS, test=True)

    # --------------------------------------------------------------------- #
    def _evaluate(self, epoch: int, test: bool = False):
        self.qwen.eval(); self.bridge.eval()
        loader = self.test_loader if test else self.eval_loader
        preds, refs = [], []
        with torch.no_grad():
            for batch in loader:
                pixel = batch["pixel"].to(self.device)
                if VISION_ENCODER == "vit":
                    img_emb = self.vision_encoder(pixel)[0][:, 0]
                else:
                    img_emb = self.vision_encoder(pixel).image_embeds
                vis_tokens = self.bridge(img_emb)
                
                # Generate text tokens only (excluding visual embedding positions)
                outputs = self.qwen.generate(
                    inputs_embeds=vis_tokens,
                    max_new_tokens=32,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=False
                )
                
                # Extract only the newly generated token IDs (skip visual token positions)
                generated_ids = outputs.sequences[:, vis_tokens.shape[1]:]  # Skip visual tokens
                
                # Decode predictions and references
                batch_preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                batch_refs = self.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
                
                preds.extend(batch_preds)
                refs.extend(batch_refs)
                
        metrics = self._compute_metrics(preds, refs)
        tag = "test" if test else "eval"
        
        log_dict = {f"{tag}/{k}": v for k, v in metrics.items()}
        log_dict["epoch"] = epoch + 1
        wandb.log(log_dict)
        
        log.info(f"📊 {tag.title()} metrics at epoch {epoch+1}: {metrics}")

    # --------------------------------------------------------------------- #
    def _save_checkpoint(self, epoch: int):
        fname = OUTPUT_DIR_MODELS / f"{epoch+1:02d}_{VISION_ENCODER}_top{TOP_K}.pt"
        torch.save({
            "qwen": self.qwen.state_dict(),
            "bridge": self.bridge.state_dict(),
        }, fname)
        log.info(f"💾 Saved checkpoint → {fname}")

###############################################################################
# 6.  CLI entry‑point                                                        #
###############################################################################

def parse_args():
    p = argparse.ArgumentParser(description="Multimodal Transfer‑Learning runner")
    p.add_argument("--train", action="store_true", help="Train then evaluate")
    p.add_argument("--test", action="store_true", help="Run standalone test using last checkpoint")
    p.add_argument("--data", action="store_true", help="Prepare and cache dataset only")
    return p.parse_args()


def main():
    args = parse_args()
    
    if args.data:
        log.info("🔄 Preparing dataset...")
        engine = MultimodalTransferLearning(prepare_data_only=True)
        log.info("✅ Dataset preparation complete!")
    elif args.train:
        engine = MultimodalTransferLearning()
        engine.train()
    elif args.test:
        engine = MultimodalTransferLearning()
        engine._evaluate(epoch=EPOCHS, test=True)
    else:
        log.error("❌ Must specify either --data, --train, or --test")


if __name__ == "__main__":
    main()
