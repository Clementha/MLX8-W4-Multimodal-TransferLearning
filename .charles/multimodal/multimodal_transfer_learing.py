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
import time
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

OUTPUT_DIR_CACHE: Path = Path(os.getenv("OUTPUT_DIR_CACHE", "../.data/cache"))
OUTPUT_DIR_CACHE.mkdir(parents=True, exist_ok=True)

TRAINING_DATASET: str = os.getenv("TRAINING_DATASET", "flickr30k")

WANDB_ENTITY: str = os.getenv("WANDB_ENTITY", "charles-cai")
WANDB_PROJECT: str = os.getenv("WANDB_PROJECT", "mlx8-w4-multimodal-transferlearning")

# Inference settings
INFERENCE_PROMPT: str = os.getenv("INFERENCE_PROMPT", "Examine the image carefully and provide a detailed description covering objects, people, actions, background, and any notable visual elements")

BATCH_SAMPLING: int = int(os.getenv("BATCH_SAMPLING", "200"))  # Convert to int for proper usage

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
    """Two‑layer MLP that maps CLS / image_embeds → Qwen hidden dim with cross-entropy option."""

    def __init__(self, in_dim: int, out_dim: int = 1024, hidden: int = 1024, n_tokens: int = 16, use_cross_entropy: bool = False):
        super().__init__()
        self.n_tokens = n_tokens
        self.out_dim = out_dim
        self.use_cross_entropy = use_cross_entropy
        
        if use_cross_entropy:
            # Alternative: predict discrete tokens directly
            self.mapper = nn.Sequential(
                nn.Linear(in_dim, hidden), 
                nn.Tanh(), 
                nn.Linear(hidden, n_tokens * out_dim),
                nn.Softmax(dim=-1)
            )
        else:
            # Current approach: continuous embeddings
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
    def __init__(self, mode="train"):
        """
        Initialize based on mode:
        - embedding: Only vision encoder needed
        - train: Only Qwen + bridge needed  
        - test: Only Qwen + bridge needed
        - run: Load all components for inference
        """
        self.mode = mode
        self._set_seed(SEED)
        
        # Debug logging for environment variables
        log.info(f"🔧 Environment: VISION_ENCODER={VISION_ENCODER}, IMG_EMB_DIM={IMG_EMB_DIM}")
        
        if mode == "embedding":
            log.info("🔄 Embedding mode: Initializing vision encoder only...")
            self._init_vision()
            self._generate_embeddings()
            log.info("✅ Embedding generation complete!")
            return
            
        elif mode in ["train", "test"]:
            log.info(f"🔄 {mode.title()} mode: Initializing Qwen + bridge...")
            self._init_wandb()
            self._init_qwen()
            log.info(f"🔧 Creating ImageAdapter: in_dim={IMG_EMB_DIM} -> out_dim={self.qwen_hidden_size}")
            # Add cross-entropy option for bridge
            use_cross_entropy = os.getenv("BRIDGE_CROSS_ENTROPY", "false").lower() == "true"
            self.bridge = ImageAdapter(IMG_EMB_DIM, out_dim=self.qwen_hidden_size, use_cross_entropy=use_cross_entropy).to(self.device)
            self._load_cached_embeddings()
            self._prepare_data()
            
            if mode == "train":
                self._init_optimisers()
                log.info("🚀 Training setup complete!")
            else:
                log.info("🚀 Test setup complete!")
                
        elif mode == "run":
            log.info("🔄 Inference mode: Initializing all components...")
            self._init_vision()
            self._init_qwen()
            log.info(f"🔧 Creating ImageAdapter: in_dim={IMG_EMB_DIM} -> out_dim={self.qwen_hidden_size}")
            use_cross_entropy = os.getenv("BRIDGE_CROSS_ENTROPY", "false").lower() == "true"
            self.bridge = ImageAdapter(IMG_EMB_DIM, out_dim=self.qwen_hidden_size, use_cross_entropy=use_cross_entropy).to(self.device)
            self._load_latest_checkpoint()
            log.info("🚀 Inference setup complete!")
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'embedding', 'train', 'test', or 'run'")

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
        # Set device first
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
        
        # Fix pad_token issue: Use a different token than eos_token
        if self.tokenizer.pad_token is None:
            # Add a new pad token instead of using eos_token
            self.tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
            log.info(f"🔧 Added new pad token: {self.tokenizer.pad_token}")
        
        self.qwen = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B-Base", torch_dtype="auto"
        ).to(self.device)
        
        # Resize token embeddings to accommodate new pad token
        if self.tokenizer.pad_token != self.tokenizer.eos_token:
            self.qwen.resize_token_embeddings(len(self.tokenizer))
            log.info(f"🔧 Resized token embeddings to {len(self.tokenizer)}")
        
        # Get actual Qwen hidden size and log detailed info
        self.qwen_hidden_size = self.qwen.config.hidden_size
        self.qwen_dtype = next(self.qwen.parameters()).dtype  # Store Qwen's dtype
        embed_tokens = self.qwen.get_input_embeddings()
        
        log.info(f"🔧 Qwen model dimensions:")
        log.info(f"  Hidden size: {self.qwen_hidden_size}")
        log.info(f"  Embedding dim: {embed_tokens.embedding_dim}")
        log.info(f"  Vocab size: {self.qwen.config.vocab_size}")
        log.info(f"  Dtype: {self.qwen_dtype}")
        log.info(f"  Pad token: '{self.tokenizer.pad_token}' (ID: {self.tokenizer.pad_token_id})")
        log.info(f"  EOS token: '{self.tokenizer.eos_token}' (ID: {self.tokenizer.eos_token_id})")
        log.info(f"  Tokens are different: {self.tokenizer.pad_token_id != self.tokenizer.eos_token_id}")
        log.info(f"  Dimensions match: {embed_tokens.embedding_dim == self.qwen_hidden_size}")
        
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
    def _generate_embeddings(self):
        """Generate embeddings, prepare data splits, and cache everything together"""
        cache_file = OUTPUT_DIR_CACHE / f"{VISION_ENCODER}_{TRAINING_DATASET}_complete.pt"
        
        if cache_file.exists():
            log.warning(f"⚠️ Cache file already exists: {cache_file}")
            response = input("Overwrite? (y/N): ").strip().lower()
            if response != 'y':
                log.info("❌ Aborting embedding generation.")
                return
        
        log.info("📥 Loading raw dataset...")
        raw = load_dataset("lmms-lab/flickr30k", split="test", token=os.getenv("HF_TOKEN"))
        
        log.info(f"🔄 Generating embeddings for {len(raw)} images...")
        embeddings = {}
        
        # Process in batches for efficiency
        batch_size = 32 if self.device.type == "cuda" else 8
        
        for i in tqdm(range(0, len(raw), batch_size), desc="Processing image batches"):
            batch_end = min(i + batch_size, len(raw))
            batch_images = [raw[j]["image"] for j in range(i, batch_end)]
            
            try:
                # Process batch
                if VISION_ENCODER == "vit":
                    batch_pixels = self.processor(images=batch_images, return_tensors="pt").pixel_values
                else:
                    batch_pixels = self.processor(images=batch_images, return_tensors="pt").pixel_values
                
                batch_pixels = batch_pixels.to(self.device)
                
                # Generate embeddings
                with torch.no_grad():
                    if VISION_ENCODER == "vit":
                        batch_embeddings = self.vision_encoder(batch_pixels)[0][:, 0]  # CLS token
                    else:
                        # Use CLIP's get_image_features method for proper 512-dim embeddings
                        batch_embeddings = self.vision_encoder.get_image_features(pixel_values=batch_pixels)
                
                # Store embeddings (move to CPU to save GPU memory)
                for j, embedding in enumerate(batch_embeddings):
                    img_id = i + j
                    embeddings[img_id] = embedding.cpu()
                    
            except Exception as e:
                log.warning(f"⚠️ Batch processing failed at {i}, falling back to individual: {e}")
                # Fallback to individual processing
                for j in range(i, batch_end):
                    try:
                        img_id = j
                        image = raw[img_id]["image"]
                        
                        if VISION_ENCODER == "vit":
                            pixels = self.processor(images=image, return_tensors="pt").pixel_values
                        else:
                            pixels = self.processor(images=image, return_tensors="pt").pixel_values
                        
                        pixels = pixels.to(self.device)
                        
                        with torch.no_grad():
                            if VISION_ENCODER == "vit":
                                embedding = self.vision_encoder(pixels)[0][:, 0]  # CLS token
                            else:
                                # Use CLIP's get_image_features method for proper 512-dim embeddings
                                embedding = self.vision_encoder.get_image_features(pixel_values=pixels)[0]
                        
                        embeddings[img_id] = embedding.cpu()
                        
                    except Exception as e2:
                        log.error(f"❌ Failed to process image {j}: {e2}")
                        # Create zero embedding as fallback
                        embeddings[j] = torch.zeros(IMG_EMB_DIM)
        
        # Keep embeddings in memory for data preparation
        self.embeddings_cache = embeddings
        log.info(f"✅ Generated {len(embeddings)} embeddings, keeping in memory")
        
        # Now prepare data splits with embeddings
        log.info("⏳ Creating image-caption pairs with embeddings...")
        processed_examples = []
        
        for img_id, example in enumerate(tqdm(raw, desc="Processing examples")):
            img_embedding = embeddings[img_id]
            for caption in example["caption"]:
                processed_examples.append({
                    "embedding": img_embedding,
                    "caption": caption
                })
        
        # Convert to dataset and split
        log.info("📊 Creating dataset and splitting 80/10/10...")
        flat_dataset = Dataset.from_list(processed_examples)
        
        train_test = flat_dataset.train_test_split(test_size=0.2, seed=SEED)
        eval_test = train_test["test"].train_test_split(test_size=0.5, seed=SEED)
        
        processed_dataset = DatasetDict({
            "train": train_test["train"],
            "eval": eval_test["train"], 
            "test": eval_test["test"],
        })
        
        log.info(
            f"📚 Dataset splits created: "
            f"{len(processed_dataset['train'])} train / {len(processed_dataset['eval'])} eval / {len(processed_dataset['test'])} test"
        )
        
        # Save everything together
        cache_data = {
            'dataset': processed_dataset,
            'metadata': {
                'vision_encoder': VISION_ENCODER,
                'dataset': TRAINING_DATASET,
                'total_images': len(raw),
                'total_examples': len(processed_examples),
                'embedding_dim': IMG_EMB_DIM,
                'encoder_id': ENCODER_ID,
                'created_at': torch.tensor([time.time()], dtype=torch.float64)
            }
        }
        
        log.info(f"💾 Saving complete dataset with embeddings to {cache_file}")
        torch.save(cache_data, cache_file)
        log.info(f"✅ Complete dataset cached successfully!")
        log.info(f"📊 Cache info: {len(processed_examples)} examples, {IMG_EMB_DIM}D embeddings")

    def _load_cached_embeddings(self):
        """Load cached complete dataset for training/testing"""
        cache_file = OUTPUT_DIR_CACHE / f"{VISION_ENCODER}_{TRAINING_DATASET}_complete.pt"
        
        if not cache_file.exists():
            log.error(f"❌ Complete dataset cache not found: {cache_file}")
            log.error(f"💡 Please run: python {__file__} --embedding")
            raise FileNotFoundError(f"Missing complete dataset cache: {cache_file}")
        
        log.info(f"📁 Loading cached complete dataset from {cache_file}")
        try:
            # Load cache with weights_only=False to handle HuggingFace datasets
            cache_data = torch.load(cache_file, map_location='cpu', weights_only=False)
            
            self.dataset = cache_data['dataset']
            metadata = cache_data['metadata']
            
            # Validate cache metadata
            if metadata['vision_encoder'] != VISION_ENCODER:
                log.error(f"❌ Cache vision encoder mismatch: {metadata['vision_encoder']} != {VISION_ENCODER}")
                log.error(f"💡 Delete cache file or set VISION_ENCODER={metadata['vision_encoder']}")
                raise ValueError(f"Cache vision encoder mismatch: {metadata['vision_encoder']} != {VISION_ENCODER}")
            if metadata['dataset'] != TRAINING_DATASET:
                raise ValueError(f"Cache dataset mismatch: {metadata['dataset']} != {TRAINING_DATASET}")
            if metadata['embedding_dim'] != IMG_EMB_DIM:
                log.error(f"❌ Cache embedding dimension mismatch: {metadata['embedding_dim']} != {IMG_EMB_DIM}")
                log.error(f"💡 Delete cache file and regenerate with correct VISION_ENCODER")
                raise ValueError(f"Cache embedding dimension mismatch: {metadata['embedding_dim']} != {IMG_EMB_DIM}")
            
            log.info(f"✅ Loaded complete dataset with {metadata['total_examples']} examples")
            log.info(f"📊 Cache metadata: {metadata['vision_encoder']}, {metadata['total_images']} images, {metadata['embedding_dim']}D")
            
        except Exception as e:
            log.error(f"❌ Failed to load complete dataset cache: {e}")
            log.error(f"💡 Please regenerate cache: python {__file__} --embedding")
            raise

    def _prepare_data(self):
        # Data is already prepared and loaded, just setup data loaders
        self._setup_data_loaders()

    def _setup_data_loaders(self):
        """Setup data loaders using pre-processed dataset with embeddings"""
        log.info("⏳ Setting up data loaders...")
        
        def preprocess(example):
            # For training: tokenize just the caption (no prompt)
            # This ensures the model learns to generate clean captions
            tokens = self.tokenizer(
                example['caption'], 
                truncation=True, 
                padding=False,
                return_tensors="pt"
            )
            example["input_ids"] = tokens.input_ids[0]
            example["original_caption"] = example["caption"]
            return example

        # Custom collate function to handle variable-length sequences
        def collate_fn(batch):
            # Stack embeddings (these are all the same size)
            embeddings = torch.stack([item["embedding"] for item in batch])
            
            # Collect original captions
            original_captions = [item["original_caption"] for item in batch]
            
            # Pad input_ids to the same length within the batch
            input_ids = [item["input_ids"] for item in batch]
            max_len = max(len(ids) for ids in input_ids)
            
            # Pad sequences and create attention masks
            padded_input_ids = []
            attention_masks = []
            
            for ids in input_ids:
                # Create attention mask (1 for real tokens, 0 for padding)
                attention_mask = torch.ones(len(ids), dtype=torch.long)
                
                if len(ids) < max_len:
                    # Pad with pad_token_id
                    padding_length = max_len - len(ids)
                    padding = torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=ids.dtype)
                    padded_ids = torch.cat([ids, padding])
                    
                    # Extend attention mask with zeros for padding
                    padding_mask = torch.zeros(padding_length, dtype=torch.long)
                    attention_mask = torch.cat([attention_mask, padding_mask])
                else:
                    padded_ids = ids
                    
                padded_input_ids.append(padded_ids)
                attention_masks.append(attention_mask)
            
            return {
                "embedding": embeddings,
                "input_ids": torch.stack(padded_input_ids),
                "attention_mask": torch.stack(attention_masks),
                "original_caption": original_captions
            }

        # Process datasets to add tokenized captions
        log.info("Tokenizing captions with prompts...")
        self.dataset["train"] = self.dataset["train"].map(
            preprocess, 
            remove_columns=["caption"],
            batch_size=1000,
            desc="Processing train"
        )
        
        self.dataset["eval"] = self.dataset["eval"].map(
            preprocess, 
            remove_columns=["caption"],
            batch_size=1000,
            desc="Processing eval"
        )
        
        self.dataset["test"] = self.dataset["test"].map(
            preprocess, 
            remove_columns=["caption"],
            batch_size=1000,
            desc="Processing test"
        )
        
        self.dataset.set_format(type="torch")
        self.train_loader = DataLoader(self.dataset["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        self.eval_loader = DataLoader(self.dataset["eval"], batch_size=BATCH_SIZE, collate_fn=collate_fn)
        self.test_loader = DataLoader(self.dataset["test"], batch_size=BATCH_SIZE, collate_fn=collate_fn)
        
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
        # Always use cached embedding (no vision encoder needed)
        img_emb = batch["embedding"].to(self.device)
        caption_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        
        # Bridge: img_emb (B, 768/512) -> vis_tokens (B, 16, qwen_hidden_size)
        vis_tokens = self.bridge(img_emb)
        # Convert to Qwen's dtype to avoid dtype mismatch
        vis_tokens = vis_tokens.to(dtype=self.qwen_dtype)
        
        # Text embeddings: caption_ids (B, seq_len) -> text_emb (B, seq_len, qwen_hidden_size)
        text_emb = self.qwen.get_input_embeddings()(caption_ids)
        
        # Debug: Log shapes on first batch to verify dimensions
        if not hasattr(self, '_logged_shapes'):
            log.info(f"🔍 Forward pass dimensions:")
            log.info(f"  img_emb: {img_emb.shape} ({img_emb.dtype})")
            log.info(f"  vis_tokens: {vis_tokens.shape} ({vis_tokens.dtype})")
            log.info(f"  text_emb: {text_emb.shape} ({text_emb.dtype})")
            log.info(f"  attention_mask: {attention_mask.shape}")
            self._logged_shapes = True
        
        # Concatenate: inputs (B, 16+seq_len, qwen_hidden_size)
        inputs = torch.cat([vis_tokens, text_emb], dim=1)
        
        # Create attention mask for visual + text tokens
        batch_size, vis_seq_len = vis_tokens.shape[:2]
        vis_attention = torch.ones((batch_size, vis_seq_len), device=self.device, dtype=attention_mask.dtype)
        full_attention_mask = torch.cat([vis_attention, attention_mask], dim=1)
        
        # Create labels for loss calculation:
        # - Visual tokens get -100 (ignored in CrossEntropyLoss)
        # - Text tokens get their actual token IDs (used for loss)
        vis_labels = torch.full((batch_size, vis_seq_len), -100, device=self.device)
        labels = torch.cat([vis_labels, caption_ids], dim=1)
        
        # Forward pass through Qwen:
        outputs = self.qwen(
            inputs_embeds=inputs, 
            attention_mask=full_attention_mask,
            labels=labels
        )
        return outputs.loss

    # --------------------------------------------------------------------- #
    def _compute_metrics(self, preds: List[str], refs: List[str]) -> Dict[str, float]:
        if not preds or not refs:
            return dict(BLEU4=0.0, CIDEr=0.0, SPICE=0.0, ROUGE_L=0.0)
        
        # Clean predictions and references
        preds_clean = [pred.strip() for pred in preds if pred.strip()]
        refs_clean = [ref.strip() for ref in refs if ref.strip()]
        
        if not preds_clean or not refs_clean or len(preds_clean) != len(refs_clean):
            log.warning(f"⚠️ Metrics issue: preds={len(preds_clean)}, refs={len(refs_clean)}")
            return dict(BLEU4=0.0, CIDEr=0.0, SPICE=0.0, ROUGE_L=0.0)
            
        try:
            # Load metrics from evaluate
            bleu_metric = evaluate.load("bleu")
            rouge_metric = evaluate.load("rouge")
            
            # CIDEr and SPICE - need special handling
            try:
                from pycocoevalcap.cider.cider import Cider
                from pycocoevalcap.spice.spice import Spice
                
                # Format for COCO eval (expects dict format)
                gts = {i: [ref] for i, ref in enumerate(refs_clean)}
                res = {i: [pred] for i, pred in enumerate(preds_clean)}
                
                cider_scorer = Cider()
                spice_scorer = Spice()
                
                cider_score, _ = cider_scorer.compute_score(gts, res)
                spice_score, _ = spice_scorer.compute_score(gts, res)
                
                cider = cider_score * 100  # Convert to percentage
                spice = spice_score * 100
                
            except ImportError:
                log.warning("⚠️ COCO eval tools not available, using approximations")
                # Fallback: use sentence similarity as proxy
                cider = 0.0
                spice = 0.0
            except Exception as e:
                log.warning(f"⚠️ CIDEr/SPICE computation failed: {e}")
                cider = 0.0
                spice = 0.0

            # Convert single strings to lists of references as expected by evaluate
            bleu_refs = [[ref] for ref in refs_clean]
            
            bleu_result = bleu_metric.compute(predictions=preds_clean, references=bleu_refs)
            rouge_result = rouge_metric.compute(predictions=preds_clean, references=refs_clean)

            bleu = bleu_result["bleu"] * 100  # Convert to percentage
            rouge_l = rouge_result["rougeL"] * 100 # Convert to percentage

            log.info(f"📊 Metrics: BLEU={bleu:.2f}, CIDEr={cider:.2f}, SPICE={spice:.2f}, ROUGE-L={rouge_l:.2f}")
        except Exception as e:
            log.warning(f"⚠️ Metrics computation failed: {e}")
            bleu = cider = spice = rouge_l = 0.0
            
        return dict(
            BLEU4=bleu, 
            CIDEr=cider,
            SPICE=spice,
            ROUGE_L=rouge_l, 
            Samples=len(preds_clean)
        )

    def _compute_lightweight_metrics(self, preds: List[str], refs: List[str]) -> Dict[str, float]:
        """Compute lightweight metrics suitable for frequent batch-level evaluation"""
        if not preds or not refs:
            return dict(avg_length=0.0, non_empty_ratio=0.0, sample_count=0)
        
        # Clean predictions and references
        preds_clean = [pred.strip() for pred in preds if pred.strip()]
        refs_clean = [ref.strip() for ref in refs if ref.strip()]
        
        if not preds_clean or not refs_clean:
            return dict(avg_length=0.0, non_empty_ratio=0.0, sample_count=0)
        
        # Lightweight metrics that don't require heavy computation
        avg_pred_length = sum(len(pred.split()) for pred in preds_clean) / len(preds_clean)
        avg_ref_length = sum(len(ref.split()) for ref in refs_clean) / len(refs_clean)
        non_empty_ratio = len(preds_clean) / len(preds) if preds else 0.0
        
        # Simple word overlap as a proxy for quality (much faster than BLEU)
        word_overlaps = []
        for pred, ref in zip(preds_clean, refs_clean):
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())
            if ref_words:
                overlap = len(pred_words.intersection(ref_words)) / len(ref_words)
                word_overlaps.append(overlap)
        
        avg_word_overlap = sum(word_overlaps) / len(word_overlaps) if word_overlaps else 0.0
        
        return dict(
            avg_pred_length=avg_pred_length,
            avg_ref_length=avg_ref_length,
            length_ratio=avg_pred_length / avg_ref_length if avg_ref_length > 0 else 0.0,
            non_empty_ratio=non_empty_ratio * 100,  # Convert to percentage
            word_overlap=avg_word_overlap * 100,  # Convert to percentage
            sample_count=len(preds_clean)
        )

    def _generate_sample_during_training(self, batch, num_samples: int = 3):
        """Generate sample captions during training for debugging"""
        self.qwen.eval()
        self.bridge.eval()
        
        with torch.no_grad():
            # Take first few samples from batch
            sample_embeddings = batch["embedding"][:num_samples].to(self.device)
            
            # Get original captions for reference
            sample_refs = []
            if "original_caption" in batch:
                for i in range(min(num_samples, len(batch["original_caption"]))):
                    sample_refs.append(batch["original_caption"][i])
            else:
                # Fallback: decode from input_ids (no prompt removal needed now)
                sample_input_ids = batch["input_ids"][:num_samples]
                for ids in sample_input_ids:
                    decoded = self.tokenizer.decode(ids, skip_special_tokens=True)
                    sample_refs.append(decoded)
            
            # Generate captions using full generation pipeline
            sample_preds = self._generate_captions_from_embeddings(sample_embeddings, max_new_tokens=30)
            
            # Compute lightweight metrics for this sample
            sample_metrics = self._compute_lightweight_metrics(sample_preds, sample_refs)
            
            log.info("📝 Training samples:")
            for i, (pred, ref) in enumerate(zip(sample_preds, sample_refs)):
                log.info(f"  Sample {i+1}:")
                print(f"    \033[93mPred: {pred}\033[0m")  # Yellow
                print(f"    \033[92mRef:  {ref}\033[0m")   # Green
            
            # Log sample metrics
            log.info(f"📊 Sample metrics: len_ratio={sample_metrics['length_ratio']:.2f}, "
                    f"word_overlap={sample_metrics['word_overlap']:.1f}%, "
                    f"non_empty={sample_metrics['non_empty_ratio']:.1f}%")
            
            return sample_metrics
        
        # Return to training mode
        if TOP_K > 0:
            self.qwen.train()
        self.bridge.train()

    def train(self):
        log.info(f"🏃 Starting training for {EPOCHS} epochs...")
        for epoch in range(EPOCHS):
            # Set training mode only for trainable components
            if TOP_K > 0:
                self.qwen.train()
            self.bridge.train()
            
            epoch_losses = []
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
            
            for batch_idx, batch in enumerate(pbar):
                loss = self._forward_step(batch)
                epoch_losses.append(loss.item())
                
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Update progress bar
                avg_loss = sum(epoch_losses) / len(epoch_losses)
                pbar.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{avg_loss:.4f}")
                
                # Generate samples and compute lightweight metrics every BATCH_SAMPLING batches
                if batch_idx % BATCH_SAMPLING == 0 and batch_idx > 0:
                    log.info(f"🔍 Generating samples at batch {batch_idx}...")
                    sample_metrics = self._generate_sample_during_training(batch, num_samples=2)
                    
                    # Log sample metrics to wandb with batch-level prefix
                    wandb_sample_metrics = {f"train_sample/{k}": v for k, v in sample_metrics.items()}
                    wandb_sample_metrics["train_sample/batch_idx"] = batch_idx
                    wandb_sample_metrics["train_sample/epoch"] = epoch + 1
                    wandb.log(wandb_sample_metrics)
                
                # Log batch loss to wandb
                wandb.log({
                    "train/batch_loss": loss.item(),
                    "train/step": epoch * len(self.train_loader) + batch_idx
                })
            
            # Log epoch metrics
            epoch_avg_loss = sum(epoch_losses) / len(epoch_losses)
            wandb.log({
                "train/epoch_loss": epoch_avg_loss,
                "epoch": epoch + 1
            })
            
            log.info(f"📊 Epoch {epoch+1} complete - Average loss: {epoch_avg_loss:.4f}")
            
            # Eval after each epoch with samples
            log.info(f"📊 Running evaluation after epoch {epoch+1}...")
            self._evaluate(epoch)
            self._save_checkpoint(epoch)

        # Final test after training
        log.info("🎯 Training complete! Running final test evaluation...")
        self.test()

    def test(self):
        """Run final test evaluation"""
        log.info("🧪 Running test evaluation...")
        self._evaluate(EPOCHS, test=True)
        log.info("✅ Test evaluation complete!")

    # --------------------------------------------------------------------- #
    def _generate_captions_from_embeddings(self, img_embeddings: torch.Tensor, max_new_tokens: int = 50) -> List[str]:
        """Shared caption generation logic for both evaluation and inference"""
        captions = []
        batch_size = min(BATCH_SIZE, len(img_embeddings))
        
        for i in range(0, len(img_embeddings), batch_size):
            batch_embeddings = img_embeddings[i:i+batch_size]
            
            # Bridge: img_emb -> vis_tokens
            vis_tokens = self.bridge(batch_embeddings)
            vis_tokens = vis_tokens.to(dtype=self.qwen_dtype)
            
            # Create prompt tokens for inference consistency
            prompt_text = INFERENCE_PROMPT
            prompt_tokens = self.tokenizer(
                prompt_text, 
                return_tensors="pt", 
                add_special_tokens=False
            ).input_ids.to(self.device)
            
            # Expand prompt for batch
            batch_size_actual = vis_tokens.shape[0]
            prompt_tokens = prompt_tokens.repeat(batch_size_actual, 1)
            prompt_emb = self.qwen.get_input_embeddings()(prompt_tokens)
            
            # Add colon token after prompt
            colon_token = self.tokenizer(":", return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            colon_token = colon_token.repeat(batch_size_actual, 1)
            colon_emb = self.qwen.get_input_embeddings()(colon_token)
            
            # Concatenate: [visual_tokens, prompt_tokens, colon_token]
            full_inputs = torch.cat([vis_tokens, prompt_emb, colon_emb], dim=1)
            
            # Create attention mask
            total_seq_len = full_inputs.shape[1]
            full_attention = torch.ones(
                (batch_size_actual, total_seq_len), 
                device=self.device,
                dtype=torch.long
            )
            
            # Fixed generation parameters - avoid max_length/max_new_tokens conflict
            with torch.no_grad():
                outputs = self.qwen.generate(
                    inputs_embeds=full_inputs,
                    attention_mask=full_attention,
                    max_new_tokens=max_new_tokens,  # Only use max_new_tokens
                    min_new_tokens=3,  # Reduced from 5 to be less restrictive
                    do_sample=True,
                    temperature=0.8,  # Slightly higher for more diversity
                    top_p=0.9,
                    top_k=50,  # Add top_k for better control
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True
                )
            
            # Extract only the newly generated tokens
            skip_len = total_seq_len
            generated_ids = outputs.sequences[:, skip_len:]
            
            # Enhanced debug logging (only once)
            if not hasattr(self, '_logged_generation'):
                log.info(f"🔍 Generation debug:")
                log.info(f"  Input length: {total_seq_len}")
                log.info(f"  Output length: {outputs.sequences.shape[1]}")
                log.info(f"  Generated tokens: {generated_ids.shape[1]}")
                log.info(f"  Generation config: max_new_tokens={max_new_tokens}, min_new_tokens=3")
                log.info(f"  Pad token ID: {self.tokenizer.pad_token_id}")
                log.info(f"  EOS token ID: {self.tokenizer.eos_token_id}")
                if generated_ids.shape[1] > 0:
                    log.info(f"  First generated tokens: {generated_ids[0][:10].tolist()}")
                self._logged_generation = True
            
            # Handle generation failure with better fallback
            if generated_ids.numel() == 0 or outputs.sequences.shape[1] <= total_seq_len:
                log.warning("⚠️ No new tokens generated, using simplified approach")
                # Simplified generation: just visual tokens without prompt
                simple_outputs = self.qwen.generate(
                    inputs_embeds=vis_tokens,
                    max_new_tokens=15,  # Shorter for better success rate
                    min_new_tokens=1,   # Very permissive
                    do_sample=True,
                    temperature=1.0,    # More random for better generation
                    top_p=0.95,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True
                )
                simple_generated = simple_outputs.sequences[:, vis_tokens.shape[1]:]
                
                if simple_generated.numel() > 0:
                    batch_captions = self.tokenizer.batch_decode(simple_generated, skip_special_tokens=True)
                    batch_captions_clean = [f"[Fallback] {cap.strip()}" for cap in batch_captions if cap.strip()]
                    # Fill in empty results
                    while len(batch_captions_clean) < batch_size_actual:
                        batch_captions_clean.append("[No generation]")
                else:
                    batch_captions_clean = ["[Generation failed]"] * batch_size_actual
            else:
                # Clean up generated captions
                batch_captions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                batch_captions_clean = []
                
                for caption in batch_captions:
                    # Clean and remove any prompt artifacts
                    caption_clean = caption.strip()
                    
                    # Remove common prompt artifacts
                    prompt_artifacts = [
                        INFERENCE_PROMPT.lower(),
                        "examine the image carefully",
                        "provide a detailed description",
                        "covering objects, people, actions"
                    ]
                    
                    for artifact in prompt_artifacts:
                        if artifact in caption_clean.lower():
                            idx = caption_clean.lower().find(artifact)
                            caption_clean = caption_clean[idx + len(artifact):].strip()
                    
                    # Remove leading punctuation
                    while caption_clean and caption_clean[0] in ":.,!?- ":
                        caption_clean = caption_clean[1:].strip()
                    
                    # Fallback for empty captions
                    if not caption_clean:
                        caption_clean = "[Empty generation]"
                    
                    batch_captions_clean.append(caption_clean)
            
            captions.extend(batch_captions_clean)

        return captions

   # --------------------------------------------------------------------- #
    def _save_checkpoint(self, epoch: int):
        """Save checkpoint with proper metadata and validation"""
        # Create filename with all relevant parameters
        fname = OUTPUT_DIR_MODELS / f"{epoch+1:02d}_{VISION_ENCODER}_top{TOP_K}_{TRAINING_DATASET}.pt"
        
        # Prepare checkpoint data with metadata
        checkpoint_data = {
            "epoch": epoch + 1,
            "qwen_state_dict": self.qwen.state_dict(),
            "bridge_state_dict": self.bridge.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": {
                "VISION_ENCODER": VISION_ENCODER,
                "ENCODER_ID": ENCODER_ID,
                "IMG_EMB_DIM": IMG_EMB_DIM,
                "TOP_K": TOP_K,
                "TRAINING_DATASET": TRAINING_DATASET,
                "BATCH_SIZE": BATCH_SIZE,
                "LR_ADAPTER": LR_ADAPTER,
                "LR_QWEN": LR_QWEN,
                "qwen_hidden_size": self.qwen_hidden_size,
                "bridge_n_tokens": self.bridge.n_tokens,
                "bridge_out_dim": self.bridge.out_dim,
            },
            "model_info": {
                "qwen_trainable_params": sum(p.numel() for p in self.qwen.parameters() if p.requires_grad),
                "bridge_trainable_params": sum(p.numel() for p in self.bridge.parameters() if p.requires_grad),
                "total_trainable_params": sum(p.numel() for p in list(self.qwen.parameters()) + list(self.bridge.parameters()) if p.requires_grad),
            }
        }
        
        try:
            torch.save(checkpoint_data, fname)
            log.info(f"💾 Saved checkpoint → {fname}")
            log.info(f"   Epoch: {epoch + 1}, TOP_K: {TOP_K}, Dataset: {TRAINING_DATASET}")
            log.info(f"   Trainable params: {checkpoint_data['model_info']['total_trainable_params']:,}")
        except Exception as e:
            log.error(f"❌ Failed to save checkpoint: {e}")
            raise

    def _load_latest_checkpoint(self):
        """Load the latest checkpoint with comprehensive validation"""
        # Pattern matching for current configuration
        pattern = f"*_{VISION_ENCODER}_top{TOP_K}_{TRAINING_DATASET}.pt"
        checkpoint_files = list(OUTPUT_DIR_MODELS.glob(pattern))
        
        # Fallback to older naming convention if no files found
        if not checkpoint_files:
            old_pattern = f"*_{VISION_ENCODER}_top{TOP_K}.pt"
            checkpoint_files = list(OUTPUT_DIR_MODELS.glob(old_pattern))
            if checkpoint_files:
                log.warning(f"⚠️ Using old checkpoint naming convention: {old_pattern}")
        
        if not checkpoint_files:
            log.error(f"❌ No checkpoints found in {OUTPUT_DIR_MODELS}")
            log.error(f"   Searched for: {pattern}")
            log.error(f"   Available files: {list(OUTPUT_DIR_MODELS.glob('*.pt'))}")
            raise FileNotFoundError(f"No checkpoints found for {VISION_ENCODER}_top{TOP_K}_{TRAINING_DATASET}")
        
        # Sort by epoch number (extract from filename)
        def extract_epoch(fname):
            try:
                return int(fname.stem.split('_')[0])
            except (ValueError, IndexError):
                return 0
        
        latest_checkpoint = max(checkpoint_files, key=extract_epoch)
        log.info(f"📁 Loading checkpoint: {latest_checkpoint}")
        
        try:
            checkpoint = torch.load(latest_checkpoint, map_location=self.device, weights_only=False)
            
            # Validate checkpoint structure
            required_keys = ["qwen_state_dict", "bridge_state_dict"]
            missing_keys = [key for key in required_keys if key not in checkpoint]
            
            if missing_keys:
                # Handle old checkpoint format
                if "qwen" in checkpoint and "bridge" in checkpoint:
                    log.warning("⚠️ Loading old checkpoint format")
                    self.qwen.load_state_dict(checkpoint["qwen"])
                    self.bridge.load_state_dict(checkpoint["bridge"])
                else:
                    raise KeyError(f"Missing required keys in checkpoint: {missing_keys}")
            else:
                # Load new checkpoint format
                self.qwen.load_state_dict(checkpoint["qwen_state_dict"])
                self.bridge.load_state_dict(checkpoint["bridge_state_dict"])
                
                # Validate configuration if available
                if "config" in checkpoint:
                    config = checkpoint["config"]
                    
                    # Check critical configuration matches
                    config_checks = [
                        ("VISION_ENCODER", VISION_ENCODER),
                        ("TOP_K", TOP_K),
                        ("IMG_EMB_DIM", IMG_EMB_DIM),
                        ("qwen_hidden_size", self.qwen_hidden_size),
                    ]
                    
                    for key, expected in config_checks:
                        if key in config and config[key] != expected:
                            log.warning(f"⚠️ Config mismatch - {key}: checkpoint={config[key]}, current={expected}")
                
                # Log checkpoint info
                if "model_info" in checkpoint:
                    info = checkpoint["model_info"]
                    log.info(f"   Trainable params: {info.get('total_trainable_params', 'unknown'):,}")
                
                if "epoch" in checkpoint:
                    log.info(f"   Trained for: {checkpoint['epoch']} epochs")
            
            self.qwen.eval()
            self.bridge.eval()
            log.info("✅ Checkpoint loaded successfully")
            
        except Exception as e:
            log.error(f"❌ Failed to load checkpoint {latest_checkpoint}: {e}")
            raise

    def _encode_images_batch(self, image_paths: List[str]) -> torch.Tensor:
        """Encode a batch of images to embeddings"""
        log.info(f"🖼️ Processing {len(image_paths)} images...")
        
        # Load images
        from PIL import Image
        images = []
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
            except Exception as e:
                log.error(f"❌ Failed to load {path}: {e}")
                raise
        
        # Process in batches for efficiency
        batch_size = 16 if self.device.type == "cuda" else 4
        embeddings = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            
            # Process batch
            if VISION_ENCODER == "vit":
                batch_pixels = self.processor(images=batch_images, return_tensors="pt").pixel_values
            else:
                batch_pixels = self.processor(images=batch_images, return_tensors="pt").pixel_values
            
            batch_pixels = batch_pixels.to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                if VISION_ENCODER == "vit":
                    batch_embeddings = self.vision_encoder(batch_pixels)[0][:, 0]  # CLS token
                else:
                    batch_embeddings = self.vision_encoder.get_image_features(pixel_values=batch_pixels)
            
            embeddings.extend(batch_embeddings.cpu())
        
        return torch.stack(embeddings)

    def run_inference(self, image_paths: List[str]) -> List[str]:
        """Run inference on a batch of images"""
        log.info(f"🔮 Running inference on {len(image_paths)} images...")
        
        # Encode images
        img_embeddings = self._encode_images_batch(image_paths)
        img_embeddings = img_embeddings.to(self.device)
        
        # Generate captions using shared method
        captions = self._generate_captions_from_embeddings(img_embeddings, max_new_tokens=64)
        
        return captions
    
    def _evaluate(self, epoch: int, test: bool = False):
        """Evaluate model on eval/test set with full metrics"""
        loader = self.test_loader if test else self.eval_loader
        split_name = "test" if test else "eval"
        
        log.info(f"📊 Running {split_name} evaluation...")
        
        # Set to eval mode
        self.qwen.eval()
        self.bridge.eval()
        
        eval_losses = []
        all_preds = []
        all_refs = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"{split_name.title()} evaluation"):
                # Compute loss
                loss = self._forward_step(batch)
                eval_losses.append(loss.item())
                
                # Generate captions for metrics
                img_embeddings = batch["embedding"].to(self.device)
                preds = self._generate_captions_from_embeddings(img_embeddings, max_new_tokens=50)
                
                # Get reference captions
                if "original_caption" in batch:
                    refs = batch["original_caption"]
                else:
                    # Decode from input_ids as fallback
                    refs = []
                    for ids in batch["input_ids"]:
                        decoded = self.tokenizer.decode(ids, skip_special_tokens=True)
                        refs.append(decoded)
                
                all_preds.extend(preds)
                all_refs.extend(refs)
        
        # Compute metrics
        avg_loss = sum(eval_losses) / len(eval_losses)
        metrics = self._compute_metrics(all_preds, all_refs)
        
        # Log results
        log.info(f"📊 {split_name.title()} Results:")
        log.info(f"  Loss: {avg_loss:.4f}")
        log.info(f"  BLEU-4: {metrics['BLEU4']:.2f}")
        log.info(f"  CIDEr: {metrics['CIDEr']:.2f}")
        log.info(f"  SPICE: {metrics['SPICE']:.2f}")
        log.info(f"  ROUGE-L: {metrics['ROUGE_L']:.2f}")
        
        # Log to wandb
        wandb_metrics = {f"{split_name}/{k.lower()}": v for k, v in metrics.items()}
        wandb_metrics[f"{split_name}/loss"] = avg_loss
        wandb_metrics["epoch"] = epoch + 1
        wandb.log(wandb_metrics)
        
        # Show some sample predictions
        log.info(f"📝 Sample {split_name} predictions:")
        for i in range(min(3, len(all_preds))):
            print(f"  Sample {i+1}:")
            print(f"    \033[93mPred: {all_preds[i]}\033[0m")  # Yellow
            print(f"    \033[92mRef:  {all_refs[i]}\033[0m")   # Green
        
        # Return to training mode if needed
        if not test and TOP_K > 0:
            self.qwen.train()
        if not test:
            self.bridge.train()
        
        return metrics

###############################################################################
# 6.  CLI entry‑point                                                        #
###############################################################################

def parse_args():
    p = argparse.ArgumentParser(description="Multimodal Transfer‑Learning runner")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--embedding", action="store_true", help="Generate and cache vision embeddings")
    group.add_argument("--train", action="store_true", help="Train model using cached embeddings (includes final test)")
    group.add_argument("--test", action="store_true", help="Run standalone test evaluation using cached embeddings")
    group.add_argument("--run", nargs="+", metavar="IMAGE", help="Run inference on images (e.g., --run image1.jpg image2.jpg)")
    return p.parse_args()

def main():
    args = parse_args()
    
    if args.embedding:
        log.info("🔄 Mode: Embedding generation")
        engine = MultimodalTransferLearning(mode="embedding")
    elif args.train:
        log.info("🔄 Mode: Training (with final test)")
        engine = MultimodalTransferLearning(mode="train")
        engine.train()
    elif args.test:
        log.info("🔄 Mode: Test evaluation")
        engine = MultimodalTransferLearning(mode="test")
        engine.test()
        log.info("✅ Evaluation complete!")
    elif args.run:
        log.info(f"🔄 Mode: Inference on {len(args.run)} images")
        engine = MultimodalTransferLearning(mode="run")
        
        # Validate image paths
        valid_paths = []
        for path in args.run:
            if os.path.exists(path):
                valid_paths.append(path)
            else:
                log.warning(f"⚠️ Image not found: {path}")
        
        if not valid_paths:
            log.error("❌ No valid images found")
            return
        
        # Run inference
        captions = engine.run_inference(valid_paths)
        
        # Display results
        log.info("🎯 Inference Results:")
        for path, caption in zip(valid_paths, captions):
            log.info(f"📸 {os.path.basename(path)}: {caption}")
    else:
        log.error("❌ Must specify --embedding, --train, --test, or --run")

if __name__ == "__main__":
    main()