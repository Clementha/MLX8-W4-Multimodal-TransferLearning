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
import ast  # Add this import for parsing the prompt list
import pandas as pd  # Add for parquet storage

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

import random

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

# Training prompt settings
USE_PROMPT_VARIATION: bool = os.getenv("USE_PROMPT_VARIATION", "False").lower() == "true"
PROMPT_DROPOUT_RATE: float = float(os.getenv("PROMPT_DROPOUT_RATE", "0.15"))
PROMPT_RANDOM_SELECTION: bool = os.getenv("PROMPT_RANDOM_SELECTION", "True").lower() == "true"
PROMPT_SHUFFLE_PER_EPOCH: bool = os.getenv("PROMPT_SHUFFLE_PER_EPOCH", "True").lower() == "true"

BATCH_SAMPLING: int = int(os.getenv("BATCH_SAMPLING", "200"))  # Convert to int for proper usage

# Caption processing settings
SPLIT_CAPTIONS: bool = os.getenv("SPLIT_CAPTIONS", "True").lower() == "true"

# Training dataset limiting factor (1.0 = full dataset, 0.3 = 30% of dataset)
LIMITING_FACTOR: float = float(os.getenv("LIMITING_FACTOR", "1.0"))

# Eval and test dataset limiting factors
LIMITING_FACTOR_EVAL: float = float(os.getenv("LIMITING_FACTOR_EVAL", "1.0"))
LIMITING_FACTOR_TEST: float = float(os.getenv("LIMITING_FACTOR_TEST", "1.0"))

PROMPT_VARIATION_POOL = [
    "Describe this image.",
    "What do you see?",
    "Analyze the visual content.",
    "Provide an image description.",
    "Tell me about this picture.",
    "What's happening in this image?",
    "Explain what's shown here.",
    "Detail the contents of this image.",
    "Describe the scene.",
    "What are the main elements?",
    "Identify objects in the image.",
    "Summarize this visual.",
    "Caption this image.",
    "What's visible here?",
    "Describe the composition.",
    "Analyze this photograph.",
    "What can you observe?",
    "Explain the visual elements.",
    "Detail what you notice.",
    "Describe the setting.",
    "What's the main focus?",
    "Interpret this image.",
    "Provide visual details.",
    "What's depicted here?",
    "Describe the environment.",
    "Analyze the scene composition.",
    "What objects are present?",
    "Explain the visual narrative.",
    "Detail the image content.",
    "What's the overall scene?",
    "Describe key features.",
    "Analyze visual components.",
    "What stands out visually?",
    "Provide scene description.",
    "Explain what's pictured.",
    "Detail visual observations.",
    "What's in this photo?",
    "Describe image elements.",
    "Analyze the picture.",
    "What's the visual story?",
    "Explain the imagery.",
    "Detail scene contents.",
    "What do you notice?"
]
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
        # Add suffix to cache file name based on SPLIT_CAPTIONS
        split_suffix = "_split" if SPLIT_CAPTIONS else "_no_split"
        cache_file = OUTPUT_DIR_CACHE / f"{VISION_ENCODER}_{TRAINING_DATASET}_complete{split_suffix}.pt"
        
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
        
        # Now prepare data splits with embeddings based on SPLIT_CAPTIONS setting
        if SPLIT_CAPTIONS:
            log.info("⏳ Creating image-caption pairs with caption splitting (current logic)...")
            processed_examples = []
            
            for img_id, example in enumerate(tqdm(raw, desc="Processing examples with splitting")):
                img_embedding = embeddings[img_id]
                for caption in example["caption"]:
                    processed_examples.append({
                        "embedding": img_embedding,
                        "caption": caption
                    })
        else:
            log.info("⏳ Creating one image per caption (no splitting)...")
            processed_examples = []
            
            for img_id, example in enumerate(tqdm(raw, desc="Processing examples without splitting")):
                img_embedding = embeddings[img_id]
                # Use all captions joined as one string
                if example["caption"]:
                    caption = " ".join(example["caption"])
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
        log.info(f"📝 Caption splitting: {'enabled' if SPLIT_CAPTIONS else 'disabled'}")
        
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
                'split_captions': SPLIT_CAPTIONS,
                'created_at': torch.tensor([time.time()], dtype=torch.float64)
            }
        }
        
        log.info(f"💾 Saving complete dataset with embeddings to {cache_file}")
        torch.save(cache_data, cache_file)
        log.info(f"✅ Complete dataset cached successfully!")
        log.info(f"📊 Cache info: {len(processed_examples)} examples, {IMG_EMB_DIM}D embeddings, split_captions={SPLIT_CAPTIONS}")

    def _load_cached_embeddings(self):
        """Load cached complete dataset for training/testing"""
        split_suffix = "_split" if SPLIT_CAPTIONS else "_no_split"
        cache_file = OUTPUT_DIR_CACHE / f"{VISION_ENCODER}_{TRAINING_DATASET}_complete{split_suffix}.pt"
        
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
        
        def get_random_prompt():
            """Get a random prompt based on configuration"""
            if not USE_PROMPT_VARIATION:
                return INFERENCE_PROMPT
            
            # Apply prompt dropout
            if random.random() < PROMPT_DROPOUT_RATE:
                return ""  # No prompt (SOS/EOS style)
            
            # Select random prompt from pool
            if PROMPT_RANDOM_SELECTION and PROMPT_VARIATION_POOL:
                return random.choice(PROMPT_VARIATION_POOL)
            
            return INFERENCE_PROMPT
        
        def preprocess(example):
            # For training: use random prompts for variety
            prompt = get_random_prompt()
            
            # Add colon separator if prompt is not empty
            prompt_text = prompt + ": " if prompt else ""
            
            tokens = self.tokenizer(
                prompt_text + example['caption'], 
                truncation=True, 
                padding=False,
                return_tensors="pt"
            )
            example["input_ids"] = tokens.input_ids[0]
            example["original_caption"] = example["caption"]
            example["used_prompt"] = prompt  # Track which prompt was used
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
        log.info(f"Tokenizing captions with {'random prompts' if USE_PROMPT_VARIATION else 'fixed prompt'}...")
        log.info(f"📝 Prompt settings: variation={USE_PROMPT_VARIATION}, dropout={PROMPT_DROPOUT_RATE:.2f}, pool_size={len(PROMPT_VARIATION_POOL)}")
        
        # Shuffle prompts per epoch if enabled
        if PROMPT_SHUFFLE_PER_EPOCH and USE_PROMPT_VARIATION:
            random.shuffle(PROMPT_VARIATION_POOL)
            log.info("🔀 Shuffled prompt pool for this epoch")
        
        self.dataset["train"] = self.dataset["train"].map(
            preprocess, 
            remove_columns=["caption"],
            batch_size=1000,
            desc="Processing train"
        )
        
        # For eval/test, always use the inference prompt for consistency
        def preprocess_eval(example):
            prompt = INFERENCE_PROMPT + ": "
            tokens = self.tokenizer(
                prompt + example['caption'], 
                truncation=True, 
                padding=False,
                return_tensors="pt"
            )
            example["input_ids"] = tokens.input_ids[0]
            example["original_caption"] = example["caption"]
            example["used_prompt"] = INFERENCE_PROMPT
            return example
        
        self.dataset["eval"] = self.dataset["eval"].map(
            preprocess_eval, 
            remove_columns=["caption"],
            batch_size=1000,
            desc="Processing eval"
        )
        
        self.dataset["test"] = self.dataset["test"].map(
            preprocess_eval, 
            remove_columns=["caption"],
            batch_size=1000,
            desc="Processing test"
        )
        
        self.dataset.set_format(type="torch")
        
        # Apply limiting factor to training dataset only
        if LIMITING_FACTOR < 1.0:
            original_train_size = len(self.dataset["train"])
            limited_size = int(original_train_size * LIMITING_FACTOR)
            
            # Randomly sample a subset of training data
            indices = list(range(original_train_size))
            random.shuffle(indices)
            selected_indices = indices[:limited_size]
            
            # Create limited training dataset
            self.dataset["train"] = self.dataset["train"].select(selected_indices)
            
            log.info(f"🎯 Applied LIMITING_FACTOR={LIMITING_FACTOR}: {original_train_size} → {limited_size} training samples")
        else:
            log.info(f"📚 Using full training dataset: {len(self.dataset['train'])} samples")
        
        # Apply limiting factor to eval dataset
        if LIMITING_FACTOR_EVAL < 1.0:
            original_eval_size = len(self.dataset["eval"])
            limited_eval_size = int(original_eval_size * LIMITING_FACTOR_EVAL)
            
            # Randomly sample a subset of eval data
            eval_indices = list(range(original_eval_size))
            random.shuffle(eval_indices)
            selected_eval_indices = eval_indices[:limited_eval_size]
            
            # Create limited eval dataset
            self.dataset["eval"] = self.dataset["eval"].select(selected_eval_indices)
            
            log.info(f"🎯 Applied LIMITING_FACTOR_EVAL={LIMITING_FACTOR_EVAL}: {original_eval_size} → {limited_eval_size} eval samples")
        else:
            log.info(f"📚 Using full eval dataset: {len(self.dataset['eval'])} samples")
        
        # Apply limiting factor to test dataset
        if LIMITING_FACTOR_TEST < 1.0:
            original_test_size = len(self.dataset["test"])
            limited_test_size = int(original_test_size * LIMITING_FACTOR_TEST)
            
            # Randomly sample a subset of test data
            test_indices = list(range(original_test_size))
            random.shuffle(test_indices)
            selected_test_indices = test_indices[:limited_test_size]
            
            # Create limited test dataset
            self.dataset["test"] = self.dataset["test"].select(selected_test_indices)
            
            log.info(f"🎯 Applied LIMITING_FACTOR_TEST={LIMITING_FACTOR_TEST}: {original_test_size} → {limited_test_size} test samples")
        else:
            log.info(f"📚 Using full test dataset: {len(self.dataset['test'])} samples")
        
        self.train_loader = DataLoader(self.dataset["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        self.eval_loader = DataLoader(self.dataset["eval"], batch_size=BATCH_SIZE, collate_fn=collate_fn)
        self.test_loader = DataLoader(self.dataset["test"], batch_size=BATCH_SIZE, collate_fn=collate_fn)
        
        log.info("✅ Data loaders ready!")
        log.info(f"📊 Final dataset sizes: train={len(self.dataset['train'])}, eval={len(self.dataset['eval'])}, test={len(self.dataset['test'])}")

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
        # Store original training state
        qwen_training = self.qwen.training
        bridge_training = self.bridge.training
        
        # Set to eval mode for generation
        self.qwen.eval()
        self.bridge.eval()
        
        try:
            with torch.no_grad():
                # Take first few samples from batch
                sample_embeddings = batch["embedding"][:num_samples].to(self.device)
                
                # Get original captions for reference
                sample_refs = []
                if "original_caption" in batch:
                    for i in range(min(num_samples, len(batch["original_caption"]))):
                        sample_refs.append(batch["original_caption"][i])
                else:
                    # Fallback: decode from input_ids
                    sample_input_ids = batch["input_ids"][:num_samples]
                    for ids in sample_input_ids:
                        decoded = self.tokenizer.decode(ids, skip_special_tokens=True)
                        sample_refs.append(decoded)
                
                # Generate captions with shorter max_new_tokens for training samples
                sample_preds = self._generate_captions_with_prompt(sample_embeddings, prompt="", max_new_tokens=50)
                
                # Ensure we have the right number of predictions
                while len(sample_preds) < len(sample_refs):
                    sample_preds.append("no prediction")
                
                # Compute lightweight metrics for this sample
                sample_metrics = self._compute_lightweight_metrics(sample_preds, sample_refs)
                
                log.info("📝 Training samples:")
                for i, (pred, ref) in enumerate(zip(sample_preds, sample_refs)):
                    log.info(f"  Sample {i+1}:")
                    print(f"    \033[92mRef:  {ref}\033[0m")   # Green (reference first)
                    print(f"    \033[93mPred: {pred}\033[0m")  # Yellow (prediction second)
            
                # Log sample metrics
                log.info(f"📊 Sample metrics: len_ratio={sample_metrics['length_ratio']:.2f}, "
                        f"word_overlap={sample_metrics['word_overlap']:.1f}%," 
                        f" non_empty={sample_metrics['non_empty_ratio']:.1f}%")
                
                return sample_metrics
                
        except Exception as e:
            log.error(f"❌ Sample generation failed: {e}")
            # Return empty metrics
            return {
                'avg_pred_length': 0.0,
                'avg_ref_length': 0.0, 
                'length_ratio': 0.0,
                'non_empty_ratio': 0.0,
                'word_overlap': 0.0,
                'sample_count': 0
            }
        finally:
            # Restore original training state
            if qwen_training:
                self.qwen.train()
            if bridge_training:
                self.bridge.train()

    def train(self):
        log.info(f"🏃 Starting training for {EPOCHS} epochs...")
        log.info(f"📝 Using {'random prompts' if USE_PROMPT_VARIATION else 'fixed prompt'} during training")
        
        for epoch in range(EPOCHS):
            # Shuffle prompts at the beginning of each epoch if enabled
            if PROMPT_SHUFFLE_PER_EPOCH and USE_PROMPT_VARIATION:
                random.shuffle(PROMPT_VARIATION_POOL)
                log.info(f"🔀 Epoch {epoch+1}: Shuffled prompt pool")
            
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

    def _save_checkpoint(self, epoch: int):
        """
        Save a checkpoint containing the Qwen and bridge model states, along with config and metadata.
        """
        try:
            # Compose checkpoint filename
            filename = f"{epoch+1:02d}_{VISION_ENCODER}_top{TOP_K}_{TRAINING_DATASET}.pt"
            checkpoint_path = OUTPUT_DIR_MODELS / filename

            # Gather model state dicts
            checkpoint = {
                "epoch": epoch + 1,
                "qwen_state_dict": self.qwen.state_dict(),
                "bridge_state_dict": self.bridge.state_dict(),
                "config": {
                    "VISION_ENCODER": VISION_ENCODER,
                    "TOP_K": TOP_K,
                    "IMG_EMB_DIM": IMG_EMB_DIM,
                    "TRAINING_DATASET": TRAINING_DATASET,
                    "qwen_hidden_size": self.qwen_hidden_size,
                },
                "model_info": {
                    "total_trainable_params": sum(p.numel() for p in list(self.qwen.parameters()) + list(self.bridge.parameters()) if p.requires_grad),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            }

            torch.save(checkpoint, checkpoint_path)
            log.info(f"💾 Saved checkpoint: {checkpoint_path}")

        except Exception as e:
            log.error(f"❌ Failed to save checkpoint: {e}")

    def test(self):
        """Run final test evaluation"""
        log.info("🧪 Running test evaluation...")
        self._evaluate(EPOCHS, test=True)
        log.info("✅ Test evaluation complete!")

    # --------------------------------------------------------------------- #
    def _generate_captions_with_prompt(self, img_embeddings: torch.Tensor, prompt: str = None, max_new_tokens: int = 100) -> List[str]:
        """Shared caption generation logic with optional prompt"""
        captions = []
        batch_size = min(BATCH_SIZE, len(img_embeddings))
        
        # Use provided prompt or default inference prompt
        if prompt is None:
            prompt = INFERENCE_PROMPT
        
        # Ensure model is in eval mode
        self.qwen.eval()
        self.bridge.eval()
        
        for i in range(0, len(img_embeddings), batch_size):
            batch_embeddings = img_embeddings[i:i+batch_size]
            
            try:
                # Bridge: img_emb -> vis_tokens
                vis_tokens = self.bridge(batch_embeddings)
                vis_tokens = vis_tokens.to(dtype=self.qwen_dtype)
                
                batch_size_actual = vis_tokens.shape[0]
                
                if prompt and prompt.strip():
                    # With prompt: visual tokens + prompt embeddings
                    prompt_text = prompt + ": "
                    prompt_tokens = self.tokenizer(
                        [prompt_text] * batch_size_actual, 
                        return_tensors="pt", 
                        padding=True
                    ).to(self.device)
                    
                    # Get prompt embeddings
                    prompt_emb = self.qwen.get_input_embeddings()(prompt_tokens.input_ids)
                    
                    # Concatenate visual tokens + prompt embeddings
                    inputs = torch.cat([vis_tokens, prompt_emb], dim=1)
                    
                    # Create attention mask for visual + prompt tokens
                    vis_seq_len = vis_tokens.shape[1]
                    vis_attention = torch.ones((batch_size_actual, vis_seq_len), device=self.device, dtype=prompt_tokens.attention_mask.dtype)
                    full_attention_mask = torch.cat([vis_attention, prompt_tokens.attention_mask], dim=1)

                     # Skip only the prompt tokens (visual tokens are not in the sequence)
                    skip_length = prompt_tokens.input_ids.shape[1]
                else:
                    # Without prompt: visual tokens only
                    inputs = vis_tokens
                    full_attention_mask = torch.ones((batch_size_actual, vis_tokens.shape[1]), device=self.device, dtype=torch.long)
    
                    # No prompt to skip
                    skip_length = 0
                
                with torch.no_grad():
                    try:
                        # Generate with context
                        outputs = self.qwen.generate(
                            inputs_embeds=inputs,
                            attention_mask=full_attention_mask,
                            max_new_tokens=max_new_tokens,
                            min_new_tokens=1,
                            do_sample=True,
                            temperature=0.1,
                            top_p=0.92,
                            top_k=50,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            repetition_penalty=1.5,
                            no_repeat_ngram_size=3,
                            use_cache=True,
                            return_dict_in_generate=True
                        )
                        
                        # Extract generated tokens (skip input length)
                        input_length = inputs.shape[1]
                        generated_ids = outputs.sequences[:, skip_length:] #input_length:]

                        actual_input_length = 0
                        if outputs.sequences.shape[1] > 0:
                            # Try to decode first few tokens to see what they are
                            first_tokens = outputs.sequences[0, :min(20, outputs.sequences.shape[1])]
                            decoded_first = self.tokenizer.decode(first_tokens, skip_special_tokens=True)
                            print(f"DEBUG: First tokens decoded: '{decoded_first}'")
                            
                            # Check if this contains your prompt
                            if prompt and prompt.strip():
                                prompt_text = prompt + ": "
                                if prompt_text in decoded_first:
                                    print(f"DEBUG: Found prompt in sequence, visual tokens NOT included")
                                    # Only skip prompt tokens, not visual tokens
                                    prompt_tokens = self.tokenizer(prompt_text, return_tensors="pt")
                                    actual_input_length = prompt_tokens.input_ids.shape[1]
                                else:
                                    print(f"DEBUG: Prompt not found, visual tokens might be included")
                                    actual_input_length = input_length
                            else:
                                print(f"DEBUG: No prompt used")
                                actual_input_length = 0  # No tokens to skip if no prompt

                        generated_ids = outputs.sequences[:, actual_input_length:]
                        generated_ids = outputs.sequences[:, 0:]
                        
                        # Decode generated tokens
                        if generated_ids.numel() > 0 and generated_ids.shape[1] > 0:
                            batch_captions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                            batch_captions_clean = []
                            
                            for caption in batch_captions:
                                caption_clean = caption.strip()
                                
                                # Remove common artifacts
                                artifacts_to_remove = [
                                    "image shows", "the image", "this image", 
                                    "in the image", "picture shows", "photo shows"
                                ]
                                
                                # Simple cleaning
                                for artifact in artifacts_to_remove:
                                    if caption_clean.lower().startswith(artifact):
                                        caption_clean = caption_clean[len(artifact):].strip()
                                
                                # Remove leading punctuation/symbols
                                while caption_clean and caption_clean[0] in ":.,!?-() ":
                                    caption_clean = caption_clean[1:].strip()
                                
                                # Ensure we have something
                                if not caption_clean or len(caption_clean) < 3:
                                    caption_clean = "a scene"
                                
                                batch_captions_clean.append(caption_clean)
                        else:
                            # No tokens generated - use fallback
                            batch_captions_clean = ["no description available"] * batch_size_actual
                    
                    except Exception as gen_error:
                        log.warning(f"⚠️ Generation failed: {gen_error}")
                        # Fallback generation
                        batch_captions_clean = ["generation failed"] * batch_size_actual
                
                captions.extend(batch_captions_clean)
                
            except Exception as batch_error:
                log.error(f"❌ Batch processing failed: {batch_error}")
                # Emergency fallback
                batch_size_actual = len(batch_embeddings)
                emergency_captions = ["processing error"] * batch_size_actual
                captions.extend(emergency_captions)

        return captions

    def _save_evaluation_results(self, preds: List[str], refs: List[str], img_ids: List[int] = None, 
                                epoch: int = None, split: str = "eval", metrics: Dict = None):
        """Save evaluation results to parquet file"""
        try:
            # Create results dataframe
            if img_ids is None:
                img_ids = list(range(len(preds)))
            
            df_data = {
                'img_id': img_ids,
                'reference': refs,
                'prediction': preds,
                'split': [split] * len(preds)
            }
            
            # Add metrics as columns (repeated for all rows)
            if metrics:
                for key, value in metrics.items():
                    df_data[f'metric_{key.lower()}'] = [value] * len(preds)
            
            df = pd.DataFrame(df_data)
            
            # Create filename similar to wandb run name
            if epoch is not None:
                filename = f"{epoch+1:02d}_{VISION_ENCODER}_top{TOP_K}_{TRAINING_DATASET}_{split}_results.parquet"
            else:
                timestamp = int(time.time())
                filename = f"inference_{VISION_ENCODER}_top{TOP_K}_{TRAINING_DATASET}_{timestamp}_results.parquet"
            
            results_file = OUTPUT_DIR_MODELS / filename
            df.to_parquet(results_file, index=False)
            
            log.info(f"💾 Saved {split} results → {results_file}")
            log.info(f"   Samples: {len(df)}, Columns: {list(df.columns)}")
            
        except Exception as e:
            log.error(f"❌ Failed to save evaluation results: {e}")

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
        """Run inference on a batch of images using optimized shared logic"""
        log.info(f"🔮 Running inference on {len(image_paths)} images...")
        log.info(f"📝 Using inference prompt: '{INFERENCE_PROMPT[:50]}...'")
        
        # Encode images using existing method
        img_embeddings = self._encode_images_batch(image_paths)
        img_embeddings = img_embeddings.to(self.device)
        
        # Use shared caption generation method
        captions = self._generate_captions_with_prompt(img_embeddings, prompt=INFERENCE_PROMPT, max_new_tokens=100)
        
        # Save inference results to parquet
        img_ids = [f"inference_{i}" for i in range(len(image_paths))]
        image_names = [os.path.basename(path) for path in image_paths]
        
        # Create a simple dataframe for inference results
        try:
            df_data = {
                'img_id': img_ids,
                'image_path': image_paths,
                'image_name': image_names,
                'prediction': captions,
                'prompt_used': [INFERENCE_PROMPT] * len(captions)
            }
            
            df = pd.DataFrame(df_data)
            timestamp = int(time.time())
            filename = f"inference_{VISION_ENCODER}_top{TOP_K}_{TRAINING_DATASET}_{timestamp}_results.parquet"
            results_file = OUTPUT_DIR_MODELS / filename
            df.to_parquet(results_file, index=False)
            
            log.info(f"💾 Saved inference results → {results_file}")
            
        except Exception as e:
            log.warning(f"⚠️ Failed to save inference results: {e}")
        
        return captions
    
    def _evaluate(self, epoch: int, test: bool = False):
        """Evaluate model on eval/test set with full metrics and parquet storage"""
        loader = self.test_loader if test else self.eval_loader
        split_name = "test" if test else "eval"
        
        log.info(f"📊 Running {split_name} evaluation...")
        
        # Set to eval mode
        self.qwen.eval()
        self.bridge.eval()
        
        eval_losses = []
        all_preds = []
        all_refs = []
        all_img_ids = []
        
        with torch.no_grad():
            current_img_id = 0
            for batch in tqdm(loader, desc=f"{split_name.title()} evaluation"):
                # Compute loss
                loss = self._forward_step(batch)
                eval_losses.append(loss.item())
                
                # Generate captions using shared method with inference prompt
                img_embeddings = batch["embedding"].to(self.device)
                preds = self._generate_captions_with_prompt(img_embeddings, prompt=INFERENCE_PROMPT, max_new_tokens=100)
                
                # Get reference captions (don't split for test/eval)
                if "original_caption" in batch:
                    refs = batch["original_caption"]
                else:
                    # Decode from input_ids as fallback
                    refs = []
                    for ids in batch["input_ids"]:
                        decoded = self.tokenizer.decode(ids, skip_special_tokens=True)
                        # Remove prompt prefix if present
                        if INFERENCE_PROMPT in decoded:
                            decoded = decoded.replace(INFERENCE_PROMPT + ": ", "").strip()
                        refs.append(decoded)
                
                # Create image IDs for this batch
                batch_img_ids = [f"{split_name}_{current_img_id + i}" for i in range(len(preds))]
                current_img_id += len(preds)
                
                all_preds.extend(preds)
                all_refs.extend(refs)
                all_img_ids.extend(batch_img_ids)

                # Show some sample predictions every 200 images
                if current_img_id % 200 < len(preds):
                    log.info(f"📝 Sample {split_name} predictions at image {current_img_id}:")
                    for i in range(min(3, len(preds))):
                        print(f"  Sample {i+1}:")
                        print(f"    \033[92mRef:  {refs[i]}\033[0m")   # Green (reference first)
                        print(f"    \033[93mPred: {preds[i]}\033[0m")  # Yellow (prediction second)
        
        # Compute metrics
        avg_loss = sum(eval_losses) / len(eval_losses)
        metrics = self._compute_metrics(all_preds, all_refs)
        
        # Save results to parquet
        self._save_evaluation_results(all_preds, all_refs, all_img_ids, epoch, split_name, metrics)
        
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
        # log.info(f"📝 Sample {split_name} predictions:")
        # for i in range(min(3, len(all_preds))):
        #     print(f"  Sample {i+1}:")
        #     print(f"    \033[92mRef:  {all_refs[i]}\033[0m")   # Green (reference first)
        #     print(f"    \033[93mPred: {all_preds[i]}\033[0m")  # Yellow (prediction second)
        
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