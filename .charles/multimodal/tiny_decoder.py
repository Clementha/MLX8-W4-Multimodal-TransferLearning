"""Tiny causal Transformer decoder (GPT‑style) written **from scratch** in PyTorch.

This is *not* the full Qwen‑3 model—just a minimal, readable clone of the
core architectural ideas so you can experiment freely without the heavy
Transformers dependency.

Usage example >>>
    from tiny_decoder import TinyGPTDecoder

    model = TinyGPTDecoder(
        vocab_size=32000,
        d_model=512,
        n_heads=8,
        n_layers=6,
        max_seq_len=256,
    ).cuda()

    tokens = torch.randint(0, 32000, (4, 128)).cuda()
    logits = model(tokens)  # (B, T, vocab_size)

Author: Charles Cai 2025
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Building blocks                                                              #
# --------------------------------------------------------------------------- #

class CausalSelfAttention(nn.Module):
    """Multi‑head causal self‑attention written with linear ops only."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        B, T, D = x.shape
        qkv = self.qkv(x)  # (B, T, 3D)
        qkv = qkv.view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, h, T, d)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, h, T, T)
        attn = attn.masked_fill(attn_mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v  # (B, h, T, d)
        out = out.transpose(1, 2).reshape(B, T, D)  # (B, T, D)
        out = self.proj_drop(self.out(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model: int, expand: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, expand * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expand * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask):
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.ff(self.ln2(x))
        return x

# --------------------------------------------------------------------------- #
# Tiny GPT‑style decoder                                                       #
# --------------------------------------------------------------------------- #

class TinyGPTDecoder(nn.Module):
    """Pure‑PyTorch causal decoder + token/positional embeddings + LM head."""

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def _causal_mask(self, T: int, device: torch.device):
        mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)
        return mask  # (1, 1, T, T)

    def forward(self, tokens: torch.Tensor):  # tokens (B, T)
        B, T = tokens.shape
        device = tokens.device
        tok = self.token_emb(tokens)
        pos = self.pos_emb(torch.arange(T, device=device))
        x = self.drop(tok + pos)
        mask = self._causal_mask(T, device)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln_final(x)
        logits = self.head(x)  # (B, T, vocab)
        return logits

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new_tokens: int = 20):
        self.eval()
        for _ in range(max_new_tokens):
            logits = self.forward(prompt)[:, -1]  # (B, vocab)
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            prompt = torch.cat([prompt, next_id], dim=-1)
        return prompt


# --------------------------------------------------------------------------- #
# Sanity‑check                                                                 #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    torch.manual_seed(0)
    model = TinyGPTDecoder().cuda()
    dummy = torch.randint(0, 32000, (2, 64)).cuda()
    print(model(dummy).shape)        # => torch.Size([2, 64, 32000])
    out = model.generate(dummy[:, :10])
    print(out.shape)                 # => (2, 30)



# Clone Qwen's architecutre, then load wedights

#Adapt your code skeleton

from tiny_decoder import TinyGPTDecoder       # reuse your blocks

qwen_cfg = dict(
    vocab_size = 151_936,        # from config.json
    d_model    = 1024,
    n_layers   = 28,
    n_heads    = 16,             # implement GQA: 16 Q heads, 8 KV heads
    max_seq_len= 32_768,
)
model = TinyGPTDecoder(**qwen_cfg).cuda()

# Changes you must implement:

# Grouped-Query Attention (GQA).
# Qwen splits Q versus KV heads (16 / 8) 
# huggingface.co
# . Add two projection sets or follow Llama-2’s pattern.

# Rotary embeddings.
# Replace the learned pos_emb table with RoPE; otherwise most attention weights will be nonsense 
# huggingface.co
# .

# SiLU FFN with 3× expand.
# Qwen uses silu(x)·x and intermediate_size = 3 072 (not 4 096) 
# huggingface.co
# .

# RMSNorm instead of LayerNorm.
# Swap LayerNorm for torch.nn.functional.normalize with ε≈1e-5; Qwen omits mean-centering.

# Weight names.
# Follow the Hugging Face naming (model.layers.0.self_attn.q_proj.weight, …) so load_state_dict can match keys.



# # Download & load the pretrained state-dict
# Any non-empty missing list means some tensors still disagree.
# Refactor until you see missing: 0 unexpected: 0 – then forward passes will reproduce Qwen logits within numerical noise.
from transformers import AutoModelForCausalLM
import torch, tempfile, os

# 1) fetch reference weights
ref = AutoModelForCausalLM.from_pretrained(
          "Qwen/Qwen3-0.6B-Base",
          torch_dtype="bf16",
          device_map="cpu")         # no GPU RAM needed
qwen_sd = ref.state_dict()

# 2) load into your replica
missing, unexpected = model.load_state_dict(qwen_sd, strict=False)
print("missing:", len(missing), "unexpected:", len(unexpected))


# Partial boot-strapping (research sandbox)
# If you only want to re-use the embeddings or lower attention blocks while keeping a slim decoder:

sub_sd = {k: v for k, v in qwen_sd.items() if k.startswith("model.layers.0.")}
model.load_state_dict(sub_sd, strict=False)   # ignore size mismatch elsewhere :contentReference[oaicite:9]{index=9}

# This trick lets you “steal” Qwen’s token embeddings (embedding.word_embeddings.weight, shape 151 936 × 1 024) 
# and maybe the first N transformer blocks, then fine-tune the rest from scratch – but expect quality to be lower than a faithful replica.


# PyTorch will happily skip tens of megabytes of incompatible tensors without warning when strict=False discuss.pytorch.org; 
# the resulting network may run but its output will be garbage.
# Always inspect the missing and unexpected lists, or print shapes, before assuming the port worked.



# verifying logits parity
# Values below 1e-3 indicate the clone matches the official implementation.

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
sample = tok("Hello world", return_tensors="pt").input_ids.cuda()

with torch.no_grad():
    ref_logits = ref(sample).logits[:, -1]
    my_logits  = model(sample).logits[:, -1]
    print("Max abs diff:", (ref_logits - my_logits).abs().max())
