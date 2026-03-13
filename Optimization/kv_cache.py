"""
KV Cache implementation for EzeLLM.

Provides a KVCache class and modified attention/generation methods that
avoid recomputing the full sequence at every decode step.
"""

import sys
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from rotary_embedding_torch import RotaryEmbedding

# Add parent directory so we can import the original model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dev'))
from ezellm import EzeLLMConfig, SwiGLU, MLP, Block, EzeLLM


class KVCache:
    """Stores cached K and V tensors per layer for autoregressive generation."""

    def __init__(self, num_layers: int, max_seq_len: int, n_kv_heads: int,
                 head_dim: int, batch_size: int, device: torch.device,
                 dtype: torch.dtype):
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        # Pre-allocate cache tensors: (batch, n_kv_heads, max_seq_len, head_dim)
        self.k_cache = [
            torch.zeros(batch_size, n_kv_heads, max_seq_len, head_dim,
                        device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        self.v_cache = [
            torch.zeros(batch_size, n_kv_heads, max_seq_len, head_dim,
                        device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        # Current position (number of tokens cached so far)
        self.seq_len = 0

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """
        Append new K/V to the cache for a given layer.
        k, v: (batch, n_kv_heads, new_tokens, head_dim) — already RoPE-rotated for k
        """
        new_tokens = k.shape[2]
        start = self.seq_len
        end = self.seq_len + new_tokens
        self.k_cache[layer_idx][:, :, start:end, :] = k
        self.v_cache[layer_idx][:, :, start:end, :] = v

    def get(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cached K, V up to current seq_len for a given layer."""
        return (
            self.k_cache[layer_idx][:, :, :self.seq_len, :],
            self.v_cache[layer_idx][:, :, :self.seq_len, :],
        )

    def advance(self, num_tokens: int):
        """Advance the sequence position after all layers have been updated."""
        self.seq_len += num_tokens

    def reset(self):
        """Clear the cache for a new conversation."""
        self.seq_len = 0
        for i in range(self.num_layers):
            self.k_cache[i].zero_()
            self.v_cache[i].zero_()


def attn_forward_cached(attn_module, x: torch.Tensor, cache: Optional[KVCache],
                        layer_idx: int, offset: int) -> torch.Tensor:
    """
    Modified attention forward that works with KV cache.

    Args:
        attn_module: The _attn_ module from a Block
        x: Input tensor, (B, T, embed_size) where T is:
            - full prompt length during prefill
            - 1 during decode
        cache: KVCache instance (None = no caching, same as original)
        layer_idx: Index of the current transformer layer
        offset: Position offset for RoPE (0 during prefill, cache.seq_len during decode)
    """
    B, T, _ = x.size()

    # Generate q, k, v
    qkv = attn_module.qkv_gen(x)
    q, k, v = torch.split(
        qkv,
        [attn_module.n_heads * attn_module.head_dim,
         attn_module.n_kv_heads * attn_module.head_dim,
         attn_module.n_kv_heads * attn_module.head_dim],
        dim=2
    )

    # Reshape for multi-head attention
    q = q.view(B, T, attn_module.n_heads, attn_module.head_dim).transpose(1, 2)
    k = k.view(B, T, attn_module.n_kv_heads, attn_module.head_dim).transpose(1, 2)
    v = v.view(B, T, attn_module.n_kv_heads, attn_module.head_dim).transpose(1, 2)

    # Apply rotary embeddings with position offset
    q = attn_module.rotary_emb.rotate_queries_or_keys(q, offset=offset)
    k = attn_module.rotary_emb.rotate_queries_or_keys(k, offset=offset)

    if cache is not None:
        # Store new K/V in cache (K already has RoPE applied)
        cache.update(layer_idx, k, v)

        # Get full cached K/V (includes the tokens we just added)
        # We need to get them AFTER advance is called at the end, so we
        # compute the full range manually
        full_len = cache.seq_len + T  # seq_len hasn't been advanced yet
        k_full = cache.k_cache[layer_idx][:, :, :full_len, :]
        v_full = cache.v_cache[layer_idx][:, :, :full_len, :]

        # Replicate K/V heads for GQA
        k_full = k_full.repeat_interleave(attn_module.n_rep, dim=1)
        v_full = v_full.repeat_interleave(attn_module.n_rep, dim=1)

        if T == 1:
            # Decode: single query token, attend to all cached tokens
            # No causal mask needed — the single query is always the last position
            y = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=False)
        else:
            # Prefill: standard causal attention over the full prompt
            y = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=True)
    else:
        # No cache — original behavior
        k = k.repeat_interleave(attn_module.n_rep, dim=1)
        v = v.repeat_interleave(attn_module.n_rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    y = y.transpose(1, 2).contiguous().view(B, T, -1)
    y = attn_module.projection_attn1(y)
    return y


def forward_cached(model: EzeLLM, idx: torch.Tensor,
                   cache: Optional[KVCache] = None,
                   offset: int = 0) -> torch.Tensor:
    """
    Modified forward pass that uses KV cache.

    Args:
        model: EzeLLM model instance
        idx: Token indices (B, T)
        cache: KVCache instance
        offset: Position offset for RoPE
    Returns:
        logits: (B, T, vocab_size)
    """
    B, T = idx.size()
    tok_emb = model.transformer.wte(idx)
    x = tok_emb

    for i, block in enumerate(model.transformer.hiddens):
        # Pre-norm + attention with cache
        residual = x
        x_norm = block.ln1(x)
        attn_out = attn_forward_cached(block.attn, x_norm, cache, i, offset)
        x = residual + attn_out

        # Pre-norm + MLP (no caching needed)
        x = x + block.mlp(block.ln2(x))

    x = model.transformer.ln_f(x)

    # Fix the lm_head bug: use Sequential directly instead of looping
    logits = model.lm_head(x)

    return logits


def generate_cached(
    model: EzeLLM,
    input_: str = "I'm a",
    temperature: float = 1.0,
    topk: int = 50,
    max_l: int = 2048,
    num_return_seq: int = 1,
    verbose: bool = True,
) -> str:
    """
    Generate text using KV cache for efficient autoregressive decoding.

    Args:
        model: EzeLLM model instance
        input_: Prompt string
        temperature: Sampling temperature
        topk: Top-k filtering
        max_l: Maximum generation length
        num_return_seq: Number of sequences (only 1 supported with cache)
        verbose: Print generation stats
    Returns:
        Decoded text string
    """
    device = model.device
    tokenizer = model.tokenizer
    eot_id = model.eot_id

    tokens = tokenizer.encode(input_)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device)
    tokens = tokens.unsqueeze(0).repeat(num_return_seq, 1)
    x = tokens

    config = model.config
    cache = KVCache(
        num_layers=config.hidden_count,
        max_seq_len=max_l,
        n_kv_heads=config.n_kv_heads,
        head_dim=config.embed_size // config.head_count,
        batch_size=num_return_seq,
        device=device,
        dtype=next(model.parameters()).dtype,
    )

    gen_start = time.time()

    with torch.no_grad():
        # === Prefill: process entire prompt at once ===
        logits = forward_cached(model, x, cache=cache, offset=0)
        cache.advance(x.shape[1])

        # Sample first new token from last position
        logits_last = logits[:, -1, :] / temperature
        probs = F.softmax(logits_last, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, topk, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        next_token = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, next_token), dim=-1)

        if next_token.item() == eot_id:
            pass
        else:
            # === Decode: one token at a time ===
            while x.size(1) < max_l:
                offset = cache.seq_len
                logits = forward_cached(model, next_token, cache=cache, offset=offset)
                cache.advance(1)

                logits_last = logits[:, -1, :] / temperature
                probs = F.softmax(logits_last, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, topk, dim=-1)
                ix = torch.multinomial(topk_probs, 1)
                next_token = torch.gather(topk_indices, -1, ix)
                x = torch.cat((x, next_token), dim=-1)

                if next_token.item() == eot_id:
                    break

    gen_time = time.time() - gen_start
    total_tokens = x.shape[1]
    tokens_list = x[0, :max_l].tolist()
    decoded = tokenizer.decode(tokens_list)

    if verbose:
        print(f"Generated {total_tokens} tokens in {gen_time:.2f}s, "
              f"{total_tokens / gen_time:.2f} tokens/sec")

    return decoded


def generate_no_cache(
    model: EzeLLM,
    input_: str = "I'm a",
    temperature: float = 1.0,
    topk: int = 50,
    max_l: int = 2048,
    num_return_seq: int = 1,
    verbose: bool = True,
) -> str:
    """
    Generate text WITHOUT KV cache (original behavior but with lm_head bug fix).
    Used as baseline for verification.
    """
    device = model.device
    tokenizer = model.tokenizer
    eot_id = model.eot_id

    tokens = tokenizer.encode(input_)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device)
    tokens = tokens.unsqueeze(0).repeat(num_return_seq, 1)
    x = tokens

    gen_start = time.time()

    while x.size(1) < max_l:
        with torch.no_grad():
            logits = forward_cached(model, x, cache=None, offset=0)
            logits_last = logits[:, -1, :] / temperature
            probs = F.softmax(logits_last, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, topk, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            next_token = torch.gather(topk_indices, -1, ix)
            x = torch.cat((x, next_token), dim=-1)

            if next_token.item() == eot_id:
                break

    gen_time = time.time() - gen_start
    total_tokens = x.shape[1]
    tokens_list = x[0, :max_l].tolist()
    decoded = tokenizer.decode(tokens_list)

    if verbose:
        print(f"Generated {total_tokens} tokens in {gen_time:.2f}s, "
              f"{total_tokens / gen_time:.2f} tokens/sec")

    return decoded


def generate_deterministic(model, input_str, max_tokens, use_cache, temperature=1.0, topk=50):
    """
    Deterministic generation for verification: uses argmax (greedy) decoding
    to ensure reproducibility. Returns list of token IDs.
    """
    device = model.device
    tokenizer = model.tokenizer

    tokens = tokenizer.encode(input_str)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    config = model.config

    if use_cache:
        cache = KVCache(
            num_layers=config.hidden_count,
            max_seq_len=len(tokens[0]) + max_tokens + 1,
            n_kv_heads=config.n_kv_heads,
            head_dim=config.embed_size // config.head_count,
            batch_size=1,
            device=device,
            dtype=next(model.parameters()).dtype,
        )

        with torch.no_grad():
            # Prefill
            logits = forward_cached(model, tokens, cache=cache, offset=0)
            cache.advance(tokens.shape[1])

            generated = []
            next_tok_logits = logits[:, -1, :]
            next_token = torch.argmax(next_tok_logits, dim=-1, keepdim=True)
            generated.append(next_token.item())

            for _ in range(max_tokens - 1):
                offset = cache.seq_len
                logits = forward_cached(model, next_token, cache=cache, offset=offset)
                cache.advance(1)
                next_tok_logits = logits[:, -1, :]
                next_token = torch.argmax(next_tok_logits, dim=-1, keepdim=True)
                generated.append(next_token.item())

        return generated
    else:
        # No cache — full recompute each step
        x = tokens
        generated = []

        with torch.no_grad():
            for _ in range(max_tokens):
                logits = forward_cached(model, x, cache=None, offset=0)
                next_tok_logits = logits[:, -1, :]
                next_token = torch.argmax(next_tok_logits, dim=-1, keepdim=True)
                generated.append(next_token.item())
                x = torch.cat((x, next_token), dim=-1)

        return generated
