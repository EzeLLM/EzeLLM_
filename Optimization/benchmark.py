"""
Benchmark script for EzeLLM optimization variants.

Measures tokens/second, peak VRAM, model size, and output quality
across all optimized model variants.
"""

import sys
import os
import time
import gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dev'))

import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

from ezellm import EzeLLM
from kv_cache import (
    forward_cached, generate_cached, generate_no_cache, KVCache
)
from quantize import load_int8_calibrated


PROMPT = "The theory of general relativity explains"
NUM_GENERATE_TOKENS = 100
TOPK = 50
TEMPERATURE = 1.0


def get_file_size_mb(path: str) -> float:
    """Get file size in MB."""
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    return 0.0


def measure_generation(model, prompt, max_tokens, use_cache, device,
                       warmup_runs=1):
    """
    Measure generation speed and VRAM usage.
    Returns (tokens_per_sec, peak_vram_mb, generated_text).
    """
    tokenizer = model.tokenizer

    # Warmup
    for _ in range(warmup_runs):
        if use_cache:
            generate_cached(model, prompt, temperature=TEMPERATURE, topk=TOPK,
                            max_l=len(tokenizer.encode(prompt)) + 10,
                            verbose=False)
        else:
            generate_no_cache(model, prompt, temperature=TEMPERATURE, topk=TOPK,
                              max_l=len(tokenizer.encode(prompt)) + 10,
                              verbose=False)
        if device == 'cuda':
            torch.cuda.synchronize()

    # Reset VRAM tracking
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    prompt_len = len(tokenizer.encode(prompt))
    max_l = prompt_len + max_tokens

    # Timed generation
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()

    if use_cache:
        text = generate_cached(
            model, prompt, temperature=TEMPERATURE, topk=TOPK,
            max_l=max_l, verbose=False
        )
    else:
        text = generate_no_cache(
            model, prompt, temperature=TEMPERATURE, topk=TOPK,
            max_l=max_l, verbose=False
        )

    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # Count generated tokens (total - prompt)
    total_tokens = len(tokenizer.encode(text))
    gen_tokens = total_tokens - prompt_len
    tokens_per_sec = gen_tokens / elapsed if elapsed > 0 else 0

    # Peak VRAM
    peak_vram_mb = 0
    if device == 'cuda':
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return tokens_per_sec, peak_vram_mb, text


def measure_int8_dynamic(model_path, prompt, max_tokens, device):
    """
    Benchmark INT8 dynamic quantized model.
    Dynamic quantization only works on CPU, so we measure CPU speed.
    """
    from quantize import load_int8_dynamic

    model = load_int8_dynamic(model_path, device='cpu')
    model.eval()
    model.device = 'cpu'

    tokenizer = model.tokenizer

    prompt_len = len(tokenizer.encode(prompt))
    max_l = prompt_len + max_tokens

    # Warmup
    generate_no_cache(model, prompt, temperature=TEMPERATURE, topk=TOPK,
                      max_l=prompt_len + 5, verbose=False)

    start = time.perf_counter()
    text = generate_no_cache(
        model, prompt, temperature=TEMPERATURE, topk=TOPK,
        max_l=max_l, verbose=False
    )
    elapsed = time.perf_counter() - start

    total_tokens = len(tokenizer.encode(text))
    gen_tokens = total_tokens - prompt_len
    tokens_per_sec = gen_tokens / elapsed if elapsed > 0 else 0

    return tokens_per_sec, 0, text


def compute_perplexity(model, text, device, use_cache=False):
    """Compute perplexity on a short text sample."""
    tokenizer = model.tokenizer
    tokens = tokenizer.encode(text)
    # Limit to 512 tokens for speed
    tokens = tokens[:512]
    if len(tokens) < 2:
        return float('inf')

    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = forward_cached(model, input_ids, cache=None, offset=0)

    # Shift: predict next token from each position
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    return torch.exp(loss).item()


def print_table(results):
    """Print a formatted comparison table."""
    # Column widths
    name_w = 25
    tok_w = 12
    vram_w = 12
    size_w = 12
    ppl_w = 14

    border_top = (
        f"╔{'═' * name_w}╦{'═' * tok_w}╦{'═' * vram_w}"
        f"╦{'═' * size_w}╦{'═' * ppl_w}╗"
    )
    border_mid = (
        f"╠{'═' * name_w}╬{'═' * tok_w}╬{'═' * vram_w}"
        f"╬{'═' * size_w}╬{'═' * ppl_w}╣"
    )
    border_bot = (
        f"╚{'═' * name_w}╩{'═' * tok_w}╩{'═' * vram_w}"
        f"╩{'═' * size_w}╩{'═' * ppl_w}╝"
    )

    header = (
        f"║{'Variant':^{name_w}}║{'tok/sec':^{tok_w}}║{'VRAM MB':^{vram_w}}"
        f"║{'Size MB':^{size_w}}║{'Perplexity':^{ppl_w}}║"
    )

    print()
    print(border_top)
    print(header)
    print(border_mid)

    for r in results:
        vram_str = f"{r['vram_mb']:.0f}" if r['vram_mb'] > 0 else "N/A (CPU)"
        ppl_str = f"{r['perplexity']:.2f}" if r['perplexity'] < 1e6 else "N/A"
        row = (
            f"║{r['name']:<{name_w}}║{r['tok_sec']:^{tok_w}.1f}║{vram_str:^{vram_w}}"
            f"║{r['size_mb']:^{size_w}.1f}║{ppl_str:^{ppl_w}}║"
        )
        print(row)

    print(border_bot)

    # Print generated texts
    print("\n=== Generated Text Samples ===")
    for r in results:
        print(f"\n--- {r['name']} ---")
        # Show only the generated part (after prompt)
        print(r['text'][:500])


def run_benchmarks(model_dir: str = None):
    """Run all benchmarks and print comparison table."""
    if model_dir is None:
        model_dir = os.path.dirname(os.path.abspath(__file__))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running benchmarks on {device.upper()}")

    fp32_path = os.path.join(model_dir, 'model.pt')
    fp16_path = os.path.join(model_dir, 'model_fp16.pt')
    int8_dyn_path = os.path.join(model_dir, 'model_int8_dynamic.pt')
    int8_cal_path = os.path.join(model_dir, 'model_int8_calibrated.pt')

    results = []

    # Perplexity test text
    ppl_text = (
        "The quick brown fox jumps over the lazy dog. In the field of artificial "
        "intelligence, machine learning has become a dominant paradigm. Neural "
        "networks, inspired by biological brain structures, have proven remarkably "
        "effective at tasks ranging from image recognition to natural language "
        "processing. Deep learning, a subset of machine learning, uses multiple "
        "layers of artificial neurons to progressively extract higher-level features "
        "from raw input."
    )

    # --- 1. FP32 Baseline (no cache) ---
    print("\n[1/5] Benchmarking FP32 baseline (no KV cache)...")
    checkpoint = torch.load(fp32_path, map_location=device)
    model = EzeLLM(checkpoint['config'], device=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    tok_sec, vram, text = measure_generation(
        model, PROMPT, NUM_GENERATE_TOKENS, use_cache=False, device=device
    )
    ppl = compute_perplexity(model, ppl_text, device)

    results.append({
        'name': 'FP32 (baseline)',
        'tok_sec': tok_sec,
        'vram_mb': vram,
        'size_mb': get_file_size_mb(fp32_path),
        'perplexity': ppl,
        'text': text,
    })

    del model
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    # --- 2. FP32 + KV Cache ---
    print("\n[2/5] Benchmarking FP32 + KV Cache...")
    checkpoint = torch.load(fp32_path, map_location=device)
    model = EzeLLM(checkpoint['config'], device=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    tok_sec, vram, text = measure_generation(
        model, PROMPT, NUM_GENERATE_TOKENS, use_cache=True, device=device
    )

    results.append({
        'name': 'FP32 + KV Cache',
        'tok_sec': tok_sec,
        'vram_mb': vram,
        'size_mb': get_file_size_mb(fp32_path),
        'perplexity': ppl,  # Same model, same perplexity
        'text': text,
    })

    del model
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    # --- 3. FP16 + KV Cache ---
    if os.path.exists(fp16_path):
        print("\n[3/5] Benchmarking FP16 + KV Cache...")
        checkpoint = torch.load(fp16_path, map_location=device)
        model = EzeLLM(checkpoint['config'], device=device)
        model.load_state_dict(checkpoint['model'])
        model.eval()

        tok_sec, vram, text = measure_generation(
            model, PROMPT, NUM_GENERATE_TOKENS, use_cache=True, device=device
        )
        ppl_fp16 = compute_perplexity(model, ppl_text, device)

        results.append({
            'name': 'FP16 + KV Cache',
            'tok_sec': tok_sec,
            'vram_mb': vram,
            'size_mb': get_file_size_mb(fp16_path),
            'perplexity': ppl_fp16,
            'text': text,
        })

        del model
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
    else:
        print("\n[3/5] Skipping FP16 (not found)")

    # --- 4. INT8 Dynamic + KV Cache ---
    if os.path.exists(int8_dyn_path):
        print("\n[4/5] Benchmarking INT8 Dynamic (CPU only)...")
        tok_sec, vram, text = measure_int8_dynamic(
            int8_dyn_path, PROMPT, NUM_GENERATE_TOKENS, device
        )

        results.append({
            'name': 'INT8 Dynamic (CPU)',
            'tok_sec': tok_sec,
            'vram_mb': vram,
            'size_mb': get_file_size_mb(int8_dyn_path),
            'perplexity': float('inf'),  # Can't easily compute on quantized
            'text': text,
        })
    else:
        print("\n[4/5] Skipping INT8 Dynamic (not found)")

    # --- 5. INT8 Calibrated + KV Cache ---
    if os.path.exists(int8_cal_path):
        print("\n[5/5] Benchmarking INT8 Calibrated + KV Cache...")
        model = load_int8_calibrated(int8_cal_path, device=device)

        tok_sec, vram, text = measure_generation(
            model, PROMPT, NUM_GENERATE_TOKENS, use_cache=True, device=device
        )
        ppl_cal = compute_perplexity(model, ppl_text, device)

        results.append({
            'name': 'INT8 Calib + KV Cache',
            'tok_sec': tok_sec,
            'vram_mb': vram,
            'size_mb': get_file_size_mb(int8_cal_path),
            'perplexity': ppl_cal,
            'text': text,
        })

        del model
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
    else:
        print("\n[5/5] Skipping INT8 Calibrated (not found)")

    # Print results
    print_table(results)
    return results


if __name__ == '__main__':
    run_benchmarks()
