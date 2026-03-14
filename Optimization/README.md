# EzeLLM Optimization

Performance optimizations for the EzeLLM 327M-parameter LLaMA-style language model.

## Results

| Variant | Throughput (tok/s @ 256 tokens) |
|---------|------:|
| FP32 no cache | 93.7 |
| FP32 + KV Cache | 92.9 |
| **FP16 + KV Cache** | **101.6** |
| **Rust CUDA** | **346.1** |
| Rust CPU | 27.4 |

Full benchmark results across 32–2048 tokens: [new.md](new.md)

## Python Optimizations

1. **KV Cache** — Caches key/value tensors per layer, avoiding full-sequence recomputation. Token-for-token identical output.
2. **FP16 Conversion** — Halves model size and VRAM usage with near-zero quality loss.

### Quick Start (Python)

```bash
cd Optimization
pip install -r requirements.txt
python run_all.py
```

This downloads the model, verifies KV cache correctness, creates the FP16 variant, and runs benchmarks.

## Rust Inference Engine

A native Rust implementation using [Candle](https://github.com/huggingface/candle). The model, tokenizer, and config are embedded directly into the binary — no external files needed at runtime.

**3.4x faster** than the best Python variant at typical sequence lengths.

### Building

```bash
cd Optimization/rust-inference
./build.sh
```

The build script automatically detects CUDA, Metal, or falls back to CPU.

### Usage

```bash
# Default: CUDA if available
./target/release/ezellm-rs --prompt "The theory of relativity"

# More options
./target/release/ezellm-rs --prompt "Hello" --max-tokens 512 --temperature 0.8 --top-k 50

# Force CPU
./target/release/ezellm-rs --cpu
```

### Prerequisites

- Rust toolchain (rustup)
- For CUDA: NVIDIA GPU + CUDA toolkit
- Model must be exported first: `python export_safetensors.py` (done automatically by `build.sh`)

## File Structure

| File | Description |
|------|-------------|
| `run_all.py` | Master script — runs everything end-to-end |
| `kv_cache.py` | KV cache implementation + cached generation |
| `benchmark.py` | Speed / memory / quality comparison |
| `verify_kv_cache.py` | Token-for-token identity test |
| `export_safetensors.py` | Export .pt to safetensors for Rust |
| `rust-inference/` | Rust inference engine (Candle + CUDA) |
| `new.md` | Full benchmark results (32–2048 tokens) |
| `report.tex` | Workshop paper with analysis |

## Hardware

Tested on RTX 4090 + Ryzen 9 7950X. Requires NVIDIA GPU with CUDA for GPU variants.
