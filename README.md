# EzeLLM

A custom 327M-parameter LLM with a LLaMA-style architecture (GQA, RoPE, SwiGLU, RMSNorm), trained on 100B tokens of English educational content using a single NVIDIA RTX 4090.

## Quick Start

```bash
pip install -r requirements.txt

# Generate text (downloads model automatically on first run)
python dev/ezellm.py --prompt "The theory of relativity"

# With options
python dev/ezellm.py --prompt "Hello world" --max-tokens 512 --temperature 0.8
```

The default mode uses **FP16 + KV cache** for optimal throughput (~100 tok/s on RTX 4090).

## Rust Inference (3.4x faster)

A native Rust binary with the model embedded — no Python, no external files:

```bash
cd Optimization/rust-inference
./build.sh
./target/release/ezellm-rs --prompt "The theory of relativity"
```

Achieves **~346 tok/s** on RTX 4090 (vs ~100 tok/s Python FP16).

## Architecture

- **Parameters**: 327M (1024 hidden, 20 layers)
- **Attention**: Grouped Query Attention (16 query heads, 8 KV heads)
- **Position encoding**: Rotary Position Embeddings (RoPE)
- **Activation**: SwiGLU
- **Normalization**: RMSNorm
- **Context length**: 2048 tokens
- **Vocabulary**: 50,304 tokens (GPT-2 BPE)

## Repository Structure

```
EzeLLM_/
├── dev/
│   └── ezellm.py          # Model definition + inference
├── config/
│   ├── config.toml         # Model download URL
│   └── memory.toml         # Local model paths
├── Optimization/
│   ├── run_all.py          # Full optimization pipeline
│   ├── kv_cache.py         # KV cache implementation
│   ├── quantize.py         # FP16 / INT8 quantization
│   ├── benchmark.py        # Benchmarking suite
│   ├── export_safetensors.py  # Export for Rust engine
│   ├── rust-inference/     # Rust + Candle inference engine
│   ├── report.tex          # Workshop paper
│   └── new.md              # Full benchmark results
└── requirements.txt
```

## Optimization Results

| Variant | tok/s (256 tokens) | Speedup |
|---------|---:|---:|
| FP32 no cache | 93.7 | 1.0x |
| FP32 + KV Cache | 92.9 | 1.0x |
| FP16 + KV Cache | 101.6 | 1.1x |
| **Rust CUDA** | **346.1** | **3.7x** |

See [Optimization/new.md](Optimization/new.md) for full results across 32–2048 tokens.

## Requirements

- Python 3.10+
- PyTorch 2.1+
- NVIDIA GPU with CUDA (for GPU inference)
- Rust toolchain (for Rust engine only)
