# EzeLLM Rust Inference Engine

Native Rust inference binary for EzeLLM using [Candle](https://github.com/huggingface/candle). Loads FP16 weights from safetensors and runs inference with KV-cached autoregressive generation.

## Prerequisites

- **Rust toolchain** (1.70+): `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- **CUDA toolkit** (optional, for GPU): CUDA 11.8+ with `nvcc` on PATH
- **Python** (for model export): Python 3.8+ with `safetensors`, `tiktoken`, `tokenizers` packages

## Quick Start

### 1. Export the model

First, ensure you have the FP32 model at `Optimization/model.pt` (run `python run_all.py` if not).

```bash
cd Optimization
pip install safetensors tokenizers tiktoken
python export_safetensors.py --model model.pt --output ./exported/
```

This produces:
- `exported/model.safetensors` (FP16 weights, ~700 MB)
- `exported/config.json` (model hyperparameters)
- `exported/tokenizer.json` (GPT-2 BPE tokenizer)

### 2. Build

```bash
cd rust-inference
chmod +x build.sh
./build.sh
```

Or manually:

```bash
# With CUDA (GPU)
RUSTFLAGS="-C target-cpu=native" cargo build --release --features cuda

# CPU only
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

### 3. Run

```bash
./target/release/ezellm-rs \
    --model-dir ../exported/ \
    --prompt "The theory of general relativity explains" \
    --max-tokens 200 \
    --temperature 0.8 \
    --top-k 50
```

#### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-dir` | (required) | Path to exported model directory |
| `--prompt` | "The theory of relativity" | Input prompt |
| `--max-tokens` | 256 | Maximum tokens to generate |
| `--temperature` | 0.8 | Sampling temperature (lower = more deterministic) |
| `--top-k` | 50 | Top-k filtering |
| `--cpu` | false | Force CPU inference (even if CUDA available) |

## Architecture

The engine implements the full EzeLLM architecture in Rust:

- **Embedding** → 20x **Transformer Blocks** → **RMSNorm** → **LM Head**
- Each block: RMSNorm → GQA Attention (16 query, 8 KV heads) → RMSNorm → SwiGLU MLP
- LM Head: Linear → RMSNorm → SwiGLU → Linear → RMSNorm → SwiGLU → Linear (tied weights)
- RoPE positional encoding with precomputed sin/cos tables
- KV cache for O(1) per-token decode (vs O(n) without cache)

## Verification

To verify the Rust output matches Python (greedy decoding):

```bash
# Rust (near-greedy)
./target/release/ezellm-rs --model-dir ../exported/ \
    --prompt "The theory of relativity" \
    --temperature 0.001 --top-k 1 --max-tokens 50

# Python (greedy, from Optimization/)
python -c "
import sys; sys.path.insert(0, '../dev')
from ezellm import EzeLLM
from kv_cache import generate_cached
import torch
model = EzeLLM.from_pretrained('model.pt')
model.eval()
print(generate_cached(model, 'The theory of relativity', temperature=0.001, topk=1, max_l=60, verbose=False))
"
```

The first ~20 tokens should match exactly. Minor divergence after that is expected due to FP16 vs FP32 precision differences.

## Performance

Expected performance on RTX 4090 (24GB VRAM):

| Variant | tok/s |
|---|---|
| Python FP32 no cache | ~5-15 |
| Python FP16 + KV cache | ~80-200 |
| **Rust FP16 + KV cache (CUDA)** | **~200-500** |
| **Rust FP16 + KV cache (CPU)** | **~30-80** |
