# EzeLLM Optimization

Performance optimizations for the EzeLLM ~356M-parameter LLaMA-style language model.

## Optimizations

1. **KV Cache** — Caches key/value tensors per layer during autoregressive generation, avoiding full-sequence recomputation at each step. Produces token-for-token identical output to the original model.

2. **FP16 Conversion** — Converts FP32 weights to FP16, halving model size and VRAM usage with near-zero quality loss.

3. **INT8 Dynamic Quantization** — PyTorch-native weight-only INT8 quantization of all Linear layers. No calibration data needed.

4. **INT8 Calibrated Quantization** — Per-channel INT8 quantization using activation statistics from 128 calibration samples (Nemotron dataset). Better precision than naive rounding.

## Quick Start

```bash
cd Optimization
pip install -r requirements.txt
python run_all.py
```

This will:
1. Download the model weights (~1.4 GB)
2. Verify KV cache correctness (greedy decode identity test)
3. Create FP16, INT8 dynamic, and INT8 calibrated model variants
4. Run benchmarks and print a comparison table

## Hardware Requirements

- NVIDIA GPU with CUDA support (tested on RTX 4090, 24GB VRAM)
- ~5 GB disk space for all model variants

## File Structure

| File | Description |
|------|-------------|
| `run_all.py` | Master script — runs everything end-to-end |
| `kv_cache.py` | KV cache implementation + cached generation |
| `quantize.py` | FP16, INT8 dynamic, INT8 calibrated conversion |
| `benchmark.py` | Speed / memory / quality comparison |
| `verify_kv_cache.py` | Token-for-token identity test |
| `requirements.txt` | Python dependencies |

## Individual Scripts

### Verify KV Cache
```bash
python verify_kv_cache.py [model_path]
```

### Create Quantized Models
```bash
python quantize.py
```

### Run Benchmarks Only
```bash
python benchmark.py
```

## Notes

- This is a custom PyTorch model, not a HuggingFace transformers model. Tools like AutoGPTQ, AutoAWQ, and llama.cpp will not work.
- INT8 dynamic quantization uses `torch.ao.quantization.quantize_dynamic` and runs on CPU only.
- The KV cache implementation correctly handles RoPE position offsets and GQA (16 query heads, 8 KV heads).
- The lm_head bug in the original model (loop overwriting logits instead of chaining through Sequential) is fixed in the cached forward pass.
