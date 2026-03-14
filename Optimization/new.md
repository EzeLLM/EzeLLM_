# EzeLLM Full Benchmark Results (up to 2048 tokens)

All results: greedy decoding, temperature ≈ 0, top-k = 50, averaged over **10 diverse prompts** (4–16 input tokens each). Hardware: RTX 4090 + Ryzen 9 7950X.

---

## Throughput Table (tokens/second)

| Tokens | FP32 No Cache | FP32 + KV Cache | FP16 + KV Cache | Rust CUDA | Rust CPU |
|-------:|--------------:|----------------:|----------------:|----------:|---------:|
|     32 |          91.7 |            92.5 |            92.8 |   **315.3** |     29.2 |
|     64 |         110.9 |            94.3 |            94.0 |   **335.3** |     28.7 |
|    128 |         106.0 |            93.0 |           100.5 |   **347.1** |     28.1 |
|    256 |          93.7 |            92.9 |           101.6 |   **346.1** |     27.4 |
|    512 |          92.5 |            92.0 |           100.9 |   **338.5** |     26.3 |
|   1024 |          66.0 |            88.9 |           100.5 |   **319.6** |     24.5 |
|   2048 |          35.9 |            91.5 |           100.2 |   **261.9** |     21.8 |

---

## Speedup: Rust CUDA vs Best Python (FP16 + KV Cache)

| Tokens | FP16 Python | Rust CUDA | Speedup |
|-------:|------------:|----------:|--------:|
|     32 |        92.8 |     315.3 |   3.4×  |
|     64 |        94.0 |     335.3 |   3.6×  |
|    128 |       100.5 |     347.1 |   3.5×  |
|    256 |       101.6 |     346.1 |   3.4×  |
|    512 |       100.9 |     338.5 |   3.4×  |
|   1024 |       100.5 |     319.6 |   3.2×  |
|   2048 |       100.2 |     261.9 |   2.6×  |

---

## Observations

### 1. Rust CUDA is 2.6–3.6× faster across all sequence lengths

The speedup is highest at medium lengths (64–512 tokens) where the per-token compute is small and Python's dispatch overhead dominates. At 2048 tokens, attention computation grows and narrows the gap slightly, but Rust still holds a 2.6× advantage.

### 2. Rust CUDA throughput degrades at long sequences

Decode throughput drops from 347 tok/s at 128 tokens to 262 tok/s at 2048 tokens (–25%). This is expected: each decode step performs a matmul of Q against the full KV cache. At 2048 cached tokens, the attention matmul is 16× larger than at 128 tokens. This is the regime where flash attention would help—but its compilation cost was prohibitive on this machine.

### 3. Python throughput is nearly flat (KV cache variants)

Both FP32+Cache (~92 tok/s) and FP16+Cache (~100 tok/s) show almost no degradation from 32 to 2048 tokens. This means Python's per-token overhead (~10ms) is so large that the actual GPU compute increase from longer KV caches is invisible. The Python runtime is the bottleneck, not the GPU.

### 4. FP32 No Cache collapses at longer sequences

Without KV cache, throughput drops from 111 tok/s at 64 tokens to 92.5 at 512, **66.0 at 1024**, and **35.9 at 2048** tokens. The O(n²) recomputation cost is devastating: at 2048 tokens, no-cache is 2.8× slower than FP32 with cache (35.9 vs 91.5 tok/s) and **9.7× slower than Rust CUDA** (35.9 vs 346 tok/s). This makes the KV cache essential for any practical generation beyond a few hundred tokens.

### 5. Rust CPU is 4× slower than Python GPU

Rust CPU (21–29 tok/s) cannot compete with Python+CUDA (~100 tok/s), which is expected—CPU matmuls are bandwidth-limited. The Rust CPU path exists as a fallback, not as a performance target.

### 6. FP16 gives ~8% improvement over FP32 in Python

At 128+ tokens, FP16+Cache averages ~100 tok/s vs ~92 tok/s for FP32+Cache. The improvement is modest because Python overhead dominates. The same FP16 weights in Rust yield 3.5× throughput, showing the FP16 benefit is fully realized only when the runtime overhead is eliminated.

---

## Scaling Behavior

**Rust CUDA**: Throughput peaks around 128–256 tokens then declines. The decline follows the expected attention compute scaling—attention is O(n) per decode step, so total work grows linearly with cached sequence length.

**Python variants**: Flat throughput regardless of sequence length. Python's ~10ms per-token overhead masks the compute scaling entirely. Even doubling the attention work (1024→2048) produces no measurable slowdown.

**Implication**: For this 327M model on an RTX 4090, Rust's advantage is largest in the "interactive" regime (32–512 tokens). At 2048 tokens, the GPU compute starts to matter and the gap narrows. For larger models or longer contexts, the relative speedup would be smaller as GPU compute dominates over runtime overhead.

---

## Wall-Clock Time (seconds, average per prompt)

| Tokens | FP32 No Cache | FP32 + Cache | FP16 + Cache | Rust CUDA | Rust CPU |
|-------:|--------------:|-------------:|-------------:|----------:|---------:|
|     32 |         0.349 |        0.346 |        0.345 |     0.10  |     1.10 |
|     64 |         0.577 |        0.679 |        0.681 |     0.19  |     2.23 |
|    128 |         1.208 |        1.377 |        1.274 |     0.37  |     4.56 |
|    256 |         2.732 |        2.757 |        2.521 |     0.74  |     9.34 |
|    512 |          5.5  |        5.562 |        5.074 |     1.51  |    19.47 |
|   1024 |         15.5  |       11.519 |       10.192 |     3.20  |    41.80 |
|   2048 |         56.8  |       22.374 |       20.434 |     7.83  |    93.94 |

At 2048 tokens: Rust CUDA completes in **7.8 seconds**. FP32 No Cache takes **56.8 seconds** — 7.3× slower.
