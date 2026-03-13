"""
Master script for EzeLLM optimization.

Downloads the model, verifies KV cache correctness, creates all optimized
variants, and runs benchmarks.

Usage:
    cd Optimization
    pip install -r requirements.txt
    python run_all.py
"""

import sys
import os
import subprocess

# Ensure we can import from this directory and from dev/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dev'))

import torch
import __main__
import requests
from tqdm import tqdm
from ezellm import EzeLLMConfig

# Checkpoint pickles config as __main__.EzeLLMConfig — make it findable
__main__.EzeLLMConfig = EzeLLMConfig

MODEL_URL = "https://huggingface.co/TerminatorPower/EzeLLM-base-text-fp32/resolve/main/model.pt"
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "model.pt")


def download_model(url: str = MODEL_URL, output_path: str = MODEL_PATH):
    """Download model weights if not already present."""
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Model already exists at {output_path} ({size_mb:.1f} MB)")
        return output_path

    print(f"Downloading model from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))

    with open(output_path, 'wb') as f, tqdm(
        desc="Downloading model",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                size = f.write(chunk)
                pbar.update(size)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Model downloaded to {output_path} ({size_mb:.1f} MB)")
    return output_path


def main():
    print("=" * 60)
    print("  EzeLLM Optimization Pipeline")
    print("=" * 60)

    # Step 0: Download model
    print("\n=== Step 0: Download Model ===")
    model_path = download_model()

    # Step 1: Verify KV cache correctness
    print("\n=== Step 1: Verify KV Cache ===")
    from verify_kv_cache import verify_kv_cache
    success = verify_kv_cache(model_path, num_tokens=50)
    if not success:
        print("\nERROR: KV cache verification failed! Aborting.")
        sys.exit(1)
    print("KV cache verification passed!")

    # Step 2: Create optimized variants
    print("\n=== Step 2: Create Optimized Variants ===")
    from quantize import convert_fp16, convert_int8_dynamic, convert_int8_calibrated

    fp16_path = os.path.join(MODEL_DIR, "model_fp16.pt")
    int8_dyn_path = os.path.join(MODEL_DIR, "model_int8_dynamic.pt")
    int8_cal_path = os.path.join(MODEL_DIR, "model_int8_calibrated.pt")

    print("\n--- Creating FP16 variant ---")
    convert_fp16(model_path, fp16_path)

    print("\n--- Creating INT8 dynamic variant ---")
    convert_int8_dynamic(model_path, int8_dyn_path)

    print("\n--- Creating INT8 calibrated variant ---")
    convert_int8_calibrated(model_path, int8_cal_path, num_calibration_samples=128)

    # Step 3: Benchmark everything
    print("\n=== Step 3: Run Benchmarks ===")
    from benchmark import run_benchmarks
    run_benchmarks(MODEL_DIR)

    print("\n" + "=" * 60)
    print("  Optimization pipeline complete!")
    print("=" * 60)
    print(f"\nOptimized models saved in: {MODEL_DIR}")
    print("  - model.pt (FP32 baseline)")
    print("  - model_fp16.pt (FP16)")
    print("  - model_int8_dynamic.pt (INT8 dynamic)")
    print("  - model_int8_calibrated.pt (INT8 calibrated)")


if __name__ == '__main__':
    main()
