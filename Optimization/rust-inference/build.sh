#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Building EzeLLM Rust Inference ==="

# Step 1: Export model to safetensors (if not already done)
EXPORTED_DIR="$SCRIPT_DIR/../exported"
if [ ! -f "$EXPORTED_DIR/model.safetensors" ]; then
    echo "Exporting model to safetensors..."
    cd "$SCRIPT_DIR/.."
    python export_safetensors.py --model model.pt --output ./exported/
    cd "$SCRIPT_DIR"
else
    echo "Exported model already exists at $EXPORTED_DIR"
fi

# Step 2: Build Rust binary
echo ""
echo "Building Rust binary (release mode)..."

# Detect CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected, building with GPU support"
    RUSTFLAGS="-C target-cpu=native" cargo build --release --features cuda
else
    echo "No CUDA detected, building CPU-only"
    RUSTFLAGS="-C target-cpu=native" cargo build --release
fi

echo ""
echo "=== Build complete ==="
echo "Binary: $SCRIPT_DIR/target/release/ezellm-rs"
echo ""
echo "Run with:"
echo "  ./target/release/ezellm-rs --model-dir ../exported/ --prompt 'Your prompt here'"
