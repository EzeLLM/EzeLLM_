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

# Detect GPU backend
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected, building with NVIDIA GPU support"
    RUSTFLAGS="-C target-cpu=native" cargo build --release --features cuda
elif [[ "$(uname)" == "Darwin" ]] && system_profiler SPDisplaysDataType 2>/dev/null | grep -q "Metal"; then
    echo "Metal detected, building with Apple GPU support"
    RUSTFLAGS="-C target-cpu=native" cargo build --release --features metal
else
    echo "No GPU detected, building CPU-only"
    RUSTFLAGS="-C target-cpu=native" cargo build --release
fi

echo ""
echo "=== Build complete ==="
echo "Binary: $SCRIPT_DIR/target/release/ezellm-rs"
echo ""
echo "Usage:"
echo "  ./target/release/ezellm-rs --prompt 'Your prompt here'"
echo "  ./target/release/ezellm-rs --prompt 'Hello' --max-tokens 512 --temperature 0.8"
echo "  ./target/release/ezellm-rs --cpu   # force CPU instead of CUDA"
