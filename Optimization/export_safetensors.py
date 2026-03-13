"""
Exports EzeLLM checkpoint from .pt to .safetensors format.
Also exports the tokenizer to HuggingFace tokenizers JSON format.

Usage:
    python export_safetensors.py --model model.pt --output ./exported/

Produces:
    exported/model.safetensors   (FP16 weights)
    exported/config.json         (model config)
    exported/tokenizer.json      (HF tokenizers format, converted from tiktoken)
"""

import argparse
import json
import os
import sys
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dev'))

import torch
from safetensors.torch import save_file


# Key renaming map: PyTorch name prefix → safetensors name prefix
KEY_RENAMES = [
    # Embedding
    ("transformer.wte.weight", "wte.weight"),
    # Final layer norm
    ("transformer.ln_f.weight", "ln_f.weight"),
]

# Patterns for transformer blocks (layer index captured)
BLOCK_RENAMES = [
    (r"transformer\.hiddens\.(\d+)\.ln1\.weight", r"layers.\1.ln1.weight"),
    (r"transformer\.hiddens\.(\d+)\.ln2\.weight", r"layers.\1.ln2.weight"),
    (r"transformer\.hiddens\.(\d+)\.attn\.qkv_gen\.weight", r"layers.\1.attn.qkv.weight"),
    (r"transformer\.hiddens\.(\d+)\.attn\.qkv_gen\.bias", r"layers.\1.attn.qkv.bias"),
    (r"transformer\.hiddens\.(\d+)\.attn\.projection_attn1\.weight", r"layers.\1.attn.o_proj.weight"),
    (r"transformer\.hiddens\.(\d+)\.attn\.projection_attn1\.bias", r"layers.\1.attn.o_proj.bias"),
    (r"transformer\.hiddens\.(\d+)\.mlp\.ba_dense\.weight", r"layers.\1.mlp.up.weight"),
    (r"transformer\.hiddens\.(\d+)\.mlp\.ba_dense\.bias", r"layers.\1.mlp.up.bias"),
    (r"transformer\.hiddens\.(\d+)\.mlp\.aa_proj\.weight", r"layers.\1.mlp.down.weight"),
    (r"transformer\.hiddens\.(\d+)\.mlp\.aa_proj\.bias", r"layers.\1.mlp.down.bias"),
]

# lm_head keys pass through with same names
# lm_head.0.weight, lm_head.0.bias  (Linear 1024->1024)
# lm_head.1.weight                   (RMSNorm)
# lm_head.3.weight, lm_head.3.bias  (Linear 1024->1024)
# lm_head.4.weight                   (RMSNorm)
# lm_head.6.weight                   (Linear 1024->50304, tied to wte)

# Keys to skip (RoPE frequencies are recomputed in Rust)
SKIP_PATTERNS = [
    r"transformer\.hiddens\.\d+\.attn\.rotary_emb\.",
]


def rename_key(key: str) -> str | None:
    """Rename a PyTorch state dict key to safetensors format. Returns None to skip."""
    # Check skip patterns
    for pattern in SKIP_PATTERNS:
        if re.match(pattern, key):
            return None

    # Check exact renames
    for old, new in KEY_RENAMES:
        if key == old:
            return new

    # Check block pattern renames
    for pattern, replacement in BLOCK_RENAMES:
        m = re.match(pattern, key)
        if m:
            return re.sub(pattern, replacement, key)

    # lm_head keys pass through
    if key.startswith("lm_head."):
        return key

    print(f"  WARNING: Unknown key '{key}', skipping")
    return None


def export_model(model_path: str, output_dir: str):
    """Export .pt checkpoint to .safetensors + config.json."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading checkpoint from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    state_dict = checkpoint['model']

    print(f"Model config: {config}")
    print(f"Total keys in state dict: {len(state_dict)}")

    # Rename and convert to FP16
    renamed = {}
    skipped = 0
    tied_weight_key = None

    for key, tensor in state_dict.items():
        new_key = rename_key(key)
        if new_key is None:
            skipped += 1
            continue

        # Convert to FP16
        if tensor.is_floating_point():
            tensor = tensor.half()

        renamed[new_key] = tensor

        # Track the wte weight for verification of tying
        if new_key == "wte.weight":
            tied_weight_key = new_key

    # The lm_head.6.weight should be tied to wte.weight
    # In safetensors we store both — they should be identical
    if "lm_head.6.weight" in renamed and tied_weight_key:
        if torch.equal(renamed["lm_head.6.weight"], renamed[tied_weight_key]):
            print("  Confirmed: lm_head.6.weight is tied to wte.weight")
            # Remove duplicate to save space — Rust side will handle tying
            del renamed["lm_head.6.weight"]
        else:
            print("  WARNING: lm_head.6.weight differs from wte.weight!")

    print(f"Renamed {len(renamed)} tensors, skipped {skipped}")

    # Save safetensors
    safetensors_path = os.path.join(output_dir, "model.safetensors")
    save_file(renamed, safetensors_path)
    size_mb = os.path.getsize(safetensors_path) / (1024 * 1024)
    print(f"Saved {safetensors_path} ({size_mb:.1f} MB)")

    # Export config.json
    config_dict = {
        "hidden_size": config.embed_size,
        "num_hidden_layers": config.hidden_count,
        "num_attention_heads": config.head_count,
        "num_key_value_heads": config.n_kv_heads,
        "intermediate_size": int(5 * config.embed_size),
        "vocab_size": config.vocab_size,
        "max_position_embeddings": config.block_size,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
    }
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"Saved {config_path}")

    # Print all exported keys for verification
    print("\nExported tensor keys:")
    for key in sorted(renamed.keys()):
        t = renamed[key]
        print(f"  {key}: {list(t.shape)} {t.dtype}")


def export_tokenizer(output_dir: str):
    """Export GPT-2 tokenizer to HuggingFace tokenizers JSON format."""
    try:
        from tokenizers import Tokenizer
        # Use the pre-existing GPT-2 tokenizer from HuggingFace
        # It's the exact same BPE vocabulary that tiktoken's gpt2 encoding uses
        tokenizer = Tokenizer.from_pretrained("gpt2")
        tokenizer_path = os.path.join(output_dir, "tokenizer.json")
        tokenizer.save(tokenizer_path)
        print(f"Saved {tokenizer_path}")
    except Exception as e:
        print(f"Warning: Could not export tokenizer via HuggingFace: {e}")
        print("Falling back to manual GPT-2 tokenizer export...")
        export_tokenizer_manual(output_dir)


def export_tokenizer_manual(output_dir: str):
    """
    Fallback: manually construct a tokenizer.json compatible with
    the HuggingFace tokenizers Rust crate from tiktoken's GPT-2 encoding.
    """
    import tiktoken
    import base64

    enc = tiktoken.get_encoding('gpt2')

    # Get the vocabulary: token bytes → rank
    # tiktoken stores this as mergeable_ranks
    vocab = {}
    for token_bytes, rank in enc._mergeable_ranks.items():
        # Decode bytes to string for JSON (using latin-1 for byte-level)
        try:
            token_str = token_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # Use the GPT-2 byte-level encoding scheme
            token_str = ''.join(
                chr(b) if 32 <= b < 127 and b != 92 else f'\\x{b:02x}'
                for b in token_bytes
            )
        vocab[token_str] = rank

    # Add special tokens
    for token_str, rank in enc._special_tokens.items():
        vocab[token_str] = rank

    # Save a minimal tokenizer.json
    tokenizer_config = {
        "version": "1.0",
        "model": {
            "type": "BPE",
            "vocab": vocab,
            "merges": [],  # Merges are implicit in rank ordering
        },
        "added_tokens": [
            {
                "id": enc._special_tokens['<|endoftext|>'],
                "content": "<|endoftext|>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        ],
    }

    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    with open(tokenizer_path, 'w') as f:
        json.dump(tokenizer_config, f)
    print(f"Saved {tokenizer_path} (manual fallback)")


def main():
    parser = argparse.ArgumentParser(description="Export EzeLLM to safetensors")
    parser.add_argument("--model", type=str, default="model.pt",
                        help="Path to .pt checkpoint")
    parser.add_argument("--output", type=str, default="./exported/",
                        help="Output directory")
    args = parser.parse_args()

    # Resolve paths relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = args.model if os.path.isabs(args.model) else os.path.join(script_dir, args.model)
    output_dir = args.output if os.path.isabs(args.output) else os.path.join(script_dir, args.output)

    export_model(model_path, output_dir)
    export_tokenizer(output_dir)

    print("\nExport complete!")
    print(f"Files in {output_dir}:")
    for f in os.listdir(output_dir):
        size = os.path.getsize(os.path.join(output_dir, f)) / (1024 * 1024)
        print(f"  {f} ({size:.1f} MB)")


if __name__ == '__main__':
    main()
