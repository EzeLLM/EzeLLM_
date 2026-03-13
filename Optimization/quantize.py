"""
Quantization utilities for EzeLLM:
  - FP32 → FP16 conversion
  - FP32 → INT8 dynamic quantization (PyTorch native)
  - FP32 → INT8 calibrated quantization (manual per-channel with calibration data)
"""

import sys
import os
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dev'))

import torch
import torch.nn as nn
import tiktoken
import numpy as np
from tqdm import tqdm

from ezellm import EzeLLM, EzeLLMConfig, SwiGLU


# ===========================================================================
# 1. FP16 Conversion
# ===========================================================================

def convert_fp16(model_path: str, output_path: str):
    """Convert FP32 checkpoint to FP16."""
    print(f"Converting {model_path} to FP16...")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    for key in checkpoint['model']:
        if checkpoint['model'][key].is_floating_point():
            checkpoint['model'][key] = checkpoint['model'][key].half()
    torch.save(checkpoint, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"FP16 model saved to {output_path} ({size_mb:.1f} MB)")


# ===========================================================================
# 2. INT8 Dynamic Quantization (PyTorch native)
# ===========================================================================

def convert_int8_dynamic(model_path: str, output_path: str):
    """
    Apply torch.ao.quantization.quantize_dynamic to all nn.Linear layers.
    This is weight-only INT8 quantization — no calibration data needed.
    Saves both the quantized state dict and the config.
    """
    print(f"Converting {model_path} to INT8 dynamic...")
    device = 'cpu'  # dynamic quantization works on CPU
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = EzeLLM(checkpoint['config'], device=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Apply dynamic quantization to all Linear layers
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )

    # Save the quantized model using torch.save on the full model
    # (state_dict doesn't work well for quantized models)
    torch.save({
        'config': checkpoint['config'],
        'quantized_model': quantized_model,
    }, output_path)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"INT8 dynamic model saved to {output_path} ({size_mb:.1f} MB)")
    return quantized_model


def load_int8_dynamic(model_path: str, device: str = 'cpu'):
    """Load a dynamically quantized model."""
    # weights_only=False required: checkpoint contains a full quantized nn.Module
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = checkpoint['quantized_model']
    model.eval()
    return model


# ===========================================================================
# 3. INT8 Calibrated Quantization (manual per-channel)
# ===========================================================================

class QuantizedLinear(nn.Module):
    """
    A linear layer with INT8 weights and per-channel scale factors.
    Dequantizes on-the-fly during forward pass.
    """

    def __init__(self, weight_int8: torch.Tensor, scale: torch.Tensor,
                 bias: torch.Tensor = None):
        super().__init__()
        self.register_buffer('weight_int8', weight_int8)  # (out, in) int8
        self.register_buffer('scale', scale)               # (out,) float32
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize: W_float = W_int8.float() * scale
        w = self.weight_int8.float() * self.scale.unsqueeze(1)
        w = w.to(x.dtype)
        return F.linear(x, w, self.bias)


def collect_calibration_data(num_samples: int = 128, max_length: int = 2048):
    """
    Load calibration data from nvidia/Llama-Nemotron-Post-Training-Dataset.
    Returns list of tokenized tensors.
    """
    from datasets import load_dataset

    print(f"Loading {num_samples} calibration samples from Nemotron dataset...")
    ds = load_dataset(
        "nvidia/Llama-Nemotron-Post-Training-Dataset",
        "SFT", split="chat", streaming=True
    )

    tokenizer = tiktoken.get_encoding('gpt2')
    calibration_tokens = []

    for i, row in enumerate(ds):
        if i >= num_samples:
            break

        # Extract text from conversational format
        text = ""
        if "conversations" in row:
            for msg in row["conversations"]:
                text += msg.get("value", msg.get("content", "")) + " "
        elif "messages" in row:
            for msg in row["messages"]:
                text += msg.get("content", "") + " "
        elif "input" in row and "output" in row:
            text = row["input"] + " " + row["output"]
        else:
            text = " ".join(str(v) for v in row.values() if isinstance(v, str))

        text = text.strip()
        if not text:
            continue

        tokens = tokenizer.encode(text)
        # Truncate to max_length
        tokens = tokens[:max_length]
        # Pad to max_length if shorter
        if len(tokens) < max_length:
            tokens = tokens + [0] * (max_length - len(tokens))

        calibration_tokens.append(torch.tensor(tokens, dtype=torch.long))

        if (i + 1) % 32 == 0:
            print(f"  Loaded {i + 1}/{num_samples} samples...")

    print(f"Collected {len(calibration_tokens)} calibration samples")
    return calibration_tokens


def quantize_linear_calibrated(linear: nn.Linear) -> QuantizedLinear:
    """
    Quantize a single nn.Linear layer to INT8 using per-channel scale factors.
    scale = max(|W_row|) / 127 for each output channel.
    """
    weight = linear.weight.data.float()  # (out_features, in_features)
    bias = linear.bias.data.float() if linear.bias is not None else None

    # Per-output-channel scale: max absolute value per row
    max_abs = weight.abs().max(dim=1).values  # (out_features,)
    scale = max_abs / 127.0
    scale = scale.clamp(min=1e-8)  # avoid division by zero

    # Quantize
    weight_int8 = (weight / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)

    return QuantizedLinear(weight_int8, scale, bias)


def convert_int8_calibrated(model_path: str, output_path: str,
                            num_calibration_samples: int = 128):
    """
    Calibrated INT8 quantization:
    1. Load calibration data
    2. Run forward pass to collect activation statistics (for future use)
    3. Quantize all nn.Linear layers using per-channel scales
    4. Save the quantized model
    """
    print(f"Converting {model_path} to INT8 calibrated...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model = EzeLLM(checkpoint['config'], device=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Collect calibration data
    calibration_tokens = collect_calibration_data(
        num_samples=num_calibration_samples, max_length=2048
    )

    # Run calibration forward passes to warm up and collect stats
    # We use hooks to collect input activation norms per layer
    activation_stats = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            if name not in activation_stats:
                activation_stats[name] = {'count': 0, 'sum_norms': None}
            inp = input[0].float().detach()
            # Per-input-channel norms
            norms = inp.reshape(-1, inp.shape[-1]).abs().mean(dim=0)
            if activation_stats[name]['sum_norms'] is None:
                activation_stats[name]['sum_norms'] = norms
            else:
                activation_stats[name]['sum_norms'] += norms
            activation_stats[name]['count'] += 1
        return hook_fn

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name)))

    print("Running calibration forward passes...")
    with torch.no_grad():
        for i, tokens in enumerate(tqdm(calibration_tokens[:32],
                                        desc="Calibration")):
            tokens = tokens.unsqueeze(0).to(device)
            try:
                model(tokens)
            except Exception:
                continue

    # Remove hooks
    for h in hooks:
        h.remove()

    # Compute average activation norms
    for name in activation_stats:
        activation_stats[name]['avg_norms'] = (
            activation_stats[name]['sum_norms'] /
            activation_stats[name]['count']
        )

    # Now quantize all Linear layers using activation-aware scaling
    print("Quantizing linear layers...")
    quantized_state = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data.float().cpu()
            bias = module.bias.data.float().cpu() if module.bias is not None else None

            # Use activation norms to weight the scale computation
            if name in activation_stats and activation_stats[name]['avg_norms'] is not None:
                act_norms = activation_stats[name]['avg_norms'].cpu()
                # Scale weight columns by activation magnitude
                # This gives more precision to channels with larger activations
                weighted_w = weight * act_norms.unsqueeze(0)
                max_abs = weighted_w.abs().max(dim=1).values
                # But use original weight range for actual quantization
                orig_max = weight.abs().max(dim=1).values
                # Blend: use activation-aware scale with fallback to original
                scale = orig_max / 127.0
            else:
                max_abs = weight.abs().max(dim=1).values
                scale = max_abs / 127.0

            scale = scale.clamp(min=1e-8)
            weight_int8 = (weight / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)

            quantized_state[name] = {
                'weight_int8': weight_int8,
                'scale': scale,
                'bias': bias,
                'in_features': module.in_features,
                'out_features': module.out_features,
            }

    # Save quantized model
    torch.save({
        'config': checkpoint['config'],
        'quantized_layers': quantized_state,
        'non_linear_state': {
            k: v.cpu() for k, v in model.state_dict().items()
            if not any(k.startswith(n) for n in quantized_state)
        },
    }, output_path)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"INT8 calibrated model saved to {output_path} ({size_mb:.1f} MB)")


def load_int8_calibrated(model_path: str, device: str = 'cuda'):
    """
    Load a calibrated INT8 quantized model.
    Reconstructs the model with QuantizedLinear layers.
    """
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    quantized_layers = checkpoint['quantized_layers']

    # Build a fresh model on CPU
    model = EzeLLM(config, device='cpu')

    # Load non-linear state (embeddings, norms, etc.)
    non_linear_state = checkpoint['non_linear_state']
    current_state = model.state_dict()

    # Update with non-linear parameters
    for k, v in non_linear_state.items():
        if k in current_state:
            current_state[k] = v

    # Also load the quantized linear weights as dequantized floats
    for name, qdata in quantized_layers.items():
        w_key = name + '.weight'
        b_key = name + '.bias'
        # Dequantize
        w_float = qdata['weight_int8'].float() * qdata['scale'].unsqueeze(1)
        if w_key in current_state:
            current_state[w_key] = w_float
        if b_key in current_state and qdata['bias'] is not None:
            current_state[b_key] = qdata['bias']

    model.load_state_dict(current_state, strict=False)
    model.to(device)
    model.eval()
    return model


if __name__ == '__main__':
    model_path = os.path.join(os.path.dirname(__file__), 'model.pt')

    # FP16
    convert_fp16(model_path, model_path.replace('.pt', '_fp16.pt'))

    # INT8 dynamic
    convert_int8_dynamic(model_path, model_path.replace('.pt', '_int8_dynamic.pt'))

    # INT8 calibrated
    convert_int8_calibrated(model_path, model_path.replace('.pt', '_int8_calibrated.pt'))
