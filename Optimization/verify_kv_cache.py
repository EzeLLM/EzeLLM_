"""
Verification script: ensures KV-cached and non-cached generation produce
token-for-token identical outputs using greedy (argmax) decoding.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dev'))

import torch
from ezellm import EzeLLM
from kv_cache import generate_deterministic


def verify_kv_cache(model_path: str, num_tokens: int = 50):
    """
    Generate `num_tokens` tokens with and without KV cache using greedy decoding.
    Assert they are token-for-token identical.
    """
    print(f"Loading model from {model_path}...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = EzeLLM(checkpoint['config'], device=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    prompt = "The theory of general relativity explains"
    print(f"Prompt: '{prompt}'")
    print(f"Generating {num_tokens} tokens with greedy decoding...\n")

    # Generate without cache (baseline)
    print("Running WITHOUT KV cache...")
    tokens_no_cache = generate_deterministic(
        model, prompt, max_tokens=num_tokens, use_cache=False
    )

    # Generate with cache
    print("Running WITH KV cache...")
    tokens_with_cache = generate_deterministic(
        model, prompt, max_tokens=num_tokens, use_cache=True
    )

    # Compare token-for-token
    print(f"\nNo-cache tokens:   {tokens_no_cache}")
    print(f"With-cache tokens: {tokens_with_cache}")

    if tokens_no_cache == tokens_with_cache:
        print(f"\n[PASS] KV cache verification passed! "
              f"All {num_tokens} tokens are identical.")

        # Show decoded text for visual check
        tokenizer = model.tokenizer
        decoded = tokenizer.decode(tokens_no_cache)
        print(f"\nGenerated text: {decoded}")
        return True
    else:
        # Find first mismatch
        for i, (a, b) in enumerate(zip(tokens_no_cache, tokens_with_cache)):
            if a != b:
                print(f"\n[FAIL] First mismatch at token {i}: "
                      f"no_cache={a}, with_cache={b}")
                break
        print(f"\n[FAIL] KV cache verification FAILED! "
              f"Outputs differ.")
        return False


if __name__ == '__main__':
    model_path = os.path.join(os.path.dirname(__file__), 'model.pt')
    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    success = verify_kv_cache(model_path, num_tokens=50)
    sys.exit(0 if success else 1)
