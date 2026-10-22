"""Run attention examples and expose the reusable attention classes."""

from __future__ import annotations

import argparse

import torch

from attention_is_all_you_need.attention import (
    Attention, CrossAttention, MultiHeadAttention, SelfAttention,
    SelfAttentionEncoder, SingleHeadAttention, make_attention_mask,
    scaled_dot_product_attention,
)


def run_demo(seed: int = 7, device: str = "cpu") -> dict:
    """Print weighted averages, attention masks and model output tensors.

    Compare explicit and fused calculations using identical parameters, then
    check that masked tokens cannot affect earlier outputs. Return comparison
    errors, decoder weights and the final cross-attention output for inspection.
    """
    torch.manual_seed(seed)
    torch.set_printoptions(precision=4, sci_mode=False)
    print(f"Attention | torch {torch.__version__} | device={device} | seed={seed}")

    # A query with zero scores gives equal weight to both values.
    q = torch.zeros(1, 1, 2, device=device)
    k = torch.tensor([[[1., 0.], [0., 1.]]], device=device)
    v = torch.tensor([[[2., 4.], [6., 8.]]], device=device)
    average, weights = scaled_dot_product_attention(q, k, v)
    print("\n1. A weighted average")
    print("Weights:", weights)
    print("Context:", average, "(expected [4, 6])")
    torch.testing.assert_close(average, torch.tensor([[[4., 6.]]], device=device))

    print("\nAttention without learned projections: Q = K = V = X")
    raw_context, raw_weights = scaled_dot_product_attention(k, k, k)
    print("Input X:\n", k)
    print("Weights from X @ X.T / sqrt(2):\n", raw_weights)
    print("Context (weights @ X):\n", raw_context)
    torch.testing.assert_close(
        raw_context,
        torch.tensor([[[.66976155, .33023845], [.33023845, .66976155]]], device=device),
    )

    src = torch.randn(3, 5, 9, device=device)
    tgt = torch.randn(3, 7, 9, device=device)
    source_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 0]],
                               device=device, dtype=torch.bool)
    print("\n2. Single-head and multi-head attention")
    for module in (SingleHeadAttention(9), MultiHeadAttention(9, 3), SelfAttentionEncoder(9, 3)):
        module = module.to(device).eval()
        with torch.inference_mode():
            output = module(src)
        print(f"{type(module).__name__}: {tuple(output.shape)}")

    print("\n3. Encoder padding and decoder causality")
    print("Source mask (True = allowed key):\n", source_mask)
    decoder = SelfAttention(9, 3, causal=True).to(device).eval()
    with torch.inference_mode():
        decoder_output, decoder_weights = decoder(src, source_mask, need_weights=True)
    print("First decoder attention head:\n", decoder_weights[0, 0])
    print("Decoder output shape:", tuple(decoder_output.shape))
    cross = CrossAttention(9, 3).to(device).eval()
    with torch.inference_mode():
        cross_output = cross(src, tgt, attention_mask=source_mask)
    print("CrossAttention (target queries, source keys/values):", tuple(cross_output.shape))

    print("\n4. Explicit calculations versus fused SDPA")
    explicit = Attention(9, 3, fused_attn=False).to(device).eval()
    efficient = Attention(9, 3, fused_attn=True).to(device).eval()
    efficient.load_state_dict(explicit.state_dict())
    errors = {}
    cases = (
        ("encoder", {"attention_mask": source_mask}),
        ("decoder", {"attention_mask": source_mask, "causal": True}),
        ("cross", {"tgt": tgt, "attention_mask": source_mask}),
    )
    with torch.inference_mode():
        for name, kwargs in cases:
            output = efficient(src, **kwargs)
            reference = explicit(src, **kwargs)
            torch.testing.assert_close(output, reference, atol=1e-6, rtol=1e-5)
            errors[name] = (output - reference).abs().max().item()
            print(f"{name}: {tuple(output.shape)} | max difference={errors[name]:.2e}")
        print("Final cross-attention output, first target token:\n", output[0, 0])
        changed = src.clone()
        changed[0, 3:] += 100
        baseline = efficient(src, attention_mask=source_mask, causal=True)
        perturbed = efficient(changed, attention_mask=source_mask, causal=True)
        torch.testing.assert_close(baseline[0, :3], perturbed[0, :3])
    print("\nPASS: explicit/fused outputs agree; padded future tokens do not affect earlier outputs.")
    print("These attention-layer outputs use randomly initialized weights.")
    print("SDPA selects the attention kernel for this device.")
    return {"errors": errors, "decoder_weights": decoder_weights.cpu(), "output": output.cpu()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "xpu"), default="cpu")
    args = parser.parse_args()
    run_demo(args.seed, args.device)


if __name__ == "__main__":
    main()
