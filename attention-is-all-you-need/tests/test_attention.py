"""Check attention calculations, gradients, masks and public interfaces."""

import math

import pytest
import torch
from torch import nn

from core.attention import (
    Attention, CrossAttention, MultiHeadAttention, SelfAttention,
    SelfAttentionEncoder, SingleHeadAttention, scaled_dot_product_attention,
)


def inputs():
    torch.manual_seed(11)
    src = torch.randn(2, 5, 12, dtype=torch.float64)
    tgt = torch.randn(2, 3, 12, dtype=torch.float64)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool)
    return src, tgt, mask


def test_known_weighted_average_and_fully_masked_rows():
    q = torch.zeros(1, 2, 2, dtype=torch.float64, requires_grad=True)
    k = torch.eye(2, dtype=torch.float64)[None]
    v = torch.tensor([[[2., 4.], [6., 8.]]], dtype=torch.float64)
    mask = torch.tensor([[[True, True], [False, False]]])
    context, weights = scaled_dot_product_attention(q, k, v, mask)
    torch.testing.assert_close(context, torch.tensor([[[4., 6.], [0., 0.]]], dtype=torch.float64))
    torch.testing.assert_close(weights, torch.tensor([[[.5, .5], [0., 0.]]], dtype=torch.float64))
    context.sum().backward()
    assert torch.isfinite(q.grad).all()


def test_attention_without_projections_has_hand_calculated_context():
    x = torch.eye(2, dtype=torch.float64)[None]
    context, weights = scaled_dot_product_attention(x, x, x)
    alpha = math.exp(1 / math.sqrt(2))
    diagonal, other = alpha / (alpha + 1), 1 / (alpha + 1)
    expected = torch.tensor(
        [[[diagonal, other], [other, diagonal]]], dtype=torch.float64
    )
    torch.testing.assert_close(weights, expected)
    torch.testing.assert_close(context, expected)


@pytest.mark.parametrize("case", ["encoder", "decoder", "causal_only", "cross"])
def test_manual_and_fused_outputs_and_gradients(case):
    src, tgt, mask = inputs()
    torch.manual_seed(17)
    manual = Attention(12, 3, fused_attn=False).double()
    fused = Attention(12, 3).double()
    fused.load_state_dict(manual.state_dict())
    kwargs = {"attention_mask": mask}
    if case == "decoder":
        kwargs["causal"] = True
    elif case == "causal_only":
        kwargs = {"causal": True}
    elif case == "cross":
        kwargs["tgt"] = tgt
    outputs, gradients = [], []
    for layer in (manual, fused):
        x = src.clone().requires_grad_()
        arguments = dict(kwargs)
        target = None
        if case == "cross":
            target = tgt.clone().requires_grad_()
            arguments["tgt"] = target
        output = layer(x, **arguments)
        output.square().sum().backward()
        outputs.append(output)
        gradients.append([x.grad, *[p.grad for p in layer.parameters()]] + ([target.grad] if target is not None else []))
    torch.testing.assert_close(outputs[0], outputs[1], atol=1e-10, rtol=1e-9)
    for a, b in zip(*gradients):
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize("case", ["encoder", "decoder", "cross"])
def test_matches_independent_pytorch_multihead_attention(case):
    src, tgt, mask = inputs()
    layer = Attention(12, 3, fused_attn=False).double().eval()
    reference = nn.MultiheadAttention(12, 3, batch_first=True).double().eval()
    with torch.no_grad():
        reference.in_proj_weight.copy_(torch.cat([layer.q_proj.weight, layer.k_proj.weight, layer.v_proj.weight]))
        reference.in_proj_bias.copy_(torch.cat([layer.q_proj.bias, layer.k_proj.bias, layer.v_proj.bias]))
        reference.out_proj.load_state_dict(layer.out_proj.state_dict())
    queries = tgt if case == "cross" else src
    causal = case == "decoder"
    blocked = torch.ones(5, 5, dtype=torch.bool).triu(1) if causal else None
    actual, weights = layer(src, tgt=tgt if case == "cross" else None,
                            attention_mask=mask, causal=causal, need_weights=True)
    expected, expected_weights = reference(queries, src, src, key_padding_mask=~mask,
                                          attn_mask=blocked, average_attn_weights=False)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(weights, expected_weights, atol=1e-10, rtol=1e-9)


def test_loop_heads_equal_vectorized_heads_with_matching_parameters():
    src, _, _ = inputs()
    loop = MultiHeadAttention(12, 3).double()
    vectorized = SelfAttentionEncoder(12, 3).double()
    with torch.no_grad():
        for name, projection in (("Q", vectorized.q_proj), ("K", vectorized.k_proj), ("V", vectorized.v_proj)):
            projection.weight.copy_(torch.cat([head[name].weight for head in loop.multihead_qkv]))
            projection.bias.copy_(torch.cat([head[name].bias for head in loop.multihead_qkv]))
        vectorized.proj.load_state_dict(loop.head_mesh.state_dict())
    a, aw = loop(src, need_weights=True)
    b, bw = vectorized(src, need_weights=True)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(aw, bw)


@pytest.mark.parametrize("fused", [False, True])
def test_causal_outputs_do_not_read_future_tokens(fused):
    src, _, _ = inputs()
    layer = Attention(12, 3, fused_attn=fused).double().eval()
    changed = src.clone()
    changed[:, 3:] += 100
    torch.testing.assert_close(layer(src, causal=True)[:, :3], layer(changed, causal=True)[:, :3])


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("cross", [False, True])
def test_padding_is_ignored_with_or_without_cross_attention(fused, cross):
    src, tgt, mask = inputs()
    layer = Attention(12, 3, fused_attn=fused).double().eval()
    changed = src.clone()
    changed[~mask] += 100
    target = tgt if cross else None
    first = layer(src, target, mask)
    second = layer(changed, target, mask)
    # For encoder self-attention, only real query rows are compared.
    torch.testing.assert_close(first if cross else first[mask], second if cross else second[mask])


@pytest.mark.parametrize("fused", [False, True])
def test_empty_or_left_padded_rows_are_finite(fused):
    src, _, _ = inputs()
    src.requires_grad_()
    mask = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 1, 1, 1]], dtype=torch.bool)
    layer = Attention(12, 3, fused_attn=fused).double()
    out = layer(src, attention_mask=mask, causal=True)
    assert torch.isfinite(out).all()
    # Zero attention context passes through the output projection's bias.
    torch.testing.assert_close(out[0], layer.out_proj.bias.expand_as(out[0]))
    out.sum().backward()
    assert torch.isfinite(src.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in layer.parameters())


@pytest.mark.parametrize("fused", [False, True])
def test_dropout_is_disabled_in_eval_and_active_in_train(fused):
    src, _, mask = inputs()
    layer = Attention(12, 3, attn_dropout=1.0, fused_attn=fused).double()
    layer.train()
    out = layer(src, attention_mask=mask)
    torch.testing.assert_close(out, layer.out_proj.bias.expand_as(out))
    layer.eval()
    first = layer(src, attention_mask=mask)
    torch.testing.assert_close(first, layer(src, attention_mask=mask))
    assert not torch.allclose(first, out)


def test_attention_interfaces_and_numeric_padding_mask():
    src, tgt, mask = inputs()
    assert SingleHeadAttention(12).double()(src).shape == src.shape
    assert SelfAttention(12, 3, causal=True).double()(src, mask).shape == src.shape
    assert CrossAttention(12, 3).double()(src=src, tgt=tgt, attention_mask=mask).shape == tgt.shape
    layer = Attention(12, 3).double()
    torch.testing.assert_close(layer(src, attention_mask=mask), layer(src, attention_mask=mask.long()))
    assert set(layer.state_dict()) == {f"{name}.{kind}" for name in ("q_proj", "k_proj", "v_proj", "out_proj") for kind in ("weight", "bias")}


@pytest.mark.parametrize("args", [(10, 3), (0, 3), (12, 0)])
def test_invalid_head_dimensions(args):
    with pytest.raises(ValueError):
        Attention(*args)


def test_invalid_inputs_fail_clearly():
    src, tgt, mask = inputs()
    layer = Attention(12, 3).double()
    with pytest.raises(ValueError, match="attention_mask"):
        layer(src, attention_mask=mask[:, :3])
    with pytest.raises(ValueError, match="causal"):
        layer(src, tgt, causal=True)
    with pytest.raises(ValueError, match="batch"):
        layer(src, tgt[:1])
    with pytest.raises(ValueError, match="0/1"):
        layer(src, attention_mask=torch.full_like(mask, 2, dtype=torch.long))
    with pytest.raises(ValueError, match="shape"):
        layer(src[0])
