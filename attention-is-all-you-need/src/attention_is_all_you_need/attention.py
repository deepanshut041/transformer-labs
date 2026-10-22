from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

__all__ = [
    "Attention", "SingleHeadAttention", "MultiHeadAttention",
    "SelfAttentionEncoder", "SelfAttention", "CrossAttention",
    "make_attention_mask", "scaled_dot_product_attention",
]


def _dimensions(embed_dim: int, num_heads: int) -> int:
    """Check that equal-width heads fit the embedding and return head width."""
    if embed_dim <= 0 or num_heads <= 0 or embed_dim % num_heads:
        raise ValueError("embedding dimension must be positive and divisible by num_heads")
    return embed_dim // num_heads


def _sequence(x: Tensor, embed_dim: int, name: str) -> None:
    """Validate a floating-point (batch, sequence, embedding) tensor."""
    if x.ndim != 3 or x.shape[-1] != embed_dim or x.shape[1] == 0:
        raise ValueError(f"{name} must have shape (batch, nonempty sequence, {embed_dim})")
    if not x.is_floating_point():
        raise TypeError(f"{name} must contain floating-point embeddings")


def make_attention_mask(
    attention_mask: Tensor | None,
    query_length: int,
    key_length: int,
    *,
    causal: bool = False,
    device: torch.device | str | None = None,
) -> Tensor | None:
    """Build the allowed query-key pairs from padding and causal constraints.

    attention_mask has shape (batch, keys), with True/1 for valid source keys.
    Padding produces a (batch, 1, 1, keys) mask that broadcasts across heads
    and queries. Causality combines it with a lower triangle, so query i can
    read only keys j <= i. The query and key lengths must then be equal.

    Return None when neither constraint is present. A padding mask excludes
    key columns; it does not zero the outputs of padded query positions.
    """
    allowed = None
    if attention_mask is not None:
        if attention_mask.ndim != 2 or attention_mask.shape[1] != key_length:
            raise ValueError("attention_mask must have shape (batch, source sequence)")
        if attention_mask.dtype != torch.bool:
            if not torch.all((attention_mask == 0) | (attention_mask == 1)):
                raise ValueError("attention_mask values must be boolean or 0/1")
        if device is None:
            device = attention_mask.device
        allowed = attention_mask.to(device=device, dtype=torch.bool)[:, None, None, :]
    if causal:
        if query_length != key_length:
            raise ValueError("causal attention requires self-attention with equal lengths")
        triangle = torch.ones(query_length, key_length, dtype=torch.bool, device=device).tril()
        triangle = triangle[None, None, :, :]
        allowed = triangle if allowed is None else allowed & triangle
    return allowed


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attention_mask: Tensor | None = None,
    *,
    dropout_p: float = 0.0,
    training: bool = False,
) -> tuple[Tensor, Tensor]:
    """Compute softmax(Q K^T / sqrt(head_dim)) V and return context and weights.

    q has shape (..., queries, head_dim), k has (..., keys, head_dim), and
    v has (..., keys, value_dim). The result has (..., queries, value_dim)
    shape; weights have (..., queries, keys) shape.

    The boolean mask must broadcast to the scores, with True for allowed
    pairs. Softmax runs across keys. Fully blocked rows produce zero weights
    and context. Returned weights are before dropout, so valid rows sum to
    one even when dropout is applied to the weights used for the context.
    """
    scores = (q @ k.transpose(-2, -1)) * q.shape[-1] ** -0.5
    has_keys = None
    if attention_mask is not None:
        scores = scores.masked_fill(~attention_mask, float("-inf"))
        has_keys = attention_mask.any(dim=-1, keepdim=True)
        # softmax([-inf, ...]) is undefined. Replace only fully blocked rows.
        scores = torch.where(has_keys, scores, 0.0)
    weights = scores.softmax(dim=-1)
    if has_keys is not None:
        weights = torch.where(has_keys, weights, 0.0)
    context = F.dropout(weights, p=dropout_p, training=training) @ v
    return context, weights


def _heads(x: Tensor, num_heads: int, head_dim: int) -> Tensor:
    """Split (batch, sequence, embedding) into (batch, heads, sequence, width)."""
    batch, length, _ = x.shape
    return x.reshape(batch, length, num_heads, head_dim).transpose(1, 2)


def _context(
    q: Tensor, k: Tensor, v: Tensor, attention_mask: Tensor | None,
    *, causal: bool, dropout_p: float, training: bool,
    fused_attn: bool, need_weights: bool,
) -> tuple[Tensor, Tensor | None]:
    """Select SDPA or explicit attention with the same masking and dropout rules.

    Requesting weights selects the explicit path. SDPA handles a causal mask
    internally when no padding mask is present; otherwise combine the masks
    before the call. Dropout applies only during training.
    """
    mask = make_attention_mask(
        attention_mask, q.shape[-2], k.shape[-2],
        # SDPA can build its own causal mask when there is no padding mask.
        causal=causal and not (fused_attn and not need_weights and attention_mask is None),
        device=q.device,
    )
    if fused_attn and not need_weights:
        context = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask,
            dropout_p=dropout_p if training else 0.0,
            is_causal=causal and attention_mask is None,
        )
        return context, None
    return scaled_dot_product_attention(q, k, v, mask, dropout_p=dropout_p, training=training)


class SingleHeadAttention(nn.Module):
    """Learn one attention distribution per query over the same input sequence.

    Three linear layers project input embeddings into queries, keys and
    values. Scaled query-key scores become softmax weights, which mix the
    values. Input and output both have (batch, sequence, embedding) shape.
    need_weights=True also returns (batch, sequence, sequence) weights.
    """

    def __init__(self, embedding_dimension: int):
        super().__init__()
        _dimensions(embedding_dimension, 1)
        self.embed_dim = embedding_dimension
        self.query = nn.Linear(embedding_dimension, embedding_dimension)
        self.key = nn.Linear(embedding_dimension, embedding_dimension)
        self.value = nn.Linear(embedding_dimension, embedding_dimension)

    def forward(self, x: Tensor, *, need_weights: bool = False):
        _sequence(x, self.embed_dim, "x")
        output, weights = scaled_dot_product_attention(self.query(x), self.key(x), self.value(x))
        return (output, weights) if need_weights else output


class MultiHeadAttention(nn.Module):
    """Compute independent attention heads with separate projection layers.

    Each head projects the full embedding to embedding_dimension / num_heads
    features and computes its own attention weights. Concatenating the heads
    restores the embedding width; head_mesh then mixes their information.

    Heads run in a loop, which makes their separate calculations easy to
    inspect. Input/output have (batch, sequence, embedding) shape. Optional
    weights have (batch, heads, sequence, sequence) shape.
    """

    def __init__(self, embedding_dimension: int, num_heads: int):
        super().__init__()
        self.head_dim = _dimensions(embedding_dimension, num_heads)
        self.embed_dim = embedding_dimension
        self.num_heads = num_heads
        self.multihead_qkv = nn.ModuleList([
            nn.ModuleDict({name: nn.Linear(embedding_dimension, self.head_dim)
                           for name in ("Q", "K", "V")})
            for _ in range(num_heads)
        ])
        self.head_mesh = nn.Linear(embedding_dimension, embedding_dimension)

    def forward(self, x: Tensor, *, need_weights: bool = False):
        _sequence(x, self.embed_dim, "x")
        outputs, weights = [], []
        for head in self.multihead_qkv:
            output, weight = scaled_dot_product_attention(head["Q"](x), head["K"](x), head["V"](x))
            outputs.append(output)
            weights.append(weight)
        output = self.head_mesh(torch.cat(outputs, dim=-1))
        return (output, torch.stack(weights, dim=1)) if need_weights else output


class SelfAttention(nn.Module):
    """Compute all self-attention heads together using batched tensor operations.

    Queries, keys and values come from the same input sequence. Padding masks
    exclude invalid keys; causal=True additionally prevents reading future
    positions. Attention dropout acts on weights, and projection dropout acts
    on the final output. Both are disabled by eval().

    Input/output have (batch, sequence, embed_dim) shape. fused_attn=True uses
    PyTorch SDPA. need_weights=True selects the explicit calculation and also
    returns (batch, heads, sequence, sequence) weights.
    """

    def __init__(self, embed_dim=768, num_heads=12, attn_p=0.0, proj_p=0.0,
                 causal=False, fused_attn=False):
        super().__init__()
        self.head_dim = _dimensions(embed_dim, num_heads)
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.causal, self.fused = causal, fused_attn
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_p)

    def _forward(self, src, tgt, attention_mask, need_weights):
        _validate_inputs(src, tgt, attention_mask, self.embed_dim)
        query_input = src if tgt is None else tgt
        q = _heads(self.q_proj(query_input), self.num_heads, self.head_dim)
        k = _heads(self.k_proj(src), self.num_heads, self.head_dim)
        v = _heads(self.v_proj(src), self.num_heads, self.head_dim)
        context, weights = _context(
            q, k, v, attention_mask, causal=self.causal,
            dropout_p=self.attn_drop.p, training=self.training,
            fused_attn=self.fused, need_weights=need_weights,
        )
        output = self.proj_drop(self.proj(context.transpose(1, 2).flatten(2)))
        return (output, weights) if need_weights else output

    def forward(self, x, attention_mask=None, *, need_weights=False):
        return self._forward(x, None, attention_mask, need_weights)


class SelfAttentionEncoder(SelfAttention):
    """Bidirectional self-attention for encoding a complete input sequence.

    Every query can attend to every valid source key, including positions
    before and after it. A supplied padding mask excludes padded keys.
    Uses the vectorized SelfAttention calculation without a causal mask.
    """

    def __init__(self, embed_dim=768, num_heads=12, attn_p=0.0, proj_p=0.0):
        super().__init__(embed_dim, num_heads, attn_p, proj_p)


class CrossAttention(SelfAttention):
    """Let target queries retrieve information from source keys and values.

    src has (batch, source_length, embed_dim) shape and tgt has
    (batch, target_length, embed_dim) shape; their lengths may differ.
    attention_mask has (batch, source_length) shape and excludes source keys.

    Output follows the target length. Optional weights have
    (batch, heads, target_length, source_length) shape. No causal mask is
    applied to the source sequence.
    """

    def __init__(self, embed_dim=768, num_heads=12, attn_p=0.0, proj_p=0.0):
        super().__init__(embed_dim, num_heads, attn_p, proj_p)

    def forward(self, src, tgt, attention_mask=None, *, need_weights=False):
        return self._forward(src, tgt, attention_mask, need_weights)


def _validate_inputs(src, tgt, attention_mask, embed_dim):
    """Check embedding shapes, matching batches/devices/dtypes and source mask."""
    _sequence(src, embed_dim, "src")
    if tgt is not None:
        _sequence(tgt, embed_dim, "tgt")
        if tgt.shape[0] != src.shape[0] or tgt.dtype != src.dtype or tgt.device != src.device:
            raise ValueError("src and tgt must share batch size, dtype and device")
    if attention_mask is not None and attention_mask.shape != src.shape[:2]:
        raise ValueError("attention_mask must have shape (batch, source sequence)")


class Attention(nn.Module):
    """Multi-head attention supporting self-attention and cross-attention.

    src: (batch, source length, embedding_dimension)
    tgt: optional (batch, target length, embedding_dimension)
    attention_mask: optional (batch, source length), True/1 = allowed key

    Without tgt, all three projections use src. With tgt, queries come from
    tgt and keys/values come from src, so output length follows tgt. Split
    each projection into equal-width heads, compute attention, concatenate
    the head contexts and apply out_proj to mix them.

    causal=True applies only to self-attention. Padding masks always refer
    to source keys, with True/1 meaning allowed. The default fused path uses
    PyTorch SDPA; fused_attn=False computes scores and weights explicitly.
    Dropout applies only during training. need_weights=True uses the explicit
    path and returns (output, weights), with weights shaped
    (batch, heads, query_length, source_length).
    """

    def __init__(self, embedding_dimension=768, num_heads=12, attn_dropout=0.0,
                 *, fused_attn=True):
        super().__init__()
        self.head_dim = _dimensions(embedding_dimension, num_heads)
        if not 0 <= attn_dropout <= 1:
            raise ValueError("attn_dropout must lie between zero and one")
        self.embedding_dimension = embedding_dimension
        self.num_heads, self.attn_dropout = num_heads, attn_dropout
        self.fused_attn = fused_attn
        self.q_proj = nn.Linear(embedding_dimension, embedding_dimension)
        self.k_proj = nn.Linear(embedding_dimension, embedding_dimension)
        self.v_proj = nn.Linear(embedding_dimension, embedding_dimension)
        self.out_proj = nn.Linear(embedding_dimension, embedding_dimension)

    def forward(self, src, tgt=None, attention_mask=None, causal=False, *, need_weights=False):
        _validate_inputs(src, tgt, attention_mask, self.embedding_dimension)
        if tgt is not None and causal:
            raise ValueError("causal masking is only supported for self-attention")
        query_input = src if tgt is None else tgt
        q = _heads(self.q_proj(query_input), self.num_heads, self.head_dim)
        k = _heads(self.k_proj(src), self.num_heads, self.head_dim)
        v = _heads(self.v_proj(src), self.num_heads, self.head_dim)
        context, weights = _context(
            q, k, v, attention_mask, causal=causal,
            dropout_p=self.attn_dropout, training=self.training,
            fused_attn=self.fused_attn, need_weights=need_weights,
        )
        output = self.out_proj(context.transpose(1, 2).flatten(2))
        return (output, weights) if need_weights else output
