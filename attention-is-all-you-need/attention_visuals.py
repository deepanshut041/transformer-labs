"""Draw matrix operations and inspect attention tensors without model-side plotting."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon
import ipywidgets as widgets
from IPython import get_ipython
from IPython.display import display, clear_output

from attention_is_all_you_need.attention import Attention, SingleHeadAttention, scaled_dot_product_attention, make_attention_mask

PALETTE = ["#168a91", "#8862bd", "#de844a", "#3664aa", "#bd4c67"]
QA = Path(__file__).resolve().parent / "runs" / "attention-visuals"
figure_names: list[str] = []
_save_figures = True


def configure(*, save_figures=True):
    """Set readable plot defaults and inline display in an active notebook kernel."""
    global _save_figures
    _save_figures = save_figures
    figure_names.clear()
    shell = get_ipython()
    if shell is not None:
        shell.run_line_magic("matplotlib", "inline")
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "font.size": 12,
        "axes.titlesize": 14, "axes.labelsize": 12,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)

def matrix(ax, value, title, row_labels=None, column_labels=None,
           *, cmap="Blues", vmin=None, vmax=None, fmt=".2f", blocked=None,
           annotate=True, xlabel="Key position", ylabel="Query position"):
    """Draw matrix entries with explicit axes and a distinct blocked-cell color."""
    value = array(value)
    finite = value[np.isfinite(value)]
    if vmin is None:
        vmin = min(0.0, float(finite.min())) if finite.size else 0.0
    if vmax is None:
        vmax = max(float(finite.max()), vmin + 1e-9) if finite.size else 1.0
    colors = plt.get_cmap(cmap).copy()
    colors.set_bad("#e5e7eb")
    image = ax.imshow(np.ma.masked_invalid(value), cmap=colors,
                      vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_title(title, pad=13)
    ax.set_xticks(range(value.shape[1]), column_labels or [str(i) for i in range(value.shape[1])])
    ax.set_yticks(range(value.shape[0]), row_labels or [str(i) for i in range(value.shape[0])])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(-.5, value.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, value.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(value.shape[0]):
        for j in range(value.shape[1]):
            is_blocked = blocked is not None and not bool(array(blocked)[i, j])
            if is_blocked:
                ax.add_patch(Rectangle((j-.5, i-.5), 1, 1, facecolor="#e5e7eb", edgecolor="white", zorder=2))
            if annotate:
                item = value[i, j]
                label = "−∞" if np.isneginf(item) else ("undefined" if np.isnan(item) else format(item, fmt))
                level = (item - vmin) / (vmax - vmin) if np.isfinite(item) else 0.0
                dark = level > .55 if cmap == "Blues" else abs(level - .5) > .32
                color = "white" if dark and not is_blocked else "#182433"
                ax.text(j, i, label, ha="center", va="center", color=color, fontsize=12, zorder=3)
    return image

def box(ax, x, y, width, height, title, detail, color):
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=.03,rounding_size=.10",
                               facecolor=color, edgecolor="#cbd5e1", linewidth=1.1))
    ax.text(x + width/2, y + height*.68, title, ha="center", va="center", weight="bold", fontsize=13)
    ax.text(x + width/2, y + height*.28, detail, ha="center", va="center", fontsize=11)

def arrow(ax, start, end, *, color="#405369", rad=0, label=None):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15,
                                linewidth=1.5, color=color, connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((start[0]+end[0])/2, (start[1]+end[1])/2+.18, label,
                ha="center", fontsize=11, color=color)

def finish(fig, name):
    """Save a local inspection copy and embed the figure in the notebook."""
    if name not in figure_names:
        figure_names.append(name)
    if _save_figures:
        QA.mkdir(parents=True, exist_ok=True)
        fig.savefig(QA / f"{name}.png", bbox_inches="tight")
    plt.show()
    plt.close(fig)

def controls(callback, **fields):
    """Connect controls to a plot; a separate static figure is the default view."""
    output = widgets.Output()
    def update(change=None):
        with output:
            clear_output(wait=True)
            callback(**{name: control.value for name, control in fields.items()})
    for control in fields.values():
        control.observe(update, names="value")
    display(widgets.VBox([widgets.HBox(list(fields.values())), output]))
    return fields

def show_weighted_average(weights, v, context):
    """Plot two-key weights and their contributions to the context features."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    a = weights[0, 0].detach().numpy()
    values = v[0].detach().numpy()
    axes[0].bar(["Key 0", "Key 1"], a, color=PALETTE[:2])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Attention weight")
    axes[0].set_title("Equal scores → equal weights")
    for i, weight in enumerate(a):
        axes[0].text(i, weight+.04, f"{weight:.2f}", ha="center")
    bottom = np.zeros(2)
    for i in range(2):
        contribution = a[i] * values[i]
        axes[1].bar(["Feature 0", "Feature 1"], contribution, bottom=bottom,
                    color=PALETTE[i], label=f"{a[i]:.1f} × value {i}")
        for j, item in enumerate(contribution):
            axes[1].text(j, bottom[j]+item/2, f"{item:.1f}", ha="center", va="center", color="white", weight="bold")
        bottom += contribution
    axes[1].set_ylim(0, 7)
    axes[1].set_ylabel("Contribution to context")
    axes[1].set_title("Each output feature is a weighted sum")
    axes[1].legend(loc="upper left")
    for j, total in enumerate(bottom):
        axes[1].text(j, total+.12, f"Total = {total:.0f}", ha="center", weight="bold")
    np.testing.assert_allclose(bottom, context[0, 0].numpy())
    finish(fig, "01-weighted-values")

def show_dot_products():
    """Inspect a selected query/key dot product, scores, weights and mixed values."""
    q_toy = torch.tensor([[[1., 0.], [0., 1.], [1., 1.]]])
    k_toy = q_toy.clone()
    v_toy = torch.tensor([[[2., 0.], [0., 3.], [4., 5.]]])
    toy_context, toy_weights = scaled_dot_product_attention(q_toy, k_toy, v_toy)
    toy_scores = (q_toy @ k_toy.transpose(-2, -1)) / np.sqrt(2)

    def inspect_dot(query_row=2, key_row=2):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
        matrix(axes[0, 0], q_toy[0], "Q: select one query row", xlabel="Feature", ylabel="Query row")
        matrix(axes[0, 1], k_toy[0].T, "Kᵀ: select one key column", xlabel="Key column", ylabel="Feature")
        matrix(axes[1, 0], toy_scores[0], "QKᵀ / √2: one cell per query–key pair")
        axes[0, 0].add_patch(Rectangle((-.5, query_row-.5), 2, 1, fill=False, edgecolor="#e88926", linewidth=3))
        axes[0, 1].add_patch(Rectangle((key_row-.5, -.5), 1, 2, fill=False, edgecolor="#e88926", linewidth=3))
        axes[1, 0].add_patch(Rectangle((key_row-.5, query_row-.5), 1, 1, fill=False, edgecolor="#e88926", linewidth=3))
        products = array(q_toy[0, query_row] * k_toy[0, key_row])
        axes[1, 1].bar(["Feature 0", "Feature 1"], products, color=PALETTE[:2])
        axes[1, 1].set_ylim(0, 1.45)
        axes[1, 1].set_ylabel("Elementwise product")
        score = products.sum() / np.sqrt(2)
        axes[1, 1].set_title(f"Sum products, then scale: {products.sum():.0f} / √2 = {score:.3f}")
        for i, item in enumerate(products):
            axes[1, 1].text(i, item+.04, f"{item:.0f}", ha="center")
        np.testing.assert_allclose(score, toy_scores[0, query_row, key_row].item(), atol=1e-7)
        finish(fig, "02-dot-product-cells")
        return score

    inspect_dot()
    dot_controls = controls(inspect_dot,
        query_row=widgets.IntSlider(value=2, min=0, max=2, description="Query row", continuous_update=False),
        key_row=widgets.IntSlider(value=2, min=0, max=2, description="Key row", continuous_update=False))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), layout="constrained")
    matrix(axes[0], toy_scores[0], "Scaled scores: compare each query to keys")
    matrix(axes[1], toy_weights[0], "Softmax: each row sums to 1", vmin=0, vmax=1)
    matrix(axes[2], toy_context[0], "A @ V: mixed value features", xlabel="Value feature", ylabel="Query position")
    for i, row_sum in enumerate(toy_weights[0].sum(-1)):
        np.testing.assert_allclose(row_sum.item(), 1.0, atol=1e-7)
    finish(fig, "03-scores-weights-context")
    return dot_controls

def show_query_explorer():
    """Change query direction and temperature and inspect weights and value geometry."""
    q_toy = torch.tensor([[[1., 0.], [0., 1.], [1., 1.]]])
    k_toy = q_toy.clone()
    v_toy = torch.tensor([[[2., 0.], [0., 3.], [4., 5.]]])
    def explore_query(angle=0, temperature=1.0):
        radians = np.deg2rad(angle)
        query_vector = torch.tensor([[[np.cos(radians), np.sin(radians)]]], dtype=torch.float32)
        result, attention_weights = scaled_dot_product_attention(query_vector / temperature, k_toy, v_toy)
        probability = array(attention_weights[0, 0])
        mixed = array(result[0, 0])
        np.testing.assert_allclose(probability.sum(), 1.0, atol=1e-7)
        np.testing.assert_allclose(mixed, probability @ array(v_toy[0]), atol=1e-6)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout="constrained")
        for i, key in enumerate(array(k_toy[0])):
            axes[0].arrow(0, 0, *key, color=PALETTE[i], width=.018, length_includes_head=True)
            axes[0].text(key[0]+.06, key[1]+.06, f"K{i}", color=PALETTE[i], weight="bold")
        query_array = array(query_vector[0, 0])
        axes[0].arrow(0, 0, *query_array, color="#172b4d", width=.012, length_includes_head=True, linestyle="--")
        axes[0].text(query_array[0]*.75-.08, query_array[1]*.75-.12, "Q", weight="bold")
        axes[0].axhline(0, color="#cbd5e1", linewidth=.8)
        axes[0].axvline(0, color="#cbd5e1", linewidth=.8)
        axes[0].set(xlim=(-1.3, 1.45), ylim=(-1.3, 1.45), aspect="equal", xlabel="Feature 0", ylabel="Feature 1")
        axes[0].set_title(f"Query direction: {angle}°")
        axes[1].bar(["K0", "K1", "K2"], probability, color=PALETTE[:3])
        axes[1].set(ylim=(0, 1), ylabel="Attention weight")
        axes[1].set_title(f"Softmax distribution · temperature {temperature:.1f}")
        for i, item in enumerate(probability):
            axes[1].text(i, item+.035, f"{item:.3f}", ha="center")
        value_points = array(v_toy[0])
        axes[2].add_patch(Polygon(value_points, fill=False, edgecolor="#94a3b8", linestyle="--"))
        axes[2].scatter(value_points[:, 0], value_points[:, 1], c=PALETTE[:3], s=120)
        for i, point in enumerate(value_points):
            axes[2].annotate(f"V{i}", point, xytext=(7, 7), textcoords="offset points", color=PALETTE[i], weight="bold")
        axes[2].scatter(*mixed, marker="*", s=220, color="#172b4d", zorder=5)
        axes[2].annotate(f"Context\n[{mixed[0]:.2f}, {mixed[1]:.2f}]", mixed,
                         xytext=(12, -35), textcoords="offset points", fontsize=11)
        axes[2].set(xlim=(-.7, 5.1), ylim=(-1.2, 6), xlabel="Value feature 0", ylabel="Value feature 1")
        axes[2].set_title("Weighted context stays within the values")
        finish(fig, "04-query-and-context")
        return probability, mixed

    explore_query()
    query_controls = controls(explore_query,
        angle=widgets.IntSlider(value=0, min=-180, max=180, step=15, description="Angle", continuous_update=False),
        temperature=widgets.FloatSlider(value=1.0, min=.2, max=2, step=.1, description="Temperature", continuous_update=False))
    return query_controls

def show_head_width_scaling():
    """Compare score spread before/after scaling on independent Gaussian pairs."""
    rng = np.random.default_rng(21)
    widths = np.array([4, 16, 64, 256])
    unscaled_std, scaled_std = [], []
    for width in widths:
        queries = rng.normal(size=(8192, width))
        keys = rng.normal(size=(8192, width))
        products = (queries * keys).sum(-1)
        unscaled_std.append(products.std())
        scaled_std.append((products / np.sqrt(width)).std())
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    positions = np.arange(len(widths))
    ax.bar(positions-.18, unscaled_std, width=.36, color=PALETTE[1], label="Q·K")
    ax.bar(positions+.18, scaled_std, width=.36, color=PALETTE[0], label="Q·K / √head width")
    ax.plot(positions, np.sqrt(widths), "o--", color="#4b5563", label="√width prediction")
    ax.axhline(1, color="#64748b", linestyle=":", linewidth=1)
    ax.set_xticks(positions, [str(value) for value in widths])
    ax.set(xlabel="Head width", ylabel="Score standard deviation", title="Scaling keeps this toy score distribution comparable")
    ax.legend()
    assert np.all(np.abs(np.array(scaled_std)-1) < .06)
    finish(fig, "05-head-width-scaling")

def show_projection_flow():
    """Draw the single-head query/key/value execution path and tensor dimensions."""
    fig, ax = plt.subplots(figsize=(13, 5.2), layout="constrained")
    ax.set(xlim=(0, 14), ylim=(0, 5.5))
    ax.axis("off")
    box(ax, .1, 2.1, 1.7, 1.1, "Input X", "[3, 5, 9]", "#edf2f8")
    for y, title, color in ((4.1, "query(X)", "#e6f4f3"), (2.1, "key(X)", "#efe9f7"), (.1, "value(X)", "#fff0e4")):
        box(ax, 2.5, y, 2, 1.1, title, "[3, 5, 9]", color)
        arrow(ax, (1.8, 2.65), (2.5, y+.55))
    box(ax, 5.5, 3.05, 2.1, 1.2, "Q @ Kᵀ / √9", "[3, 5, 5]", "#edf2f8")
    arrow(ax, (4.5, 4.65), (5.5, 3.95))
    arrow(ax, (4.5, 2.65), (5.5, 3.4))
    box(ax, 8.1, 3.05, 2, 1.2, "Softmax", "over key columns", "#e6f4f3")
    arrow(ax, (7.6, 3.65), (8.1, 3.65))
    box(ax, 10.8, 2.15, 2.8, 1.2, "Weights @ V", "Context [3, 5, 9]", "#fff0e4")
    arrow(ax, (10.1, 3.65), (10.8, 3.0))
    arrow(ax, (4.5, .65), (10.8, 2.65), rad=.12)
    ax.text(7, 5.25, "One head: separate learned roles, then one weighted mix", ha="center", fontsize=16, weight="bold")
    ax.text(7, .02, "A score row belongs to a query; its columns describe the keys that query can read.", ha="center", fontsize=12)
    finish(fig, "06-projection-flow")

def show_head_packing(packed_weights):
    """Show feature reindexing into heads, real head weights and a row selector."""
    packed_features = torch.arange(18).reshape(1, 3, 6)
    heads = packed_features.reshape(1, 3, 2, 3).transpose(1, 2)
    recovered = heads.transpose(1, 2).reshape(1, 3, 6)
    assert torch.equal(packed_features, recovered)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), layout="constrained")
    matrix(axes[0], packed_features[0], "Projected tensor [1, 3, 6]", fmt=".0f", xlabel="Packed feature", ylabel="Token")
    axes[0].axvline(2.5, color="#e88926", linewidth=3)
    for h in range(2):
        matrix(axes[h+1], heads[0, h], f"Head {h}: [3 tokens, 3 features]", fmt=".0f", xlabel="Feature within head", ylabel="Token")
    fig.set_layout_engine("none")
    fig.subplots_adjust(top=.77, bottom=.17, wspace=.35)
    fig.suptitle("Reshape [B,T,D] → [B,T,H,d] · transpose → [B,H,T,d]", fontsize=15)
    finish(fig, "07-head-packing")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), layout="constrained")
    for h, ax in enumerate(axes):
        matrix(ax, packed_weights[0, h], f"Head {h}: its own 5×5 weights", vmin=0, vmax=1)
    fig.set_layout_engine("none")
    fig.subplots_adjust(top=.77, bottom=.17, wspace=.35)
    fig.suptitle("Same tokens, separate projections and weight matrices · common color scale", fontsize=15)
    finish(fig, "08-head-weight-matrices")

    def inspect_head(head=0, query_row=0):
        selected = array(packed_weights[0, head])
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), layout="constrained")
        matrix(axes[0], selected, f"Head {head}", vmin=0, vmax=1)
        axes[0].add_patch(Rectangle((-.5, query_row-.5), 5, 1, fill=False, edgecolor="#e88926", linewidth=3))
        axes[1].bar(range(5), selected[query_row], color=PALETTE[head])
        axes[1].set(ylim=(0, 1), xlabel="Key position", ylabel="Weight", title=f"Query {query_row}: choose which values to read")
        axes[1].set_xticks(range(5))
        np.testing.assert_allclose(selected[query_row].sum(), 1.0, atol=1e-6)
        finish(fig, "09-selected-head")
        return selected[query_row]

    inspect_head()
    head_controls = controls(inspect_head,
        head=widgets.IntSlider(value=0, min=0, max=2, description="Head", continuous_update=False),
        query_row=widgets.IntSlider(value=0, min=0, max=4, description="Query row", continuous_update=False))
    return head_controls

def show_padding(padding_mask, encoder_weights):
    """Compare allowed key columns and the corresponding encoder attention weights."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5), layout="constrained")
    allowed = padding_mask[0][None].expand(5, 5)
    matrix(axes[0], allowed.int(), "Padding: identical allowed columns", vmin=0, vmax=1, fmt=".0f", blocked=allowed)
    matrix(axes[1], encoder_weights[0, 0], "Encoder weights: padded columns are zero",
           vmin=0, vmax=1, blocked=allowed)
    fig.suptitle("Gray means blocked; query rows remain present even at padded positions", fontsize=14)
    finish(fig, "10-padding-mask-and-weights")

def show_masks(encoder, x, padding_mask):
    """Inspect padding/causal combinations through masked scores and softmax weights."""
    def inspect_masks(use_padding=True, use_causal=True, query_row=2):
        q_heads = encoder.q_proj(x).reshape(3, 5, 3, 3).transpose(1, 2)
        k_heads = encoder.k_proj(x).reshape(3, 5, 3, 3).transpose(1, 2)
        v_heads = encoder.v_proj(x).reshape(3, 5, 3, 3).transpose(1, 2)
        valid_mask = make_attention_mask(padding_mask if use_padding else None, 5, 5, causal=use_causal)
        context, probability = scaled_dot_product_attention(q_heads, k_heads, v_heads, valid_mask)
        allowed = torch.ones(5, 5, dtype=torch.bool) if valid_mask is None else valid_mask[0, 0].expand(5, 5)
        raw_scores = (q_heads @ k_heads.transpose(-2, -1)) / np.sqrt(3)
        selected_scores = raw_scores[0, 0].masked_fill(~allowed, float("-inf"))
        p = probability[0, 0]
        assert torch.count_nonzero(p[~allowed]) == 0
        assert torch.isfinite(context).all()
        fig, axes = plt.subplots(2, 2, figsize=(11, 9), layout="constrained")
        matrix(axes[0, 0], allowed.int(), "Allowed connections (1 = read this key)", vmin=0, vmax=1, fmt=".0f", blocked=allowed)
        matrix(axes[0, 1], selected_scores, "Blocked scores become −∞", cmap="coolwarm", blocked=allowed)
        matrix(axes[1, 0], p, "Softmax redistributes weight over allowed keys", vmin=0, vmax=1, blocked=allowed)
        axes[1, 0].add_patch(Rectangle((-.5, query_row-.5), 5, 1, fill=False, edgecolor="#e88926", linewidth=3))
        axes[1, 1].bar(range(5), array(p[query_row]), color=PALETTE[0])
        axes[1, 1].set(ylim=(0, 1.08), xlabel="Key position", ylabel="Weight", title=f"Query {query_row}: row sum = {p[query_row].sum():.2f}")
        axes[1, 1].set_xticks(range(5))
        fig.suptitle(f"Padding = {use_padding} · Causal = {use_causal} · Same scores, different allowed connections", fontsize=15)
        finish(fig, "11-mask-score-weight-trace")
        return array(p), array(allowed)

    inspect_masks()
    mask_controls = controls(inspect_masks,
        use_padding=widgets.Checkbox(value=True, description="Padding"),
        use_causal=widgets.Checkbox(value=True, description="Causal"),
        query_row=widgets.IntSlider(value=2, min=0, max=4, description="Query row", continuous_update=False))
    return mask_controls

def show_future_isolation(decoder, x):
    """Measure output changes with identical weights and different causal constraints."""
    bidirectional = copy.deepcopy(decoder)
    bidirectional.causal = False
    amplitudes = np.array([0., 1., 5., 20., 100.])
    causal_changes, unmasked_changes = [], []
    with torch.inference_mode():
        causal_base = decoder(x)
        unmasked_base = bidirectional(x)
        for amplitude in amplitudes:
            changed = x.clone()
            changed[:, -1] += amplitude
            causal_changes.append((decoder(changed)[:, :-1] - causal_base[:, :-1]).abs().max().item())
            unmasked_changes.append((bidirectional(changed)[:, :-1] - unmasked_base[:, :-1]).abs().max().item())
        last_change = x.clone()
        last_change[:, -1] += 20
        per_position = (decoder(last_change) - causal_base).abs().amax(dim=(0, 2)).numpy()
    assert max(causal_changes) == 0
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    axes[0].plot(amplitudes, unmasked_changes, "o-", color=PALETTE[1], label="No causal mask")
    axes[0].plot(amplitudes, causal_changes, "o-", color=PALETTE[0], label="Causal mask")
    axes[0].set(xlabel="Change added to the final input token", ylabel="Largest change at earlier outputs", title="Same parameters; only the causal mask changes")
    axes[0].legend()
    axes[1].bar(range(5), per_position, color=[PALETTE[0]]*4 + [PALETTE[2]])
    axes[1].set_xticks(range(5))
    axes[1].set(xlabel="Output query position", ylabel="Maximum absolute change", title="Only the final position is allowed to change")
    finish(fig, "12-future-token-isolation")

def show_cross_attention(padding_mask, cross_weights):
    """Draw target-query/source-key flow and the rectangular cross-attention weights."""
    fig, ax = plt.subplots(figsize=(13, 5), layout="constrained")
    ax.set(xlim=(0, 14), ylim=(0, 5.2))
    ax.axis("off")
    box(ax, .1, 3.4, 2.5, 1.1, "Source", "[3, 5, 9]", "#efe9f7")
    box(ax, .1, .7, 2.5, 1.1, "Target", "[3, 7, 9]", "#e6f4f3")
    box(ax, 3.4, 3.4, 2.5, 1.1, "K and V", "5 source keys", "#efe9f7")
    box(ax, 3.4, .7, 2.5, 1.1, "Q", "7 target queries", "#e6f4f3")
    arrow(ax, (2.6, 3.95), (3.4, 3.95))
    arrow(ax, (2.6, 1.25), (3.4, 1.25))
    box(ax, 6.7, 2.05, 2.7, 1.3, "Q @ Kᵀ / √3", "[3, 3, 7, 5]", "#edf2f8")
    arrow(ax, (5.9, 3.95), (6.7, 3.1))
    arrow(ax, (5.9, 1.25), (6.7, 2.3))
    box(ax, 10.3, 2.05, 3.2, 1.3, "Mix V, merge + project", "Output [3, 7, 9]", "#fff0e4")
    arrow(ax, (9.4, 2.7), (10.3, 2.7))
    ax.text(9.85, 1.6, "Softmax over\nsource keys", ha="center", va="center", fontsize=11)
    arrow(ax, (5.9, 4.2), (11.5, 3.35), rad=-.25, label="V")
    ax.text(7, 4.95, "Cross-attention: output positions follow the target, not the source", ha="center", fontsize=16, weight="bold")
    finish(fig, "13-cross-attention-flow")
    fig, ax = plt.subplots(figsize=(7.5, 7.5), layout="constrained")
    valid_source = padding_mask[0][None].expand(7, 5)
    matrix(ax, cross_weights[0, 0], "7 target queries × 5 source keys",
           row_labels=[f"target {i}" for i in range(7)],
           column_labels=["src 0", "src 1", "src 2", "PAD", "PAD"],
           vmin=0, vmax=1, blocked=valid_source,
           xlabel="Source keys (columns)", ylabel="Target queries (rows)")
    fig.suptitle("Each row reads the source; gray columns cannot contribute", fontsize=14)
    finish(fig, "14-rectangular-cross-attention")

def show_path_comparison(manual, efficient, x, target, padding_mask):
    """Compare explicit and fused output tensors under matching settings."""
    with torch.inference_mode():
        explicit_output = manual(x, tgt=target, attention_mask=padding_mask)
        efficient_output = efficient(x, tgt=target, attention_mask=padding_mask)
    error = (explicit_output - efficient_output).abs()
    torch.testing.assert_close(explicit_output, efficient_output, atol=1e-6, rtol=1e-5)
    maximum = max(float(explicit_output.abs().max()), float(efficient_output.abs().max()))
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), layout="constrained")
    for ax, output, title in ((axes[0], explicit_output, "Explicit attention"), (axes[1], efficient_output, "Fused SDPA")):
        matrix(ax, output[0], title, cmap="coolwarm", vmin=-maximum, vmax=maximum,
               annotate=False, xlabel="Output feature", ylabel="Target query")
    image = matrix(axes[2], error[0], f"Absolute difference · max {error.max():.2e}",
                   vmin=0, vmax=max(float(error.max()), 1e-9), annotate=False,
                   xlabel="Output feature", ylabel="Target query")
    fig.colorbar(image, ax=axes[2], shrink=.75, label="Absolute error (different scale)")
    fig.suptitle("Matching inputs, parameters, masks and dropout settings", fontsize=16)
    finish(fig, "15-explicit-fused-parity")

def show_gradient_flow(trainable):
    """Draw the backward path and plot actual projection-gradient norms."""
    labels = []
    norms = []
    for name, parameter in trainable.named_parameters():
        labels.append(name.replace("_proj", "").replace(".weight", " W").replace(".bias", " b"))
        norms.append(parameter.grad.norm().item())
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=(1, 1.4), layout="constrained")
    ax = axes[0]
    ax.set(xlim=(0, 12), ylim=(0, 3))
    ax.axis("off")
    for start, title, detail, color in ((.1, "Input", "x", "#edf2f8"), (3.1, "Attention", "Q, K, V → context", "#e6f4f3"), (6.1, "Projection", "out_proj(context)", "#efe9f7"), (9.1, "Objective", "mean(output²)", "#fff0e4")):
        box(ax, start, 1.2, 2.7, 1.1, title, detail, color)
    for start in (2.8, 5.8, 8.8):
        arrow(ax, (start, 1.75), (start+.3, 1.75))
    arrow(ax, (10.4, 1.1), (1.4, 1.1), color=PALETTE[2], rad=-.15)
    ax.text(6, .06, "Backward: gradients reach the projections and input · no optimizer update here", ha="center", fontsize=12)
    axes[1].bar(labels, norms, color=[PALETTE[0], PALETTE[0], PALETTE[1], PALETTE[1], PALETTE[2], PALETTE[2], PALETTE[3], PALETTE[3]])
    axes[1].set_yscale("log")
    axes[1].set(ylabel="Gradient L2 norm (log scale)", xlabel="Projection parameter", title="Gradients from the actual backward pass")
    assert np.isfinite(norms).all()
    finish(fig, "16-gradient-flow")


def matrix_product_data(left, right, bias=None):
    """Return operands and result for a two-dimensional matrix multiplication."""
    left = torch.as_tensor(left, dtype=torch.float64).detach().cpu()
    right = torch.as_tensor(right, dtype=torch.float64).detach().cpu()
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
        raise ValueError("left columns must match right rows for two-dimensional matrices")
    result = left @ right
    if bias is not None:
        bias = torch.as_tensor(bias, dtype=torch.float64).detach().cpu()
        if bias.ndim != 1 or bias.shape[0] != result.shape[1]:
            raise ValueError("bias must contain one value per output column")
        result = result + bias
    return {"left": left, "right": right, "bias": bias, "result": result}


def projection_example():
    """Compute a controlled three-token Q/K/V example with the real layer."""
    x = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], dtype=torch.float64)
    weights = {
        "Q": torch.tensor([[1., 1.], [0., 1.]], dtype=torch.float64),
        "K": torch.tensor([[1., 0.], [1., -1.]], dtype=torch.float64),
        "V": torch.tensor([[2., 0.], [0., 3.]], dtype=torch.float64),
    }
    biases = {"Q": torch.zeros(2, dtype=torch.float64),
              "K": torch.zeros(2, dtype=torch.float64),
              "V": torch.tensor([.5, -.5], dtype=torch.float64)}
    # Fixture construction must not change the caller's random initialization.
    with torch.random.fork_rng(devices=[]):
        layer = SingleHeadAttention(2).double().eval()
    result = {"X": x}
    with torch.no_grad():
        for name, projection in (("Q", layer.query), ("K", layer.key), ("V", layer.value)):
            projection.weight.copy_(weights[name])
            projection.bias.copy_(biases[name])
            result["W_" + name] = projection.weight.T.detach().clone()
            result["b_" + name] = projection.bias.detach().clone()
            result[name] = projection(x)
        result["K_T"] = result["K"].T
        result["raw_scores"] = result["Q"] @ result["K_T"]
        result["scores"] = result["raw_scores"] / np.sqrt(2)
        context, attention_weights = scaled_dot_product_attention(result["Q"], result["K"], result["V"])
        result["A"], result["context"] = attention_weights, context
        torch.testing.assert_close(layer(x[None])[0], context)
    return result


def _number(value):
    value = float(value)
    return str(int(round(value))) if abs(value-round(value)) < 1e-9 else f"{value:.3f}"


def _bracket_matrix(ax, data, x, center, name, role="", *, row=None, column=None, cell=None):
    """Draw a numbered matrix in brackets and highlight selected operands."""
    data = array(data)
    rows, columns = data.shape
    step_x, step_y = .88, .72
    width, height = columns*step_x+.35, rows*step_y
    bottom = center-height/2
    for i in range(rows):
        for j in range(columns):
            y = center + height/2 - (i+.5)*step_y
            selected = (row == i or column == j or cell == (i, j))
            if selected:
                ax.add_patch(Rectangle((x+.16+j*step_x, y-step_y*.41), step_x, step_y*.82,
                                       facecolor="#fff0c5", edgecolor="#e88926", linewidth=1.3))
            ax.text(x+.16+(j+.5)*step_x, y, _number(data[i,j]),
                    ha="center", va="center", fontsize=15, color="#162433")
    for edge, sign in ((x, 1), (x+width, -1)):
        ax.plot([edge+sign*.13, edge, edge, edge+sign*.13],
                [bottom+height, bottom+height, bottom, bottom], color="#25364d", linewidth=1.8)
    ax.text(x+width/2, center+height/2+.50, name, ha="center", weight="bold", fontsize=16)
    ax.text(x+width/2, bottom-.38, f"{rows} × {columns}", ha="center", fontsize=13)
    if role:
        ax.text(x+width/2, bottom-.78, role, ha="center", fontsize=11)
    return width


def show_matrix_product(left=None, right=None, *, left_name="A", right_name="B", result_name="C",
                        bias=None, cell=(0, 1), title="A row × B column gives one output cell",
                        roles=("left rows", "right columns", "result rows"), figure_name="17-basic-matrix-product"):
    """Display A × B (+ bias) = C as numbered matrices with one scalar expansion."""
    if left is None:
        left = [[1., 2.], [3., 4.]]
    if right is None:
        right = [[5., 6.], [7., 8.]]
    data = matrix_product_data(left, right, bias)
    a, b, c, bias = data["left"], data["right"], data["result"], data["bias"]
    i, j = cell
    if not (0 <= i < c.shape[0] and 0 <= j < c.shape[1]):
        raise ValueError("selected cell must lie within the result matrix")
    sizes = [a.shape[1]*.88+.35, b.shape[1]*.88+.35, c.shape[1]*.88+.35]
    if bias is not None:
        sizes.insert(2, bias.shape[0]*.88+.35)
    total_width = sum(sizes) + (len(sizes)-1)*.95 + .4
    fig, ax = plt.subplots(figsize=(min(16, max(11, total_width*.85)), 5.4), layout="constrained")
    center = 1.8
    max_rows = max(a.shape[0], b.shape[0], c.shape[0])
    ax.set(xlim=(-.05,total_width), ylim=(-1.9, center+max_rows*.36+1.45))
    ax.axis("off")
    pos = .1
    pos += _bracket_matrix(ax, a, pos, center, left_name, roles[0], row=i)
    ax.text(pos+.47, center, "×", ha="center", va="center", fontsize=24)
    pos += .95
    pos += _bracket_matrix(ax, b, pos, center, right_name, roles[1], column=j)
    if bias is not None:
        ax.text(pos+.47, center, "+", ha="center", va="center", fontsize=24)
        pos += .95
        pos += _bracket_matrix(ax, bias[None], pos, center, "bias", "broadcast over rows", column=j)
    ax.text(pos+.47, center, "=", ha="center", va="center", fontsize=24)
    pos += .95
    _bracket_matrix(ax, c, pos, center, result_name, roles[2], cell=(i,j))
    terms = [f"{_number(a[i,k])}×({_number(b[k,j])})" for k in range(a.shape[1])]
    if bias is not None:
        terms.append(f"({_number(bias[j])})")
    ax.text(total_width/2, -1.35,
            f"{result_name}[{i}, {j}] = " + " + ".join(terms) + f" = {_number(c[i,j])}",
            ha="center", fontsize=14, color="#182433")
    fig.suptitle(title, fontsize=17)
    finish(fig, figure_name)


def show_projection(name="Q"):
    """Display X × learned weight + bias for the query, key or value role."""
    if name not in ("Q", "K", "V"):
        raise ValueError("projection must be Q, K or V")
    data = projection_example()
    title = {"Q": "Query vectors: a learned transformation of each input token",
             "K": "Key vectors: the representations compared against queries",
             "V": "Value vectors: the payload that attention weights will mix"}[name]
    cell = {"Q": (2,0), "K": (2,1), "V": (1,1)}[name]
    product = matrix_product_data(data["X"], data["W_"+name], data["b_"+name])
    torch.testing.assert_close(product["result"], data[name])
    show_matrix_product(data["X"], data["W_"+name], bias=data["b_"+name],
                        left_name="X", right_name="W_"+name, result_name=name, cell=cell, title=title,
                        roles=("tokens × input features", "input × output features", "tokens × role features"),
                        figure_name={"Q":"18-query-projection", "K":"19-key-projection", "V":"20-value-projection"}[name])


def show_unprojected_attention(x, weights, context):
    """Show the score and value products when input tokens fill all three roles."""
    x, weights, context = x[0], weights[0], context[0]
    torch.testing.assert_close(weights @ x, context)
    show_matrix_product(
        x, x.T, left_name="X", right_name="Xᵀ", result_name="R", cell=(2, 1),
        title="Before learned projections: compare input tokens directly",
        roles=("tokens × features", "features × tokens", "queries × keys"),
        figure_name="27-unprojected-scores",
    )
    show_matrix_product(
        weights, x, left_name="A", right_name="X", result_name="Context", cell=(0, 1),
        title="After scaling and softmax: mix the original input vectors",
        roles=("queries × source keys", "source keys × features", "queries × features"),
        figure_name="28-unprojected-context",
    )


def show_key_transpose():
    """Show how key rows become columns before the query/key multiplication."""
    data = projection_example()
    fig, ax = plt.subplots(figsize=(10.5, 5), layout="constrained")
    ax.set(xlim=(0,8.5), ylim=(-1.25,4.5))
    ax.axis("off")
    _bracket_matrix(ax, data["K"], .1, 1.8, "K", "rows = keys; columns = features", cell=(0,1))
    arrow(ax, (2.5,1.8), (4.4,1.8), label="transpose")
    _bracket_matrix(ax, data["K_T"], 4.8,1.8,"Kᵀ", "rows = features; columns = keys", cell=(1,0))
    ax.text(4.2,-.9,"K[0,1] moves to Kᵀ[1,0] · same value, exchanged axes",ha="center",fontsize=13)
    fig.suptitle("The inner feature axes must meet: Q [tokens × features] × Kᵀ [features × keys]",fontsize=15)
    finish(fig,"21-key-transpose")


def show_score_product():
    """Display every query/key dot product as Q × Kᵀ = raw scores."""
    data = projection_example()
    show_matrix_product(data["Q"],data["K_T"],left_name="Q",right_name="Kᵀ",result_name="R",cell=(1,1),
                        title="Each score pairs one query row with one key column",
                        roles=("queries × features","features × keys","queries × keys"),
                        figure_name="22-query-key-multiplication")


def show_score_normalization():
    """Display raw scores, scaling and row-wise softmax as numbered matrices."""
    data=projection_example()
    fig,ax=plt.subplots(figsize=(13,6),layout="constrained")
    ax.set(xlim=(0,12.2),ylim=(-2.4,4.5))
    ax.axis("off")
    _bracket_matrix(ax,data["raw_scores"],.1,1.8,"R = QKᵀ","raw pair scores",row=0)
    ax.text(3.35,1.8,"÷ √2",ha="center",fontsize=18)
    _bracket_matrix(ax,data["scores"],4.1,1.8,"S","scaled scores",row=0)
    arrow(ax,(7.2,1.8),(8.3,1.8))
    ax.text(7.75,2.55,"softmax\nover each row",ha="center",fontsize=11)
    _bracket_matrix(ax,data["A"],8.7,1.8,"A","query-specific weights",row=0)
    exponentials=data["scores"][0].exp()
    ax.text(6.1,-1.10,"Row 0 exponentials: ["+", ".join(_number(v) for v in exponentials)+"]",ha="center",fontsize=13)
    ax.text(6.1,-1.60,f"Divide by their sum {_number(exponentials.sum())} → ["+", ".join(_number(v) for v in data["A"][0])+ "]",ha="center",fontsize=13)
    ax.text(6.1,-2.10,"Each weight row sums to 1; rows are normalized separately.",ha="center",fontsize=13)
    fig.suptitle("Raw score → divide by √head width → exponentiate and normalize",fontsize=16)
    torch.testing.assert_close(data["A"].sum(-1),torch.ones(3,dtype=torch.float64))
    finish(fig,"23-scaling-and-softmax-matrices")


def show_value_product():
    """Display attention weights × value payloads = the context matrix."""
    data=projection_example()
    product=matrix_product_data(data["A"],data["V"])
    torch.testing.assert_close(product["result"],data["context"])
    show_matrix_product(data["A"],data["V"],left_name="A",right_name="V",result_name="Context",cell=(0,1),
                        title="One weight row mixes the value rows into one context vector",
                        roles=("queries × source keys","source keys × value features","queries × value features"),
                        figure_name="24-weights-times-values")


def output_projection_example():
    """Return two synthetic head contexts and a controlled output projection."""
    first=torch.tensor([[1.,0.],[0.,1.]],dtype=torch.float64)
    second=torch.tensor([[2.,1.],[1.,2.]],dtype=torch.float64)
    heads=torch.stack((first,second))[None]
    merged=heads.transpose(1,2).flatten(2)[0]
    weight=torch.tensor([[1.,0.,1.,0.],[0.,1.,0.,1.],[1.,0.,0.,1.],[0.,1.,1.,0.]],dtype=torch.float64)
    bias=torch.zeros(4,dtype=torch.float64)
    with torch.random.fork_rng(devices=[]):
        projection=Attention(embedding_dimension=4, num_heads=2).double().out_proj
    with torch.no_grad():
        projection.weight.copy_(weight.T)
        projection.bias.copy_(bias)
        output=projection(merged)
    torch.testing.assert_close(merged,torch.cat((first,second),dim=-1))
    return {"head0":first,"head1":second,"merged":merged,"W_O":weight,"bias":bias,"output":output}


def show_head_merge():
    """Show concatenation along the feature axis before the output projection."""
    data=output_projection_example()
    fig,ax=plt.subplots(figsize=(12,5),layout="constrained")
    ax.set(xlim=(0,11.5),ylim=(-1.6,4))
    ax.axis("off")
    _bracket_matrix(ax,data["head0"],.1,1.8,"Head 0","tokens × head features")
    ax.text(2.8,1.8,"‖",ha="center",fontsize=24)
    _bracket_matrix(ax,data["head1"],3.35,1.8,"Head 1","tokens × head features")
    ax.text(6.05,1.8,"=",ha="center",fontsize=24)
    _bracket_matrix(ax,data["merged"],6.6,1.8,"Merged context","tokens × all head features")
    ax.text(5.75,-1.15,"Concatenation keeps every number and places the head features side by side.",ha="center",fontsize=13)
    fig.suptitle("transpose(1, 2).flatten(2): merge the head and feature axes",fontsize=16)
    finish(fig,"25-head-context-concatenation")


def show_output_projection():
    """Display the learned linear mix applied after head contexts are merged."""
    data=output_projection_example()
    product=matrix_product_data(data["merged"],data["W_O"],data["bias"])
    torch.testing.assert_close(product["result"],data["output"])
    show_matrix_product(data["merged"],data["W_O"],bias=data["bias"],left_name="Merged",right_name="W_O",result_name="Output",cell=(0,0),
                        title="The output projection mixes information across head features",
                        roles=("tokens × merged features","input × output features","tokens × output features"),
                        figure_name="26-output-projection-multiplication")
