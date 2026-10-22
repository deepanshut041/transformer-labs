# Attention Is All You Need

I'm following Priyam Mazumdar's three videos to build an English-to-French Transformer in Python, from attention through to training and translation.

Video 1 is implemented: attention without learned projections, single-head and multi-head attention, padding and causal masks, cross-attention, and the final efficient attention class. The code stays in Python files; [attention_mechanism.ipynb](attention_mechanism.ipynb) explains the basic concepts and runs the same code. Video 2's data preparation is next.

## Run video 1

From this project's directory:

```bash
uv run --frozen python attention.py
```

It prints the weighted-average example, attention directly on input vectors, attention matrices and final output tensors. Encoder/decoder outputs have shape `(3, 5, 9)`; cross-attention has shape `(3, 7, 9)`. The explicit and efficient paths are checked against each other, ending with `PASS`. These are attention-layer outputs with random weights; real translation comes after the later model and training videos.

To run the checks:

```bash
uv run --frozen pytest -q
```

Include `--extra notebook` to check the matrix-example helpers too:

```bash
uv run --frozen --extra notebook pytest -q
```

To open the concept notebook:

```bash
uv run --frozen --extra notebook python -m ipykernel install --sys-prefix --name python3 --display-name "Attention (Python 3.12)"
uv run --frozen --extra notebook jupyter lab attention_mechanism.ipynb
```

Use the Attention/Python kernel and run the cells from the top. Saved outputs include twenty-eight diagrams and plots, so the notebook is readable before rerunning. The matrix walkthroughs show `X × Xᵀ` and `A × X` before learned projections, then `X × W + b = Q/K/V`, `Q × Kᵀ`, scaling and softmax, `A × V`, concatenated heads and the output projection. Each multiplication highlights a row, a column and the resulting cell, with numbers and dimensions.

All drawing code, control setup and extra visualization calculations live in [attention_visuals.py](attention_visuals.py). The notebook calls functions such as `vis.show_projection("Q")` and `vis.show_value_product()` alongside the attention examples. Four sets of controls select dot-product cells, query direction/temperature, heads and masks. Run the cells in Jupyter to use those controls. The attention classes stay in their own Python files.

## Videos

| Part | Video | What I'll work on |
| --- | --- | --- |
| 1 | [Attention](https://www.youtube.com/watch?v=JXY5CmiK3LI) | Self-attention, multiple heads, padding and causal masks, cross-attention |
| 2 | [Data preparation](https://www.youtube.com/watch?v=UkBggwQsbF8) | English/French tokenizers, parallel text, shifted targets and batches |
| 3 | [Transformer and training](https://www.youtube.com/watch?v=DOqg_QxInd8) | Embeddings, encoder, decoder, training, translation and BLEU evaluation |

The [source notes](docs/SOURCES.md) have chapter timestamps and links to the paper and reference code.

## Files

The top-level files follow the video sequence. I'll keep the shared implementation in `src/attention_is_all_you_need/`.

```text
attention.py                    Attention
attention_visuals.py            Matrix walkthroughs, plots and interactive controls
tokenizer.py                    Tokenizer preparation
prepare_data.py                 Dataset preparation
train.py                        Training
inference.py                    Translation
compute_bleu.py                 BLEU evaluation
src/attention_is_all_you_need/  Attention implementation; later model/data files
configs/                        Draft local CPU and Colab Free settings
tests/                          Attention checks and known answers for matrix examples
attention_mechanism.ipynb        Basic concepts and worked examples
docs/                           Sources and Colab setup
```

Datasets, tokenizer files, weights, checkpoints and run output go in the ignored `data/`, `artifacts/`, `weights/`, `checkpoints/` and `runs/` directories.

## Local setup

From this directory:

```bash
uv sync --frozen
```

This installs the package, PyTorch, NumPy and development tools. Linux uses the locked CPU PyTorch build for these examples. The notebook tools are optional; the launcher doesn't need them. To add PyTorch and the tokenizer, dataset, training and evaluation libraries for the video build:

```bash
uv sync --frozen --extra video
```

The configs are starting points for small CPU checks and Colab Free training. I'll check the model and memory use as I implement it.

## Colab

The [Colab notes](docs/COLAB.md) cover uploading the source ZIP, installing the environment and running the Python files. I'll use notebook cells to launch the files and keep the implementation in this repo.

## Workspace launcher

Select **P00 · Translation Transformer** in the workspace TUI, then choose the file for the video you're working through.

From workspace home:

```bash
./ai run attention-is-all-you-need
./ai run attention-is-all-you-need --entrypoint inference
```

The default file is `attention.py`, which runs the video 1 examples above. The other choices are `tokenizer`, `data`, `train`, `inference` and `bleu`; those later-video files still contain notes only. The launcher settings are in `.ai-project.toml`.

## Reusing attention in the next videos

Here `src` and `tgt` are floating-point tensors shaped `(batch, sequence, embedding_dimension)`. Their masks have shape `(batch, sequence)`, with True for real tokens.

```python
from attention import Attention

attention = Attention(embedding_dimension=9, num_heads=3, attn_dropout=0.0)
encoder_output = attention(src, attention_mask=source_mask)
decoder_output = attention(tgt, attention_mask=target_mask, causal=True)
cross_output = attention(src, tgt=tgt, attention_mask=source_mask)
```

The final class, argument names and `q_proj`, `k_proj`, `v_proj`, `out_proj` names follow the video. The implementation is in [the attention module](src/attention_is_all_you_need/attention.py); `attention.py` exposes the same classes and adds the runnable demo. Inside the package, import it with `from .attention import Attention`.

For inspecting the calculation, use `Attention(..., fused_attn=False)` or request `need_weights=True` in a forward call. The default uses PyTorch SDPA. Its selected kernel depends on the device; the checked CPU run doesn't establish GPU FlashAttention performance. Source corrections and the MIT reference are recorded in [SOURCES.md](docs/SOURCES.md).
