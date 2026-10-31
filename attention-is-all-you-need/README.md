# Attention Is All You Need

I'm following Priyam Mazumdar's three videos to build an English-to-French Transformer in Python, from attention through to training and translation.

Video 1 is implemented: attention without learned projections, single-head and multi-head attention, padding/causal masks, cross-attention and the reusable SDPA-backed class. Video 2's tokenizer/data preparation and Video 3's complete model/training remain pending. The tutorial explains the implementation. The notebook runs the Python project on Google Colab.

## Project structure

```text
src/
  attention.py              Runnable attention demo
  tokenizer.py              Future tokenizer preparation command
  prepare_data.py           Future dataset preparation command
  train.py                  Future training command
  inference.py              Future translation command
  compute_bleu.py           Future BLEU command
  core/                     attention.py, model.py, config.py
  data/                     tokenizer.py, dataset.py
  training/                 trainer.py, checkpoint.py
  utils/                    attention_visuals.py
notebooks/
  attention_colab.ipynb
```

The source groups are importable packages after setup. There is no additional project-named package below `src/`. The package installer exposes `core`, `data`, `training` and `utils`, plus the runnable modules directly under `src/`; the notebook and commands share those modules. Datasets, artifacts, weights, checkpoints and generated runs stay in ignored runtime directories.

## Set up and run attention

From this project directory:

```bash
uv sync --frozen
uv run --frozen attention-demo
uv run --frozen pytest -q
```

The console command runs `src/attention.py`. The direct file command is equivalent:

```bash
uv run --frozen python src/attention.py
```

The demo prints the weighted-average example, attention directly on input vectors, attention matrices and final output tensors. Encoder/decoder attention outputs have shape `(3,5,9)` and cross-attention has shape `(3,7,9)`. Explicit and SDPA paths are compared, ending with `PASS`. These randomly initialized attention-layer outputs are component checks; real translation needs the later data/model/training implementation.

The locked Python version is 3.12 and Linux uses CPU PyTorch for these examples. The draft profiles do not establish GPU fit or hosted training. Optional future video libraries are selected with `uv sync --frozen --extra video` when needed.

## Run on Google Colab

Open the [notebook](notebooks/attention_colab.ipynb) using **File → Upload notebook** in [Google Colab](https://colab.research.google.com/). Run its six code cells in order. Upload the matching source ZIP and manifest when prompted; both are under `.ai/exports/attention-is-all-you-need/` at workspace home.

The runner verifies the archive and every file, installs pinned uv and locked Python 3.12.13 dependencies in an isolated environment, records runtime versions, runs `src/attention.py`, runs all 35 component/matrix checks, and downloads a result archive. CPU is sufficient. It stops on a failed command or skipped test. [Colab guidance](docs/COLAB.md) gives the complete transfer and execution steps.

The former concept notebook has been removed; the attention tutorial owns the explanations, worked maths and code. [Graph helpers](src/utils/attention_visuals.py) remain reusable Python utilities for those figures and matrix checks. To run all 35 checks locally, including the visualization cases:

```bash
uv sync --frozen --extra notebook
uv run --frozen --extra notebook pytest -q
```

The replacement runner is checked locally with uploaded source and a simulated Colab upload/download interface. Actual Google-hosted execution is still unverified.

## Study the implemented project

The [Attention tutorial](../../../docs/tutorials/projects/attention-is-all-you-need/attention-lab/v1/lessons/01-weighted-information.md) begins with why a position needs information from others. Eleven connected steps follow one numerical example through weighted retrieval, comparisons, projections, heads, masks, cross-attention, learning and the complete project class. Code blocks include the helpers they call. Setup is a separate run reference. The rewritten text was approved on 24 October 2026. Eleven new captioned companions now follow the same route in the local notes and tutorial chapters.

The local [project documentation](docs/README.md) separates phase overviews from setup and attention implementation records. The [continuous attention notes](docs/implementation/01_attention-mechanism/02-notes-to-understanding.md) present the same authored explanations in one local view. Later lessons are added as those phases are implemented; learner understanding remains a separate review.

## Source videos

| Part | Video | Implementation scope |
| --- | --- | --- |
| 1 | [Attention](https://www.youtube.com/watch?v=JXY5CmiK3LI) | Implemented attention component; Colab execution notebook |
| 2 | [Data preparation](https://www.youtube.com/watch?v=UkBggwQsbF8) | Planned English/French tokenizers, paired data, shifted labels and batches |
| 3 | [Transformer and training](https://www.youtube.com/watch?v=DOqg_QxInd8) | Planned full encoder/decoder, training, translation, BLEU and recovery |

[Source notes](docs/SOURCES.md) retain the reference revision, MIT attribution, captions and chapter timestamps. [Colab guidance](docs/COLAB.md) explains source transfer and the pending persistent-recovery contract. Actual hosted training/recovery remains unverified.

## Workspace launcher

Select **P00 · Translation Transformer** and its file in the TUI, or run from workspace home:

```bash
./ai run attention-is-all-you-need
./ai run attention-is-all-you-need --entrypoint inference
```

The default launches `src/attention.py`. The other named commands still contain notes only. Their paths and notebook/utility resources are configured in `.ai-project.toml`; a placeholder's successful process exit is not a model result.

## Reuse the attention component

```python
from core.attention import Attention

attention = Attention(embedding_dimension=9, num_heads=3, attn_dropout=0.0)
encoder_output = attention(src, attention_mask=source_mask)
decoder_output = attention(tgt, attention_mask=target_mask, causal=True)
cross_output = attention(src, tgt=tgt, attention_mask=source_mask)
```

Inputs are floating `(batch,sequence,embedding_dimension)` tensors; masks are `(batch,source_sequence)` with True for allowed keys. Cross-attention takes queries from `tgt`, keys/values from `src`, and returns the target length. See [the core module](src/core/attention.py) for the complete public contract.

The final `Attention` signature and `q_proj`, `k_proj`, `v_proj`, `out_proj` names are retained. Use `fused_attn=False` or `need_weights=True` to inspect explicit attention. The default uses PyTorch SDPA, whose selected backend depends on device/inputs; CPU checks do not establish GPU FlashAttention performance.
