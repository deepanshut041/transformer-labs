# Running in Colab

The attention implementation stays in Python files. I can use Colab cells to install the environment and run them, or open `attention_mechanism.ipynb` for the basic concepts. The notebook imports the same Python implementation.

The attention demo, twenty-five attention checks, ten matrix-example checks and all thirty-seven notebook code cells pass locally on CPU. I haven't run this in Colab yet. Translation training and recovery from Drive still need to be implemented and tested. Colab's [FAQ](https://research.google.com/colaboratory/faq.html) covers its notebook interface and temporary runtimes.

## Upload the source

This repo is local, so the Colab setup uses a source ZIP. From workspace home, the export is:

```text
.ai/exports/attention-is-all-you-need/attention-is-all-you-need-source.zip
.ai/exports/attention-is-all-you-need/source-manifest.json
```

The manifest records the source revision, local changes and hashes for the ZIP and its files. The ZIP contains source, configs, docs and dependency files. It leaves out captions, Git files, environments, data, credentials, weights and run output.

1. Open [a new Colab session](https://colab.research.google.com/). CPU is enough to check the current files.
2. Upload `attention-is-all-you-need-source.zip` through the Files panel. Use a fresh runtime or an empty extraction directory when changing source versions.
3. Print the ZIP hash and compare it with `archive_sha256` in the manifest before extracting:

```text
!sha256sum /content/attention-is-all-you-need-source.zip
```

4. Extract the source and install the base environment:

```text
!unzip -q -n /content/attention-is-all-you-need-source.zip -d /content
%cd /content/attention-is-all-you-need
!python -m pip install uv==0.10.10
!uv sync --frozen --no-dev --python 3.12
```

Python 3.12 matches `.python-version` and `uv.lock`. The base environment includes PyTorch and NumPy. The locked Linux build is CPU-only, so these attention examples use CPU even if Colab offers a GPU; a CUDA training environment will be selected and checked for the later training video.

5. Launch the attention file:

```text
!uv run --frozen --no-dev python attention.py
```

It prints the worked examples, attention matrices and final encoder/decoder/cross-attention tensors, then `PASS`. To use the concept notebook in Colab, open `attention_mechanism.ipynb`, upload/extract the source ZIP and run the setup commands above before its concept cells. After the `%cd`, add the installed package to Colab's current kernel with:

```python
import sys
sys.path.insert(0, '/content/attention-is-all-you-need/src')
```

Colab's own kernel may use a different Python from the locked command environment. The file-launch commands use the locked environment; notebook cells use Colab's preinstalled PyTorch, whose version must be recorded separately. As I implement the next videos, I'll also use `tokenizer.py`, `prepare_data.py`, `train.py`, `inference.py` and `compute_bleu.py`. Their command-line arguments still need to be written.

## Model dependencies

When the implementation needs the video libraries:

```text
!uv sync --frozen --no-dev --extra video --python 3.12
```

Then an implemented training script can be launched with:

```text
!uv run --frozen --no-dev --extra video python train.py
```

Keep `uv.lock` with the source version used for the run, and record the actual environment before training. The lock was resolved locally; the video extra hasn't been installed or checked for hosted compatibility or memory use in Colab.

## Training and recovery checklist

`configs/local.json` and `configs/colab-free.json` are draft settings; there's no config loader or trainer yet. I'll start with CPU correctness checks, then a small English-to-French model on one device. Dataset size, splits, license, vocabulary sizes and run arguments still need decisions.

Before increasing the model size, record RAM, GPU/VRAM if available, parameter count, step time and checkpoint size. Colab Free may provide only a CPU.

For real training, mount Drive using Colab's supported authorization flow and verify read/write access and available storage. Choose a stable run ID and a persistent directory such as `MyDrive/ai-runs/attention-is-all-you-need/<run-id>/`. Keep code and small data caches on `/content`, and recovery files in persistent storage.

The trainer and `src/attention_is_all_you_need/checkpoint.py` need to meet the [workspace recovery requirements](../../../../docs/guide/COLAB.md):

- Use explicit **new** and **resume** modes. Don't overwrite an existing run or start a new run when resume fails.
- Save model and optimizer state, scheduler, precision/scaler, update/token counts, data order and cursor, Python/NumPy/PyTorch CPU/CUDA RNG state, source/config/tokenizer/data/dependency identity and monitoring state.
- Start with a session limit of 60 minutes or 1,000 additional completed updates. Save every 5 minutes or 100 updates and at normal stop. Keep at least two verified generations; measure save time and how much progress can be lost.
- Save after completed updates. Write each generation separately, verify every file, hash and load, then write completion metadata last. Skip incomplete or corrupt saves, report fallback to an older valid save, and stop if persistence fails.
- Check deterministic local resume, interrupted-save fallback, identity rejection and CPU/device-aware loading.
- Test recovery in Colab Free: train and save, **Disconnect and delete runtime**, upload the same source, reinstall, remount storage and resume the same run. Record the restored update count and subsequent progress.

These training/recovery checks are still open. The current local evidence covers the attention component and concept notebook.

The visual notebook imports its displays from `attention_visuals.py`, included beside it in the source ZIP. That module uses Matplotlib and ipywidgets. For local Jupyter they are included in `--extra notebook`. In a Colab notebook kernel, install them if needed with `%pip install matplotlib ipywidgets`, then run from the top after source extraction. Saved figures are available even if widget support is unavailable; the controls need a live kernel and widget frontend. Hosted interaction is still untested.
