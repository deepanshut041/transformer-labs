# Running in Google Colab

[attention_colab.ipynb](../notebooks/attention_colab.ipynb) is the project runner. It transfers the current source into Colab and executes the Python files. The Attention Lab tutorial in Projects Docs contains the worked explanations and original figures; the old concept notebook has been removed.

## Open the notebook and source export

The current local export at workspace home contains:

```text
.ai/exports/attention-is-all-you-need/
  attention_colab.ipynb
  attention-is-all-you-need-source.zip
  source-manifest.json
```

The notebook is also in the project's `notebooks/` folder and available from the project reader. The ZIP includes current source, tests, configs, dependency files and public reference/Colab docs. It excludes captions, Git history, environments, private study records, data, weights and run output. The matching manifest identifies the source revision, uncommitted changes and hashes of the archive and every file. Use the ZIP and manifest from the same export.

1. Open [Google Colab](https://colab.research.google.com/) and choose **File → Upload notebook**. Upload `attention_colab.ipynb`.
2. Connect to a CPU runtime. Run the notebook from its first cell.
3. When the upload picker appears, select **both** `attention-is-all-you-need-source.zip` and `source-manifest.json` together.
4. Wait for dependency installation, then run the environment, demo and test cells in order.
5. Run the final cell and save `attention-colab-results.zip` before ending the runtime.

Colab supports uploading notebooks and runs code in temporary hosted machines; see its [official FAQ](https://research.google.com/colaboratory/faq.html). There is no public repository URL for this local working tree, so the notebook uses the reviewed source export.

## What the six code cells do

| Cell | Operation | Expected result |
| --- | --- | --- |
| Upload and verify | Match ZIP SHA-256, all file hashes and safe member paths before extraction | Fresh source directory under `/content/attention-colab-…/` |
| Install | Isolated uv 0.10.10 tools environment; `uv sync --frozen --extra notebook --python 3.12.13` | Project environment built from the supplied lock |
| Runtime | Query the installed project's Python, PyTorch, NumPy and Matplotlib | Actual environment versions recorded |
| Demo | Execute `src/attention.py` | Component tensors and `PASS` |
| Checks | Execute pytest with a machine-readable report | All 35 current tests pass; failures/skips stop the run |
| Download | Save source identity, environment, demo/test logs and receipt | `attention-colab-results.zip` |

The `notebook` dependency extra is retained for the visualization/math checks; it includes Matplotlib, ipywidgets and local Jupyter tools. The Colab kernel only coordinates subprocesses. It never imports the project's Python modules, so its preinstalled PyTorch does not substitute for the locked environment. uv's [locking/sync documentation](https://docs.astral.sh/uv/concepts/projects/sync/) explains frozen synchronization.

The Linux lock uses CPU PyTorch. Selecting a GPU does not change it. The expected demo shapes are `(3,5,9)` for encoder/decoder and `(3,7,9)` for cross-attention. These are initialized component outputs; they are not translated text or a task score.

## Reruns and failures

Run cells in order. If the source hashes disagree, obtain a matching ZIP/manifest pair and repeat the first cell. Each upload creates a new directory, so a changed source version is not mixed with a prior extraction. If installation fails, read the command's printed output and rerun that cell after correcting the runtime/network issue. A failed or skipped test stops before the success receipt is written.

After **Disconnect and delete runtime**, reconnect and rerun from the top, uploading the pair again. No Drive mount is required for this component check; download the logs before deleting a runtime. Local execution of these cells with a simulated upload/download interface is verified. I have not executed this notebook on a Google-hosted runtime, and no training/recovery result is claimed.

## Model dependencies

When the implementation needs the video libraries:

```text
!uv sync --frozen --no-dev --extra video --python 3.12
```

Then an implemented training script can be launched with:

```text
!uv run --frozen --no-dev --extra video python src/train.py
```

Keep `uv.lock` with the source version used for the run, and record the actual environment before training. The lock was resolved locally; the video extra hasn't been installed or checked for hosted compatibility or memory use in Colab.

## Training and recovery checklist

`configs/local.json` and `configs/colab-free.json` are draft settings; there's no config loader or trainer yet. I'll start with CPU correctness checks, then a small English-to-French model on one device. Dataset size, splits, license, vocabulary sizes and run arguments still need decisions.

Before increasing the model size, record RAM, GPU/VRAM if available, parameter count, step time and checkpoint size. Colab Free may provide only a CPU.

For real training, mount Drive using Colab's supported authorization flow and verify read/write access and available storage. Choose a stable run ID and a persistent directory such as `MyDrive/ai-runs/attention-is-all-you-need/<run-id>/`. Keep code and small data caches on `/content`, and recovery files in persistent storage.

The trainer and `src/training/checkpoint.py` need to meet the [workspace recovery requirements](../../../../docs/guide/COLAB.md):

- Use explicit **new** and **resume** modes. Don't overwrite an existing run or start a new run when resume fails.
- Save model and optimizer state, scheduler, precision/scaler, update/token counts, data order and cursor, Python/NumPy/PyTorch CPU/CUDA RNG state, source/config/tokenizer/data/dependency identity and monitoring state.
- Start with a session limit of 60 minutes or 1,000 additional completed updates. Save every 5 minutes or 100 updates and at normal stop. Keep at least two verified generations; measure save time and how much progress can be lost.
- Save after completed updates. Write each generation separately, verify every file, hash and load, then write completion metadata last. Skip incomplete or corrupt saves, report fallback to an older valid save, and stop if persistence fails.
- Check deterministic local resume, interrupted-save fallback, identity rejection and CPU/device-aware loading.
- Test recovery in Colab Free: train and save, **Disconnect and delete runtime**, upload the same source, reinstall, remount storage and resume the same run. Record the restored update count and subsequent progress.

These training/recovery checks are still open. The current local evidence covers the attention component and the Colab runner executed with a simulated file interface. Actual hosted execution remains unverified.
