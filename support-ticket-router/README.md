# Support Ticket Router

I want to build an application that reads a support request, suggests its category and shows similar requests. I'll train the models on Colab and compare how tokenization and embeddings affect the application's predictions.

**Status:** standalone boilerplate prepared on 2026-10-13 — locked Python environment, importable package, runnable scripts with explicit inputs, proposed profiles and a Colab notebook template. Dataset acquisition, tokenizers, embeddings, classifier, training and the routing application remain unimplemented.

**Core goal:** Train embeddings and an intent classifier; compare word tokenization, BPE and subwords.

I'll train the embeddings and classifier myself on Colab. The comparison will cover word-level tokens, byte-level BPE and another subword tokenizer such as SentencePiece Unigram. Character n-gram embeddings will be a separate representation comparison. The support router is the application I'll use to measure their effect.

Start with the [project overview and phases](phase.md) for an easy-language explanation of the application, request-to-prediction flow, tokenizers, embeddings, training and all ten phases. Detailed requirements, inputs, outputs, whiteboard chunks and completion checks live in the locally ignored [plan/](plan/00-resume.md) folder. We establish requirements before each explanation and implementation; the assistant writes the code/tests, then we review them together. A phase can be covered in one piece or several chunks according to its scope.

## Run directly

Inside this project:

```bash
uv sync --frozen
uv run --frozen python demo.py
uv run --frozen python check_setup.py
uv run --frozen python show_config.py
uv run --frozen python colab_config.py
uv run --frozen pytest -q
```

Every runnable file declares its inputs inside its `if __name__ == "__main__":` block. I can read and edit those values, then run the file directly. Importable functions receive the same inputs explicitly.

For example, `train_embeddings.py` ends with:

```python
if __name__ == "__main__":
    config_path = Path(__file__).resolve().parent / "configs" / "local.json"
    method = "word2vec"
    raise SystemExit(main(config_path=config_path, method=method))
```

| File | Inputs in its main block | Current behavior |
| --- | --- | --- |
| demo.py | Configuration and sample request | Run the boilerplate and show its readiness; no prediction |
| check_setup.py | Project root | Validate project files and proposed settings |
| show_config.py / colab_config.py | Configuration path | Show proposed CPU or Colab settings |
| prepare_data.py | Configuration and dataset revision | Explicit unimplemented-stage error |
| fit_tokenizers.py | Configuration, method and vocabulary size | Explicit unimplemented-stage error |
| train_embeddings.py | Configuration and embedding method | Explicit unimplemented-stage error |
| train_classifier.py | Configuration, tokenizer and embedding method | Explicit unimplemented-stage error |
| train.py | Configuration and resume checkpoint | Explicit unimplemented-stage error |
| compare.py | Configuration and tokenizer choices | Explicit unimplemented-stage error |
| inference.py | Request, model directory and top-k | Explicit unimplemented-stage error |
| retrieve.py | Query, archive directory and top-k | Explicit unimplemented-stage error |

The future-stage scripts exit with code 2 and a clear message. They don't download data, fit tokenizers, train or produce fabricated results. The demo and setup/profile scripts run successfully without starting those stages.

## What I'll build

The first version will let me enter a request such as “My payment went through twice” and see:

- The three highest-scoring categories, with their model scores.
- Similar labelled requests from the training archive.
- A way to record a corrected category for later review.
- A comparison view for the trained tokenizers and embedding models.

The first task is **intent classification**: choose the kind of help the customer needs. A category can eventually map to a support queue, but those mappings need to be defined for the application. Low-scoring or ambiguous requests should be flagged for review using a threshold chosen on validation data. Model scores won't be described as calibrated confidence without checking calibration.

This is an application project. Tokenizer and embedding comparisons are experiments inside it, with accuracy, mistakes and runtime measured on the same task.

## First dataset

I'll start with [PolyAI's Banking77 dataset](https://github.com/PolyAI-LDN/task-specific-datasets#banking), which contains 10,003 training requests, 3,080 test requests and 77 banking intents. The repository provides the dataset and its CC BY 4.0 license. I'll pin the source revision and record file hashes, attribution and acquisition instructions when I start.

Banking77 contains requests and intent labels, not ticket resolutions. The first similarity view will retrieve **related labelled examples**. Showing resolved cases or suggested solutions can follow later with a suitable archive and relevance labels.

I'll create a seeded, stratified validation split from the official training set and reserve the official test set for the final comparison. Duplicate or near-duplicate requests will be inspected before freezing the split. Vocabulary fitting, tokenizer training and embedding training will use training data only. The retrieval archive will also use training examples; held-out requests will be queries.

Downloaded data and generated artifacts will stay out of Git. Public dataset acquisition comes first; adapting the router to another support domain is a later step.

## Models and comparisons

I'll begin with a small CPU baseline, then train the learned models from scratch.

| Stage | What I'll compare | What it tells me |
| --- | --- | --- |
| Baseline | TF-IDF with a linear classifier | How far a simple lexical model gets |
| Tokenizers | Word tokens, byte-level BPE and SentencePiece Unigram | Effects of vocabulary size, sequence length, rare words and spelling changes |
| Learned representations | Randomly initialized embeddings trained with the classifier, embeddings I train with Word2Vec and fastText-style character n-gram embeddings I train | Whether context or subword training helps this application |
| Similar requests | Lexical similarity and pooled learned embeddings | Which approach retrieves useful related requests |

The first neural classifier will pool token embeddings and feed them through a small classification head. Padding must be excluded from pooling. I'll train it with cross-entropy over the intent labels and check that it can overfit a tiny fixed batch before a longer run.

Tokenizer comparisons will use the same classifier design, data splits and recorded training budget. I'll report actual vocabulary sizes, parameter counts, truncation rates and sequence lengths so changes in model size or input coverage are visible. Word2Vec versus fastText will be a separate comparison with a consistent word-level input policy. A tiny Transformer encoder can be added later once the simpler application works.

The small request corpus may be insufficient for strong Word2Vec embeddings. Any extra unlabelled training corpus will need its own provenance and a clearly recorded experiment; held-out requests won't become pretraining data.

## How I'll evaluate it

- **Accuracy and macro-F1:** overall routing quality and performance across all intents.
- **Per-intent results and a confusion matrix:** categories that are repeatedly mixed up.
- **Top-three accuracy:** whether the correct category appears among the suggestions.
- **Robustness examples:** fixed spelling changes, rare terms and unsupported requests, kept separate from the ordinary test set.
- **Application cost:** training time, model size, memory use and inference latency on recorded hardware.

Similarity evaluation will start with examples I inspect. A matching intent is a useful proxy, but doesn't establish that two tickets have the same problem or solution. Judged relevant-request pairs will be needed before claiming retrieval quality.

I'll freeze settings using validation results, evaluate the test set after selection and keep concrete successes and failures. A correction saved through the demo won't silently retrain the model or become a test label.

## Colab training and recovery

I'll target a bounded **Colab Free** profile. A tiny local CPU run comes first; hosted memory use, step time and checkpoint cost will determine the larger training budget. Colab fit has not been verified yet.

Python/PyTorch implementation files will own preparation, training, evaluation and inference. A thin, output-cleared Colab notebook will call that same code using a portable source package.

I'll follow the [workspace Colab contract](../../../docs/guide/COLAB.md):

- Stage reproducible data caches on runtime-local disk and save recovery artifacts in Drive.
- Save model, optimizer, scheduler, any gradient scaler, completed progress, RNG and data-order state, plus source/config/data/tokenizer/environment identities.
- Verify immutable checkpoint generations and retain an older valid fallback.
- Keep the best validation model separate from the latest resumable checkpoint.
- Delete the runtime, restore the same run in a fresh one and prove that training continues from its saved update.

Tokenizer artifacts need deterministic save/load checks. Hosted training and recovery are future acceptance checks, not results established by this README.

## Scaffold layout

The source files now exist. Configuration validation and the boilerplate demo work; the learning and application functions remain explicit placeholders.

```text
pyproject.toml + uv.lock     Locked Python environment and development tools
demo.py + check_setup.py    Direct boilerplate run and setup validation
show_config.py             Proposed CPU configuration
colab_config.py            Proposed Colab configuration
prepare_data.py            Future dataset acquisition and splitting
fit_tokenizers.py          Future word/BPE/Unigram fitting
train_embeddings.py        Future embedding training
train_classifier.py        Future intent-classifier training
train.py                   Future common training/resume
compare.py                 Future tokenizer/model comparison
inference.py + retrieve.py  Future request routing and related examples
src/support_ticket_router/ Importable project functions and placeholders
configs/                   Proposed CPU and Colab Free settings
notebooks/colab.ipynb       Output-cleared workflow template
tests/                     Configuration, standalone and deferred-stage checks
docs/                      Project workflow and Colab requirements
```

This is an ordinary folder tracked by `ai_projects` inside Transformer Labs. Datasets, trained tokenizers, weights, environments, checkpoints and generated runs remain outside Git.

## First implementation phase

Start with [Phase 01 — labelled data and stable intent IDs](plan/phase-01-data-contract.md). Its requirements are prepared; implementation has not started.

1. Explain the pinned source, CSV schema, input/output requirements and failure checks.
2. Implement and review Chunk A: acquire, load and validate raw training records.
3. Explain how intent names become fixed classifier output IDs.
4. Implement and review Chunk B: save/reload the label map and inspect a small example preview.

This phase ends with validated input/target records and reusable source/label identities. Splits and the lexical baseline follow in Phase 2; tokenization, embeddings and model training follow in their own phases.

The project is complete when I've trained embeddings and an intent classifier, compared word tokenization, BPE and another subword method on the same task, and built an application that routes fresh requests, shows related examples and records corrections. I'll retain the held-out comparison and proof that an actual Colab training run resumes after runtime loss.

Implementation references for later: [fastText classification guide](https://fasttext.cc/docs/en/supervised-tutorial.html), [SentencePiece](https://github.com/google/sentencepiece) and the [workspace Colab contract](../../../docs/guide/COLAB.md).
