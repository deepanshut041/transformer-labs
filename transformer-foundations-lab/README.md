# Tokenization and Transformer Foundations

P00-M1 in-progress. Lesson 2 setup complete. Continue with [Lesson 3 — numbers](../../../docs/tutorials/projects/p00/transformer-foundations/v1/lessons/03-numbers.md), starting with `linear()`.

This ordinary project folder in Transformer Labs owns P00 implementation and is tracked by ai_projects. The package, Python environment and saved arithmetic exercise are ready. Tensor/autograd/MLP, tokenizer, attention, complete model, training and Colab recovery remain to build and verify.

## Run the saved program

From this project folder:

```bash
uv sync --frozen
uv run --frozen python first_prediction.py
```

Expected output:

```text
prediction: 6
target: 1
error: 5
```

## Continue work

- [Implementation checkpoint](docs/implementation/00_p00-foundations-implementation-and-learning/00-resume.md) and [verification](docs/implementation/00_p00-foundations-implementation-and-learning/06-review.md).
- [Owning roadmap](../../../docs/guide/roadmap.md), [progress](../../../docs/guide/progress.md), [resume](../../../docs/guide/resume.md) and [decisions](../../../docs/guide/decisions.md).
- [Colab Free training and persistent recovery requirements](../../../docs/guide/COLAB.md).

`src/transformer_foundations_lab/` holds future reusable functions; `tests/` holds independent checks. `demo.py` is the scaffold caller until Lesson 3. Meaningful pytest tests are pending. Launcher entry `p00` runs the teaching reference, not these files.

Later use the same local trainer in a thin Colab notebook. Keep data, weights, checkpoints, credentials, environments and generated runs ignored. Local checks, learner explanation, training quality, hosted recovery and publication are separate gates.
