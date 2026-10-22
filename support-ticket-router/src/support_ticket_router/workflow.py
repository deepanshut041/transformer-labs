"""Descriptions of future work, not executable training steps."""

from dataclasses import dataclass


class StageNotImplemented(NotImplementedError):
    """The user has authorized boilerplate, not this implementation stage."""


@dataclass(frozen=True)
class Stage:
    id: str
    label: str
    module: str
    next_action: str
    evidence: str


STAGES = (
    Stage("data", "Dataset and splits", "data.py",
          "Pin Banking77, inspect labels and freeze leakage-checked splits.",
          "Source/license and split manifest; no held-out fitting."),
    Stage("tokenizers", "Word / BPE / subwords", "tokenizers.py",
          "Implement word tokenization, byte-level BPE and Unigram with training-only fitting.",
          "Save/load checks, vocabulary sizes, sequence lengths and truncation rates."),
    Stage("embeddings", "Embedding training", "embeddings.py",
          "Train joint embeddings, Word2Vec and character n-gram representations.",
          "Independent numerical checks and actual training artifacts."),
    Stage("classifier", "Intent classifier", "model.py",
          "Build masked pooling and a classification head, then verify a tiny overfit.",
          "Forward/loss/gradient checks and trained intent predictions."),
    Stage("training", "Colab training and resume", "train.py",
          "Use one trainer locally and on Colab with full-state persistent recovery.",
          "Measured budget and restored/advanced updates after runtime deletion."),
    Stage("comparison", "Held-out comparison", "evaluate.py",
          "Compare word tokens, BPE and Unigram under recorded training budgets.",
          "Accuracy, macro-F1, top-three accuracy, errors and inference cost."),
    Stage("application", "Routing and related requests", "app.py",
          "Show category suggestions, related labelled examples and corrections.",
          "A usable application backed by actual trained artifacts."),
)


def stage_named(name: str) -> Stage:
    for stage in STAGES:
        if stage.id == name:
            return stage
    raise ValueError(f"Unknown workflow stage: {name}")


def pending(name: str) -> None:
    stage = stage_named(name)
    raise StageNotImplemented(
        f"{stage.label} is not implemented. This project contains boilerplate only. "
        f"Next action when project work starts: {stage.next_action}"
    )
