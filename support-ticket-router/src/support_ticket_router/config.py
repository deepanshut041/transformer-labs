"""Validate proposed profiles without loading data or allocating a model."""
from pathlib import Path
import json


PROFILES = ("local", "colab-free")


def load_profile(root: Path, name: str) -> dict:
    if name not in PROFILES:
        raise ValueError(f"Unknown profile: {name}")
    path = root / "configs" / f"{name}.json"
    config = json.loads(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("Profile must be a JSON object")
    required = {"schema_version", "project_id", "phase", "profile", "data_download",
                "training_enabled", "dataset", "experiment", "runtime", "paths", "recovery"}
    if set(config) != required:
        raise ValueError("Profile fields do not match schema version 1")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("Unsupported profile schema version")
    if config["project_id"] != "support-ticket-router" or config["profile"] != name:
        raise ValueError("Profile identity does not match the selected project/profile")
    if config["phase"] != "boilerplate" or config["data_download"] is not False or config["training_enabled"] is not False:
        raise ValueError("Boilerplate profiles must keep data download and training disabled")
    for field in ("dataset", "experiment", "runtime", "paths", "recovery"):
        if not isinstance(config[field], dict):
            raise ValueError(f"{field} must be an object")
    data = config["dataset"]
    if data.get("id") != "banking77" or data.get("acquired") is not False:
        raise ValueError("Dataset is a planned Banking77 input, not acquired data")
    fraction = data.get("validation_fraction")
    if type(fraction) not in (float, int) or not 0 < fraction < 1:
        raise ValueError("Validation fraction must be between zero and one")
    experiment = config["experiment"]
    if experiment.get("tokenizers") != ["word", "byte-bpe", "unigram"]:
        raise ValueError("The planned comparison must retain word, BPE and Unigram")
    if experiment.get("embedding_methods") != ["joint", "word2vec", "character-ngram"]:
        raise ValueError("The planned embedding methods must be retained")
    if experiment.get("classifier") != "masked-mean-mlp":
        raise ValueError("Initial classifier is the proposed masked-mean MLP")
    for field in ("embedding_dim", "batch_size", "max_tokens"):
        if type(experiment.get(field)) is not int or experiment[field] <= 0:
            raise ValueError(f"{field} must be a positive integer")
    if type(experiment.get("seed")) is not int or experiment["seed"] < 0:
        raise ValueError("Seed must be a nonnegative integer")
    if config["runtime"].get("device") not in ("cpu", "auto"):
        raise ValueError("Proposed device must be cpu or auto")
    if set(config["paths"]) != {"data", "runs", "checkpoints", "artifacts"}:
        raise ValueError("Declare data, runs, checkpoints and artifacts paths")
    for label, value in config["paths"].items():
        if not isinstance(value, str) or not value or Path(value).is_absolute() or not (root / value).resolve().is_relative_to(root.resolve()):
            raise ValueError(f"{label} must stay inside the project")
    if config["recovery"].get("full_state_required") is not True:
        raise ValueError("Future training must retain full-state recovery")
    expected_storage = "google-drive" if name == "colab-free" else "local"
    if config["recovery"].get("persistent_storage") != expected_storage:
        raise ValueError("Recovery storage does not match the proposed profile")
    return config
