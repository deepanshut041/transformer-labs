"""Read-only inspection of the scaffold and its future work."""
from pathlib import Path
import json

from . import CORE_GOAL
from .config import PROFILES, load_profile
from .workflow import STAGES, StageNotImplemented


def check(root: Path) -> dict:
    required = ["README.md", "pyproject.toml", "uv.lock", "demo.py", "check_setup.py",
                "show_config.py", "colab_config.py", "prepare_data.py", "fit_tokenizers.py",
                "train_embeddings.py", "train_classifier.py", "train.py", "compare.py",
                "inference.py", "retrieve.py",
                ".python-version", ".gitignore", "docs/WORKFLOW.md", "docs/COLAB.md",
                "notebooks/colab.ipynb", "tests/test_scaffold.py"]
    modules = ["__init__.py", "__main__.py", "demo.py", "config.py", "scaffold.py", "workflow.py",
               "data.py", "tokenizers.py", "embeddings.py", "model.py", "train.py",
               "checkpoint.py", "retrieve.py", "evaluate.py", "app.py"]
    required += [f"src/support_ticket_router/{name}" for name in modules]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ValueError(f"Missing scaffold files: {', '.join(missing)}")
    if (root / ".git").exists():
        raise ValueError("This project must remain an ordinary ai_projects folder")
    profiles = {name: load_profile(root, name) for name in PROFILES}
    notebook = json.loads((root / "notebooks/colab.ipynb").read_text())
    if notebook.get("nbformat") != 4:
        raise ValueError("Colab template must be a version-4 notebook")
    for cell in notebook.get("cells", []):
        if cell["cell_type"] == "code" and (cell.get("outputs") or cell.get("execution_count") is not None):
            raise ValueError("Keep the notebook template output-cleared")
    return {"project": "Support Ticket Router", "phase": "boilerplate", "core_goal": CORE_GOAL,
            "checked_files": len(required), "profiles": list(profiles),
            "stages": [{"id": stage.id, "status": "not-implemented"} for stage in STAGES],
            "data_acquired": False, "training_started": False, "colab_verified": False}


def load_configuration(config_path: Path) -> dict:
    """Read a selected project profile without preparing data or a model."""
    path = Path(config_path).resolve()
    root = path.parent.parent
    if path != root / "configs" / f"{path.stem}.json":
        raise ValueError("Select a JSON profile from the project's configs folder")
    return load_profile(root, path.stem)


def run_stage(action, *args, **inputs) -> int:
    """Report a deferred stage without pretending to produce model results."""
    try:
        action(*args, **inputs)
    except StageNotImplemented as exc:
        print(f"Support Ticket Router: {exc}")
        return 2
    return 0
