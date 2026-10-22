"""Verify proposed settings, direct source execution and deferred stages."""
import ast
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from support_ticket_router.config import load_profile
from support_ticket_router.demo import run
from support_ticket_router.scaffold import check, load_configuration
from support_ticket_router.workflow import StageNotImplemented

ROOT = Path(__file__).resolve().parents[1]
SUCCESSFUL_FILES = ("demo.py", "check_setup.py", "show_config.py", "colab_config.py")
PENDING_FILES = ("prepare_data.py", "fit_tokenizers.py", "train_embeddings.py", "train_classifier.py",
                 "train.py", "compare.py", "inference.py", "retrieve.py")


@pytest.fixture
def copy_project(tmp_path):
    root = tmp_path / "project"
    shutil.copytree(ROOT, root, ignore=shutil.ignore_patterns(
        ".venv", "__pycache__", ".pytest_cache", "dist", "data", "runs", "artifacts", "checkpoints"))
    return root


def test_readiness_check_does_not_create_work_artifacts(copy_project):
    before = {p.relative_to(copy_project) for p in copy_project.rglob("*")}
    report = check(copy_project)
    assert report["phase"] == "boilerplate"
    assert report["training_started"] is False and report["data_acquired"] is False
    assert all(stage["status"] == "not-implemented" for stage in report["stages"])
    assert {p.relative_to(copy_project) for p in copy_project.rglob("*")} == before


@pytest.mark.parametrize("name", ["local", "colab-free"])
def test_proposed_profiles_keep_execution_disabled(name):
    config = load_configuration(ROOT / "configs" / f"{name}.json")
    assert config["profile"] == name
    assert config["data_download"] is False and config["training_enabled"] is False


@pytest.mark.parametrize("field,value", [("training_enabled", True), ("data_download", True),
                                        ("schema_version", 2), ("phase", "training")])
def test_config_rejects_accidental_execution_flags_and_unknown_versions(field, value, copy_project):
    path = copy_project / "configs/local.json"
    config = json.loads(path.read_text())
    config[field] = value
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_profile(copy_project, "local")


@pytest.mark.parametrize("value", ["../outside", "/tmp/outside"])
def test_config_rejects_paths_outside_the_project(value, copy_project):
    path = copy_project / "configs/local.json"
    config = json.loads(path.read_text())
    config["paths"]["runs"] = value
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="inside the project"):
        load_profile(copy_project, "local")


@pytest.mark.parametrize("filename", SUCCESSFUL_FILES + PENDING_FILES)
def test_direct_source_files_work_from_an_unrelated_directory(filename, tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / filename)], cwd=tmp_path,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == (0 if filename in SUCCESSFUL_FILES else 2), result.stdout + result.stderr
    assert "Traceback" not in result.stderr
    if filename in PENDING_FILES:
        assert "not implemented" in result.stdout
    assert not any((ROOT / name).exists() for name in ("data", "runs", "artifacts", "checkpoints", "weights"))


def test_runnable_files_declare_inputs_in_their_main_blocks():
    for filename in SUCCESSFUL_FILES + PENDING_FILES:
        tree = ast.parse((ROOT / filename).read_text())
        blocks = [node for node in tree.body if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "__name__ == '__main__'"]
        assert len(blocks) == 1, filename
        assert any(isinstance(node, ast.Assign) for node in blocks[0].body), filename
        assert any(isinstance(node, ast.Raise) for node in blocks[0].body), filename


@pytest.mark.parametrize("module,function,args", [
    ("data", "prepare_data", ({},)),
    ("tokenizers", "fit_tokenizers", ({},)),
    ("embeddings", "train_embeddings", ({},)),
    ("model", "build_classifier", ({},)),
    ("train", "train", ({},)),
    ("train", "train_classifier", ({},)),
    ("checkpoint", "save_checkpoint", ({}, "unused")),
    ("checkpoint", "resume_checkpoint", ("unused",)),
    ("retrieve", "related_requests", ("sample", {})),
    ("evaluate", "compare_models", ({},)),
    ("app", "route_request", ("sample", {})),
])
def test_future_functions_do_not_return_fabricated_results(module, function, args):
    entry = getattr(importlib.import_module(f"support_ticket_router.{module}"), function)
    with pytest.raises(StageNotImplemented, match="not implemented"):
        entry(*args)


def test_demo_accepts_explicit_inputs_and_reports_no_trained_prediction(capsys):
    assert run(ROOT / "configs/local.json", "A custom request") == 0
    output = capsys.readouterr().out
    assert "A custom request" in output and "Prediction: unavailable" in output


def test_missing_source_file_is_rejected(copy_project):
    (copy_project / "demo.py").unlink()
    with pytest.raises(ValueError, match="Missing scaffold files"):
        check(copy_project)


def test_configuration_path_must_select_the_project_configs_folder(tmp_path):
    with pytest.raises(ValueError, match="configs folder"):
        load_configuration(tmp_path / "local.json")


def test_notebook_template_reports_scope_without_model_or_hosted_work(capsys):
    notebook = json.loads((ROOT / "notebooks/colab.ipynb").read_text())
    namespace = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            exec(compile("".join(cell["source"]), "colab-template", "exec"), namespace)
    assert "no data download, training, predictions or Drive access" in capsys.readouterr().out
