"""Run from the project directory; the sample inputs are declared below."""
from pathlib import Path

from .demo import run


if __name__ == "__main__":
    config_path = Path.cwd() / "configs" / "local.json"
    sample_request = "My payment went through twice."
    raise SystemExit(run(config_path=config_path, sample_request=sample_request))
