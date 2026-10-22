"""Run the project boilerplate with the explicit example inputs below."""
from pathlib import Path

from support_ticket_router.demo import run


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    config_path = project_root / "configs" / "local.json"
    sample_request = "My payment went through twice."
    raise SystemExit(run(config_path=config_path, sample_request=sample_request))
