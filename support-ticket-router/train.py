"""Future common training/resume entry; implementation is deferred."""
from pathlib import Path

from support_ticket_router.scaffold import load_configuration, run_stage
from support_ticket_router.train import train


def main(config_path: Path, resume_from: Path | None) -> int:
    config = load_configuration(config_path)
    return run_stage(train, config, resume_from=resume_from)


if __name__ == "__main__":
    config_path = Path(__file__).resolve().parent / "configs" / "local.json"
    resume_from = None  # Later: a verified full-state checkpoint path.
    raise SystemExit(main(config_path=config_path, resume_from=resume_from))
