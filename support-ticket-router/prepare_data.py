"""Future dataset preparation. Inputs are explicit; implementation is deferred."""
from pathlib import Path

from support_ticket_router.data import prepare_data
from support_ticket_router.scaffold import load_configuration, run_stage


def main(config_path: Path, source_revision: str | None) -> int:
    config = load_configuration(config_path)
    return run_stage(prepare_data, config, source_revision=source_revision)


if __name__ == "__main__":
    config_path = Path(__file__).resolve().parent / "configs" / "local.json"
    source_revision = None  # Pin a dataset revision when implementation starts.
    raise SystemExit(main(config_path=config_path, source_revision=source_revision))
