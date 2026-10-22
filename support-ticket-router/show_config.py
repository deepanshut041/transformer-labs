"""Display a proposed profile using the config_path input below."""
import json
from pathlib import Path

from support_ticket_router.scaffold import load_configuration


def main(config_path: Path) -> int:
    print(json.dumps(load_configuration(config_path), indent=2))
    return 0


if __name__ == "__main__":
    config_path = Path(__file__).resolve().parent / "configs" / "local.json"
    raise SystemExit(main(config_path=config_path))
