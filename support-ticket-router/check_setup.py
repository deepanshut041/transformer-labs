"""Validate project files and proposed profiles."""
import json
from pathlib import Path

from support_ticket_router.scaffold import check


def main(project_root: Path) -> int:
    print(json.dumps(check(project_root), indent=2))
    return 0


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    raise SystemExit(main(project_root=project_root))
