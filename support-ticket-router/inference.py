"""Future request routing. The text/model inputs are explicit below."""
from pathlib import Path

from support_ticket_router.app import route_request
from support_ticket_router.scaffold import run_stage


def main(text: str, model_dir: Path, top_k: int) -> int:
    return run_stage(route_request, text, model_dir, top_k=top_k)


if __name__ == "__main__":
    text = "My payment went through twice."
    model_dir = Path(__file__).resolve().parent / "artifacts" / "best"
    top_k = 3
    raise SystemExit(main(text=text, model_dir=model_dir, top_k=top_k))
