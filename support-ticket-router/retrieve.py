"""Future related labelled requests with visible query/archive inputs."""
from pathlib import Path

from support_ticket_router.retrieve import related_requests
from support_ticket_router.scaffold import run_stage


def main(query: str, archive_dir: Path, top_k: int) -> int:
    return run_stage(related_requests, query, archive_dir, top_k=top_k)


if __name__ == "__main__":
    query = "My payment went through twice."
    archive_dir = Path(__file__).resolve().parent / "artifacts" / "request-archive"
    top_k = 5
    raise SystemExit(main(query=query, archive_dir=archive_dir, top_k=top_k))
