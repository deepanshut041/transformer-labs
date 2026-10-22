"""A standalone boilerplate run; it never returns a trained prediction."""
from pathlib import Path

from . import CORE_GOAL
from .scaffold import check, load_configuration


def run(config_path: Path, sample_request: str) -> int:
    config_path = Path(config_path).resolve()
    config = load_configuration(config_path)
    report = check(config_path.parent.parent)
    print("Support Ticket Router | boilerplate run")
    print(CORE_GOAL)
    print(f"Profile: {config['profile']} | Tokenizers: {', '.join(config['experiment']['tokenizers'])}")
    print(f"Example input: {sample_request}")
    print(f"Scaffold ready: {report['checked_files']} files; both profiles validated")
    print("Prediction: unavailable; embeddings and intent classifier are not implemented.")
    print("Data preparation, training and Colab recovery have not started.")
    return 0
