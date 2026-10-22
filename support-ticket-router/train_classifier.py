"""Future intent-classifier training with explicit tokenizer/embedding inputs."""
from pathlib import Path

from support_ticket_router.scaffold import load_configuration, run_stage
from support_ticket_router.train import train_classifier


def main(config_path: Path, tokenizer_method: str, embedding_method: str) -> int:
    config = load_configuration(config_path)
    return run_stage(train_classifier, config, tokenizer_method=tokenizer_method, embedding_method=embedding_method)


if __name__ == "__main__":
    config_path = Path(__file__).resolve().parent / "configs" / "local.json"
    tokenizer_method = "word"
    embedding_method = "joint"
    raise SystemExit(main(config_path=config_path, tokenizer_method=tokenizer_method, embedding_method=embedding_method))
