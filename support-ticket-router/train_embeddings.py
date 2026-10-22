"""Future embedding training with visible inputs; no training is implemented."""
from pathlib import Path

from support_ticket_router.embeddings import train_embeddings
from support_ticket_router.scaffold import load_configuration, run_stage


def main(config_path: Path, method: str) -> int:
    config = load_configuration(config_path)
    return run_stage(train_embeddings, config, method=method)


if __name__ == "__main__":
    config_path = Path(__file__).resolve().parent / "configs" / "local.json"
    method = "word2vec"  # Later choices: joint, word2vec, character-ngram.
    raise SystemExit(main(config_path=config_path, method=method))
