"""Future word/BPE/Unigram fitting. Edit the inputs below before running."""
from pathlib import Path

from support_ticket_router.scaffold import load_configuration, run_stage
from support_ticket_router.tokenizers import fit_tokenizers


def main(config_path: Path, method: str, vocabulary_size: int) -> int:
    config = load_configuration(config_path)
    return run_stage(fit_tokenizers, config, method=method, vocabulary_size=vocabulary_size)


if __name__ == "__main__":
    config_path = Path(__file__).resolve().parent / "configs" / "local.json"
    method = "word"  # Later choices: word, byte-bpe, unigram.
    vocabulary_size = 8192
    raise SystemExit(main(config_path=config_path, method=method, vocabulary_size=vocabulary_size))
