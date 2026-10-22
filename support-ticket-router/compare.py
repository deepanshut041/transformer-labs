"""Future held-out comparison with its tokenizer choices visible below."""
from pathlib import Path

from support_ticket_router.evaluate import compare_models
from support_ticket_router.scaffold import load_configuration, run_stage


def main(config_path: Path, tokenizer_methods: tuple[str, ...]) -> int:
    config = load_configuration(config_path)
    return run_stage(compare_models, config, tokenizer_methods=tokenizer_methods)


if __name__ == "__main__":
    config_path = Path(__file__).resolve().parent / "configs" / "local.json"
    tokenizer_methods = ("word", "byte-bpe", "unigram")
    raise SystemExit(main(config_path=config_path, tokenizer_methods=tokenizer_methods))
