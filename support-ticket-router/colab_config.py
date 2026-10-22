"""Display the proposed Colab profile without starting a hosted job."""
from pathlib import Path

from show_config import main


if __name__ == "__main__":
    config_path = Path(__file__).resolve().parent / "configs" / "colab-free.json"
    raise SystemExit(main(config_path=config_path))
