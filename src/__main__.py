"""Run the RAG CLI with ``python -m src``."""
import fire

from src.cli import Rag


def main() -> None:
    """Dispatch command-line arguments with Python Fire."""
    fire.Fire(Rag)


if __name__ == "__main__":
    main()
