"""JSON I/O helpers for the RAG pipeline."""
import json
from pathlib import Path
from typing import Any, Union

from pydantic import BaseModel

from src.models import RagDataset


def load_json(path: Union[str, Path]) -> Any:
    """Load and parse a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        The parsed JSON content.

    Raises:
        ValueError: If the file does not exist, cannot be read, or is not
            valid JSON.
    """
    target = Path(path)
    try:
        with target.open(
            "r", encoding="utf-8", errors="replace"
        ) as handle:
            return json.load(handle)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Could not load JSON file {target}: {exc}"
        ) from exc


def load_rag_dataset(path: Union[str, Path]) -> RagDataset:
    """Load a RAG dataset from a JSON file.

    Args:
        path: Path to the dataset JSON file.

    Returns:
        The validated RagDataset.

    Raises:
        ValueError: If the file is missing, malformed, or does not validate
            against RagDataset.
    """
    target = Path(path)
    data = load_json(target)
    try:
        return RagDataset.model_validate(data)
    except ValueError as exc:
        raise ValueError(
            f"Invalid RAG dataset in {target}: {exc}"
        ) from exc


def save_model(model: BaseModel, path: Union[str, Path]) -> None:
    """Serialize a pydantic model to a JSON file.

    Creates parent directories if needed. Writes compact-but-readable JSON
    (indent=2) with UTF-8 encoding.

    Args:
        model: The model instance to persist.
        path: Destination JSON path.

    Raises:
        ValueError: If the destination cannot be written.
    """
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            handle.write(model.model_dump_json(indent=2))
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Could not save model to {target}: {exc}"
        ) from exc


def to_posix_path(path: Union[str, Path]) -> str:
    """Normalize a filesystem path to POSIX separators.

    Args:
        path: Path to normalize.

    Returns:
        The path as a POSIX-style string (forward slashes).
    """
    return str(path).replace("\\", "/")
