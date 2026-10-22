"""Persistence for the lexical index: NPZ arrays plus JSON metadata."""
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from src.config import IndexConfig
from src.indexing.bm25 import _BUNDLE_KEYS, BM25Index
from src.io_utils import load_json, save_model
from src.models import IndexedChunk

INDEX_FORMAT_VERSION: int = 1
INDEX_ARRAYS_FILENAME: str = "index.npz"
INDEX_META_FILENAME: str = "index_meta.json"


class IndexMeta(BaseModel):
    """JSON metadata persisted next to the index arrays."""

    model_config = ConfigDict(extra="ignore")

    format_version: int
    config: IndexConfig
    chunks: List[IndexedChunk]
    avgdl: Dict[str, float]


def save_index(
    index: BM25Index,
    chunks: Sequence[IndexedChunk],
    config: IndexConfig,
    directory: Union[str, Path],
) -> None:
    """Write ``index.npz`` and ``index_meta.json`` under ``directory``.

    Args:
        index: A built index.
        chunks: One record per doc id, in doc id order.
        config: The configuration the index was built with.
        directory: Destination; created with parents if needed.

    Raises:
        ValueError: If the index is not built, ``len(chunks)`` differs
            from ``index.doc_count``, or the destination cannot be
            written.
    """
    if not index.is_built:
        raise ValueError("cannot save an index that has not been built")
    if len(chunks) != index.doc_count:
        raise ValueError(
            f"received {len(chunks)} chunk records for "
            f"{index.doc_count} documents"
        )
    target = Path(directory)
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(
            f"Could not create index directory {target}: {exc}"
        ) from exc
    arrays_path = target / INDEX_ARRAYS_FILENAME
    try:
        np.savez_compressed(
            arrays_path, allow_pickle=False, **index.to_bundle()
        )
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Could not save index arrays to {arrays_path}: {exc}"
        ) from exc
    meta = IndexMeta(
        format_version=INDEX_FORMAT_VERSION,
        config=config,
        chunks=list(chunks),
        avgdl=index.avgdl,
    )
    save_model(meta, target / INDEX_META_FILENAME)


def load_index(
    directory: Union[str, Path],
) -> Tuple[BM25Index, List[IndexedChunk], IndexConfig]:
    """Read back a saved index, its chunk records and its configuration.

    Args:
        directory: Directory holding the two artifacts.

    Returns:
        ``(index, chunks, config)`` where ``chunks[doc_id]`` is the record
        of that document.

    Raises:
        ValueError: If either file is missing or unreadable, the JSON is
            malformed or fails validation, ``format_version`` differs from
            ``INDEX_FORMAT_VERSION``, the arrays fail ``from_bundle``
            validation, or ``len(chunks) != index.doc_count``.
    """
    target = Path(directory)
    arrays_path = target / INDEX_ARRAYS_FILENAME
    meta_path = target / INDEX_META_FILENAME
    if not arrays_path.is_file():
        raise ValueError(f"Missing index arrays file {arrays_path}")
    if not meta_path.is_file():
        raise ValueError(f"Missing index metadata file {meta_path}")
    try:
        meta = IndexMeta.model_validate(load_json(meta_path))
    except ValueError as exc:
        raise ValueError(
            f"Could not load index metadata {meta_path}: {exc}"
        ) from exc
    if meta.format_version != INDEX_FORMAT_VERSION:
        raise ValueError(
            f"Index format version {meta.format_version} does not match "
            f"the supported version {INDEX_FORMAT_VERSION}"
        )
    try:
        with np.load(arrays_path, allow_pickle=False) as data:
            bundle: Dict[str, NDArray[Any]] = {
                key: np.asarray(data[key]).copy()
                for key in _BUNDLE_KEYS
            }
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Could not load index arrays {arrays_path}: {exc}"
        ) from exc
    index = BM25Index.from_bundle(bundle, meta.avgdl, meta.config)
    chunks = list(meta.chunks)
    if len(chunks) != index.doc_count:
        raise ValueError(
            f"metadata holds {len(chunks)} chunks for "
            f"{index.doc_count} documents"
        )
    return index, chunks, meta.config


__all__ = [
    "INDEX_ARRAYS_FILENAME",
    "INDEX_FORMAT_VERSION",
    "INDEX_META_FILENAME",
    "IndexMeta",
    "load_index",
    "save_index",
]
