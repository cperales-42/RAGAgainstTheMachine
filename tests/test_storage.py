"""Tests for lexical index persistence in src.indexing.storage."""
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from src.config import IndexConfig
from src.indexing.bm25 import _BUNDLE_KEYS, BM25Index
from src.indexing.storage import (
    INDEX_ARRAYS_FILENAME,
    INDEX_FORMAT_VERSION,
    INDEX_META_FILENAME,
    load_index,
    save_index,
)
from src.models import IndexedChunk

_VERBATIM_PATH = "data/raw/vllm-0.10.1/docs/features/lora.md"
_QUERIES = ["openai server", "lora adapters"]


def _sample() -> Tuple[BM25Index, List[IndexedChunk], IndexConfig]:
    """Build a deterministic three-document index with its records.

    Returns:
        A built index, its three IndexedChunk records and the config.
    """
    config = IndexConfig()
    index = BM25Index(
        k1=config.k1,
        b=config.b,
        weight_body=config.weight_body,
        weight_header=config.weight_header,
        weight_path=config.weight_path,
    )
    index.add_document(
        0,
        {
            "body": ["openai", "server", "configuration"],
            "header": ["openai", "compatible", "server"],
            "path": ["docs", "serving", "openai", "compatible", "server"],
        },
    )
    index.add_document(
        1,
        {
            "body": ["lora", "adapters", "serving"],
            "header": ["lora"],
            "path": ["docs", "features", "lora"],
        },
    )
    index.add_document(
        2,
        {
            "body": ["sampler", "skip", "output"],
            "header": ["sampler"],
            "path": ["vllm", "model_executor", "layers", "sampler"],
        },
    )
    index.build()
    records = [
        IndexedChunk(
            file_path=(
                "data/raw/vllm-0.10.1/docs/serving/"
                "openai_compatible_server.md"
            ),
            first_character_index=0,
            last_character_index=120,
            kind="markdown",
        ),
        IndexedChunk(
            file_path=_VERBATIM_PATH,
            first_character_index=0,
            last_character_index=200,
            kind="markdown",
        ),
        IndexedChunk(
            file_path=(
                "data/raw/vllm-0.10.1/vllm/model_executor/"
                "layers/sampler.py"
            ),
            first_character_index=10,
            last_character_index=500,
            kind="python",
        ),
    ]
    return index, records, IndexConfig()


def test_case37_save_load_round_trip(tmp_path: Path) -> None:
    """Case 37: a saved index reloads with identical behaviour."""
    index, records, config = _sample()
    save_index(index, records, config, tmp_path)
    reloaded, loaded_records, loaded_config = load_index(tmp_path)
    assert reloaded.doc_count == index.doc_count
    assert loaded_config == config
    assert loaded_records == records
    assert loaded_records[1].file_path == _VERBATIM_PATH
    for query in _QUERIES:
        assert reloaded.search(query, 5) == index.search(query, 5)


def test_case38_artifact_contract(tmp_path: Path) -> None:
    """Case 38: the npz and JSON keys are exactly the documented sets."""
    index, records, config = _sample()
    save_index(index, records, config, tmp_path)
    with np.load(tmp_path / INDEX_ARRAYS_FILENAME) as data:
        assert set(data.files) == set(_BUNDLE_KEYS)
    raw = json.loads((tmp_path / INDEX_META_FILENAME).read_text("utf-8"))
    assert set(raw) == {"format_version", "config", "chunks", "avgdl"}


def test_case39_creates_deep_directory(tmp_path: Path) -> None:
    """Case 39: a missing parent chain is created on save."""
    index, records, config = _sample()
    destination = tmp_path / "a" / "b" / "c"
    save_index(index, records, config, destination)
    assert (destination / INDEX_ARRAYS_FILENAME).is_file()
    assert (destination / INDEX_META_FILENAME).is_file()


def test_case40_unbuilt_index_does_not_create_directory(
    tmp_path: Path,
) -> None:
    """Case 40: an unbuilt index fails before the directory is made."""
    destination = tmp_path / "never"
    with pytest.raises(ValueError):
        save_index(BM25Index(), [], IndexConfig(), destination)
    assert not destination.exists()


def test_case41_record_count_mismatch(tmp_path: Path) -> None:
    """Case 41: a chunk count differing from doc_count raises."""
    index, records, config = _sample()
    with pytest.raises(ValueError):
        save_index(index, records[:2], config, tmp_path)


def test_case42_destination_under_regular_file(tmp_path: Path) -> None:
    """Case 42: a blocker file yields ValueError, not OSError."""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    index, records, config = _sample()
    with pytest.raises(ValueError):
        save_index(index, records, config, blocker / "sub")


def test_case43_load_missing_directory(tmp_path: Path) -> None:
    """Case 43: loading an absent directory names the path."""
    missing = tmp_path / "absent"
    with pytest.raises(ValueError, match="absent"):
        load_index(missing)


def test_case44_malformed_metadata(tmp_path: Path) -> None:
    """Case 44: invalid JSON in the metadata file raises ValueError."""
    index, records, config = _sample()
    save_index(index, records, config, tmp_path)
    (tmp_path / INDEX_META_FILENAME).write_text("{not json", "utf-8")
    with pytest.raises(ValueError):
        load_index(tmp_path)


def test_case45_unsupported_format_version(tmp_path: Path) -> None:
    """Case 45: a future format version names both versions."""
    index, records, config = _sample()
    save_index(index, records, config, tmp_path)
    meta_path = tmp_path / INDEX_META_FILENAME
    raw = json.loads(meta_path.read_text("utf-8"))
    raw["format_version"] = 999
    meta_path.write_text(json.dumps(raw), "utf-8")
    with pytest.raises(ValueError) as error:
        load_index(tmp_path)
    assert "999" in str(error.value)
    assert str(INDEX_FORMAT_VERSION) in str(error.value)


def test_case46_chunk_count_tampering(tmp_path: Path) -> None:
    """Case 46: metadata with fewer chunks than documents raises."""
    index, records, config = _sample()
    save_index(index, records, config, tmp_path)
    meta_path = tmp_path / INDEX_META_FILENAME
    raw = json.loads(meta_path.read_text("utf-8"))
    raw["chunks"] = raw["chunks"][:-1]
    meta_path.write_text(json.dumps(raw), "utf-8")
    with pytest.raises(ValueError):
        load_index(tmp_path)


def _mangle(kind: str) -> Dict[str, Any]:
    """Return a corrupted bundle reproducing one broken invariant.

    Args:
        kind: One of ``missing``, ``indptr`` or ``float_docs``.

    Returns:
        The bundle dict to feed ``from_bundle``.
    """
    index, _, _ = _sample()
    bundle: Dict[str, Any] = dict(index.to_bundle())
    if kind == "missing":
        bundle.pop("idf")
    elif kind == "indptr":
        indptr = np.asarray(bundle["indptr"])
        bundle["indptr"] = indptr[:-1]
    else:
        bundle["docs"] = np.asarray(bundle["docs"]).astype(np.float64)
    return bundle


@pytest.mark.parametrize("kind", ["missing", "indptr", "float_docs"])
def test_case47_from_bundle_rejects_broken_arrays(
    kind: str,
) -> None:
    """Case 47: each broken array invariant raises ValueError."""
    bundle = _mangle(kind)
    with pytest.raises(ValueError):
        BM25Index.from_bundle(bundle, _sample()[0].avgdl, IndexConfig())
