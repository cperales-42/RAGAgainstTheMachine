"""Tests for corpus ingestion in src.indexing.indexer."""
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
from pydantic import ValidationError

from src.chunking.base import read_file_text
from src.config import IndexConfig
from src.indexing import index_corpus, load_index
from src.indexing.indexer import IndexStats
from src.indexing.storage import (
    INDEX_ARRAYS_FILENAME,
    INDEX_META_FILENAME,
)
from src.models import IndexedChunk

_ALPHA = "data/raw/tiny/alpha.py"


def _spans_by_file(
    chunks: List[IndexedChunk],
) -> Dict[str, List[Tuple[int, int]]]:
    """Group chunk spans by recorded file path.

    Args:
        chunks: Loaded IndexedChunk records.

    Returns:
        Mapping from file path to a list of ``(first, last)`` tuples.
    """
    grouped: Dict[str, List[Tuple[int, int]]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.file_path, []).append(
            (
                chunk.first_character_index,
                chunk.last_character_index,
            )
        )
    return grouped


def _merge(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping intervals into disjoint sorted ranges.

    Args:
        intervals: Candidate intervals.

    Returns:
        Disjoint, sorted intervals covering the input.
    """
    merged: List[Tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def test_case48_indexes_the_tiny_corpus(tiny_project: Path) -> None:
    """Case 48: the default run reports counts and writes both artifacts."""
    stats = index_corpus(IndexConfig())
    assert isinstance(stats, IndexStats)
    assert stats.files_indexed == 3
    assert stats.chunks_indexed >= 3
    assert stats.tokens_indexed > 0
    assert stats.elapsed_seconds > 0
    assert stats.index_directory == "data/processed"
    processed = Path("data/processed")
    assert (processed / INDEX_ARRAYS_FILENAME).is_file()
    assert (processed / INDEX_META_FILENAME).is_file()
    _, records, _ = load_index("data/processed")
    assert all(
        record.file_path.startswith("data/raw/tiny/") for record in records
    )


def test_case49_offsets_are_consistent(tiny_project: Path) -> None:
    """Case 49: every span slices the file and tiles it exactly."""
    index_corpus(IndexConfig(), show_progress=False)
    _, records, _ = load_index("data/processed")
    grouped = _spans_by_file(list(records))
    assert set(grouped) == {
        "data/raw/tiny/alpha.py",
        "data/raw/tiny/beta.py",
        "data/raw/tiny/guide.md",
    }
    for file_path, spans in grouped.items():
        text = read_file_text(file_path)
        for first, last in spans:
            assert 0 <= first < last <= len(text)
            assert last - first <= 2000
            assert text[first:last]
        assert _merge(spans) == [(0, len(text))]


def test_case50_query_finds_the_made_up_identifier(
    tiny_project: Path,
) -> None:
    """Case 50: a rare identifier ranks its defining chunk first."""
    index_corpus(IndexConfig(), show_progress=False)
    index, records, _ = load_index("data/processed")
    hits = index.search("quixotic_tokenizer_v2", 3)
    assert hits
    top = records[hits[0].doc_id]
    assert top.file_path == _ALPHA
    text = read_file_text(top.file_path)
    assert "def quixotic_tokenizer_v2" in text[
        top.first_character_index:top.last_character_index
    ]


def test_case51_smaller_chunks_respect_the_limit(
    tiny_project: Path,
) -> None:
    """Case 51: max_chunk_size=64 widens the spans and adds chunks."""
    default_stats = index_corpus(IndexConfig(), show_progress=False)
    small_stats = index_corpus(
        IndexConfig(max_chunk_size=64, processed_path="out/small"),
        show_progress=False,
    )
    _, small_records, _ = load_index("out/small")
    assert all(
        record.last_character_index - record.first_character_index <= 64
        for record in small_records
    )
    assert small_stats.chunks_indexed > default_stats.chunks_indexed


def test_case52_two_runs_are_deterministic(tiny_project: Path) -> None:
    """Case 52: repeated runs produce identical records and scores."""
    index_corpus(IndexConfig(), show_progress=False)
    first_index, first_records, _ = load_index("data/processed")
    index_corpus(IndexConfig(), show_progress=False)
    second_index, second_records, _ = load_index("data/processed")
    assert first_records == second_records
    assert first_index.search("frosting guide", 5) == second_index.search(
        "frosting guide", 5
    )


def test_case53_absolute_corpus_path_still_relative(
    tiny_project: Path,
) -> None:
    """Case 53: an absolute corpus path yields relative recorded paths."""
    corpus = tiny_project / "data" / "raw" / "tiny"
    index_corpus(IndexConfig(), show_progress=False)
    _, relative_records, _ = load_index("data/processed")
    index_corpus(
        IndexConfig(
            corpus_path=str(corpus), processed_path="out/absolute"
        ),
        show_progress=False,
    )
    _, absolute_records, _ = load_index("out/absolute")
    assert [record.file_path for record in absolute_records] == [
        record.file_path for record in relative_records
    ]


def test_case54_missing_corpus_names_the_path(tiny_project: Path) -> None:
    """Case 54: a missing corpus path raises and names it."""
    with pytest.raises(ValueError, match="does/not/exist"):
        index_corpus(
            IndexConfig(corpus_path="does/not/exist"), show_progress=False
        )


@pytest.mark.parametrize("name", ["empty", "notes"])
def test_case55_no_supported_files(tiny_project: Path, name: str) -> None:
    """Case 55: a corpus without supported files mentions extensions."""
    target = tiny_project / "data" / "raw" / name
    target.mkdir(parents=True)
    if name == "notes":
        (target / "notes.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="supported"):
        index_corpus(
            IndexConfig(corpus_path=f"data/raw/{name}"),
            show_progress=False,
        )


def test_case56_corpus_outside_project_root(tiny_project: Path) -> None:
    """Case 56: an absolute corpus outside the project root is rejected."""
    outside = tiny_project.parent / "outside"
    outside.mkdir()
    (outside / "module.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="project root"):
        index_corpus(
            IndexConfig(corpus_path=str(outside)), show_progress=False
        )


def test_case57_project_root_is_rejected(tiny_project: Path) -> None:
    """Case 57: pointing corpus_path at the project root raises."""
    with pytest.raises(ValueError):
        index_corpus(IndexConfig(corpus_path="."), show_progress=False)


def test_case58_silent_run_matches_loud_run(tiny_project: Path) -> None:
    """Case 58: show_progress=False reports the same counts."""
    loud = index_corpus(IndexConfig())
    silent = index_corpus(IndexConfig(), show_progress=False)
    assert silent.files_indexed == loud.files_indexed
    assert silent.chunks_indexed == loud.chunks_indexed


def test_case59_invalid_config_is_rejected_early(
    tiny_project: Path,
) -> None:
    """Case 59: bad chunk size and overlap fail before any file access."""
    with pytest.raises(ValidationError) as zero:
        IndexConfig(max_chunk_size=0)
    assert isinstance(zero.value, ValueError)
    with pytest.raises(ValidationError):
        IndexConfig(max_chunk_size=10, overlap=10)
