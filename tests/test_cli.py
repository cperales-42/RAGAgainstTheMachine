"""Tests for the Python Fire command-line interface."""
from pathlib import Path

import fire
import pytest

from src.cli import Rag
from src.indexing import load_index


def test_case60_index_with_explicit_paths(tiny_project: Path) -> None:
    """Case 60: explicit corpus and output paths index successfully."""
    result = fire.Fire(
        Rag,
        command=[
            "index",
            "--corpus_path",
            "data/raw/tiny",
            "--processed_path",
            "out/idx",
        ],
        name="rag",
    )
    assert result is None
    assert Path("out/idx/index.npz").is_file()


def test_case61_index_uses_config_defaults(tiny_project: Path) -> None:
    """Case 61: bare `index` writes to the default data/processed."""
    fire.Fire(Rag, command=["index"], name="rag")
    assert Path("data/processed/index.npz").is_file()
    assert Path("data/processed/index_meta.json").is_file()


def test_case62_space_separated_flags(tiny_project: Path) -> None:
    """Case 62: subject-style space-separated flags are honoured."""
    fire.Fire(
        Rag,
        command=[
            "index",
            "--max_chunk_size",
            "64",
            "--corpus_path",
            "data/raw/tiny",
        ],
        name="rag",
    )
    _, records, _ = load_index("data/processed")
    assert all(
        record.last_character_index - record.first_character_index <= 64
        for record in records
    )


def test_case63_missing_corpus_exits_cleanly(
    tiny_project: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Case 63: a missing corpus exits 1 without any traceback."""
    with pytest.raises(SystemExit) as error:
        fire.Fire(
            Rag,
            command=["index", "--corpus_path", "does/not/exist"],
            name="rag",
        )
    assert error.value.code == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("error:")
    assert "Traceback" not in captured.err
    assert "Traceback" not in captured.out


def test_case64_invalid_chunk_size_exits_cleanly(
    tiny_project: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Case 64: max_chunk_size=0 exits 1 with a clean message."""
    with pytest.raises(SystemExit) as error:
        fire.Fire(
            Rag,
            command=["index", "--max_chunk_size=0"],
            name="rag",
        )
    assert error.value.code == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("error:")
    assert "Traceback" not in captured.err
    assert "Traceback" not in captured.out


def test_case65_documentation_surface() -> None:
    """Case 65: Fire renders help from the non-empty docstrings."""
    assert Rag.__doc__
    assert Rag.index.__doc__
