"""Shared pytest fixtures for the RAG test suite."""
from pathlib import Path

import pytest


@pytest.fixture
def tiny_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a three-file corpus under data/raw/tiny and chdir to its root.

    Args:
        tmp_path: Pytest per-test temporary directory.
        monkeypatch: Pytest monkeypatch helper for chdir.

    Returns:
        The project root holding ``data/raw/tiny``.
    """
    root = tmp_path / "project"
    corpus = root / "data" / "raw" / "tiny"
    corpus.mkdir(parents=True)
    (corpus / "alpha.py").write_text(
        '"""Alpha module."""\n\n\n'
        "def quixotic_tokenizer_v2(text: str) -> str:\n"
        '    """Return the text with a marker appended."""\n'
        '    return text + "!"\n',
        encoding="utf-8",
    )
    (corpus / "beta.py").write_text(
        '"""Beta module."""\n\n\n'
        "def mundane_helper(n: int) -> int:\n"
        '    """Return n plus one."""\n'
        "    return n + 1\n",
        encoding="utf-8",
    )
    (corpus / "guide.md").write_text(
        "# Frosting Guide\n\n"
        "A short guide about frosting cakes with buttercream.\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(root)
    return root
