"""Tests for the chunking package in src.chunking."""
import dataclasses
from pathlib import Path
from typing import List, Tuple
from unittest import mock

import pytest

from src.chunking import SUPPORTED_EXTENSIONS, chunk_file
from src.chunking.base import (
    CHUNK_KINDS,
    Chunk,
    line_start_offsets,
    read_file_text,
    sliding_window,
    validate_chunks,
)
from src.chunking.markdown_chunker import chunk_markdown
from src.chunking.python_chunker import chunk_python
from src.io_utils import to_posix_path

CORPUS_PY_PATH = Path(
    "data/raw/vllm-0.10.1/vllm/entrypoints/llm.py"
)
CORPUS_MD_PATH = Path(
    "data/raw/vllm-0.10.1/docs/serving/openai_compatible_server.md"
)


def _merged_spans(chunks: List[Chunk]) -> List[Tuple[int, int]]:
    """Merge chunk spans into a sorted list of disjoint intervals.

    Args:
        chunks: The chunks whose spans are merged.

    Returns:
        Disjoint, sorted [start, end) intervals covering every chunk.
    """
    ordered = sorted((chunk.start, chunk.end) for chunk in chunks)
    merged: List[Tuple[int, int]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def _assert_full_coverage(text: str, chunks: List[Chunk]) -> None:
    """Assert the chunks tile exactly [0, len(text)).

    Args:
        text: The full file content the chunks were cut from.
        chunks: The chunks that must cover the whole text.
    """
    if not text:
        assert chunks == []
        return
    assert _merged_spans(chunks) == [(0, len(text))]


def test_chunk_valid_construction_frozen_equality() -> None:
    """Case 1: identical chunks compare equal and are immutable."""
    first = Chunk(
        file_path="data/raw/a.py",
        start=0,
        end=5,
        text="hello",
        kind="python",
    )
    second = Chunk(
        file_path="data/raw/a.py",
        start=0,
        end=5,
        text="hello",
        kind="python",
    )
    assert first == second
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(first, "start", 3)


def test_chunk_rejects_empty_file_path() -> None:
    """Case 2: an empty file_path raises ValueError."""
    with pytest.raises(ValueError):
        Chunk(file_path="", start=0, end=5, text="hello", kind="python")


def test_chunk_rejects_negative_start() -> None:
    """Case 3: a negative start raises ValueError."""
    with pytest.raises(ValueError):
        Chunk(file_path="a.py", start=-1, end=5, text="hello", kind="python")


def test_chunk_rejects_inverted_range() -> None:
    """Case 4: end <= start raises ValueError."""
    with pytest.raises(ValueError):
        Chunk(file_path="a.py", start=0, end=0, text="", kind="python")
    with pytest.raises(ValueError):
        Chunk(file_path="a.py", start=5, end=3, text="x", kind="python")


def test_chunk_rejects_unknown_kind() -> None:
    """Case 5: a kind outside CHUNK_KINDS raises ValueError."""
    assert CHUNK_KINDS == ("python", "markdown", "sliding")
    with pytest.raises(ValueError):
        Chunk(file_path="a.py", start=0, end=5, text="hello", kind="banana")


def test_line_start_offsets_empty_string() -> None:
    """Case 6: line_start_offsets('') returns [0]."""
    assert line_start_offsets("") == [0]


def test_line_start_offsets_trailing_newline() -> None:
    """Case 7: line_start_offsets('a\\n') returns [0, 2]."""
    assert line_start_offsets("a\n") == [0, 2]


def test_line_start_offsets_multiple_lines() -> None:
    """Case 8: line_start_offsets counts every line start."""
    assert line_start_offsets("a\nbb\nccc") == [0, 2, 5]


def test_read_file_text_round_trip_unicode(tmp_path: Path) -> None:
    """Case 9: read_file_text round-trips accents and emoji."""
    path = tmp_path / "unicode.txt"
    expected = "café 🚀 naïve\n"
    path.write_text(expected, encoding="utf-8")
    assert read_file_text(path) == expected


def test_read_file_text_missing_path_raises_value_error(
    tmp_path: Path,
) -> None:
    """Case 10: a missing path raises ValueError naming the path."""
    missing = tmp_path / "not_here.txt"
    with pytest.raises(ValueError) as excinfo:
        read_file_text(missing)
    assert not isinstance(excinfo.value, FileNotFoundError)
    assert str(missing) in str(excinfo.value)


def test_read_file_text_replaces_invalid_bytes(tmp_path: Path) -> None:
    """Case 11: one invalid UTF-8 byte becomes U+FFFD, no crash."""
    path = tmp_path / "broken.txt"
    path.write_bytes(b"caf\xe9")
    content = read_file_text(path)
    assert "\ufffd" in content


def test_read_file_text_uses_context_manager(tmp_path: Path) -> None:
    """Case 12: read_file_text enters and exits the file handle."""
    path = tmp_path / "sample.txt"
    path.write_text("hello", encoding="utf-8")
    handle = mock.MagicMock()
    handle.read.return_value = "hello"
    handle.__enter__.return_value = handle
    with mock.patch.object(Path, "open", return_value=handle):
        content = read_file_text(path)
    assert content == "hello"
    handle.__enter__.assert_called_once()
    handle.__exit__.assert_called_once()


def test_sliding_window_exact_division() -> None:
    """Case 13: exact division yields adjacent sliding chunks."""
    text = "abcdefghij"
    chunks = list(sliding_window(text, "f.py", 0, 10, 5))
    assert len(chunks) == 2
    assert chunks[0].end == chunks[1].start
    assert chunks[0].kind == "sliding"
    assert chunks[1].kind == "sliding"
    _assert_full_coverage(text, chunks)


def test_sliding_window_remainder() -> None:
    """Case 14: a remainder yields 5/5/2 chunks with full coverage."""
    text = "a" * 12
    chunks = list(sliding_window(text, "f.py", 0, 12, 5))
    assert [chunk.end - chunk.start for chunk in chunks] == [5, 5, 2]
    _assert_full_coverage(text, chunks)


def test_sliding_window_overlap_awkward() -> None:
    """Case 15: overlap=2 on len=23/max=7 respects every guarantee."""
    text = "".join(chr(97 + index) for index in range(23))
    chunks = list(sliding_window(text, "f.py", 0, 23, 7, 2))
    assert chunks
    for chunk in chunks:
        assert chunk.end - chunk.start <= 7
        assert text[chunk.start:chunk.end] == chunk.text
    for previous, current in zip(chunks, chunks[1:]):
        assert previous.end - current.start == 2
    _assert_full_coverage(text, chunks)


def test_sliding_window_rejects_zero_max_chunk_size() -> None:
    """Case 16: max_chunk_size=0 raises ValueError."""
    with pytest.raises(ValueError):
        list(sliding_window("abc", "f.py", 0, 3, 0))


def test_sliding_window_rejects_invalid_overlap() -> None:
    """Case 17: overlap < 0 or overlap >= max_chunk_size raises."""
    with pytest.raises(ValueError):
        list(sliding_window("abc", "f.py", 0, 3, 5, -1))
    with pytest.raises(ValueError):
        list(sliding_window("abc", "f.py", 0, 3, 5, 5))


def test_sliding_window_empty_range_yields_nothing() -> None:
    """Case 18: start == end is valid and yields no chunks."""
    assert list(sliding_window("abc", "f.py", 2, 2, 5)) == []


def test_sliding_window_rejects_invalid_ranges() -> None:
    """Case 19: start < 0, end > len(text) or start > end raise."""
    text = "abcdefghij"
    with pytest.raises(ValueError):
        list(sliding_window(text, "f.py", -1, 5, 5))
    with pytest.raises(ValueError):
        list(sliding_window(text, "f.py", 0, 11, 5))
    with pytest.raises(ValueError):
        list(sliding_window(text, "f.py", 5, 3, 5))


def test_validate_chunks_accepts_good_list() -> None:
    """Case 20: validate_chunks returns None for a good chunk list."""
    text = "abcdefghij"
    chunks = list(sliding_window(text, "f.py", 0, 10, 4))
    validate_chunks(text, chunks, 4)


def test_validate_chunks_rejects_tampered_list() -> None:
    """Case 21: out-of-range end and out-of-order starts raise."""
    text = "abcdefghij"
    oversized = Chunk(
        file_path="f.py", start=0, end=50, text="hello", kind="python"
    )
    with pytest.raises(ValueError) as excinfo:
        validate_chunks(text, [oversized], 10)
    assert "chunk 0" in str(excinfo.value)
    assert "range" in str(excinfo.value)
    first = Chunk(
        file_path="f.py", start=6, end=8, text="gh", kind="sliding"
    )
    second = Chunk(
        file_path="f.py", start=3, end=4, text="d", kind="sliding"
    )
    with pytest.raises(ValueError) as excinfo:
        validate_chunks(text, [first, second], 5)
    assert "chunk 1" in str(excinfo.value)
    assert "order" in str(excinfo.value)


def test_chunk_python_module_partition() -> None:
    """Case 22: import and defs partition a module in file order."""
    text = (
        "import os\n"
        "\n"
        "def foo():\n"
        "    return 1\n"
        "\n"
        "def bar():\n"
        "    return 2\n"
    )
    chunks = chunk_python(text, "data/raw/vllm-0.10.1/a.py", 2000)
    validate_chunks(text, chunks, 2000)
    _assert_full_coverage(text, chunks)
    starts = [chunk.start for chunk in chunks]
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts)
    def_chunks = [
        chunk for chunk in chunks if "def " in chunk.text
    ]
    assert def_chunks
    assert all(chunk.kind == "python" for chunk in def_chunks)
    assert any("import os" in chunk.text for chunk in chunks)


def test_chunk_python_decorator_starts_chunk() -> None:
    """Case 23: the function chunk starts at the decorator line."""
    text = (
        "import os\n"
        "@decorate\n"
        "def foo():\n"
        "    return 1\n"
    )
    chunks = chunk_python(text, "data/raw/vllm-0.10.1/a.py", 2000)
    _assert_full_coverage(text, chunks)
    function_chunk = next(
        chunk for chunk in chunks if "def foo" in chunk.text
    )
    assert function_chunk.text.startswith("@decorate")


def test_chunk_python_small_class_single_chunk() -> None:
    """Case 24: a small class fits in one python chunk."""
    text = (
        "class Foo:\n"
        "    def method_one(self):\n"
        "        return 1\n"
        "\n"
        "    def method_two(self):\n"
        "        return 2\n"
    )
    chunks = chunk_python(text, "data/raw/vllm-0.10.1/a.py", 2000)
    assert len(chunks) == 1
    assert chunks[0].kind == "python"
    assert "class" in chunks[0].text
    assert "def method_one" in chunks[0].text
    assert "def method_two" in chunks[0].text
    _assert_full_coverage(text, chunks)


def test_chunk_python_oversized_class_split_by_members() -> None:
    """Case 25: an oversized class splits into header and members."""
    text = (
        "class Foo(Base):\n"
        "    '''Class docstring.'''\n"
        "\n"
        "    @staticmethod\n"
        "    def method_one():\n"
        "        value = 1\n"
        "        other = 2\n"
        "        return value + other\n"
        "\n"
        "    def method_two(self):\n"
        "        first = 10\n"
        "        second = 20\n"
        "        return first + second\n"
    )
    max_chunk_size = 80
    chunks = chunk_python(text, "data/raw/vllm-0.10.1/a.py", max_chunk_size)
    validate_chunks(text, chunks, max_chunk_size)
    _assert_full_coverage(text, chunks)
    assert len(chunks) >= 3
    assert any("class " in chunk.text for chunk in chunks)
    ordered = sorted(chunks, key=lambda chunk: chunk.start)
    for previous, current in zip(ordered, ordered[1:]):
        assert previous.end == current.start
    assert all(
        chunk.start < chunk.end for chunk in ordered
    )


def test_chunk_python_oversized_function_uses_sliding() -> None:
    """Case 26: a giant top-level function becomes sliding windows."""
    body = "\n".join(
        f"    value_{index} = {index} * 2" for index in range(10)
    )
    text = f"def giant():\n{body}\n"
    max_chunk_size = 60
    chunks = chunk_python(text, "data/raw/vllm-0.10.1/a.py", max_chunk_size)
    sliding_chunks = [
        chunk for chunk in chunks if chunk.kind == "sliding"
    ]
    assert len(sliding_chunks) >= 2
    assert all(chunk.end - chunk.start <= max_chunk_size for chunk in chunks)
    validate_chunks(text, chunks, max_chunk_size)
    _assert_full_coverage(text, chunks)


def test_chunk_python_syntax_error_fallback() -> None:
    """Case 27: broken source falls back to sliding chunks."""
    text = "def broken(:\n    pass\n"
    chunks = chunk_python(text, "data/raw/vllm-0.10.1/a.py", 2000)
    assert len(chunks) >= 1
    assert all(chunk.kind == "sliding" for chunk in chunks)
    validate_chunks(text, chunks, 2000)
    _assert_full_coverage(text, chunks)


def test_chunk_python_comments_only_fallback() -> None:
    """Case 28: an empty AST body falls back to sliding chunks."""
    text = "# just a comment\n"
    chunks = chunk_python(text, "data/raw/vllm-0.10.1/a.py", 2000)
    assert len(chunks) >= 1
    assert all(chunk.kind == "sliding" for chunk in chunks)
    _assert_full_coverage(text, chunks)


def test_chunk_python_unicode_offsets_are_character_based() -> None:
    """Case 29: emoji and accents keep slice consistency."""
    text = (
        "# café 🚀 launch\n"
        "def alpha():\n"
        "    return 1\n"
        "\n"
        "# naïve 🚀 step\n"
        "def beta():\n"
        "    return 2\n"
    )
    chunks = chunk_python(text, "data/raw/vllm-0.10.1/a.py", 2000)
    for chunk in chunks:
        assert text[chunk.start:chunk.end] == chunk.text
    validate_chunks(text, chunks, 2000)
    _assert_full_coverage(text, chunks)


def test_chunk_python_docstring_stays_in_function_chunk() -> None:
    """Case 30: the function chunk contains its docstring."""
    text = (
        "def documented():\n"
        '    """Return the answer."""\n'
        "    return 42\n"
    )
    chunks = chunk_python(text, "data/raw/vllm-0.10.1/a.py", 2000)
    function_chunk = next(
        chunk for chunk in chunks if "def documented" in chunk.text
    )
    assert '"""Return the answer."""' in function_chunk.text


def test_chunk_python_empty_text_returns_empty_list() -> None:
    """Case 31: empty text produces no chunks."""
    assert chunk_python("", "data/raw/vllm-0.10.1/a.py", 2000) == []


def test_chunk_python_rejects_zero_max_chunk_size() -> None:
    """Case 32: max_chunk_size=0 raises ValueError."""
    with pytest.raises(ValueError):
        chunk_python("import os\n", "data/raw/vllm-0.10.1/a.py", 0)


def test_chunk_markdown_header_sections() -> None:
    """Case 33: each ATX header becomes its own markdown chunk."""
    text = "# A\n## B\n## C\n"
    chunks = chunk_markdown(text, "data/raw/vllm-0.10.1/a.md", 2000)
    assert len(chunks) == 3
    assert [chunk.text.split("\n")[0] for chunk in chunks] == [
        "# A",
        "## B",
        "## C",
    ]
    assert all(chunk.kind == "markdown" for chunk in chunks)
    _assert_full_coverage(text, chunks)


def test_chunk_markdown_leading_prose_before_header() -> None:
    """Case 34: prose before the first header forms a leading chunk."""
    text = "Intro paragraph here.\n\n# Title\nbody\n"
    chunks = chunk_markdown(text, "data/raw/vllm-0.10.1/a.md", 2000)
    assert chunks[0].start == 0
    assert chunks[0].end == text.index("# Title")
    assert chunks[0].text == "Intro paragraph here.\n\n"
    _assert_full_coverage(text, chunks)


def test_chunk_markdown_fenced_code_comment_is_not_header() -> None:
    """Case 35: a # comment inside a fence creates no boundary."""
    text = (
        "# Intro\n"
        "\n"
        "```python\n"
        "# not a header\n"
        "x = 1\n"
        "```\n"
        "\n"
        "## Second\n"
        "body\n"
    )
    chunks = chunk_markdown(text, "data/raw/vllm-0.10.1/a.md", 2000)
    assert len(chunks) == 2
    assert "# not a header" in chunks[0].text
    assert chunks[1].text.startswith("## Second")
    _assert_full_coverage(text, chunks)


def test_chunk_markdown_oversized_section_paragraph_groups() -> None:
    """Case 36: an oversized section splits into paragraph groups."""
    text = (
        "# Big Section\n"
        "\n"
        "First paragraph line one.\n"
        "First paragraph line two.\n"
        "\n"
        "Second paragraph alpha beta gamma delta.\n"
        "\n"
        "Third paragraph one two three four five.\n"
    )
    max_chunk_size = 60
    chunks = chunk_markdown(text, "data/raw/vllm-0.10.1/a.md", max_chunk_size)
    assert len(chunks) >= 2
    assert "# Big Section" in chunks[0].text
    assert all(
        chunk.end - chunk.start <= max_chunk_size for chunk in chunks
    )
    assert all(chunk.kind in ("markdown", "sliding") for chunk in chunks)
    validate_chunks(text, chunks, max_chunk_size)
    _assert_full_coverage(text, chunks)


def test_chunk_markdown_single_oversized_paragraph_slides() -> None:
    """Case 37: one paragraph longer than max becomes sliding chunks."""
    text = " ".join(["lorem ipsum dolor sit amet"] * 30)
    max_chunk_size = 100
    chunks = chunk_markdown(text, "data/raw/vllm-0.10.1/a.md", max_chunk_size)
    assert len(chunks) >= 2
    assert all(chunk.kind == "sliding" for chunk in chunks)
    assert all(
        chunk.end - chunk.start <= max_chunk_size for chunk in chunks
    )
    _assert_full_coverage(text, chunks)


def test_chunk_markdown_headerless_paragraph_groups() -> None:
    """Case 38: headerless text groups paragraphs as markdown chunks."""
    text = "Para one.\n\nPara two.\n\nPara three.\n"
    max_chunk_size = 15
    chunks = chunk_markdown(text, "data/raw/vllm-0.10.1/a.md", max_chunk_size)
    assert len(chunks) >= 2
    assert all(chunk.kind == "markdown" for chunk in chunks)
    validate_chunks(text, chunks, max_chunk_size)
    _assert_full_coverage(text, chunks)


def test_chunk_markdown_setext_style_treated_as_plain_text() -> None:
    """Case 39: setext underlines are not special and never crash."""
    text = "Title\n=====\nbody text here\n"
    chunks = chunk_markdown(text, "data/raw/vllm-0.10.1/a.md", 2000)
    assert len(chunks) == 1
    assert chunks[0].kind == "markdown"
    assert "Title" in chunks[0].text
    _assert_full_coverage(text, chunks)


def test_chunk_markdown_empty_and_invalid_params() -> None:
    """Case 40: empty text returns [] and max_chunk_size=0 raises."""
    assert chunk_markdown("", "data/raw/vllm-0.10.1/a.md", 2000) == []
    with pytest.raises(ValueError):
        chunk_markdown("# H\n", "data/raw/vllm-0.10.1/a.md", 0)


def test_chunk_file_python_records_posix_path(tmp_path: Path) -> None:
    """Case 41: a tmp .py file records the POSIX-normalized path."""
    path = tmp_path / "sample.py"
    path.write_text("def foo():\n    return 1\n", encoding="utf-8")
    chunks = chunk_file(path, max_chunk_size=2000)
    assert chunks
    assert all(
        chunk.file_path == to_posix_path(str(path)) for chunk in chunks
    )
    assert all(chunk.kind == "python" for chunk in chunks)


def test_chunk_file_markdown_extensions_dispatch(tmp_path: Path) -> None:
    """Case 42: .md, .txt and .rst files use the markdown strategy."""
    for suffix in (".md", ".txt", ".rst"):
        path = tmp_path / f"sample{suffix}"
        path.write_text("# Head\n\nSome prose.\n", encoding="utf-8")
        chunks = chunk_file(path, max_chunk_size=2000)
        assert chunks
        assert all(chunk.kind == "markdown" for chunk in chunks)


def test_chunk_file_unsupported_extension_raises(tmp_path: Path) -> None:
    """Case 43: an unsupported extension raises with the supported set."""
    path = tmp_path / "data.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        chunk_file(path, max_chunk_size=2000)
    message = str(excinfo.value)
    assert ".json" in message
    assert ".py" in message
    assert ".md" in message


def test_chunk_file_missing_python_path_raises(tmp_path: Path) -> None:
    """Case 44: a missing .py file raises ValueError, not the OS error."""
    missing = tmp_path / "gone.py"
    with pytest.raises(ValueError) as excinfo:
        chunk_file(missing, max_chunk_size=2000)
    assert not isinstance(excinfo.value, FileNotFoundError)
    assert str(missing) in str(excinfo.value)


def test_supported_extensions_constant() -> None:
    """Case 45: SUPPORTED_EXTENSIONS lists the four suffixes."""
    assert SUPPORTED_EXTENSIONS == (".py", ".md", ".txt", ".rst")


def test_chunk_file_uppercase_suffix_dispatches(tmp_path: Path) -> None:
    """Case 46: uppercase suffixes dispatch case-insensitively."""
    py_path = tmp_path / "UP.PY"
    py_path.write_text("def foo():\n    return 1\n", encoding="utf-8")
    md_path = tmp_path / "UP.MD"
    md_path.write_text("# Head\n\nProse.\n", encoding="utf-8")
    py_chunks = chunk_file(py_path, max_chunk_size=2000)
    md_chunks = chunk_file(md_path, max_chunk_size=2000)
    assert py_chunks
    assert all(chunk.kind == "python" for chunk in py_chunks)
    assert md_chunks
    assert all(chunk.kind == "markdown" for chunk in md_chunks)


@pytest.mark.skipif(
    not CORPUS_PY_PATH.exists(),
    reason="vllm corpus not present in checkout",
)
def test_real_corpus_llm_py_full_coverage() -> None:
    """Case 47: llm.py chunks validate and cover the whole file."""
    text = read_file_text(CORPUS_PY_PATH)
    chunks = chunk_file(CORPUS_PY_PATH, max_chunk_size=2000)
    validate_chunks(text, chunks, 2000)
    _assert_full_coverage(text, chunks)
    assert len(chunks) >= 2


@pytest.mark.skipif(
    not CORPUS_MD_PATH.exists(),
    reason="vllm corpus not present in checkout",
)
def test_real_corpus_openai_server_md_full_coverage() -> None:
    """Case 48: the server doc chunks validate and cover the file."""
    text = read_file_text(CORPUS_MD_PATH)
    chunks = chunk_file(CORPUS_MD_PATH, max_chunk_size=2000)
    validate_chunks(text, chunks, 2000)
    _assert_full_coverage(text, chunks)
    assert len(chunks) >= 2
