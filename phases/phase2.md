# Phase 2 — Chunking (`src/chunking/`)

> **Agent:** `rag-dev`
> **Source of truth:** `en.subject.md` (section VI.1 Indexing the codebase) and
> `build_plan.md` (FASE 2).
> **Status of Phase 0:** DONE and verified on 2026-10-15
> (uv sync, make lint, corpus in `data/raw/vllm-0.10.1/`, datasets in
> `data/datasets/`, moulinette renamed and working).
> **Status of Phase 1:** DONE and verified on 2026-10-15
> (`models.py` + `io_utils.py` + 23 tests green, `make lint-strict` exit 0).
> **You must read before coding:** `en.subject.md` sections V.1 and VI.1
> (two chunking strategies, `--max_chunk_size` default 2000; the moulinette
> rejects the WHOLE output if any retrieved source exceeds 2000 characters)
> and VI.2 (`file_path` compared verbatim to the ground truth).

---

## 1. Objective

Implement the two mandatory chunking strategies — Python code (AST-based) and
Markdown/text (header-based) — over a shared chunking core that guarantees
**exact character offsets**. Every chunk is a slice of the original file:

```
0 <= start < end <= len(text)
end - start <= max_chunk_size          # hard limit: the moulinette fails at 2000
text[start:end] == chunk.text          # offset integrity: retrieval, generation
                                      # context and recall@k (IoU) all depend on it
```

These three invariants are non-negotiable for every chunk of every strategy.
Downstream phases (indexer, retriever, generator) will reuse this package
verbatim, so the contract must be airtight.

At the end of this phase you must be able to:

1. Chunk any corpus file (`.py`, `.md`, `.txt`, `.rst`) into validated chunks.
2. Survive broken Python (`SyntaxError`), emoji/accents, empty files,
   headerless markdown, and code fences — without a crash.
3. Pass `pytest`, `flake8` and `mypy --strict` with zero errors.

---

## 2. Files to create

```
src/chunking/
├── __init__.py         # dispatcher: chunk_file + SUPPORTED_EXTENSIONS
├── base.py             # Chunk, line_start_offsets, read_file_text,
│                       #   sliding_window, validate_chunks
├── python_chunker.py   # AST strategy
└── markdown_chunker.py # header/paragraph strategy
tests/
└── test_chunking.py
```

Do NOT create anything else. Do NOT modify `src/models.py`, `src/io_utils.py`
(importing from it is expected), `pyproject.toml`, `Makefile`, or `README.md`.
No CLI yet (that is Phase 3). No new dependencies: stdlib only
(`ast`, `dataclasses`, `pathlib`, `re`, `typing`).

Corpus facts for context: 1763 `.py` files and 206 `.md`/`.txt`/`.rst` files
under `data/raw/vllm-0.10.1/`; longest Python line is 1564 characters.

---

## 3. `src/chunking/base.py` — exact interface

### 3.1 Imports and module docstring

```python
"""Chunk primitives shared by every chunking strategy."""
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple, Union
```

Convention (same as Phase 1): `typing.List`/`Tuple`/`Union` explicit,
Python 3.10 target.

### 3.2 Constants

```python
CHUNK_KINDS: Tuple[str, ...] = ("python", "markdown", "sliding")
```

Meaning of each kind (exact values, MANDATORY — tests assert them):

- `"python"` — emitted by AST analysis in `python_chunker`.
- `"markdown"` — emitted by header/paragraph segmentation in
  `markdown_chunker`.
- `"sliding"` — emitted by `sliding_window` (subdivision of an oversized
  unit, or the `SyntaxError` fallback).

### 3.3 `Chunk` (plain frozen dataclass, NOT pydantic)

```python
@dataclass(frozen=True)
class Chunk:
    """A contiguous, validated slice of a source file."""

    file_path: str
    start: int
    end: int
    text: str
    kind: str

    def __post_init__(self) -> None:
        """Validate path, range and kind without mutating state."""
```

`__post_init__` raises `ValueError` (never `AssertionError`) when:

- `file_path` is empty,
- `start < 0`,
- `end <= start` (empty chunks are useless and break retrieval),
- `kind not in CHUNK_KINDS`.

Rationale: the subject only mandates pydantic for the models exchanged between
pipeline stages; chunking internals are implementation detail, and a frozen
dataclass is hashable, comparable (`==` in tests) and cheap. `text` consistency
with the original file cannot be checked here (no access to the original) —
that is `validate_chunks`' job.

### 3.4 Functions (signatures are MANDATORY)

```python
def line_start_offsets(text: str) -> List[int]:
    """Character offset of the first character of every line.

    Lines are counted by splitting on ``"\\n"`` (no other separator), so the
    result always has ``text.count("\\n") + 1`` entries and starts with 0.

    Args:
        text: The full file content.

    Returns:
        One offset per line: ``offsets[i]`` is the character position where
        line ``i`` (0-based) starts. ``""`` returns ``[0]``.
    """
```

Examples (tests assert them verbatim): `""` → `[0]`; `"a\n"` → `[0, 2]`;
`"a\nbb\nccc"` → `[0, 2, 5]`.

```python
def read_file_text(path: Union[str, Path]) -> str:
    """Read a text file as UTF-8, replacing undecodable bytes.

    Opens the file in text mode with default newline handling (universal
    newlines, same as ``Path.read_text()``), inside a context manager.

    Args:
        path: File to read.

    Returns:
        The decoded file content.

    Raises:
        ValueError: If the file cannot be read; the message includes the
            offending path.
    """
```

Policy notes (do not change):

- `encoding="utf-8"`, `errors="replace"`: one bad byte must never crash the
  pipeline (Phase 8 edge case; cheap to do now).
- Default newline handling: `Path.read_text()` convention. Only 4 corpus files
  contain CRLF, and the IoU >= 0.05 recall bar makes any 1-char-per-line offset
  shift irrelevant; keep ONE policy here so Phase 7 (reading retrieved spans
  back from the original files) stays consistent with indexing.
- All raw `OSError`/`UnicodeError` are wrapped as `ValueError` with the path in
  the message. Nothing else may escape.

```python
def sliding_window(
    text: str,
    file_path: str,
    start: int,
    end: int,
    max_chunk_size: int,
    overlap: int = 0,
) -> Iterator[Chunk]:
    """Split [start, end) into chunks of at most max_chunk_size characters.

    Windows are laid out left to right: the first window starts at ``start``,
    and each next window starts exactly ``overlap`` characters before the
    previous window's end. Every yielded chunk has ``kind="sliding"``.

    Args:
        text: The full file content.
        file_path: Path recorded on every yielded chunk.
        start: Inclusive start offset.
        end: Exclusive end offset.
        max_chunk_size: Maximum number of characters per chunk.
        overlap: Characters shared between consecutive windows.

    Yields:
        Chunks covering [start, end), in order.

    Raises:
        ValueError: On invalid ranges or parameters (see below).
    """
```

Parameter validation (before yielding anything):
`max_chunk_size <= 0`, `overlap < 0`, `overlap >= max_chunk_size`,
`start < 0`, `end > len(text)`, `start > end` all raise `ValueError`.
`start == end` is valid and yields nothing.
`overlap == max_chunk_size - 1` is valid (progress is guaranteed because the
step is always >= 1).

Guarantees per yielded chunk: `0 <= c.start < c.end <= len(text)`,
`c.end - c.start <= max_chunk_size`, `text[c.start:c.end] == c.text`, and for
consecutive chunks `prev.end - cur.start == overlap`.

This is deliberately a **character-based** window, not line-aligned.
`build_plan.md` sketched a line-based fallback for broken Python files; we
deviate on purpose: line alignment plus overlap has fiddly progress edge cases
(a single very long line), while character windows preserve the <= 2000
guarantee unconditionally and broken-syntax files are rare and already
unparseable. Record this decision in the Phase 8 README ("Design decisions").

```python
def validate_chunks(
    text: str,
    chunks: Sequence[Chunk],
    max_chunk_size: int,
) -> None:
    """Raise ValueError unless every chunk satisfies the chunking invariants.

    Checks, per chunk: slice consistency (``text[start:end] == text_field``),
    ``0 <= start < end <= len(text)``, ``end - start <= max_chunk_size``.
    Checks, across chunks: strictly increasing ``start`` values (file order,
    no stagnation).

    Args:
        text: The full file content the chunks were cut from.
        chunks: The chunks to validate.
        max_chunk_size: The limit that must hold for every chunk.

    Raises:
        ValueError: On the first violation; the message names the invariant
            and the offending chunk index.
    """
```

This function is the defense-in-depth for the moulinette's hard 2000-character
rule: the Phase 3 indexer and the Phase 4 retriever will call it before
persisting/returning anything, so an over-long chunk can never silently poison
an entire evaluation run.

---

## 4. `src/chunking/python_chunker.py` — exact interface

```python
def chunk_python(
    text: str,
    file_path: str,
    max_chunk_size: int,
    overlap: int = 0,
) -> List[Chunk]:
    """Chunk Python source into AST units with exact character offsets.

    Args:
        text: The full file content.
        file_path: Path recorded on every produced chunk.
        max_chunk_size: Maximum number of characters per chunk.
        overlap: Characters shared between consecutive sliding windows.

    Returns:
        Chunks in file order, covering the whole file.

    Raises:
        ValueError: On invalid ``max_chunk_size``/``overlap`` (same rules as
            ``sliding_window``).
    """
```

### 4.1 Required behaviour

1. Validate `max_chunk_size`/`overlap` first (same rules as
   `sliding_window`), then `if not text: return []`.
2. Parse with `ast.parse(text)` inside `try/except (SyntaxError, ValueError)`
   (`ast.parse` also raises `ValueError` on source containing NUL bytes). On
   failure OR when `Module.body` is empty (e.g. a comments-only file), fall
   back to
   `list(sliding_window(text, file_path, 0, len(text), max_chunk_size, overlap))`
   — all chunks `kind="sliding"`, full coverage.
3. Otherwise partition the module (rule P1), then emit one or more chunks per
   partition span (rule E1). Chunks come back in file order and the union of
   all spans is exactly `[0, len(text))`.

### 4.2 Rule P1 — module partition (MANDATORY semantics)

Walk `Module.body` (top-level statements only; nested defs are NOT promoted —
no duplicate coverage). For each node define:

- `end_line(node)` = `node.end_lineno` (treat `end_lineno is None` — typed
  optional by mypy even though CPython >= 3.8 always sets it — as
  `node.lineno`, i.e. a single-line node),
- `after(node)` = character offset of the start of the line FOLLOWING
  `end_line(node)`, via `line_start_offsets`; `len(text)` if that line does
  not exist.

Then the partition spans are:

```
span_0 = [0, after(n_0))
span_i = [after(n_{i-1}), after(n_i))          for 0 < i < k
span_k = [after(n_{k-1}), len(text))
```

Consequences (all required): the module docstring, leading comments and
imports land in early chunks; interstitial comments/blank lines between two
top-level nodes attach to the FOLLOWING node's chunk (a comment above `def foo`
usually documents `foo`); trailing content after the last node attaches to the
last chunk; decorators sit inside the decorated node's span. Coverage is exact
and gaps are impossible.

### 4.3 Rule E1 — emitting a partition span `[cs, ce)`

- `ce - cs <= max_chunk_size` → one `Chunk(kind="python")`.
- Larger span whose node is a `ClassDef` → oversized class split (rule P2).
- Any other oversized span (function, `if`, big import block, ...) →
  `list(sliding_window(text, file_path, cs, ce, max_chunk_size, overlap))`.
  Nested functions inside an oversized function are NOT promoted (keep it
  simple; sliding windows are enough for the rare giant top-level function).

### 4.4 Rule P2 — oversized class split

Let the class live in span `[cs, ce)` with body statements
`m_0 .. m_{j-1}` (any statement type: methods, assigns, nested classes...).
Define `node_start(m)` = start of the line of `m`'s FIRST DECORATOR, or of
`m.lineno` when undecorated (use `line_start_offsets`; decorators must not be
orphaned from their member). Then:

```
header  = [cs, node_start(m_0))                       # signature + docstring
member_i = [node_start(m_i), node_start(m_{i+1}))     for i < j-1
member_{j-1} = [node_start(m_{j-1}), ce)
```

The pieces are consecutive and cover `[cs, ce)` exactly. Each piece follows
rule E1, except that an oversized piece that is itself a (nested) `ClassDef`
also splits by its members — recurse at most like this; any oversized member
(method, nested class, assign) becomes a sliding window.
Note: comments above the first member attach to the header chunk (documented
quirk of P2); comments between two members attach to the following member.

### 4.5 mypy --strict traps (solve them, do not suppress them)

- `node.end_lineno` / `node.lineno` are `int | None` in typeshed → explicit
  `None` handling as described in P1.
- `decorator_list` exists only on `FunctionDef`/`AsyncFunctionDef`/`ClassDef`
  → `isinstance` branches, no blind `getattr`.
- Do not add `# type: ignore` unless `make lint-strict` demands it and the
  ignore is justified in your report.

---

## 5. `src/chunking/markdown_chunker.py` — exact interface

```python
def chunk_markdown(
    text: str,
    file_path: str,
    max_chunk_size: int,
    overlap: int = 0,
) -> List[Chunk]:
    """Chunk Markdown/text into header sections with exact character offsets.

    Args:
        text: The full file content.
        file_path: Path recorded on every produced chunk.
        max_chunk_size: Maximum number of characters per chunk.
        overlap: Characters shared between consecutive sliding windows.

    Returns:
        Chunks in file order, covering the whole file.

    Raises:
        ValueError: On invalid ``max_chunk_size``/``overlap`` (same rules as
            ``sliding_window``).
    """
```

Applies to `.md`, `.txt`, `.rst` (dispatched in Phase 3's `__init__`).
Used for this phase on `.md`, `.txt`, `.rst` files.

### 5.1 Required behaviour

1. Validate params, then `if not text: return []`.
2. Detect **header lines** while tracking fenced code blocks:
   - a header line matches the regex `^ {0,3}#{1,6}(?:\s|$)` (ATX headers,
     up to 3 leading spaces allowed),
   - but a line inside a fenced code block is NEVER a header. Fences open at
     a line whose stripped form starts with three or more `` ` `` or `~`
     characters and close at the next line whose stripped form starts with
     the same character repeated at least as many times. A simple open/close
     counter is enough; full CommonMark fence nesting is NOT required.
     This matters: vLLM docs embed Python snippets whose `# comment` lines
     sit at column 0 and would otherwise be mistaken for headers.
   - Setext underlines (`Title` / `=====`) are NOT special: they stay plain
     text. YAML front matter (`---`) is NOT special either.
3. Partition at header line starts: sections are the spans between
   consecutive header starts; the first section starts at offset 0 (content
   before the first header, if any, forms its own leading section); the last
   section ends at `len(text)`. Sections are consecutive — full coverage,
   no gaps.
4. Emit each section (rule M1).
5. `overlap` is passed through ONLY to `sliding_window` calls. Paragraph
   groups never overlap. (Phase 6 tuning note: overlap only affects
   sliding-window chunks.)

### 5.2 Rule M1 — emitting a section `[ss, se)`

- `se - ss <= max_chunk_size` → one `Chunk(kind="markdown")`. The chunk text
  therefore always contains its own header line.
- Larger section → split by paragraphs: a *paragraph* is a maximal run of
  consecutive non-blank lines (blank = whitespace-only line). Greedily group
  consecutive paragraphs while the group span stays `<= max_chunk_size`;
  flush a group as one `Chunk(kind="markdown")` when adding the next paragraph
  would overflow. A single paragraph longer than `max_chunk_size` is flushed
  alone and then split with
  `sliding_window(...)` (those chunks get `kind="sliding"`).

### 5.3 Design decision (record it, do not "fix" it)

The chunk text is ALWAYS the raw slice `text[start:end]`. We do NOT prepend
ancestor headings (`# Parent / ## Child` context) to sub-chunk text, even
though `build_plan.md` suggested it, because synthesizing text would break the
slice invariant that retrieval offsets, generation context extraction and the
moulinette all rely on. Heading hierarchy is recovered later at INDEX time
(Phase 3 will boost header/path tokens); chunk integrity wins over inline
context.

---

## 6. `src/chunking/__init__.py` — dispatcher (exact interface)

```python
"""Chunking strategies for code and prose."""
from pathlib import Path
from typing import List, Tuple, Union

from src.chunking.base import Chunk
from src.chunking.markdown_chunker import chunk_markdown
from src.chunking.python_chunker import chunk_python

SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".py", ".md", ".txt", ".rst")


def chunk_file(
    path: Union[str, Path],
    max_chunk_size: int,
    overlap: int = 0,
) -> List[Chunk]:
    """Read a file and chunk it with the strategy matching its extension.

    Every produced chunk records ``file_path`` as the POSIX-normalized form
    of ``path`` (via ``to_posix_path``). Callers MUST pass paths relative to
    the project root (e.g. ``data/raw/vllm-0.10.1/vllm/x.py``): the grader
    compares ``file_path`` verbatim, so the Phase 3 indexer is responsible
    for rooting every path at ``data/raw/`` before calling this function.

    Args:
        path: File to chunk.
        max_chunk_size: Maximum number of characters per chunk.
        overlap: Characters shared between consecutive sliding windows.

    Returns:
        Chunks in file order, covering the whole file.

    Raises:
        ValueError: If the extension is not supported (message names the
            extension and the supported set), the file cannot be read, or
            the parameters are invalid.
    """
```

Behaviour:

- Extension check FIRST (cheapest, clearest error), on
  `Path(path).suffix.lower()` (case-insensitive).
- `.py` → `chunk_python(text, posix_path, max_chunk_size, overlap)`.
- `.md`, `.txt`, `.rst` → `chunk_markdown(text, posix_path, ...)`.
- Anything else (including no extension at all) → `ValueError`. The Phase 3
  indexer filters by `SUPPORTED_EXTENSIONS` before calling, so this only
  fires on misuse.
- Missing/unreadable file → `ValueError` from `read_file_text` (already
  wrapped, path in message).

Re-export for convenience (imports used by later phases):

```python
__all__ = ["Chunk", "SUPPORTED_EXTENSIONS", "chunk_file"]
```

`validate_chunks` and `read_file_text` are imported directly from
`src.chunking.base` where needed (indexer in Phase 3, generator in Phase 7).
`SUPPORTED_EXTENSIONS` is the single source of truth for supported suffixes;
Phase 3's `src/config.py` must import it, not redefine it.

---

## 7. Tests — `tests/test_chunking.py` (you write them, they must pass)

Style constraints (same as Phase 1): typed parameters, PEP 257 docstrings in
English on every test function, flake8 max line 79, no prints, `tmp_path`
fixtures for synthetic files. Reuse a small helper for the coverage check
(merge sorted spans and assert the union equals `[0, len(text))`).

### 7.1 `Chunk` dataclass and constants (cases 1–5)

| # | Case | Expectation |
|---|------|-------------|
| 1 | Valid construction + `frozen=True` + equality | identical chunks compare `==`; attribute assignment raises `dataclasses.FrozenInstanceError` |
| 2 | `file_path=""` | `ValueError` |
| 3 | `start=-1` | `ValueError` |
| 4 | `end <= start` | `ValueError` |
| 5 | `kind="banana"` | `ValueError` |

### 7.2 `line_start_offsets` and `read_file_text` (cases 6–12)

| # | Case | Expectation |
|---|------|-------------|
| 6 | `line_start_offsets("")` | `[0]` |
| 7 | `line_start_offsets("a\n")` | `[0, 2]` |
| 8 | `line_start_offsets("a\nbb\nccc")` | `[0, 2, 5]` |
| 9 | `read_file_text` on tmp UTF-8 file with accents/emoji | exact content round-trip |
| 10 | `read_file_text` on missing path | `ValueError` (not `FileNotFoundError`), message contains the path |
| 11 | `read_file_text` on tmp file containing one invalid UTF-8 byte | no crash; `\ufffd` present in result |
| 12 | `read_file_text` uses a context manager | file handle not left open (e.g. assert via `psutil`-free trick: read twice, or spy on `builtins.open` returning a mock with `__enter__` — any reasonable approach) |

### 7.3 `sliding_window` and `validate_chunks` (cases 13–21)

| # | Case | Expectation |
|---|------|-------------|
| 13 | Exact division: `len=10, max=5, overlap=0` | 2 chunks, adjacent (`c0.end == c1.start`), `kind="sliding"` |
| 14 | Remainder: `len=12, max=5, overlap=0` | 3 chunks of 5/5/2, full coverage |
| 15 | `overlap=2`, awkward `len=23, max=7` | every chunk ≤ 7; `prev.end - cur.start == 2` for consecutive chunks; `text[c.start:c.end] == c.text` for all |
| 16 | `max_chunk_size=0` | `ValueError` |
| 17 | `overlap < 0` or `overlap >= max_chunk_size` | `ValueError` |
| 18 | `start == end` | yields nothing (list is empty) |
| 19 | `start < 0` or `end > len(text)` or `start > end` | `ValueError` |
| 20 | `validate_chunks` on a good chunk list | returns `None`, no raise |
| 21 | `validate_chunks` on a tampered list (chunk with `end > len(text)`; and separately: out-of-order starts) | `ValueError` naming the violation |

### 7.4 `chunk_python` (cases 22–32)

Build sources inline as strings (keep them small) or via `tmp_path`.

| # | Case | Expectation |
|---|------|-------------|
| 22 | Module with `import os` + two simple defs (max big) | chunks in file order; full coverage; def chunks `kind="python"`; some chunk contains the import line |
| 23 | Decorated function (`@decorator` above `def`) | the function's chunk text starts at the `@` line |
| 24 | Small class with 2 methods, max big | exactly one chunk, `kind="python"`, containing `class` and both `def`s |
| 25 | Same class, `max_chunk_size` small (~80–120) | ≥ 3 chunks; a header chunk containing `class `; each member covered exactly once (no duplicated spans); all invariants; full coverage |
| 26 | Oversized top-level function | ≥ 2 chunks via sliding window; all ≤ max; full coverage |
| 27 | `SyntaxError` source (e.g. `def broken(:`) | ≥ 1 chunk, all `kind="sliding"`, full coverage, invariants hold |
| 28 | Comments-only file (`# just a comment`) | ≥ 1 chunk `kind="sliding"`, full coverage (empty AST body → fallback) |
| 29 | Unicode/emoji before a def (`# café 🚀` line, then def) | slice consistency `text[c.start:c.end] == c.text` holds for all chunks (byte-vs-char trap) |
| 30 | Function with a docstring | the function's chunk text contains the docstring |
| 31 | Empty text | `[]` |
| 32 | `max_chunk_size=0` | `ValueError` |

### 7.5 `chunk_markdown` (cases 33–40)

| # | Case | Expectation |
|---|------|-------------|
| 33 | `# A` / `## B` / `## C` sections (max big) | 3 chunks, each containing exactly its own header; all `kind="markdown"`; full coverage |
| 34 | Prose before the first header | first chunk starts at 0 and ends at the first header's line start |
| 35 | Fenced ```` ```python ```` block containing a `# comment` at column 0 | the comment line does NOT create a chunk boundary (chunk count matches the real headers only); full coverage |
| 36 | Oversized section with 3 paragraphs | several chunks all ≤ max; the first sub-chunk contains the section header; all sub-chunks except any sliding-window ones are `kind="markdown"`; full coverage |
| 37 | Single paragraph longer than max | sliding-window chunks `kind="sliding"`, all ≤ max, full coverage |
| 38 | Headerless multi-paragraph text | groups `kind="markdown"`, full coverage |
| 39 | Setext-style `Title\n=====\nbody` | no crash; invariants + full coverage (treated as plain text) |
| 40 | Empty text → `[]`; `max_chunk_size=0` → `ValueError` | as stated |

### 7.6 `chunk_file` dispatcher (cases 41–46)

| # | Case | Expectation |
|---|------|-------------|
| 41 | tmp `.py` file | python chunks; `chunk.file_path == to_posix_path(str(tmp_file))` |
| 42 | tmp `.md`, `.txt`, `.rst` files | markdown chunks (loop or parametrize over the three suffixes) |
| 43 | tmp `.json` file (any unsupported extension) | `ValueError` mentioning the extension and the supported set |
| 44 | Missing `.py` path | `ValueError`, not `FileNotFoundError` |
| 45 | `SUPPORTED_EXTENSIONS` | equals `(".py", ".md", ".txt", ".rst")` |
| 46 | `.PY` / `.MD` (uppercase suffix) | dispatched correctly (case-insensitive) |

### 7.7 Real corpus smoke tests (cases 47–48)

Guard with `pytest.mark.skipif(not Path(...).exists(), ...)` so the suite
passes on a clean checkout, like in Phase 1.

| # | Case | Expectation |
|---|------|-------------|
| 47 | `data/raw/vllm-0.10.1/vllm/entrypoints/llm.py`, max 2000 | `validate_chunks` passes; merged coverage == `len(text)`; ≥ 2 chunks |
| 48 | `data/raw/vllm-0.10.1/docs/serving/openai_compatible_server.md`, max 2000 | same; ≥ 2 chunks (file has headers) |

Total: at least 48 test functions (merging trivially related asserts into one
test is fine, but every listed behaviour must be covered).

---

## 8. Implementation order (do it in this order)

1. `src/chunking/base.py` — `Chunk`, `CHUNK_KINDS`, `line_start_offsets`,
   `read_file_text`, `sliding_window`, `validate_chunks`.
2. Cases 1–21 of `tests/test_chunking.py` — write and pass them.
3. `src/chunking/python_chunker.py` — partition rules P1/E1/P2 + fallback.
4. Cases 22–32 — write and pass them.
5. `src/chunking/markdown_chunker.py` — headers + fences + paragraphs (M1).
6. Cases 33–40 — write and pass them.
7. `src/chunking/__init__.py` — dispatcher + `SUPPORTED_EXTENSIONS`.
8. Cases 41–46 — write and pass them.
9. Real-corpus cases 47–48.
10. Run the full verification suite (section 9).

---

## 9. Verification (all commands must exit 0)

```bash
# 1. Unit tests — the 23 Phase-1 tests must stay green too
uv run pytest tests/ -v

# 2. Lint, exactly as the Makefile does
make lint
make lint-strict

# 3. Real-corpus smoke test (not a unit test, just for you)
uv run python -c "
from src.chunking import chunk_file
from src.chunking.base import read_file_text, validate_chunks
files = [
    'data/raw/vllm-0.10.1/vllm/entrypoints/llm.py',
    'data/raw/vllm-0.10.1/docs/serving/openai_compatible_server.md',
    'data/raw/vllm-0.10.1/examples/offline_inference/basic/README.md',
]
for f in files:
    text = read_file_text(f)
    chunks = chunk_file(f, max_chunk_size=2000)
    validate_chunks(text, chunks, 2000)
    covered = sum(c.end - c.start for c in chunks)
    biggest = max(c.end - c.start for c in chunks)
    print(f, 'chunks=', len(chunks), 'max=', biggest, 'full=', covered == len(text))
"
```

Expected smoke output shape (numbers are yours to report; `full=True` for all
three lines, `max` <= 2000 for all three lines):

```
...llm.py chunks= <n> max= <n> full= True
...openai_compatible_server.md chunks= <n> max= <n> full= True
...README.md chunks= <n> max= <n> full= True
```

---

## 10. Definition of Done (check every box before reporting back)

- [ ] Exactly four new modules under `src/chunking/` plus
      `tests/test_chunking.py`; nothing else created or modified.
- [ ] `Chunk` is a frozen dataclass with the five fields of section 3.3 and
      runtime validation (empty path, negative/inverted range, unknown kind).
- [ ] `line_start_offsets`, `read_file_text`, `sliding_window`,
      `validate_chunks` match the exact signatures of section 3.4.
- [ ] For every chunk of every strategy:
      `0 <= start < end <= len(text)`, `end - start <= max_chunk_size`,
      `text[start:end] == chunk.text` (enforced by `validate_chunks`).
- [ ] Python chunker: full-coverage partition (P1), oversized class split by
      members (P2), `SyntaxError`/empty-body fallback with `kind="sliding"`,
      decorator-aware spans, unicode-safe offsets.
- [ ] Markdown chunker: ATX headers with fenced-code-block awareness,
      paragraph grouping for oversized sections, sliding-window last resort,
      full coverage, `kind` values correct.
- [ ] Dispatcher: 4 extensions (case-insensitive), `ValueError` otherwise,
      `file_path` recorded as POSIX-normalized input path,
      `SUPPORTED_EXTENSIONS` exported.
- [ ] At least 48 tests cover cases 1–48 and pass.
- [ ] The 23 Phase-1 tests still pass (`uv run pytest tests/ -v` all green).
- [ ] `make lint` and `make lint-strict` both exit 0.
- [ ] Real-corpus smoke test prints `full=True` and `max<=2000` for all three
      files.
- [ ] Every public class and function has a PEP 257 docstring in English
      (Google style).
- [ ] Zero inline comments.
- [ ] `mypy --strict` passes on `src/` and `tests/` without new ignores.

---

## 11. Reporting back

When done, reply with:

1. Files created and line counts.
2. The test names (one-line list, numbered as in section 7 where possible).
3. Smoke numbers from section 9: for each of the three real files, the chunk
   count, the maximum chunk size and `full=True`.
4. Design decisions you made beyond the spec (if any) and why.
5. Any assumption you had to make.
6. Confirmation that all commands in section 9 exited 0, with the pytest
   summary line (e.g. `71 passed in 1.2s`).

Do not start Phase 3 (tokenizer, BM25, indexer, CLI) until this phase is
reviewed and validated.
