# Phase 3 — Lexical indexing: tokenizer, BM25F and the `index` command

> **Agent:** `rag-dev` (implementation may be delegated to `retrieval-specialist`).
> **Source of truth:** `en.subject.md` (sections VI.1 Indexing the codebase,
> VI.6 Command-Line Interface, V.1/V.2/V.4 common rules) and
> `build_plan.md` (FASE 3).
> **Status of Phases 0–2:** DONE and verified on 2026-10-15; re-checked on
> 2026-10-19: 71 tests green, `make lint-strict` exit 0, and the full corpus
> chunks into 45,227 spans, every one of them `<= 2000` characters.
> **You must read before coding:** `en.subject.md` sections VI.1 (indexing the
> whole corpus takes at most 5 minutes; at least one of TF-IDF/BM25 is
> mandatory; `--max_chunk_size` defaults to 2000), VI.6 (Python Fire CLI,
> `uv run python -m src <command>`, every path a configurable CLI argument,
> degenerate inputs must never produce a traceback), V.4 (uv, pydantic, tqdm)
> and V.2 (the exact Makefile lint commands).
> **Deliberate deviation from `build_plan.md` (approved, see 7.1):** the plan
> sketched the path/header "boost" as token duplication inside the document
> text. This phase implements **BM25F** instead: the same Okapi BM25 machinery
> with per-field length normalization and explicit field weights. It still
> satisfies the mandatory "TF-IDF or BM25" requirement of section VI.1.

---

## 1. Objective

Build the lexical index that every later stage queries: a code-aware tokenizer,
an **Okapi BM25F** inverted index over three chunk fields (body, header,
path), deterministic persistence under `data/processed/`, and the `index`
command of the Fire CLI.

Measured facts about the real corpus (2026-10-19, `overlap=0`,
`max_chunk_size=2000`, produced by the Phase-2 chunker):

| Fact | Value |
|---|---|
| supported files | 1969 (1763 `.py`, 178 `.md`, 28 `.txt`, 0 `.rst`) |
| corpus size | 20.2 MB |
| chunks | 45,227 (43,506 python + 1,721 markdown/txt), max span 2000 |
| full-corpus chunking | 3.8 s |
| BM25F prototype (tokenizer + postings + build) | 9.1 s wall, 191 MB peak RSS |
| vocabulary / postings | ~59k tokens / ~2.0M postings (3 fields) |
| persisted arrays | `savez_compressed` 0.6 s, ~3 MB, load 0.04 s |
| retrieval throughput | 5 queries in 0.096 s → ~3.9 s projected for 200 (limit 90 s) |

At the end of this phase you must be able to:

1. Run `uv run python -m src index --max_chunk_size 2000` and get
   `data/processed/index.npz` + `data/processed/index_meta.json` in well under
   the hard 5-minute limit (prototype measured ~10 s total).
2. Reload the index and search a query, getting plausible hits whose
   `file_path` is verbatim `data/raw/vllm-0.10.1/...` and whose spans are all
   `<= 2000` characters.
3. Pass `pytest`, `flake8` and `mypy --strict` with zero errors (this requires
   the two `pyproject.toml` changes of section 3: neither `fire` nor `tqdm`
   ships type information).

---

## 2. Files to create and modify

```
src/
├── config.py                 # defaults + pydantic IndexConfig (NEW)
├── models.py                 # MODIFY: append IndexedChunk (additive only)
├── cli.py                    # Fire CLI: class Rag with the index command
├── __main__.py               # entry point: fire.Fire(Rag)
└── indexing/
    ├── __init__.py           # re-exports for later phases
    ├── tokenizer.py          # tokenize(): code-aware, lowercase
    ├── bm25.py               # BM25Index: BM25F scoring + CSR arrays
    ├── storage.py            # IndexMeta, save_index, load_index
    └── indexer.py            # index_corpus(): walk, chunk, index, persist
tests/
├── conftest.py               # shared tiny_project fixture
├── test_models.py            # MODIFY: append 4 cases for IndexedChunk
├── test_tokenizer.py         # NEW
├── test_bm25.py              # NEW
├── test_storage.py           # NEW
├── test_indexer.py           # NEW
└── test_cli.py               # NEW
pyproject.toml                # MODIFY: types-tqdm + mypy override (section 3)
```

Do NOT touch: `src/chunking/**` (already validated), `src/io_utils.py`
(importing from it is expected), `Makefile`, `README.md`, `.flake8`,
`data/raw/**`. No new runtime dependencies: stdlib + `numpy` + `pydantic` +
`tqdm` + `fire`, all already present. Do NOT implement `search` /
`search_dataset` (Phase 4) nor `retriever.py`.

---

## 3. `pyproject.toml` changes (do this first)

`mypy --strict` currently fails on `import fire` and `import tqdm` with
`error: ... missing library stubs or py.typed marker`. Two changes fix it:

1. Add the community stubs for tqdm as a **dev** dependency (runtime is
   unchanged; the stubs are a separate distribution):

   ```bash
   uv add --dev types-tqdm
   ```

2. `fire` has no stubs anywhere on PyPI, so scope a mypy override to that
   single module (append to the end of `pyproject.toml`):

   ```toml
   [[tool.mypy.overrides]]
   module = ["fire"]
   ignore_missing_imports = true
   ```

Verify immediately that the suite is still green and strict-clean:

```bash
make lint-strict   # must exit 0 before writing any Phase-3 code
```

This override is a documented design decision for the README: it is limited to
one untyped third-party module and does not weaken checking of `src/`.

---

## 4. `src/config.py` — exact interface

Single source of truth for every default. `src/cli.py` and `src/indexing/*`
import these names; no default literal may be repeated anywhere else.

```python
"""Default configuration for the indexing pipeline and its CLI."""
from typing import Literal  # not needed, see note

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.chunking import SUPPORTED_EXTENSIONS

DEFAULT_CORPUS_PATH: str = "data/raw"
DEFAULT_PROCESSED_PATH: str = "data/processed"
DEFAULT_MAX_CHUNK_SIZE: int = 2000
DEFAULT_OVERLAP: int = 0
DEFAULT_K1: float = 1.5
DEFAULT_B: float = 0.75
DEFAULT_WEIGHT_BODY: float = 1.0
DEFAULT_WEIGHT_HEADER: float = 2.0
DEFAULT_WEIGHT_PATH: float = 2.0
```

(The `Literal` import in the snippet above is only a placeholder for "import
what you use": `config.py` itself needs no `typing` imports.)

```python
class IndexConfig(BaseModel):
    """Settings that define how a lexical index is built and loaded."""

    model_config = ConfigDict(extra="ignore")

    corpus_path: str = Field(default=DEFAULT_CORPUS_PATH)
    processed_path: str = Field(default=DEFAULT_PROCESSED_PATH)
    max_chunk_size: int = Field(default=DEFAULT_MAX_CHUNK_SIZE, ge=1)
    overlap: int = Field(default=DEFAULT_OVERLAP, ge=0)
    k1: float = Field(default=DEFAULT_K1, gt=0.0)
    b: float = Field(default=DEFAULT_B, ge=0.0, le=1.0)
    weight_body: float = Field(default=DEFAULT_WEIGHT_BODY, ge=0.0)
    weight_header: float = Field(default=DEFAULT_WEIGHT_HEADER, ge=0.0)
    weight_path: float = Field(default=DEFAULT_WEIGHT_PATH, ge=0.0)

    @model_validator(mode="after")
    def _overlap_fits_chunk(self) -> "IndexConfig":
        """Reject overlap values that would stall the sliding window."""
```

Requirements:

- `__all__ = ["DEFAULT_...", "IndexConfig", "SUPPORTED_EXTENSIONS"]`.
- `SUPPORTED_EXTENSIONS` is re-exported (imported from `src.chunking`, never
  redefined): `src/indexing/indexer.py` imports it from here, satisfying the
  `build_plan.md` note that config centralises extensions.
- The `_overlap_fits_chunk` validator raises
  `ValueError("overlap must be smaller than max_chunk_size")` when
  `overlap >= max_chunk_size` (same rule as `sliding_window` in Phase 2, but
  caught here — before any file is touched).
- `pydantic.ValidationError` subclasses `ValueError`, so the CLI error
  contract of section 11 catches configuration mistakes with one `except`.
- Rationale for validating at the boundary: the CLI builds `IndexConfig` from
  untrusted argv, and `storage.py` re-loads it from JSON later; one validated
  object replaces ad-hoc checks everywhere.

---

## 5. `src/models.py` — additive change: `IndexedChunk`

Append exactly one class; do not modify the eight existing models. Add
`Literal` to the `typing` import.

```python
class IndexedChunk(MinimalSource):
    """A persisted chunk location tagged with the strategy that produced it."""

    kind: Literal["python", "markdown", "sliding"]
```

Purpose: the index must remember, for every doc id, where the chunk lives
(`file_path`, `first_character_index`, `last_character_index` — already
validated by `MinimalSource`) and which chunking strategy made it (useful to
diagnose recall failures in Phase 6: sliding-window chunks are arbitrary
slices). The payload text is intentionally NOT persisted: Phase 7 re-reads
`text[first:last]` from the original file with `read_file_text`, so the corpus
stays the single source of truth for content.

Conventions this class makes explicit (cross-phase, record them):

- `first_character_index = chunk.start`, `last_character_index = chunk.end`
  with `end` **exclusive** (Python-slice semantics: the span is exactly
  `text[first:last]`). A chunk of 2000 characters therefore reports
  `last - first == 2000`, still within the moulinette limit under the
  difference interpretation the project has used since Phase 1.
- The `kind` values are the `CHUNK_KINDS` of `src/chunking/base.py`; a test
  (case 3) asserts the `Literal` and the tuple never drift apart.

---

## 6. `src/indexing/tokenizer.py` — exact interface

```python
"""Code-aware whitespace tokenizer for chunks, paths and queries."""
import re
from typing import List

_WORD_SEP = re.compile(r"\W+", flags=re.UNICODE)
_CAMEL_SEP = re.compile(
    r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])",
)


def tokenize(text: str) -> List[str]:
    """Split text into lowercase code-aware tokens.

    Splits on runs of non-word characters, emits each word in lowercase
    and additionally emits its snake_case/camelCase subtokens (each in
    lowercase, skipped when they equal the whole word). Order follows the
    input; repetitions are preserved because term frequency needs them.

    Args:
        text: Any text: chunk body, header line, file path or question.

    Returns:
        Lowercase tokens, never empty strings. ``""`` and separator-only
        input return ``[]``.
    """
```

Normative algorithm (tests assert the exact lists):

1. `for piece in _WORD_SEP.split(text)`; skip empty pieces.
2. Append `piece.lower()`.
3. For each `part` in `_CAMEL_SEP.split(piece)`, for each `sub` in
   `part.split("_")`: if `sub` is non-empty and `sub.lower() != piece.lower()`,
   append `sub.lower()`.

Properties the tests enforce:

- `\W` with `re.UNICODE` keeps accented letters and `_` (so
  `skip_sampler_cpu_output` survives as one piece to be sub-split), while
  emoji and punctuation act as separators.
- Every output token `t` satisfies `t == t.lower()` and `t != ""`.
- No deduplication, no stopword removal, no stemming, no length filter.
  Rationale: questions quote identifiers verbatim (`skip_sampler_cpu_output`)
  and code is full of words that are stopwords in English but structural in
  Python (`for`, `is`, `import`, `self` — the corpus contains them in >9k
  chunks each). The SAME function indexes documents and tokenizes queries;
  any asymmetry there silently destroys recall.

Worked examples (tests assert them verbatim):

| input | output |
|---|---|
| `""` | `[]` |
| `"-- ++ "` | `[]` |
| `"Hello World"` | `["hello", "world"]` |
| `"skip_sampler_cpu_output"` | `["skip_sampler_cpu_output", "skip", "sampler", "cpu", "output"]` |
| `"getUserName"` | `["getusername", "get", "user", "name"]` |
| `"LLMEngine"` | `["llmengine", "llm", "engine"]` |
| `"parseHTTPResponse_data"` | `["parsehttpresponse_data", "parse", "http", "response", "data"]` |
| `"v0.10.1"` | `["v0", "10", "1"]` |
| `"café naïve"` | `["café", "naïve"]` |
| `"foo foo FOO"` | `["foo", "foo", "foo"]` |
| `"chunk_2_text"` | `["chunk_2_text", "chunk", "2", "text"]` |

---

## 7. `src/indexing/bm25.py` — the BM25F core

### 7.1 Why BM25F instead of the plan's naive path boost

`build_plan.md` (FASE 3, "Boost por ruta") suggested appending `file_path`
and header tokens to the document text so they match more often. That works
crudely and we measured better structure with the standard fix, **BM25F**:

1. **Saturation destroys ad-hoc weights.** Repeating a path token 3 times to
   give it weight 3 does not yield a 3x score: with `k1=1.5` the term
   frequency `1 -> 3` only moves the factor `(k1+1)*tf/(k1+tf)` from `1.00`
   to `1.67`. The "weight" knob would interact with `k1` and be impossible to
   tune independently.
2. **Length normalization must be per field.** A match inside a 10-token
   header line is far stronger evidence than the same token inside a
   600-token body. Concatenating fields into one bag of tokens treats them
   identically; BM25F normalizes each field by its own average length before
   combining, so short fields get the boost they deserve.
3. **Duplication pollutes statistics.** Injected tokens inflate the document
   length used by `b` and distort the term frequency of the body field.
4. **Tuning stays orthogonal for Phase 6:** `k1` (saturation), `b` (length),
   `w_body`/`w_header`/`w_path` (evidence type) move one thing at a time.
5. It is still literally BM25 (Okapi family, same `idf`, same saturation), so
   section VI.1 of the subject is satisfied: we implement BM25, not a
   proprietary variant.

### 7.2 Scoring formulas (normative)

For every token `t`, document `d`, field `f` in `FIELDS`:

```
idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

norm_f(d)   = 1 - b + b * len_f(d) / avgdl_f        # 1.0 when avgdl_f <= 0
tf_hat(t,d) = SUM over f of  w_f * tf_f(t,d) / norm_f(d)
score(t,d)  = idf(t) * tf_hat(t,d) * (k1 + 1) / (k1 + tf_hat(t,d))
score(Q, d) = SUM over the UNIQUE tokens t of tokenize(Q) of score(t, d)
```

- `N` = number of documents; `df(t)` = documents where `t` appears in at
  least one field; `len_f(d)` = token count of field `f` of `d`.
- The `idf` is the smoothed variant: always strictly positive (classic IDF
  turns negative for very common terms, which would rank matching documents
  BELOW non-matching ones).
- Saturation happens **once**, on the aggregated `tf_hat` — that is what
  makes the field weights meaningful (point 1 of 7.1).
- Defaults: `k1=1.5`, `b=0.75`, `w_body=1.0`, `w_header=2.0`, `w_path=2.0`.

Query semantics (mandatory):

1. Deduplicate query tokens, preserving first-occurrence order.
2. Return only documents with `score > 0` — i.e. documents that share at
   least one term with the query. Zero-score documents share nothing and
   would only fill the top-k slots with noise (and make generation context
   worse). A doc containing a query term always scores `> 0` (positive idf,
   positive weights).
3. Sort by `(-score, doc_id)`: ties are broken by ascending doc id, so
   results are deterministic and reproducible across runs.
4. `k <= 0`, an empty/separator-only query, or queries made only of unknown
   terms all return `[]` — never an exception (degenerate-input rule).

### 7.3 Exact interface

```python
"""Okapi BM25F over an inverted index of three chunk fields."""
from typing import Any, Dict, List, Mapping, NamedTuple, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from src.config import (
    DEFAULT_B,
    DEFAULT_K1,
    DEFAULT_WEIGHT_BODY,
    DEFAULT_WEIGHT_HEADER,
    DEFAULT_WEIGHT_PATH,
    IndexConfig,
)
from src.indexing.tokenizer import tokenize

FIELDS: Tuple[str, ...] = ("body", "header", "path")

_BUNDLE_KEYS: Tuple[str, ...] = (
    "vocab",
    "indptr",
    "docs",
    "tf_body",
    "tf_header",
    "tf_path",
    "len_body",
    "len_header",
    "len_path",
    "idf",
)


class ScoredChunk(NamedTuple):
    """A ranked document id with its BM25F score."""

    doc_id: int
    score: float


class BM25Index:
    """In-memory BM25F index with CSR postings, ready to persist.

    Args:
        k1: Term-frequency saturation; must be > 0.
        b: Length normalization in [0, 1].
        weight_body: Weight of the chunk text field.
        weight_header: Weight of the first-non-blank-line field.
        weight_path: Weight of the corpus-relative path field.

    Raises:
        ValueError: On out-of-range parameters.
    """

    def __init__(
        self,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
        weight_body: float = DEFAULT_WEIGHT_BODY,
        weight_header: float = DEFAULT_WEIGHT_HEADER,
        weight_path: float = DEFAULT_WEIGHT_PATH,
    ) -> None:
        """Validate the parameters and start an empty, unbuilt index."""

    @property
    def doc_count(self) -> int:
        """Number of accumulated documents (0 until ``build``)."""

    @property
    def is_built(self) -> bool:
        """Whether ``build`` has frozen the index."""

    @property
    def avgdl(self) -> Dict[str, float]:
        """Average length per field (``{}`` until ``build``)."""

    def add_document(
        self,
        doc_id: int,
        fields: Mapping[str, Sequence[str]],
    ) -> None:
        """Accumulate one document's tokens before ``build``.

        Args:
            doc_id: Must equal the number of documents already added
                (sequential from 0), so doc id ``i`` always refers to the
                ``i``-th persisted chunk record.
            fields: Token sequences keyed by exactly ``FIELDS``. Empty
                sequences are allowed (a punctuation-only chunk).

        Raises:
            ValueError: If the index is already built, a field name is
                missing or extra, or ``doc_id`` breaks the sequence.
        """

    def build(self) -> None:
        """Freeze accumulators into sorted CSR arrays and compute idf/avgdl.

        Vocabulary becomes the lexicographically sorted list of tokens seen
        in any field (determinism), postings stay ascending by doc id, and
        the accumulation dicts are released.

        Raises:
            ValueError: If no document was added or ``build`` already ran.
        """

    def search(self, query: str, k: int) -> List[ScoredChunk]:
        """Return the top-k documents sharing at least one query term.

        Args:
            query: Natural-language question or identifier.
            k: Maximum number of results; ``k <= 0`` returns ``[]``.

        Returns:
            Ranked hits, best first, ties broken by ascending doc id;
            only documents with a strictly positive score.

        Raises:
            ValueError: If the index has not been built.
        """

    def to_bundle(self) -> Dict[str, NDArray[Any]]:
        """Export every array needed to persist the index.

        Raises:
            ValueError: If the index has not been built.
        """

    @classmethod
    def from_bundle(
        cls,
        bundle: Mapping[str, NDArray[Any]],
        avgdl: Mapping[str, float],
        config: IndexConfig,
    ) -> "BM25Index":
        """Reconstruct a built index from persisted arrays.

        Args:
            bundle: Arrays exactly as ``to_bundle`` produced them.
            avgdl: Average field lengths from the metadata file.
            config: Carries ``k1``, ``b`` and the field weights.

        Returns:
            A built index answering ``search`` immediately.

        Raises:
            ValueError: If a key of ``_BUNDLE_KEYS`` is missing, a length
                invariant is violated, ``docs`` is not an integer array,
                ``avgdl`` does not cover exactly ``FIELDS`` with
                non-negative values, or the index would be empty.
        """
```

### 7.4 Data structures

During accumulation (between `add_document` calls), keep one nested dict per
field: `Dict[str, Dict[int, int]]` mapping `token -> {doc_id: tf}`. Measured:
filling it for the whole corpus takes 4.9 s and peaks at 191 MB — simple code
and comfortably inside the budget.

`build()` converts the dicts into **CSR** (compressed sparse row) arrays over
the sorted vocabulary — one contiguous row of postings per token:

| key | dtype | shape | meaning |
|---|---|---|---|
| `vocab` | `str_` | `(V,)` | sorted unique tokens |
| `indptr` | `int64` | `(V+1,)` | row `i` spans `[indptr[i], indptr[i+1])` |
| `docs` | `int32` | `(P,)` | doc ids, ascending inside each row |
| `tf_body` | `int32` | `(P,)` | body term frequency per posting |
| `tf_header` | `int32` | `(P,)` | header term frequency per posting |
| `tf_path` | `int32` | `(P,)` | path term frequency per posting |
| `len_body` | `int64` | `(N,)` | body token count per document |
| `len_header` | `int64` | `(N,)` | header token count per document |
| `len_path` | `int64` | `(N,)` | path token count per document |
| `idf` | `float64` | `(V,)` | precomputed idf per vocabulary row |

(`V` ~59k, `P` ~2.0M, `N` = 45,227 on the real corpus; arrays total ~10 MB
in memory and ~3 MB compressed.)

`search` runs fully vectorized with numpy: per unique query token, locate its
row, build `tf_hat` from the three sliced `tf` arrays divided by the
per-document norms, add `idf * tf_hat * (k1+1)/(k1+tf_hat)` into a
`float64` score vector at the posting positions, then rank. The projected
cost measured with an equivalent pure-Python loop is 3.9 s for 200 queries;
numpy keeps a wide margin under the 90 s requirement (target: < 30 s).

### 7.5 Required behaviour checklist

- Constructor validates `k1 > 0`, `0 <= b <= 1`, `weight_* >= 0`
  (`ValueError` otherwise).
- `add_document` accepts fields exactly equal to `FIELDS` (a missing key or
  an extra key is a `ValueError` naming the offending key); `doc_id` must
  equal `doc_count` at call time.
- `build` requires `doc_count >= 1` and runs at most once.
- `search` requires `is_built` (`ValueError` if not); everything degenerate
  about the QUERY returns `[]` instead (section 7.2, points 2–4).
- `from_bundle` re-creates the `token -> row` lookup from `vocab` and
  validates: every `_BUNDLE_KEYS` entry present; `len(indptr) == V + 1`;
  `indptr[-1] == len(docs) == len(tf_body) == len(tf_header) ==
  len(tf_path)`; `len(idf) == V`; `len(len_*) == N > 0` for the three field
  lengths; `np.issubdtype(docs.dtype, np.integer)`; `avgdl` keys exactly
  `FIELDS` with values `>= 0`. Each violation raises `ValueError` naming the
  broken invariant.
- Two `BM25Index` instances built from the same documents return bitwise
  identical rankings (sorted vocab + deterministic accumulation order).

---

## 8. `src/indexing/storage.py` — exact interface

Two artifacts inside the configured `processed_path`:

- `index.npz` — the `to_bundle()` arrays, `savez_compressed`,
  `allow_pickle=False` when loading (no pickle anywhere: no code execution on
  load, no protocol/version traps, 0.04 s load time).
- `index_meta.json` — pydantic metadata written with the existing
  `save_model` helper, so all Phase-1 error handling (`ValueError` with the
  path in the message, parent directories created) is reused.

```python
"""Persistence for the lexical index: NPZ arrays plus JSON metadata."""
from pathlib import Path
from typing import List, Sequence, Tuple, Union

from pydantic import BaseModel, ConfigDict

from src.config import IndexConfig
from src.indexing.bm25 import BM25Index
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
```

Behaviour requirements:

- `save_index` fails BEFORE touching the directory when the index is unbuilt
  or the record count mismatches; then creates the directory
  (`mkdir(parents=True, exist_ok=True)` inside `try/except OSError`) and
  writes the npz inside `try/except OSError` re-raised as `ValueError`.
- `load_index` wraps `np.load` in a context manager (file handle closed) and
  copies the arrays out of the `NpzFile` before exiting it.
- JSON top-level keys are exactly `format_version`, `config`, `chunks`,
  `avgdl`; npz keys are exactly `_BUNDLE_KEYS`. A version bump must be
  deliberate: the loader compares against `INDEX_FORMAT_VERSION` and names
  both numbers in the error.
- Invariant asserted by the loader and relied on by Phases 4–7:
  **`chunks[i]` describes doc id `i`**.
- `IndexMeta.chunks` entries are pydantic-validated: corrupted offsets
  (`last < first`), empty paths or an unknown `kind` fail at load time with a
  clear message instead of surfacing as recall failures later.

---

## 9. `src/indexing/indexer.py` — exact interface

```python
"""Corpus ingestion: walk, chunk, tokenize, build and persist the index."""
from typing import List

from pydantic import BaseModel, ConfigDict

from src.config import IndexConfig


class IndexStats(BaseModel):
    """Summary of a completed indexing run."""

    model_config = ConfigDict(extra="ignore")

    files_indexed: int
    chunks_indexed: int
    tokens_indexed: int
    elapsed_seconds: float
    index_directory: str


def index_corpus(
    config: IndexConfig,
    show_progress: bool = True,
) -> IndexStats:
    """Build and persist the lexical index for the configured corpus.

    Args:
        config: Corpus path, chunking parameters, BM25 parameters and
            destination directory.
        show_progress: Whether to display the tqdm bars.

    Returns:
        Counts and timings of the run.

    Raises:
        ValueError: On any invalid or unreadable corpus location, an
            empty corpus (no supported files), a chunking/reading
            failure, or an unwritable destination.
    """
```

### 9.1 Corpus resolution (normative; every rule is tested)

All stored `file_path` values must match the ground truth **verbatim**
(`data/raw/vllm-0.10.1/docs/features/lora.md`), so the indexer is strict
about where the corpus may live:

1. `project = Path.cwd().resolve()` — the CLI is always run from the project
   root; `index_corpus` never changes the working directory itself.
2. `requested = Path(config.corpus_path)`; if relative, resolve it against
   `project`, else use it as given.
3. Missing path or not a directory -> `ValueError` naming the configured
   value.
4. `resolved = corpus.resolve()`; if it is not strictly inside `project`
   -> `ValueError` explaining the corpus must be a subdirectory of the
   project root (a corpus outside the project root cannot produce verbatim
   relative paths, and silently writing absolute or truncated paths would
   make every retrieval result miss the ground truth). `corpus_path="."` is
   rejected by the same rule (it would strip the `data/raw/` prefix).
5. `prefix = to_posix_path(resolved.relative_to(project))` — the recorded
   prefix, e.g. `data/raw` or `data/raw/vllm-0.10.1`.
6. Walk `resolved.rglob("*")` in sorted order and keep files whose
   `suffix.lower()` is in `SUPPORTED_EXTENSIONS`; if none, raise
   `ValueError` listing the supported extensions.
7. For each file: `rel = file.relative_to(resolved)` and
   `file_path = to_posix_path(Path(prefix) / rel)`; then call
   `chunk_file(file_path, ...)` with THAT project-relative string, so
   `chunk_file` records it verbatim on every chunk and reads the same bytes
   through the working directory.

### 9.2 Two-phase run (matches the reference output of section VI.1)

Phase A — **Chunking** (tqdm bar `desc="Chunking"`, `unit="file"`,
`disable=not show_progress`):

```python
text = read_file_text(file_path)
chunks = chunk_file(file_path, config.max_chunk_size, config.overlap)
validate_chunks(text, chunks, config.max_chunk_size)
```

Keep `(file_path, chunks)` pairs in memory (~25 MB for the whole corpus).
`validate_chunks` runs again here (defense in depth, Phase 2 section 3.4):
nothing that violates an invariant may ever reach the persistence layer.

Phase B — **Tokenizing** (tqdm bar `desc="Tokenizing"`, `unit="chunk"`,
`total=total_chunks`): flatten the pairs and, for `doc_id` from 0 upward,
build the three fields of each chunk:

- `body` = `tokenize(chunk.text)`,
- `header` = `tokenize(header_line)` where `header_line` is the first
  non-blank line of `chunk.text` (`text.strip().splitlines()[0]`, `""` for
  whitespace-only text). For markdown chunks that line is the section
  header; for python chunks it is the signature/decorator line. This keeps
  the Phase-2 promise that heading context is recovered at index time
  instead of being spliced into the chunk text.
- `path` = `tokenize(file_path)` with the `prefix + "/"` leading part
  stripped first, so the field holds `vllm-0.10.1/docs/serving/x.md` rather
  than the constant `data/raw/` shared by every document. (Tokens that are
  still constant across all documents — e.g. the version in
  `vllm-0.10.1` — get `df == N` and therefore `idf ~ 0`, so they cost
  nothing at query time.)

Then `index.add_document(doc_id, fields)` and append
`IndexedChunk(file_path=file_path, first_character_index=chunk.start,
last_character_index=chunk.end, kind=chunk.kind)`.

Finally: `index.build()`,
`save_index(index, records, config, config.processed_path)`, and return the
`IndexStats` (`tokens_indexed` counts body + header + path tokens;
`elapsed_seconds` from `time.monotonic()`; `index_directory` is
`to_posix_path(config.processed_path)` — the directory as configured, so the
CLI can print `under data/processed/`).

Errors: no `try/except` around the pipeline inside `index_corpus` — every
lower layer already converts its failures to `ValueError` (read, chunk,
config, storage), and section 11's CLI turns those into a clean message.
Programming errors (bugs) must still surface as tracebacks during development.

---

## 10. `src/indexing/__init__.py` — re-exports

```python
"""Lexical indexing: tokenizer, BM25F index, storage and corpus ingestion."""
from src.indexing.bm25 import BM25Index, FIELDS, ScoredChunk
from src.indexing.indexer import IndexStats, index_corpus
from src.indexing.storage import (
    INDEX_ARRAYS_FILENAME,
    INDEX_META_FILENAME,
    INDEX_FORMAT_VERSION,
    IndexMeta,
    load_index,
    save_index,
)
from src.indexing.tokenizer import tokenize

__all__ = [
    "BM25Index",
    "FIELDS",
    "INDEX_ARRAYS_FILENAME",
    "INDEX_FORMAT_VERSION",
    "INDEX_META_FILENAME",
    "IndexMeta",
    "IndexStats",
    "ScoredChunk",
    "index_corpus",
    "load_index",
    "save_index",
    "tokenize",
]
```

Import order matters (dependencies first: `bm25` and `storage` before
`indexer`); importing any submodule directly must work too, because tests do
both.

---

## 11. `src/cli.py` and `src/__main__.py`

```python
"""Command-line interface for the RAG pipeline."""
import sys

from src.config import (
    DEFAULT_B,
    DEFAULT_CORPUS_PATH,
    DEFAULT_K1,
    DEFAULT_MAX_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    DEFAULT_PROCESSED_PATH,
    DEFAULT_WEIGHT_BODY,
    DEFAULT_WEIGHT_HEADER,
    DEFAULT_WEIGHT_PATH,
    IndexConfig,
)
from src.indexing.indexer import index_corpus


class Rag:
    """RAG against the machine: index a corpus and answer questions."""

    def index(
        self,
        corpus_path: str = DEFAULT_CORPUS_PATH,
        processed_path: str = DEFAULT_PROCESSED_PATH,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
        weight_body: float = DEFAULT_WEIGHT_BODY,
        weight_header: float = DEFAULT_WEIGHT_HEADER,
        weight_path: float = DEFAULT_WEIGHT_PATH,
    ) -> None:
        """Build the lexical index over the corpus.

        Args:
            corpus_path: Corpus directory, relative to the project root.
            processed_path: Directory receiving the index artifacts.
            max_chunk_size: Maximum characters per chunk (never
                exceeded; the evaluator rejects longer sources).
            overlap: Characters shared by consecutive sliding windows.
            k1: BM25 term-frequency saturation parameter.
            b: BM25 length-normalization parameter in [0, 1].
            weight_body: BM25F weight of the chunk body field.
            weight_header: BM25F weight of the header field.
            weight_path: BM25F weight of the path field.

        Raises:
            SystemExit: Status 1 after printing ``error: ...`` to stderr
                when the configuration or the corpus is invalid.
        """
```

Body contract:

```python
try:
    config = IndexConfig(
        corpus_path=corpus_path,
        processed_path=processed_path,
        max_chunk_size=max_chunk_size,
        overlap=overlap,
        k1=k1,
        b=b,
        weight_body=weight_body,
        weight_header=weight_header,
        weight_path=weight_path,
    )
    stats = index_corpus(config)
except ValueError as exc:
    print(f"error: {exc}", file=sys.stderr)
    raise SystemExit(1) from None
print(
    f"Ingestion complete! Indexed {stats.chunks_indexed} chunks "
    f"from {stats.files_indexed} files "
    f"under {config.processed_path} in {stats.elapsed_seconds:.1f}s"
)
```

- Defaults come from `src.config` constants (never literals in `cli.py`).
- `pydantic.ValidationError` subclasses `ValueError`, so bad CLI values
  (`--max_chunk_size 0`, `overlap >= max_chunk_size`, `b=2`) share the same
  clean path: message on stderr, exit status 1, **no traceback ever**
  (subject VI.6: an unhandled traceback during review is non-functional).
- Only `ValueError` is caught; genuine bugs keep their traceback during
  development.
- Phase 4 will add `search` and `search_dataset` methods to the same class.

`src/__main__.py`:

```python
"""Run the RAG CLI with ``python -m src``."""
import fire

from src.cli import Rag


def main() -> None:
    """Dispatch command-line arguments with Python Fire."""
    fire.Fire(Rag)


if __name__ == "__main__":
    main()
```

This also makes `make run` (`uv run python -m src`) and `make debug`
(`uv run python -m pdb -m src`) meaningful.

---

## 12. Tests (you write them, they must pass)

Style constraints (same as Phases 1–2): typed parameters, PEP 257 docstrings
in English on every test function, flake8 max line 79, no prints, `tmp_path`
for synthetic corpora, real-corpus tests guarded with `pytest.mark.skipif`.
Expected suite size: 71 existing + 65 new = **136+ tests**.

### 12.1 `tests/test_models.py` — append `IndexedChunk` cases (1–4)

| # | Case | Expectation |
|---|------|-------------|
| 1 | Valid `IndexedChunk` (verbatim path, `kind="markdown"`) | fields populated; it IS a `MinimalSource` (`isinstance`) |
| 2 | `kind="banana"` | `ValidationError` |
| 3 | `typing.get_args(IndexedChunk.model_fields["kind"].annotation)` | equals `CHUNK_KINDS` imported from `src.chunking.base` (guard against drift; annotate the retrieved `annotation` before `get_args` so mypy stays strict) |
| 4 | `last_character_index < first_character_index` | `ValidationError` (inherited `MinimalSource` validator still active) |

### 12.2 `tests/test_tokenizer.py` (5–16)

| # | Case | Expectation |
|---|------|-------------|
| 5 | `tokenize("")` and `tokenize("-- ++ ")` | `[]` |
| 6 | `"Hello World"` | `["hello", "world"]` |
| 7 | `"skip_sampler_cpu_output"` | whole token first, then the four subtokens, in order |
| 8 | `"getUserName"` | `["getusername", "get", "user", "name"]` |
| 9 | `"LLMEngine"` | `["llmengine", "llm", "engine"]` |
| 10 | `"parseHTTPResponse_data"` | `["parsehttpresponse_data", "parse", "http", "response", "data"]` |
| 11 | `"v0.10.1"` | `["v0", "10", "1"]` |
| 12 | `"café naïve 🚀 ok"` | `["café", "naïve", "ok"]` (accents kept, emoji a separator) |
| 13 | `"foo foo FOO"` | `["foo", "foo", "foo"]` (repetitions preserved, no stopwords) |
| 14 | `"chunk_2_text"` | `["chunk_2_text", "chunk", "2", "text"]` |
| 15 | Invariants on a realistic 400-word string | no empty tokens, every `t == t.lower()`, input of only punctuation returns `[]` |
| 16 | Determinism | two calls on the same text return equal lists; both are plain `list[str]` |

### 12.3 `tests/test_bm25.py` (17–36)

Build tiny indexes by passing literal token lists to `add_document` (no
files). The hand-computed reference: `N=3`, docs' bodies `["hello"]`,
`["hello", "hello"]`, `["bye"]`, all other fields empty, `k1=1.5`, `b=0`,
weights 1.0. Then `idf(hello) = ln(1 + (3-2+0.5)/(2+0.5)) = ln(1.6) =
0.4700036...`, scores `doc1 = 0.6714338...`, `doc0 = 0.4700036...`.

| # | Case | Expectation |
|---|------|-------------|
| 17 | `BM25Index(k1=0)` and `k1=-1` | `ValueError` |
| 18 | `b=-0.1` and `b=1.5` | `ValueError` |
| 19 | negative `weight_body` / `weight_header` / `weight_path` | `ValueError` (parametrize over the three) |
| 20 | `add_document` with a missing field key and with an extra key | `ValueError` naming the key |
| 21 | `add_document(5, ...)` first; then `add_document(0, ...)` after one doc | `ValueError` (sequence must start at 0 and advance by 1) |
| 22 | `build()` with zero documents | `ValueError` |
| 23 | `build()` twice | `ValueError` |
| 24 | `add_document` after `build()` | `ValueError` |
| 25 | `search` before `build()` | `ValueError` |
| 26 | Reference corpus, `k=3` | results `[(1, 0.6714), (0, 0.4700)]` with `assertAlmostEqual(places=4)`; `doc 2` absent (score 0 excluded); a second identical search returns an equal list (determinism) |
| 27 | `k` values `0`, `-1`, `100` | `[]`, `[]`, and all positive-score docs respectively |
| 28 | `""` and `"   "` queries | `[]` |
| 29 | `"zebra quokka"` (unknown terms) | `[]` |
| 30 | Two identical documents | equal scores, returned in ascending `doc_id` order |
| 31 | `b=1` vs `b=0` on a length-1 vs length-5 doc with equal tf | `b=1`: short doc strictly first; `b=0`: equal scores, tie-break keeps `doc_id` order |
| 32 | Saturation with `k1=1.5`, tf 2 vs tf 4 (`b=0`) | `score(4) > score(2)` but `score(4) / score(2) < 2` (sublinear) |
| 33 | `w_header=3` vs `w_body=1`: one doc matches only in `header`, another only in `body` | the header-matching doc ranks first |
| 34 | Same setup with `w_header=0` | the header-only doc is NOT returned at all (its only evidence has weight 0) |
| 35 | Path field: doc 0 has `path=["openai", "server"]`, doc 1 has an unrelated body | query `"openai"` returns only doc 0 with `score > 0` |
| 36 | A term present in EVERY document | `idf > 0`: every document still scores `> 0` (never-negative idf) |

### 12.4 `tests/test_storage.py` (37–47)

Use a built 3-doc index and records, including one with the verbatim path
`data/raw/vllm-0.10.1/docs/features/lora.md`.

| # | Case | Expectation |
|---|------|-------------|
| 37 | `save_index` + `load_index` round trip | equal `doc_count`, equal `config`, identical `chunks` list (paths/spans/kinds byte-for-byte), and identical `search` results (scores and order) for two queries |
| 38 | Artifact contract | npz keys `== _BUNDLE_KEYS` exactly; JSON top-level keys `== {format_version, config, chunks, avgdl}` |
| 39 | Save into a deep non-existent directory | created, both files present |
| 40 | `save_index` with an unbuilt index | `ValueError` and the destination directory is NOT created |
| 41 | `len(chunks) != doc_count` | `ValueError` |
| 42 | Destination path under an existing regular file | `ValueError` (not `OSError`) |
| 43 | `load_index` on a missing directory | `ValueError` naming the path |
| 44 | `index_meta.json` containing `{not json` | `ValueError` |
| 45 | `format_version` edited to `999` | `ValueError` naming `999` and `INDEX_FORMAT_VERSION` |
| 46 | Meta tampered so `chunks` has one record fewer than the npz doc count | `ValueError` |
| 47 | `from_bundle` directly: a missing key; an `indptr` that disagrees with `vocab`; `docs` cast to float | three `ValueError`s, each naming the broken invariant (parametrize) |

### 12.5 `tests/test_indexer.py` (48–59) — shared fixture

`tests/conftest.py` provides:

```python
@pytest.fixture
def tiny_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a three-file corpus under data/raw/tiny and chdir to its root."""
```

It creates `<root>/data/raw/tiny/` with

- `alpha.py` — a module containing
  `def quixotic_tokenizer_v2(text: str) -> str:` (a made-up identifier that
  cannot collide with real corpus vocabulary),
- `beta.py` — a small unrelated module (`def mundane_helper(n: int) -> int:`),
- `guide.md` — `# Frosting Guide` plus a paragraph,

and then `monkeypatch.chdir(root)`, returning `root`. Tests that need a
corpus outside the project root build a sibling directory of `root`.

| # | Case | Expectation |
|---|------|-------------|
| 48 | `index_corpus(IndexConfig())` on the fixture | stats: `files_indexed == 3`, `chunks_indexed >= 3`, `tokens_indexed > 0`, `elapsed_seconds > 0`, `index_directory == "data/processed"`; both artifacts exist; `load_index` succeeds and every `chunk.file_path` starts with `data/raw/tiny/` |
| 49 | Offset integrity on the loaded records | for every record: `0 <= first < last <= len(text)` (text re-read from the recorded path), `last - first <= max_chunk_size`, the slice is non-empty, and the sorted spans of each file merge to exactly `[0, len(text))` (overlap 0) |
| 50 | `index.search("quixotic_tokenizer_v2", 3)` | top-1 record is `data/raw/tiny/alpha.py` and its slice contains the `def quixotic_tokenizer_v2` line |
| 51 | `IndexConfig(max_chunk_size=64)` | every persisted span width `<= 64`; more chunks than the default run |
| 52 | Two consecutive runs | identical `chunks` lists and identical scores for the same query (determinism) |
| 53 | `corpus_path` given as an absolute path inside the project root | persisted `file_path` values are still the relative verbatim `data/raw/tiny/...` strings (identical to case 48's) |
| 54 | Non-existent `corpus_path` | `ValueError` naming the path |
| 55 | Corpus directory with no supported files (empty dir; dir containing only `notes.json`) | `ValueError` mentioning supported extensions (parametrize) |
| 56 | Absolute corpus path outside the project root | `ValueError` explaining the project-root rule |
| 57 | `corpus_path="."` (the project root itself) | `ValueError` |
| 58 | `show_progress=False` | runs without error and reports the same `files_indexed`/`chunks_indexed` as case 48 |
| 59 | `IndexConfig(max_chunk_size=0)` and `IndexConfig(max_chunk_size=10, overlap=10)` | `ValidationError` (a `ValueError`) raised before any file access |

### 12.6 `tests/test_cli.py` (60–65)

Drive Fire in-process (`fire.Fire(Rag, command=[...], name="rag")`) so the
tests stay fast; the real `python -m src` entry point is exercised manually in
section 14.

| # | Case | Expectation |
|---|------|-------------|
| 60 | `["index", "--corpus_path", "data/raw/tiny", "--processed_path", "out/idx"]` on the fixture | returns `None`, `out/idx/index.npz` exists, stdout contains `Ingestion complete!` |
| 61 | `["index"]` with the fixture's DEFAULT layout (`data/raw`) | `data/processed/` created (defaults really come from `config.py`) |
| 62 | Subject-style space-separated flags: `["index", "--max_chunk_size", "64", "--corpus_path", "data/raw/tiny"]` | persisted spans all `<= 64` (Fire parses exactly the form used in `en.subject.md`) |
| 63 | `["index", "--corpus_path", "does/not/exist"]` | `SystemExit` with `code == 1`, stderr starts with `error:`, and `Traceback` appears in neither stream |
| 64 | `["index", "--max_chunk_size=0"]` | same contract: `SystemExit(1)`, `error:` on stderr, no traceback |
| 65 | Documentation surface | `Rag.__doc__` and `Rag.index.__doc__` are non-empty (Fire renders them in `--help`) |

---

## 13. Implementation order (do it in this order)

1. `pyproject.toml` changes (section 3); `make lint-strict` must stay green.
2. `src/config.py`; wire cases 17–19 and 59 expectations for it.
3. `src/models.py` append `IndexedChunk`; append tests 1–4; run them.
4. `src/indexing/tokenizer.py`; write and pass tests 5–16.
5. `src/indexing/bm25.py`; write and pass tests 17–36.
6. `src/indexing/storage.py`; write and pass tests 37–47.
7. `tests/conftest.py` + `src/indexing/indexer.py`; write and pass tests
   48–59.
8. `src/indexing/__init__.py` re-exports (import them once to prove no
   circular import).
9. `src/cli.py` + `src/__main__.py`; write and pass tests 60–65.
10. Run the full verification suite (section 14).

---

## 14. Verification (all commands must exit 0)

```bash
# 1. Full unit suite (Phase 1: 23, Phase 2: 48, Phase 3: 65 -> 136+)
uv run pytest tests/ -v

# 2. Lint, exactly as the Makefile does
make lint
make lint-strict

# 3. Index the real corpus, timed (HARD limit 300 s; prototype measured ~10 s)
time uv run python -m src index --max_chunk_size 2000
```

Expected shape of step 3 (numbers are deterministic for this corpus and
`overlap=0` — report yours and explain any deviation):

```
Chunking: 100%|████████████████| 1969/1969 [00:03<00:00, ...file/s]
Tokenizing: 100%|██████████████| 45227/45227 [00:05<00:00, ...chunk/s]
Ingestion complete! Indexed 45227 chunks from 1969 files under data/processed/ in ~10.0s
```

```bash
# 4. Artifacts
ls -la data/processed/        # index.npz (~3 MB) + index_meta.json (~10 MB)

# 5. Reload + smoke queries: paths must be verbatim, spans <= 2000
uv run python -c "
from src.indexing import load_index

index, chunks, config = load_index('data/processed')
queries = [
    'How to configure an OpenAI compatible server?',
    'How to serve multiple LoRA adapters at the same time?',
    'skip_sampler_cpu_output',
]
for q in queries:
    print('Q:', q)
    for hit in index.search(q, 3):
        m = chunks[hit.doc_id]
        width = m.last_character_index - m.first_character_index
        assert width <= 2000, width
        print(f'  {hit.score:7.3f} {m.file_path} '
              f'[{m.first_character_index}:{m.last_character_index}]')
"
```

Expected top hits (already validated end-to-end with a BM25F prototype of this
exact design over this corpus):

```
Q: How to configure an OpenAI compatible server?
  ... data/raw/vllm-0.10.1/docs/serving/openai_compatible_server.md [...]
Q: How to serve multiple LoRA adapters at the same time?
  ... data/raw/vllm-0.10.1/docs/features/lora.md [...]
Q: skip_sampler_cpu_output
  ... data/raw/vllm-0.10.1/vllm/model_executor/layers/sampler.py [...]
```

Every printed path must start with `data/raw/vllm-0.10.1/` and every span
width must be `<= 2000`.

---

## 15. Definition of Done (check every box before reporting back)

- [ ] `pyproject.toml` has `types-tqdm` in the dev group and the scoped
      `fire` mypy override; `make lint-strict` exits 0.
- [ ] `src/config.py` exposes all `DEFAULT_*` constants, re-exports
      `SUPPORTED_EXTENSIONS`, and validates `IndexConfig` (including
      `overlap < max_chunk_size`, `k1 > 0`, `0 <= b <= 1`, weights `>= 0`).
- [ ] `src/models.py` only gained `IndexedChunk`; the eight original models
      and their tests are untouched and green.
- [ ] `tokenize` matches section 6 exactly (algorithm, examples, no stopwords,
      repetitions preserved, lowercase, never empty strings).
- [ ] `BM25Index` implements BM25F exactly as formulas 7.2: per-field
      normalization, weights combined BEFORE saturation, smoothed positive
      idf, unique query tokens, only `score > 0` docs, `(-score, doc_id)`
      ordering.
- [ ] `build()` produces the ten `_BUNDLE_KEYS` arrays with the dtypes and
      invariants of section 7.4; `from_bundle` validates all of them.
- [ ] Storage writes `index.npz` + `index_meta.json`, version-checked, no
      pickle, `chunks[doc_id]` invariant enforced on load.
- [ ] The indexer roots every `file_path` at the project root (verbatim
      `data/raw/...`), rejects corpora outside it, and validates chunks with
      `validate_chunks` before persisting.
- [ ] The two tqdm bars (`Chunking` in files, `Tokenizing` in chunks) appear
      and can be disabled; `--max_chunk_size` and every other default come
      from `config.py`.
- [ ] The CLI never prints a traceback for bad input: `error: ...` on stderr
      plus exit status 1 (cases 63–64 prove it).
- [ ] All 65 new tests pass; the 71 previous tests still pass
      (`uv run pytest tests/ -v` shows 136+ passed).
- [ ] `make lint` and `make lint-strict` both exit 0.
- [ ] Real-corpus run: 1969 files, 45,227 chunks, all spans `<= 2000`,
      wall time `<< 300 s`, artifacts present under `data/processed/`.
- [ ] Smoke queries return the expected files from section 14.
- [ ] Every public class and function has a PEP 257 docstring in English;
      zero inline comments; `mypy --strict` clean with no new ignores.

---

## 16. Reporting back

When done, reply with:

1. Files created and modified, with line counts.
2. The 65 new test names (one-line list, numbered as in section 12).
3. Verification numbers: pytest summary line; the `index` run wall time; the
   artifact sizes of `index.npz` and `index_meta.json`; the vocab size and
   doc count from the smoke script; the top-3 hits of the three smoke queries.
4. Design decisions you made beyond this spec (if any) and why.
5. Any assumption you had to make.
6. Confirmation that every command in section 14 exited 0.

Do not start Phase 4 (`retriever.py`, `search`, `search_dataset`) until this
phase is reviewed and validated.
