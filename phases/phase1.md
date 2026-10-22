# Phase 1 — Data models and I/O (`src/models.py`, `src/io_utils.py`)

> **Agent:** `rag-dev`
> **Source of truth:** `en.subject.md` (section VI.4 Data Models, VI.5 Output) and
> `build_plan.md` (FASE 1).
> **Status of Phase 0:** DONE and verified on 2026-10-15
> (uv sync, make lint, corpus in `data/raw/vllm-0.10.1/`, datasets in
> `data/datasets/`, moulinette renamed and working).
> **You must read before coding:** `en.subject.md` sections VI.4 and VI.5.

---

## 1. Objective

Define the data contract between pipeline stages with pydantic v2 models
(exactly the ones the subject requires, plus allowed extras), and implement a
small, robust I/O layer to load datasets and persist results as JSON.

At the end of this phase you must be able to:

1. Load the four public datasets in `data/datasets/` without errors.
2. Validate/serialize every structure the CLI will exchange between stages.
3. Pass `pytest`, `flake8` and `mypy --strict` with zero errors.

---

## 2. Files to create

```
src/
├── models.py        # pydantic models (subject VI.4, exact names)
└── io_utils.py      # JSON loading/saving with error handling
tests/
├── __init__.py      # empty, makes pytest collection explicit
├── test_models.py
└── test_io_utils.py
```

Do NOT create anything else. Do NOT touch `pyproject.toml`, `Makefile`,
`README.md`, or any file outside `src/` and `tests/` except when a test fixture
needs a temp file (use `tmp_path`).

---

## 3. `src/models.py` — exact interface

### 3.1 Imports and module docstring

```python
"""Pydantic models exchanged between the RAG pipeline stages."""
import uuid
from typing import List, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator
```

Use `pydantic >= 2` (already pinned in `pyproject.toml`). Python target: 3.10,
so `List`/`Optional`/`Union` from `typing` (keep it explicit for mypy 3.10
compatibility).

### 3.2 Models (names, fields and types are MANDATORY)

Implement exactly these classes, in this order:

```python
class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int
```

Extra validation (allowed and recommended, see 3.3):
`first_character_index >= 0`, `last_character_index >= first_character_index`,
`file_path` non-empty. Keep `file_path` a plain `str` (paths are compared
verbatim by the grader).

```python
class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    sources: List[MinimalSource]
    answer: str
```

```python
class RagDataset(BaseModel):
    rag_questions: List[Union[AnsweredQuestion, UnansweredQuestion]]
```

```python
class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str
```

```python
class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    search_results: List[MinimalAnswer]
    k: int
```

### 3.3 Config and constraints (design decisions you must justify)

1. **`extra="ignore"`**: the ground-truth datasets contain extra keys
   (`difficulty: "synthetic"`, `is_valid: true`). Every model that can parse
   external JSON MUST tolerate them. Set on each class (or on a shared
   private base `_StrictBaseModel` if you prefer DRY — but the public class
   names above must remain the ones the subject names):

   ```python
   model_config = ConfigDict(extra="ignore")
   ```

2. **Index sanity on `MinimalSource`**: add a `@model_validator(mode="after")`
   that rejects negative `first_character_index` and
   `last_character_index < first_character_index` with a clear `ValueError`.
   This is the model that carries character offsets, and every later phase
   (chunking, retrieval, evaluation) depends on these invariants.

3. **`k` sanity on `StudentSearchResults` and `StudentSearchResultsAndAnswer`**:
   `k >= 0` (the CLI must tolerate `k=0` by returning empty lists, but a
   negative `k` in a persisted file is a validation error).

4. **Union resolution in `RagDataset`**: pydantic v2 tries members left to
   right in smart mode. Put `AnsweredQuestion` first so that a question with
   `sources` + `answer` parses as answered; questions without those keys fall
   back to `UnansweredQuestion`. Verify with a test that mixes both kinds.

5. **No defaults for required domain data**: `question`, `answer`,
   `retrieved_sources`, `sources` stay required. Only `question_id` has a
   default (UUID4 factory), exactly as the subject dictates.

### 3.4 Docstrings

Every class gets a one-line PEP 257 docstring in **English** (Google style),
e.g. `"""A single source location: file path and character range."""`.
No inline comments anywhere.

---

## 4. `src/io_utils.py` — exact interface

### 4.1 Public functions (signatures are MANDATORY)

```python
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
```

```python
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
```

```python
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
```

```python
def to_posix_path(path: Union[str, Path]) -> str:
    """Normalize a filesystem path to POSIX separators.

    Args:
        path: Path to normalize.

    Returns:
        The path as a POSIX-style string (forward slashes).
    """
```

### 4.2 Error-handling contract (mandatory)

- Use `try/except` around `open()` + `json.load()` / `model_dump_json()`.
- Use **context managers only** (`with open(...) as f:`).
- Re-raise as `ValueError` with a message that includes the offending path.
  Never let `FileNotFoundError`, `JSONDecodeError` or `OSError` escape: the
  subject marks an unhandled traceback during review as non-functional.
- Encoding: always `utf-8`. For reads, use `errors="replace"` so a single
  invalid byte never crashes the pipeline (Phase 8 edge-case requirement;
  cheap to do now).
- `save_model` must call `path.parent.mkdir(parents=True, exist_ok=True)`
  inside a `try` block before writing.

### 4.3 Imports

```python
"""JSON I/O helpers for the RAG pipeline."""
import json
from pathlib import Path
from typing import Any, Union

from pydantic import BaseModel

from src.models import RagDataset
```

No other dependencies. No global state. No hardcoded paths.

---

## 5. Tests (you write them, they must pass)

### 5.1 `tests/test_models.py` — minimum cases

| # | Case | Expectation |
|---|------|-------------|
| 1 | `MinimalSource` happy path | fields populated, types coerced |
| 2 | `MinimalSource` with `first_character_index=-1` | `ValidationError` |
| 3 | `MinimalSource` with `last < first` | `ValidationError` |
| 4 | `MinimalSource` with extra key `score` | ignored, no error |
| 5 | `UnansweredQuestion` without `question_id` | auto-generates a UUID4 string |
| 6 | two `UnansweredQuestion` defaults | different `question_id` values |
| 7 | `AnsweredQuestion` full construction | inherits `question_id`, `question` |
| 8 | `RagDataset` with 1 answered + 1 unanswered question | union resolves correctly per item |
| 9 | `MinimalSearchResults` / `MinimalAnswer` happy path | fields populated |
| 10 | `StudentSearchResults` with `k=-1` | `ValidationError` |
| 11 | `StudentSearchResultsAndAnswer` with `k=0` | valid (empty is allowed) |
| 12 | Round-trip: `model_validate(model.model_dump())` | equal models (use `==`) |

### 5.2 `tests/test_io_utils.py` — minimum cases

| # | Case | Expectation |
|---|------|-------------|
| 1 | `load_json` on a valid tmp file | returns parsed dict/list |
| 2 | `load_json` on missing path | `ValueError` (not `FileNotFoundError`) |
| 3 | `load_json` on malformed JSON | `ValueError` |
| 4 | `load_rag_dataset` on a synthetic valid dataset tmp file | `RagDataset` with correct lengths |
| 5 | `load_rag_dataset` on real `data/datasets/AnsweredQuestions/dataset_docs_public.json` | 100 questions, all `AnsweredQuestion` |
| 6 | `load_rag_dataset` on real `data/datasets/UnansweredQuestions/dataset_code_public.json` | 99 questions, all `UnansweredQuestion` |
| 7 | `save_model` to a nested non-existent tmp dir | file created, parent dirs created |
| 8 | `save_model` + `load_json` round-trip | parsed dict equals `model.model_dump()` |
| 9 | `to_posix_path` on a Windows-style path string | forward slashes returned |
| 10 | `save_model` to an invalid location (e.g. a path under a file) | `ValueError` |

Use `pytest` fixtures (`tmp_path`) for synthetic files. For the real-dataset
tests (cases 5–6), guard with `pytest.mark.skipif` if
`data/datasets/...` does not exist, so the suite also passes on a clean
checkout before the corpus is placed.

### 5.3 Style constraints for tests

- Type all fixture parameters (`tmp_path: Path`).
- PEP 257 docstrings on every test function (one line is enough).
- flake8 max line 79 — wrap long strings.
- No `print` statements.

---

## 6. Implementation order (do it in this order)

1. `src/models.py` — classes top to bottom, config, validator.
2. `tests/test_models.py` — write and pass tests 1–12.
3. `src/io_utils.py` — the four functions.
4. `tests/test_io_utils.py` — write and pass tests 1–10.
5. Run the full verification suite (section 7).

---

## 7. Verification (all commands must exit 0)

```bash
# 1. Unit tests
uv run pytest tests/ -v

# 2. Lint, exactly as the Makefile does
make lint
make lint-strict

# 3. Real-data smoke test (not a unit test, just for you)
uv run python -c "
from src.io_utils import load_rag_dataset
from pathlib import Path
base = Path('data/datasets')
for split in ('AnsweredQuestions', 'UnansweredQuestions'):
    for kind in ('docs', 'code'):
        p = base / split / f'dataset_{kind}_public.json'
        ds = load_rag_dataset(p)
        print(split, kind, len(ds.rag_questions))
"
```

Expected smoke output (docs = 100, code = 99):

```
AnsweredQuestions docs 100
AnsweredQuestions code 99
UnansweredQuestions docs 100
UnansweredQuestions code 99
```

---

## 8. Definition of Done (check every box before reporting back)

- [ ] `src/models.py` implements all 8 model classes with the exact names,
      fields and inheritance from section 3.2.
- [ ] `MinimalSource` rejects negative / inverted character indices.
- [ ] Extra keys (`difficulty`, `is_valid`) are ignored without error.
- [ ] `question_id` defaults to a fresh UUID4 string per instance.
- [ ] `RagDataset` union resolves answered vs unanswered correctly.
- [ ] `src/io_utils.py` implements `load_json`, `load_rag_dataset`,
      `save_model`, `to_posix_path` with the exact signatures of section 4.1.
- [ ] No raw `FileNotFoundError` / `JSONDecodeError` / `OSError` escapes;
      everything surfaces as `ValueError` with the path in the message.
- [ ] Context managers used everywhere; no file handle leaks.
- [ ] All 22 tests (12 + 10) pass: `uv run pytest tests/ -v` green.
- [ ] `make lint` and `make lint-strict` both exit 0.
- [ ] Real-data smoke test prints 100/99/100/99.
- [ ] Every public class and function has a PEP 257 docstring in English.
- [ ] Zero inline comments.
- [ ] `mypy --strict` passes on `src/` and `tests/`.

---

## 9. Reporting back

When done, reply with:

1. Files created and line counts.
2. The 22 test names (one-line list).
3. Design decisions you made beyond the spec (if any) and why.
4. Any assumption you had to make.
5. Confirmation that all commands in section 7 exited 0, with the pytest
   summary line (e.g. `22 passed in 0.4s`).

Do not start Phase 2 (chunking) until this phase is reviewed and validated.
