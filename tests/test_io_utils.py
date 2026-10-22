"""Tests for the JSON I/O helpers in src.io_utils."""
import json
from pathlib import Path

import pytest

from src.io_utils import (
    load_json,
    load_rag_dataset,
    save_model,
    to_posix_path,
)
from src.models import (
    AnsweredQuestion,
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
)

ANSWERED_DOCS_PATH = Path(
    "data/datasets/AnsweredQuestions/dataset_docs_public.json"
)
UNANSWERED_CODE_PATH = Path(
    "data/datasets/UnansweredQuestions/dataset_code_public.json"
)


def test_load_json_returns_parsed_content(tmp_path: Path) -> None:
    """load_json returns the parsed dict of a valid file."""
    path = tmp_path / "data.json"
    payload = {"alpha": 1, "beta": [1, 2, 3]}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    assert load_json(path) == payload


def test_load_json_missing_file_raises_value_error(
    tmp_path: Path,
) -> None:
    """load_json raises ValueError for a path that does not exist."""
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(ValueError) as excinfo:
        load_json(missing)
    assert not isinstance(excinfo.value, FileNotFoundError)
    assert "does_not_exist.json" in str(excinfo.value)


def test_load_json_malformed_json_raises_value_error(
    tmp_path: Path,
) -> None:
    """load_json raises ValueError for invalid JSON syntax."""
    path = tmp_path / "broken.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        load_json(path)
    assert not isinstance(excinfo.value, json.JSONDecodeError)
    assert "broken.json" in str(excinfo.value)


def test_load_rag_dataset_synthetic_file(tmp_path: Path) -> None:
    """load_rag_dataset validates a synthetic dataset file."""
    payload = {
        "rag_questions": [
            {
                "question": "Answered question?",
                "sources": [
                    {
                        "file_path": "data/raw/vllm-0.10.1/a.py",
                        "first_character_index": 0,
                        "last_character_index": 12,
                    },
                ],
                "answer": "The answer.",
            },
            {"question": "Unanswered question?"},
        ],
    }
    path = tmp_path / "dataset.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    dataset = load_rag_dataset(path)
    assert len(dataset.rag_questions) == 2
    assert isinstance(dataset.rag_questions[0], AnsweredQuestion)
    assert isinstance(dataset.rag_questions[1], UnansweredQuestion)
    assert not isinstance(dataset.rag_questions[1], AnsweredQuestion)


@pytest.mark.skipif(
    not ANSWERED_DOCS_PATH.exists(),
    reason="answered docs dataset not present in checkout",
)
def test_load_rag_dataset_real_answered_docs() -> None:
    """load_rag_dataset reads 100 answered questions from disk."""
    dataset = load_rag_dataset(ANSWERED_DOCS_PATH)
    assert len(dataset.rag_questions) == 100
    assert all(
        isinstance(question, AnsweredQuestion)
        for question in dataset.rag_questions
    )


@pytest.mark.skipif(
    not UNANSWERED_CODE_PATH.exists(),
    reason="unanswered code dataset not present in checkout",
)
def test_load_rag_dataset_real_unanswered_code() -> None:
    """load_rag_dataset reads 99 unanswered questions from disk."""
    dataset = load_rag_dataset(UNANSWERED_CODE_PATH)
    assert len(dataset.rag_questions) == 99
    assert all(
        isinstance(question, UnansweredQuestion)
        and not isinstance(question, AnsweredQuestion)
        for question in dataset.rag_questions
    )


def test_save_model_creates_nested_parent_dirs(tmp_path: Path) -> None:
    """save_model creates missing parent directories on write."""
    destination = tmp_path / "deep" / "nested" / "result.json"
    model = MinimalSearchResults(
        question_id="q1",
        question="How to serve a model?",
        retrieved_sources=[],
    )
    save_model(model, destination)
    assert destination.is_file()
    with destination.open(encoding="utf-8") as handle:
        parsed = json.load(handle)
    assert parsed["question_id"] == "q1"


def test_save_model_load_json_round_trip(tmp_path: Path) -> None:
    """save_model output reloads to the original model dump."""
    model = StudentSearchResultsAndAnswer(
        search_results=[
            MinimalAnswer(
                question_id="q1",
                question="How to serve a model?",
                retrieved_sources=[
                    MinimalSource(
                        file_path="data/raw/vllm-0.10.1/a.py",
                        first_character_index=0,
                        last_character_index=9,
                    ),
                ],
                answer="Start the server.",
            ),
        ],
        k=1,
    )
    path = tmp_path / "round_trip" / "result.json"
    save_model(model, path)
    assert load_json(path) == model.model_dump()


def test_to_posix_path_returns_forward_slashes() -> None:
    """to_posix_path converts Windows separators to slashes."""
    result = to_posix_path("data\\raw\\file.json")
    assert result == "data/raw/file.json"


def test_save_model_under_file_parent_raises_value_error(
    tmp_path: Path,
) -> None:
    """save_model raises ValueError when a parent is a file."""
    blocker = tmp_path / "file.txt"
    blocker.write_text("occupied", encoding="utf-8")
    model = MinimalSearchResults(
        question_id="q1",
        question="How to serve a model?",
        retrieved_sources=[],
    )
    with pytest.raises(ValueError):
        save_model(model, blocker / "out.json")
