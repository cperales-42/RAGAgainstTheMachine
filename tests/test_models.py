"""Tests for the pydantic models exchanged between pipeline stages."""
import uuid
from typing import Any, get_args

import pytest
from pydantic import ValidationError

from src.chunking.base import CHUNK_KINDS
from src.models import (
    AnsweredQuestion,
    IndexedChunk,
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
)


def test_minimal_source_happy_path_coerces_indices() -> None:
    """MinimalSource populates fields and coerces string indices."""
    source = MinimalSource.model_validate(
        {
            "file_path": "data/raw/vllm-0.10.1/README.md",
            "first_character_index": "0",
            "last_character_index": "120",
        }
    )
    assert source.file_path == "data/raw/vllm-0.10.1/README.md"
    assert source.first_character_index == 0
    assert source.last_character_index == 120
    assert isinstance(source.first_character_index, int)
    assert isinstance(source.last_character_index, int)


def test_minimal_source_rejects_negative_index() -> None:
    """MinimalSource rejects a negative first_character_index."""
    with pytest.raises(ValidationError):
        MinimalSource(
            file_path="data/raw/vllm-0.10.1/README.md",
            first_character_index=-1,
            last_character_index=10,
        )


def test_minimal_source_rejects_inverted_range() -> None:
    """MinimalSource rejects a last index below the first index."""
    with pytest.raises(ValidationError):
        MinimalSource(
            file_path="data/raw/vllm-0.10.1/README.md",
            first_character_index=10,
            last_character_index=5,
        )


def test_minimal_source_ignores_extra_score_key() -> None:
    """MinimalSource drops an unknown score key without error."""
    source = MinimalSource.model_validate(
        {
            "file_path": "data/raw/vllm-0.10.1/README.md",
            "first_character_index": 0,
            "last_character_index": 5,
            "score": 0.75,
        }
    )
    assert source.first_character_index == 0
    assert "score" not in source.model_dump()


def test_unanswered_question_generates_uuid4_default() -> None:
    """UnansweredQuestion defaults question_id to a UUID4 string."""
    question = UnansweredQuestion(question="What is vLLM?")
    generated = uuid.UUID(question.question_id)
    assert generated.version == 4


def test_unanswered_question_defaults_are_unique() -> None:
    """Two UnansweredQuestion instances get different question_ids."""
    first = UnansweredQuestion(question="First question?")
    second = UnansweredQuestion(question="Second question?")
    assert first.question_id != second.question_id


def test_answered_question_full_construction() -> None:
    """AnsweredQuestion inherits question_id and question fields."""
    source = MinimalSource(
        file_path="data/raw/vllm-0.10.1/README.md",
        first_character_index=0,
        last_character_index=9,
    )
    question = AnsweredQuestion(
        question="What is vLLM?",
        sources=[source],
        answer="An open-source LLM inference engine.",
    )
    assert uuid.UUID(question.question_id).version == 4
    assert question.question == "What is vLLM?"
    assert question.sources == [source]
    assert question.answer == "An open-source LLM inference engine."


def test_rag_dataset_union_resolves_each_item() -> None:
    """RagDataset parses answered and unanswered items per entry."""
    dataset = RagDataset.model_validate(
        {
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
    )
    assert len(dataset.rag_questions) == 2
    answered = dataset.rag_questions[0]
    unanswered = dataset.rag_questions[1]
    assert isinstance(answered, AnsweredQuestion)
    assert isinstance(unanswered, UnansweredQuestion)
    assert not isinstance(unanswered, AnsweredQuestion)


def test_minimal_search_results_happy_path() -> None:
    """MinimalSearchResults and MinimalAnswer populate their fields."""
    results = MinimalSearchResults(
        question_id="q1",
        question="How to serve a model?",
        retrieved_sources=[],
    )
    answer = MinimalAnswer(
        question_id="q1",
        question="How to serve a model?",
        retrieved_sources=[],
        answer="Start the OpenAI compatible server.",
    )
    assert results.question_id == "q1"
    assert results.question == "How to serve a model?"
    assert results.retrieved_sources == []
    assert answer.answer == "Start the OpenAI compatible server."


def test_student_search_results_rejects_negative_k() -> None:
    """StudentSearchResults rejects a negative k value."""
    with pytest.raises(ValidationError):
        StudentSearchResults(search_results=[], k=-1)


def test_student_search_results_and_answer_allows_zero_k() -> None:
    """StudentSearchResultsAndAnswer accepts k=0 with empty results."""
    results = StudentSearchResultsAndAnswer(search_results=[], k=0)
    assert results.k == 0
    assert results.search_results == []


def test_model_round_trip_preserves_equality() -> None:
    """model_validate(model.model_dump()) returns an equal model."""
    original = StudentSearchResultsAndAnswer(
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
        k=5,
    )
    restored = StudentSearchResultsAndAnswer.model_validate(
        original.model_dump()
    )
    assert restored == original


def test_minimal_source_rejects_empty_path() -> None:
    """MinimalSource rejects an empty file_path."""
    with pytest.raises(ValidationError):
        MinimalSource(
            file_path="",
            first_character_index=0,
            last_character_index=10,
        )


def test_indexed_chunk_happy_path_is_a_minimal_source() -> None:
    """Case 1: IndexedChunk keeps MinimalSource semantics plus kind."""
    chunk = IndexedChunk(
        file_path="data/raw/vllm-0.10.1/docs/features/lora.md",
        first_character_index=0,
        last_character_index=200,
        kind="markdown",
    )
    assert isinstance(chunk, MinimalSource)
    assert chunk.file_path == "data/raw/vllm-0.10.1/docs/features/lora.md"
    assert chunk.first_character_index == 0
    assert chunk.last_character_index == 200
    assert chunk.kind == "markdown"


def test_indexed_chunk_rejects_unknown_kind() -> None:
    """Case 2: kind outside the allowed Literal raises ValidationError."""
    with pytest.raises(ValidationError):
        IndexedChunk.model_validate(
            {
                "file_path": "data/raw/vllm-0.10.1/a.py",
                "first_character_index": 0,
                "last_character_index": 10,
                "kind": "banana",
            }
        )


def test_indexed_chunk_kind_matches_chunk_kinds() -> None:
    """Case 3: the kind Literal never drifts from CHUNK_KINDS."""
    annotation: Any = IndexedChunk.model_fields["kind"].annotation
    assert get_args(annotation) == CHUNK_KINDS


def test_indexed_chunk_rejects_inverted_range() -> None:
    """Case 4: the inherited span validator still rejects bad ranges."""
    with pytest.raises(ValidationError):
        IndexedChunk(
            file_path="data/raw/vllm-0.10.1/a.py",
            first_character_index=10,
            last_character_index=5,
            kind="python",
        )
