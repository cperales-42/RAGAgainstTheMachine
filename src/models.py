"""Pydantic models exchanged between the RAG pipeline stages."""
import uuid
from typing import List, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MinimalSource(BaseModel):
    """A single source location: file path and character range."""

    model_config = ConfigDict(extra="ignore")

    file_path: str
    first_character_index: int
    last_character_index: int

    @model_validator(mode="after")
    def _validate_span(self) -> "MinimalSource":
        """Reject empty paths and invalid character ranges."""
        if not self.file_path:
            raise ValueError("file_path must not be empty")
        if self.first_character_index < 0:
            raise ValueError(
                "first_character_index must be non-negative"
            )
        if self.last_character_index < self.first_character_index:
            raise ValueError(
                "last_character_index must not be smaller than "
                "first_character_index"
            )
        return self


class UnansweredQuestion(BaseModel):
    """A question that has no reference answer yet."""

    model_config = ConfigDict(extra="ignore")

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with its reference sources and answer."""

    model_config = ConfigDict(extra="ignore")

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset of RAG questions, answered or unanswered."""

    model_config = ConfigDict(extra="ignore")

    rag_questions: List[Union[AnsweredQuestion, UnansweredQuestion]]


class MinimalSearchResults(BaseModel):
    """The retrieved sources for a single question."""

    model_config = ConfigDict(extra="ignore")

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Search results extended with a generated answer."""

    model_config = ConfigDict(extra="ignore")

    answer: str


class StudentSearchResults(BaseModel):
    """Search results over a dataset with the requested k."""

    model_config = ConfigDict(extra="ignore")

    search_results: List[MinimalSearchResults]
    k: int = Field(ge=0)


class StudentSearchResultsAndAnswer(BaseModel):
    """Answered search results over a dataset with the requested k."""

    model_config = ConfigDict(extra="ignore")

    search_results: List[MinimalAnswer]
    k: int = Field(ge=0)


class IndexedChunk(MinimalSource):
    """A persisted chunk location tagged with the strategy that produced it."""

    kind: Literal["python", "markdown", "sliding"]
