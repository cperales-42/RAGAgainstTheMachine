"""Default configuration for the indexing pipeline and its CLI."""
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
        if self.overlap >= self.max_chunk_size:
            raise ValueError(
                "overlap must be smaller than max_chunk_size, "
                f"got overlap={self.overlap} "
                f"max_chunk_size={self.max_chunk_size}"
            )
        return self


__all__ = [
    "DEFAULT_B",
    "DEFAULT_CORPUS_PATH",
    "DEFAULT_K1",
    "DEFAULT_MAX_CHUNK_SIZE",
    "DEFAULT_OVERLAP",
    "DEFAULT_PROCESSED_PATH",
    "DEFAULT_WEIGHT_BODY",
    "DEFAULT_WEIGHT_HEADER",
    "DEFAULT_WEIGHT_PATH",
    "IndexConfig",
    "SUPPORTED_EXTENSIONS",
]
