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
            f"under {stats.index_directory} "
            f"in {stats.elapsed_seconds:.1f}s"
        )
        return None


__all__ = ["Rag"]
