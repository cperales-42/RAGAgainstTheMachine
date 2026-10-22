"""Corpus ingestion: walk, chunk, tokenize, build and persist the index."""
import time
from pathlib import Path
from typing import Dict, List, Literal, Tuple, cast

from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

from src.chunking import SUPPORTED_EXTENSIONS, chunk_file
from src.chunking.base import Chunk, read_file_text, validate_chunks
from src.config import IndexConfig
from src.indexing.bm25 import BM25Index
from src.indexing.storage import save_index
from src.indexing.tokenizer import tokenize
from src.io_utils import to_posix_path
from src.models import IndexedChunk


class IndexStats(BaseModel):
    """Summary of a completed indexing run."""

    model_config = ConfigDict(extra="ignore")

    files_indexed: int
    chunks_indexed: int
    tokens_indexed: int
    elapsed_seconds: float
    index_directory: str


def _resolve_corpus(config: IndexConfig) -> Tuple[Path, str]:
    """Resolve the corpus directory and its project-relative prefix.

    Args:
        config: Configuration carrying ``corpus_path``.

    Returns:
        The resolved corpus directory and its POSIX prefix relative to
        the project root.

    Raises:
        ValueError: If the corpus does not exist, is not a directory, or
            is not strictly inside the project root.
    """
    project = Path.cwd().resolve()
    requested = Path(config.corpus_path)
    if not requested.is_absolute():
        requested = project / requested
    if not requested.is_dir():
        raise ValueError(
            f"corpus_path {config.corpus_path!r} is not a directory"
        )
    resolved = requested.resolve()
    if resolved == project or not resolved.is_relative_to(project):
        raise ValueError(
            f"corpus_path {config.corpus_path!r} must be a subdirectory "
            f"of the project root {project}"
        )
    prefix = to_posix_path(resolved.relative_to(project))
    return resolved, prefix


def _supported_files(corpus: Path) -> List[Path]:
    """Return the sorted corpus files with a supported extension.

    Args:
        corpus: Resolved corpus directory.

    Returns:
        Sorted file paths whose lowercased suffix is supported.
    """
    return sorted(
        path
        for path in corpus.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _header_line(text: str) -> str:
    """Return the first non-blank line of a chunk, or an empty string.

    Args:
        text: Chunk text.

    Returns:
        The first non-blank line, stripped, or ``""`` when there is none.
    """
    stripped = text.strip()
    if not stripped:
        return ""
    return stripped.splitlines()[0]


def _local_path(file_path: str, prefix: str) -> str:
    """Strip the corpus prefix from a recorded path.

    Args:
        file_path: Project-relative path recorded on the chunk.
        prefix: Corpus prefix relative to the project root.

    Returns:
        The path relative to the corpus, unchanged when it lacks the
        prefix.
    """
    marker = prefix + "/"
    if file_path.startswith(marker):
        return file_path[len(marker):]
    return file_path


def _chunk_corpus(
    files: List[Path],
    resolved: Path,
    prefix: str,
    config: IndexConfig,
    show_progress: bool,
) -> Tuple[List[Tuple[str, List[Chunk]]], int]:
    """Chunk every corpus file and validate the produced spans.

    Args:
        files: Supported corpus files in walk order.
        resolved: Resolved corpus directory.
        prefix: Corpus prefix relative to the project root.
        config: Chunking configuration.
        show_progress: Whether to display the ``Chunking`` bar.

    Returns:
        The ``(file_path, chunks)`` pairs and the total chunk count.
    """
    pairs: List[Tuple[str, List[Chunk]]] = []
    total_chunks = 0
    bar = tqdm(
        files,
        desc="Chunking",
        unit="file",
        disable=not show_progress,
    )
    for path in bar:
        relative = path.relative_to(resolved)
        file_path = to_posix_path(Path(prefix) / relative)
        text = read_file_text(file_path)
        chunks = chunk_file(
            file_path, config.max_chunk_size, config.overlap
        )
        validate_chunks(text, chunks, config.max_chunk_size)
        pairs.append((file_path, chunks))
        total_chunks += len(chunks)
    return pairs, total_chunks


def _index_chunks(
    index: BM25Index,
    pairs: List[Tuple[str, List[Chunk]]],
    total_chunks: int,
    prefix: str,
    show_progress: bool,
) -> Tuple[List[IndexedChunk], int]:
    """Tokenize every chunk into its three fields and index it.

    Args:
        index: Empty, unbuilt index to fill.
        pairs: ``(file_path, chunks)`` pairs in doc id order.
        total_chunks: Total number of chunks to process.
        prefix: Corpus prefix relative to the project root.
        show_progress: Whether to display the ``Tokenizing`` bar.

    Returns:
        The per-document records and the number of indexed tokens.
    """
    records: List[IndexedChunk] = []
    tokens_indexed = 0
    doc_id = 0
    bar = tqdm(
        total=total_chunks,
        desc="Tokenizing",
        unit="chunk",
        disable=not show_progress,
    )
    for file_path, chunks in pairs:
        path_tokens = tokenize(_local_path(file_path, prefix))
        for chunk in chunks:
            fields: Dict[str, List[str]] = {
                "body": tokenize(chunk.text),
                "header": tokenize(_header_line(chunk.text)),
                "path": path_tokens,
            }
            index.add_document(doc_id, fields)
            records.append(
                IndexedChunk(
                    file_path=file_path,
                    first_character_index=chunk.start,
                    last_character_index=chunk.end,
                    kind=cast(
                        Literal["python", "markdown", "sliding"],
                        chunk.kind,
                    ),
                )
            )
            tokens_indexed += sum(len(tokens) for tokens in fields.values())
            doc_id += 1
            bar.update(1)
    bar.close()
    return records, tokens_indexed


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
    started = time.monotonic()
    resolved, prefix = _resolve_corpus(config)
    files = _supported_files(resolved)
    if not files:
        raise ValueError(
            f"no supported files found under {config.corpus_path!r}; "
            f"supported extensions are {SUPPORTED_EXTENSIONS}"
        )
    pairs, total_chunks = _chunk_corpus(
        files, resolved, prefix, config, show_progress
    )
    index = BM25Index(
        k1=config.k1,
        b=config.b,
        weight_body=config.weight_body,
        weight_header=config.weight_header,
        weight_path=config.weight_path,
    )
    records, tokens_indexed = _index_chunks(
        index, pairs, total_chunks, prefix, show_progress
    )
    index.build()
    save_index(index, records, config, config.processed_path)
    elapsed = time.monotonic() - started
    return IndexStats(
        files_indexed=len(files),
        chunks_indexed=len(records),
        tokens_indexed=tokens_indexed,
        elapsed_seconds=elapsed,
        index_directory=to_posix_path(config.processed_path),
    )


__all__ = ["IndexStats", "index_corpus"]
