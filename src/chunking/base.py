"""Chunk primitives shared by every chunking strategy."""
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple, Union

CHUNK_KINDS: Tuple[str, ...] = ("python", "markdown", "sliding")


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
        if not self.file_path:
            raise ValueError("file_path must not be empty")
        if self.start < 0:
            raise ValueError(f"start must be >= 0, got {self.start}")
        if self.end <= self.start:
            raise ValueError(
                "end must be greater than start, "
                f"got start={self.start} end={self.end}"
            )
        if self.kind not in CHUNK_KINDS:
            raise ValueError(
                f"unknown kind {self.kind!r}, "
                f"expected one of {CHUNK_KINDS}"
            )


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
    offsets: List[int] = [0]
    index = text.find("\n")
    while index != -1:
        offsets.append(index + 1)
        index = text.find("\n", index + 1)
    return offsets


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
    target = Path(path)
    try:
        with target.open("r", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Could not read file {target}: {exc}") from exc


def _validate_window_params(max_chunk_size: int, overlap: int) -> None:
    """Reject window parameters that would break sliding_window.

    Args:
        max_chunk_size: Maximum number of characters per chunk.
        overlap: Characters shared between consecutive windows.

    Raises:
        ValueError: If ``max_chunk_size`` is not positive, ``overlap`` is
            negative, or ``overlap`` is not smaller than ``max_chunk_size``.
    """
    if max_chunk_size <= 0:
        raise ValueError(
            f"max_chunk_size must be positive, got {max_chunk_size}"
        )
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= max_chunk_size:
        raise ValueError(
            f"overlap must be smaller than max_chunk_size, "
            f"got overlap={overlap} max_chunk_size={max_chunk_size}"
        )


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
    _validate_window_params(max_chunk_size, overlap)
    if start < 0:
        raise ValueError(f"start must be >= 0, got {start}")
    if end > len(text):
        raise ValueError(
            f"end must be <= len(text)={len(text)}, got {end}"
        )
    if start > end:
        raise ValueError(f"start must be <= end, got start={start} end={end}")
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + max_chunk_size, end)
        yield Chunk(
            file_path=file_path,
            start=cursor,
            end=chunk_end,
            text=text[cursor:chunk_end],
            kind="sliding",
        )
        if chunk_end >= end:
            break
        cursor = chunk_end - overlap


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
    previous_start = -1
    for index, chunk in enumerate(chunks):
        if not 0 <= chunk.start < chunk.end <= len(text):
            raise ValueError(
                f"chunk {index}: range invariant violated, "
                f"0 <= start < end <= len(text)={len(text)} "
                f"but start={chunk.start} end={chunk.end}"
            )
        if text[chunk.start:chunk.end] != chunk.text:
            raise ValueError(
                f"chunk {index}: slice invariant violated, "
                f"text[{chunk.start}:{chunk.end}] does not equal "
                "chunk.text"
            )
        size = chunk.end - chunk.start
        if size > max_chunk_size:
            raise ValueError(
                f"chunk {index}: size invariant violated, "
                f"size {size} exceeds max_chunk_size {max_chunk_size}"
            )
        if chunk.start <= previous_start:
            raise ValueError(
                f"chunk {index}: order invariant violated, "
                f"start {chunk.start} is not greater than previous "
                f"start {previous_start}"
            )
        previous_start = chunk.start
