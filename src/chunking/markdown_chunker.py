"""Header-based chunking for Markdown and text files."""
import re
from typing import List, Optional, Tuple

from src.chunking.base import (
    Chunk,
    _validate_window_params,
    line_start_offsets,
    sliding_window,
    validate_chunks,
)

_HEADER_RE = re.compile(r"^ {0,3}#{1,6}(?:\s|$)")


def _fence_marker(stripped: str) -> Tuple[str, int]:
    """Detect the code-fence marker a line would open.

    Args:
        stripped: The line with surrounding whitespace removed.

    Returns:
        The fence character and its run length when the line starts with
        three or more backticks or tildes, otherwise ``("", 0)``.
    """
    if len(stripped) < 3:
        return "", 0
    char = stripped[0]
    if char != "`" and char != "~":
        return "", 0
    count = 0
    for current in stripped:
        if current != char:
            break
        count += 1
    if count >= 3:
        return char, count
    return "", 0


def _header_starts(lines: List[str], offsets: List[int]) -> List[int]:
    """Locate ATX header lines while ignoring fenced code blocks.

    Args:
        lines: The file content split on ``"\\n"``.
        offsets: Character offset of the start of every line.

    Returns:
        The character offset of each header line start, in file order.
    """
    starts: List[int] = []
    fence_char = ""
    fence_length = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if fence_char:
            if stripped.startswith(fence_char * fence_length):
                fence_char = ""
                fence_length = 0
            continue
        char, length = _fence_marker(stripped)
        if char:
            fence_char = char
            fence_length = length
            continue
        if _HEADER_RE.match(line):
            starts.append(offsets[index])
    return starts


def _paragraph_spans(
    start: int,
    end: int,
    lines: List[str],
    offsets: List[int],
) -> List[Tuple[int, int]]:
    """Tile [start, end) with paragraph spans of non-blank line runs.

    Blank lines before the first paragraph attach to it, blank lines
    between two paragraphs attach to the following one, and trailing
    blanks attach to the last paragraph, so the spans tile exactly.

    Args:
        start: Inclusive start of the section.
        end: Exclusive end of the section.
        lines: The file content split on ``"\\n"``.
        offsets: Character offset of the start of every line.

    Returns:
        Consecutive paragraph spans covering [start, end) exactly.
    """
    first_lines: List[int] = []
    in_paragraph = False
    for index, line in enumerate(lines):
        line_start = offsets[index]
        if line_start < start:
            continue
        if line_start >= end:
            break
        if line.strip():
            if not in_paragraph:
                first_lines.append(index)
                in_paragraph = True
        else:
            in_paragraph = False
    if not first_lines:
        return [(start, end)]
    spans: List[Tuple[int, int]] = []
    for position, first in enumerate(first_lines):
        paragraph_start = start if position == 0 else offsets[first]
        next_position = position + 1
        if next_position == len(first_lines):
            paragraph_end = end
        else:
            paragraph_end = offsets[first_lines[next_position]]
        spans.append((paragraph_start, paragraph_end))
    return spans


def _flush_group(
    text: str,
    file_path: str,
    start: int,
    end: int,
    max_chunk_size: int,
    overlap: int,
) -> List[Chunk]:
    """Emit one paragraph group as a markdown chunk or sliding windows.

    Args:
        text: The full file content.
        file_path: Path recorded on every produced chunk.
        start: Inclusive start of the group.
        end: Exclusive end of the group.
        max_chunk_size: Maximum number of characters per chunk.
        overlap: Characters shared between consecutive sliding windows.

    Returns:
        One markdown chunk when the group fits, otherwise sliding chunks.
    """
    if end - start <= max_chunk_size:
        return [
            Chunk(
                file_path=file_path,
                start=start,
                end=end,
                text=text[start:end],
                kind="markdown",
            )
        ]
    return list(
        sliding_window(
            text, file_path, start, end, max_chunk_size, overlap
        )
    )


def _emit_section(
    text: str,
    file_path: str,
    start: int,
    end: int,
    lines: List[str],
    offsets: List[int],
    max_chunk_size: int,
    overlap: int,
) -> List[Chunk]:
    """Emit one header section following rule M1.

    Args:
        text: The full file content.
        file_path: Path recorded on every produced chunk.
        start: Inclusive start of the section.
        end: Exclusive end of the section.
        lines: The file content split on ``"\\n"``.
        offsets: Character offset of the start of every line.
        max_chunk_size: Maximum number of characters per chunk.
        overlap: Characters shared between consecutive sliding windows.

    Returns:
        Chunks covering [start, end) exactly, or nothing for an empty
        section.
    """
    if start >= end:
        return []
    if end - start <= max_chunk_size:
        return [
            Chunk(
                file_path=file_path,
                start=start,
                end=end,
                text=text[start:end],
                kind="markdown",
            )
        ]
    chunks: List[Chunk] = []
    group_start: Optional[int] = None
    group_end: Optional[int] = None
    for paragraph_start, paragraph_end in _paragraph_spans(
        start, end, lines, offsets
    ):
        if group_start is None or group_end is None:
            group_start = paragraph_start
            group_end = paragraph_end
        elif paragraph_end - group_start <= max_chunk_size:
            group_end = paragraph_end
        else:
            chunks.extend(
                _flush_group(
                    text,
                    file_path,
                    group_start,
                    group_end,
                    max_chunk_size,
                    overlap,
                )
            )
            group_start = paragraph_start
            group_end = paragraph_end
    if group_start is not None and group_end is not None:
        chunks.extend(
            _flush_group(
                text,
                file_path,
                group_start,
                group_end,
                max_chunk_size,
                overlap,
            )
        )
    return chunks


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
    _validate_window_params(max_chunk_size, overlap)
    if not text:
        return []
    offsets = line_start_offsets(text)
    lines = text.split("\n")
    chunks: List[Chunk] = []
    previous = 0
    for header_start in _header_starts(lines, offsets):
        chunks.extend(
            _emit_section(
                text,
                file_path,
                previous,
                header_start,
                lines,
                offsets,
                max_chunk_size,
                overlap,
            )
        )
        previous = header_start
    chunks.extend(
        _emit_section(
            text,
            file_path,
            previous,
            len(text),
            lines,
            offsets,
            max_chunk_size,
            overlap,
        )
    )
    validate_chunks(text, chunks, max_chunk_size)
    return chunks
