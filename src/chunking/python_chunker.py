"""AST-based chunking for Python source files."""
import ast
from typing import List, Optional, Tuple

from src.chunking.base import (
    Chunk,
    _validate_window_params,
    line_start_offsets,
    sliding_window,
    validate_chunks,
)


def _statement_end_line(node: ast.stmt) -> int:
    """1-based line where a statement ends, falling back to its start.

    Args:
        node: The statement to inspect.

    Returns:
        ``node.end_lineno`` when set, otherwise ``node.lineno``, otherwise 1.
    """
    end = node.end_lineno
    if end is None:
        start = node.lineno
        return start if start is not None else 1
    return end


def _statement_start_line(node: ast.stmt) -> int:
    """1-based first line of a statement, decorators included.

    Args:
        node: The statement to inspect.

    Returns:
        The line of the first decorator for decorated defs and classes,
        otherwise the statement's own first line.
    """
    line = node.lineno
    if isinstance(
        node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    ):
        if node.decorator_list:
            decorator_line = node.decorator_list[0].lineno
            if decorator_line is not None:
                line = decorator_line
    return line if line is not None else 1


def _after_statement(
    node: ast.stmt,
    offsets: List[int],
    text_length: int,
) -> int:
    """Character offset of the line following a statement's end line.

    Args:
        node: The statement whose end line is located.
        offsets: Character offset of the start of every line.
        text_length: Length of the full file content.

    Returns:
        The start of the line after ``node``'s end line, or ``text_length``
        when no following line exists.
    """
    end_line = _statement_end_line(node)
    if end_line < len(offsets):
        return offsets[end_line]
    return text_length


def _class_pieces(
    start: int,
    end: int,
    node: ast.ClassDef,
    offsets: List[int],
) -> List[Tuple[int, int, Optional[ast.stmt]]]:
    """Split an oversized class span into header and member pieces (P2).

    Args:
        start: Inclusive start of the class span.
        end: Exclusive end of the class span.
        node: The class statement living in the span.
        offsets: Character offset of the start of every line.

    Returns:
        Consecutive pieces tiling [start, end) with the class header first,
        or an empty list when the pieces would not tile the span (for
        example when the span only holds trailing content after the class).
    """
    member_starts: List[int] = []
    for member in node.body:
        line = _statement_start_line(member)
        if 1 <= line <= len(offsets):
            member_starts.append(offsets[line - 1])
        else:
            member_starts.append(start)
    pieces: List[Tuple[int, int, Optional[ast.stmt]]] = [
        (start, member_starts[0], None)
    ]
    for index in range(len(node.body) - 1):
        pieces.append(
            (
                member_starts[index],
                member_starts[index + 1],
                node.body[index],
            )
        )
    pieces.append((member_starts[-1], end, node.body[-1]))
    ordered = [piece for piece in pieces if piece[0] < piece[1]]
    if not ordered:
        return []
    if ordered[0][0] != start or ordered[-1][1] != end:
        return []
    for previous, current in zip(ordered, ordered[1:]):
        if previous[1] != current[0]:
            return []
    return ordered


def _emit_span(
    text: str,
    file_path: str,
    start: int,
    end: int,
    node: Optional[ast.stmt],
    max_chunk_size: int,
    overlap: int,
    offsets: List[int],
) -> List[Chunk]:
    """Emit one partition span following rules E1 and P2.

    Args:
        text: The full file content.
        file_path: Path recorded on every produced chunk.
        start: Inclusive start of the span.
        end: Exclusive end of the span.
        node: The statement the span belongs to, if any.
        max_chunk_size: Maximum number of characters per chunk.
        overlap: Characters shared between consecutive sliding windows.
        offsets: Character offset of the start of every line.

    Returns:
        Chunks covering [start, end) exactly, or nothing for an empty span.
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
                kind="python",
            )
        ]
    if isinstance(node, ast.ClassDef):
        pieces = _class_pieces(start, end, node, offsets)
        if pieces:
            emitted: List[Chunk] = []
            for piece_start, piece_end, piece_node in pieces:
                emitted.extend(
                    _emit_span(
                        text,
                        file_path,
                        piece_start,
                        piece_end,
                        piece_node,
                        max_chunk_size,
                        overlap,
                        offsets,
                    )
                )
            return emitted
    return list(
        sliding_window(
            text, file_path, start, end, max_chunk_size, overlap
        )
    )


def _fallback_chunks(
    text: str,
    file_path: str,
    max_chunk_size: int,
    overlap: int,
) -> List[Chunk]:
    """Chunk unparseable or empty-bodied source with a sliding window.

    Args:
        text: The full file content.
        file_path: Path recorded on every produced chunk.
        max_chunk_size: Maximum number of characters per chunk.
        overlap: Characters shared between consecutive sliding windows.

    Returns:
        Sliding chunks covering the whole file.
    """
    return list(
        sliding_window(
            text, file_path, 0, len(text), max_chunk_size, overlap
        )
    )


def chunk_python(
    text: str,
    file_path: str,
    max_chunk_size: int,
    overlap: int = 0,
) -> List[Chunk]:
    """Chunk Python source into AST units with exact character offsets.

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
    tree: Optional[ast.Module] = None
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        tree = None
    if tree is None or not tree.body:
        chunks = _fallback_chunks(
            text, file_path, max_chunk_size, overlap
        )
    else:
        chunks = _partition_module(
            text, tree, file_path, max_chunk_size, overlap
        )
    validate_chunks(text, chunks, max_chunk_size)
    return chunks


def _partition_module(
    text: str,
    tree: ast.Module,
    file_path: str,
    max_chunk_size: int,
    overlap: int,
) -> List[Chunk]:
    """Partition a parsed module at top-level statements and emit (P1).

    Args:
        text: The full file content.
        tree: The parsed module.
        file_path: Path recorded on every produced chunk.
        max_chunk_size: Maximum number of characters per chunk.
        overlap: Characters shared between consecutive sliding windows.

    Returns:
        Chunks in file order covering the whole file exactly.
    """
    offsets = line_start_offsets(text)
    chunks: List[Chunk] = []
    previous_after = 0
    last_node = tree.body[-1]
    for node in tree.body:
        after = _after_statement(node, offsets, len(text))
        chunks.extend(
            _emit_span(
                text,
                file_path,
                previous_after,
                after,
                node,
                max_chunk_size,
                overlap,
                offsets,
            )
        )
        previous_after = after
    chunks.extend(
        _emit_span(
            text,
            file_path,
            previous_after,
            len(text),
            last_node,
            max_chunk_size,
            overlap,
            offsets,
        )
    )
    return chunks
