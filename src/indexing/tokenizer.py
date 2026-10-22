"""Code-aware whitespace tokenizer for chunks, paths and queries."""
import re
from typing import List

_WORD_SEP = re.compile(r"\W+", flags=re.UNICODE)
_CAMEL_SEP = re.compile(
    r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])",
)


def tokenize(text: str) -> List[str]:
    """Split text into lowercase code-aware tokens.

    Splits on runs of non-word characters, emits each word in lowercase
    and additionally emits its snake_case/camelCase subtokens (each in
    lowercase, skipped when they equal the whole word). Order follows the
    input; repetitions are preserved because term frequency needs them.

    Args:
        text: Any text: chunk body, header line, file path or question.

    Returns:
        Lowercase tokens, never empty strings. ``""`` and separator-only
        input return ``[]``.
    """
    tokens: List[str] = []
    for piece in _WORD_SEP.split(text):
        if not piece:
            continue
        lowered = piece.lower()
        tokens.append(lowered)
        for part in _CAMEL_SEP.split(piece):
            for sub in part.split("_"):
                if sub and sub.lower() != lowered:
                    tokens.append(sub.lower())
    return tokens


__all__ = ["tokenize"]
