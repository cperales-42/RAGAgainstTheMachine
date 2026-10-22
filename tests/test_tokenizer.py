"""Tests for the code-aware tokenizer in src.indexing.tokenizer."""
from typing import List

import pytest

from src.indexing.tokenizer import tokenize

REALISTIC_TEXT = (
    "The LLMEngineclass handles request scheduling; "
    "call llm.generate(prompts) or LLM.embed, then "
    "skip_sampler_cpu_output splits the sampler output "
    "into per-request slices with overlap. "
    "Version v0.10.1 supports LoRA and chunk_2_text "
    "parsing, encoded in café naïve UTF-8."
)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", []),
        ("-- ++ ", []),
        ("Hello World", ["hello", "world"]),
        (
            "skip_sampler_cpu_output",
            [
                "skip_sampler_cpu_output",
                "skip",
                "sampler",
                "cpu",
                "output",
            ],
        ),
        (
            "getUserName",
            ["getusername", "get", "user", "name"],
        ),
        ("LLMEngine", ["llmengine", "llm", "engine"]),
        (
            "parseHTTPResponse_data",
            ["parsehttpresponse_data", "parse", "http", "response", "data"],
        ),
        ("v0.10.1", ["v0", "10", "1"]),
        ("café naïve", ["café", "naïve"]),
        ("foo foo FOO", ["foo", "foo", "foo"]),
        ("chunk_2_text", ["chunk_2_text", "chunk", "2", "text"]),
    ],
)
def test_tokenize_known_examples(text: str, expected: List[str]) -> None:
    """Cases 5-14: tokenize matches the normative example lists."""
    assert tokenize(text) == expected


def test_tokenize_emoji_is_a_separator() -> None:
    """Case 12: accents survive while the emoji splits its neighbors."""
    assert tokenize("café naïve 🚀 ok") == ["café", "naïve", "ok"]


def test_tokenize_invariants_on_realistic_text() -> None:
    """Case 15: no empty tokens, all lowercase, punctuation-only is empty."""
    tokens = tokenize(REALISTIC_TEXT)
    assert tokens
    assert all(token and token == token.lower() for token in tokens)
    assert tokenize("!!! ??? ... ***") == []


def test_tokenize_is_deterministic_and_returns_list() -> None:
    """Case 16: repeated calls return equal plain lists of strings."""
    first = tokenize(REALISTIC_TEXT)
    second = tokenize(REALISTIC_TEXT)
    assert isinstance(first, list)
    assert all(isinstance(token, str) for token in first)
    assert first == second


def test_tokenize_preserves_tilde_and_digits() -> None:
    """Extra: underscores keep digits attached as their own subtokens."""
    assert tokenize("model_v2_beta") == [
        "model_v2_beta",
        "model",
        "v2",
        "beta",
    ]
