"""Tests for the BM25F index in src.indexing.bm25."""
from typing import Dict, List, Sequence

import numpy as np
import pytest

from src.indexing.bm25 import BM25Index


def _fields(
    body: Sequence[str] = (),
    header: Sequence[str] = (),
    path: Sequence[str] = (),
) -> Dict[str, List[str]]:
    """Build a complete FIELDS mapping from three token sequences.

    Args:
        body: Tokens of the body field.
        header: Tokens of the header field.
        path: Tokens of the path field.

    Returns:
        A dict keyed by the three BM25F field names.
    """
    return {
        "body": list(body),
        "header": list(header),
        "path": list(path),
    }


def _build(
    documents: Sequence[Dict[str, List[str]]],
    **kwargs: float,
) -> BM25Index:
    """Accumulate and freeze a tiny index from in-memory documents.

    Args:
        documents: One FIELDS mapping per document, in doc id order.
        **kwargs: Extra keyword arguments forwarded to ``BM25Index``.

    Returns:
        A built BM25Index.
    """
    index = BM25Index(**kwargs)
    for doc_id, fields in enumerate(documents):
        index.add_document(doc_id, fields)
    index.build()
    return index


@pytest.mark.parametrize("k1", [0.0, -1.0])
def test_case17_rejects_non_positive_k1(k1: float) -> None:
    """Case 17: k1 must be strictly positive."""
    with pytest.raises(ValueError):
        BM25Index(k1=k1)


@pytest.mark.parametrize("b", [-0.1, 1.5])
def test_case18_rejects_out_of_range_b(b: float) -> None:
    """Case 18: b must lie in [0, 1]."""
    with pytest.raises(ValueError):
        BM25Index(b=b)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"weight_body": -1.0},
        {"weight_header": -0.1},
        {"weight_path": -2.0},
    ],
)
def test_case19_rejects_negative_weights(
    kwargs: Dict[str, float],
) -> None:
    """Case 19: field weights must be non-negative."""
    with pytest.raises(ValueError):
        BM25Index(**kwargs)


def test_case20_rejects_missing_and_extra_fields() -> None:
    """Case 20: fields must be exactly the FIELDS keys."""
    index = BM25Index()
    with pytest.raises(ValueError, match="path"):
        index.add_document(0, {"body": [], "header": []})
    with pytest.raises(ValueError, match="banana"):
        index.add_document(0, {**_fields(), "banana": []})


def test_case21_rejects_doc_id_out_of_sequence() -> None:
    """Case 21: doc ids must start at 0 and advance by one."""
    index = BM25Index()
    with pytest.raises(ValueError):
        index.add_document(5, _fields(body=["hello"]))
    index.add_document(0, _fields(body=["hello"]))
    with pytest.raises(ValueError):
        index.add_document(0, _fields(body=["hello"]))


def test_case22_build_requires_a_document() -> None:
    """Case 22: build on an empty index raises ValueError."""
    with pytest.raises(ValueError):
        BM25Index().build()


def test_case23_build_runs_at_most_once() -> None:
    """Case 23: a second build raises ValueError."""
    index = _build([_fields(body=["hello"])])
    with pytest.raises(ValueError):
        index.build()


def test_case24_add_after_build_is_rejected() -> None:
    """Case 24: documents cannot be added to a built index."""
    index = _build([_fields(body=["hello"])])
    with pytest.raises(ValueError):
        index.add_document(1, _fields(body=["bye"]))


def test_case25_search_before_build_is_rejected() -> None:
    """Case 25: searching an unbuilt index raises ValueError."""
    index = BM25Index()
    index.add_document(0, _fields(body=["hello"]))
    with pytest.raises(ValueError):
        index.search("hello", 1)


def test_case26_reference_scores_and_determinism() -> None:
    """Case 26: hand-computed scores, zero-score doc excluded."""
    index = _build(
        [
            _fields(body=["hello"]),
            _fields(body=["hello", "hello"]),
            _fields(body=["bye"]),
        ],
        k1=1.5,
        b=0.0,
    )
    results = index.search("hello", 3)
    assert [hit.doc_id for hit in results] == [1, 0]
    assert results[0].score == pytest.approx(0.6714338, abs=1e-4)
    assert results[1].score == pytest.approx(0.4700036, abs=1e-4)
    assert index.search("hello", 3) == results


def test_case27_k_zero_negative_and_large() -> None:
    """Case 27: k=0 and k<0 are empty; large k returns all matches."""
    index = _build(
        [
            _fields(body=["hello"]),
            _fields(body=["hello", "hello"]),
            _fields(body=["bye"]),
        ],
        b=0.0,
    )
    assert index.search("hello", 0) == []
    assert index.search("hello", -1) == []
    assert len(index.search("hello", 100)) == 2


@pytest.mark.parametrize("query", ["", "   "])
def test_case28_empty_queries_return_nothing(query: str) -> None:
    """Case 28: empty or blank queries return no results."""
    index = _build([_fields(body=["hello"])], b=0.0)
    assert index.search(query, 5) == []


def test_case29_unknown_terms_return_nothing() -> None:
    """Case 29: queries with only unknown terms return no results."""
    index = _build([_fields(body=["hello"])], b=0.0)
    assert index.search("zebra quokka", 5) == []


def test_case30_identical_documents_tie_break_by_doc_id() -> None:
    """Case 30: equal scores are returned in ascending doc id order."""
    index = _build(
        [_fields(body=["hello"]), _fields(body=["hello"])],
        b=0.0,
    )
    results = index.search("hello", 5)
    assert [hit.doc_id for hit in results] == [0, 1]
    assert results[0].score == pytest.approx(results[1].score)


def test_case31_length_normalization_b_one_vs_zero() -> None:
    """Case 31: b=1 favors the short document, b=0 keeps doc id order."""
    documents = [
        _fields(body=["hello"]),
        _fields(body=["hello", "x", "y", "z", "w"]),
    ]
    normalized = _build(documents, b=1.0).search("hello", 5)
    assert [hit.doc_id for hit in normalized] == [0, 1]
    assert normalized[0].score > normalized[1].score
    flat = _build(documents, b=0.0).search("hello", 5)
    assert [hit.doc_id for hit in flat] == [0, 1]
    assert flat[0].score == pytest.approx(flat[1].score)


def test_case32_term_frequency_saturation_is_sublinear() -> None:
    """Case 32: doubling tf raises the score by less than a factor of two."""
    index = _build(
        [_fields(body=["a", "a"]), _fields(body=["a", "a", "a", "a"])],
        k1=1.5,
        b=0.0,
    )
    results = {hit.doc_id: hit.score for hit in index.search("a", 5)}
    assert results[1] > results[0]
    assert results[1] / results[0] < 2.0


def test_case33_header_weight_beats_body_weight() -> None:
    """Case 33: a header match outweighs a body match when weighted more."""
    index = _build(
        [_fields(header=["topic"]), _fields(body=["topic"])],
        weight_body=1.0,
        weight_header=3.0,
        b=0.0,
    )
    results = index.search("topic", 5)
    assert [hit.doc_id for hit in results] == [0, 1]


def test_case34_zero_weight_removes_evidence() -> None:
    """Case 34: zero header weight drops a header-only match entirely."""
    index = _build(
        [_fields(header=["topic"]), _fields(body=["topic"])],
        weight_body=1.0,
        weight_header=0.0,
        b=0.0,
    )
    assert [hit.doc_id for hit in index.search("topic", 5)] == [1]


def test_case35_path_field_matches_queries() -> None:
    """Case 35: a path-only match is retrieved and unrelated docs are not."""
    index = _build(
        [_fields(path=["openai", "server"]), _fields(body=["banana"])],
        b=0.0,
    )
    results = index.search("openai", 5)
    assert [hit.doc_id for hit in results] == [0]
    assert results[0].score > 0.0


def test_case36_common_term_keeps_positive_idf() -> None:
    """Case 36: a term in every document still scores above zero."""
    index = _build(
        [
            _fields(body=["common"]),
            _fields(body=["common"]),
            _fields(body=["common"]),
        ],
        b=0.0,
    )
    results = index.search("common", 5)
    assert len(results) == 3
    assert all(hit.score > 0.0 for hit in results)


def test_avgdl_and_doc_count_properties() -> None:
    """Extra: doc_count, is_built and avgdl report the built state."""
    index = _build(
        [_fields(body=["a", "b"]), _fields(body=["c", "d", "e"])],
        b=0.0,
    )
    assert index.doc_count == 2
    assert index.is_built is True
    assert index.avgdl["body"] == pytest.approx(2.5)


def test_to_bundle_keys_and_dtypes() -> None:
    """Extra: to_bundle exposes the ten arrays with integer doc ids."""
    index = _build([_fields(body=["hello", "hello"])], b=0.0)
    bundle = index.to_bundle()
    assert set(bundle) == {
        "vocab",
        "indptr",
        "docs",
        "tf_body",
        "tf_header",
        "tf_path",
        "len_body",
        "len_header",
        "len_path",
        "idf",
    }
    assert np.issubdtype(bundle["docs"].dtype, np.integer)
