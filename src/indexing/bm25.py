"""Okapi BM25F over an inverted index of three chunk fields."""
from collections import Counter
from typing import Any, Dict, List, Mapping, NamedTuple, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from src.config import (
    DEFAULT_B,
    DEFAULT_K1,
    DEFAULT_WEIGHT_BODY,
    DEFAULT_WEIGHT_HEADER,
    DEFAULT_WEIGHT_PATH,
    IndexConfig,
)
from src.indexing.tokenizer import tokenize

FIELDS: Tuple[str, ...] = ("body", "header", "path")

_BUNDLE_KEYS: Tuple[str, ...] = (
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
)


class ScoredChunk(NamedTuple):
    """A ranked document id with its BM25F score."""

    doc_id: int
    score: float


def _clear_accumulators(
    postings: Mapping[str, Dict[str, Dict[int, int]]],
    lengths: Mapping[str, List[int]],
) -> None:
    """Empty the accumulation dicts in place after a build.

    Args:
        postings: Per-field token postings to clear.
        lengths: Per-field token counts to clear.
    """
    for field in FIELDS:
        postings[field].clear()
        lengths[field].clear()


class BM25Index:
    """In-memory BM25F index with CSR postings, ready to persist.

    Args:
        k1: Term-frequency saturation; must be > 0.
        b: Length normalization in [0, 1].
        weight_body: Weight of the chunk text field.
        weight_header: Weight of the first-non-blank-line field.
        weight_path: Weight of the corpus-relative path field.

    Raises:
        ValueError: On out-of-range parameters.
    """

    def __init__(
        self,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
        weight_body: float = DEFAULT_WEIGHT_BODY,
        weight_header: float = DEFAULT_WEIGHT_HEADER,
        weight_path: float = DEFAULT_WEIGHT_PATH,
    ) -> None:
        """Validate the parameters and start an empty, unbuilt index."""
        if k1 <= 0.0:
            raise ValueError(f"k1 must be > 0, got {k1}")
        if not 0.0 <= b <= 1.0:
            raise ValueError(f"b must be in [0, 1], got {b}")
        if weight_body < 0.0:
            raise ValueError(f"weight_body must be >= 0, got {weight_body}")
        if weight_header < 0.0:
            raise ValueError(
                f"weight_header must be >= 0, got {weight_header}"
            )
        if weight_path < 0.0:
            raise ValueError(f"weight_path must be >= 0, got {weight_path}")
        self._k1 = k1
        self._b = b
        self._weights: Dict[str, float] = {
            "body": weight_body,
            "header": weight_header,
            "path": weight_path,
        }
        self._postings: Dict[str, Dict[str, Dict[int, int]]] = {
            field: {} for field in FIELDS
        }
        self._lengths: Dict[str, List[int]] = {
            field: [] for field in FIELDS
        }
        self._doc_count = 0
        self._built = False
        self._token_to_row: Dict[str, int] = {}
        self._avgdl: Dict[str, float] = {}
        self._arrays: Dict[str, NDArray[Any]] = {}

    @property
    def doc_count(self) -> int:
        """Number of accumulated documents (0 until ``build``)."""
        return self._doc_count

    @property
    def is_built(self) -> bool:
        """Whether ``build`` has frozen the index."""
        return self._built

    @property
    def avgdl(self) -> Dict[str, float]:
        """Average length per field (``{}`` until ``build``)."""
        return dict(self._avgdl)

    def add_document(
        self,
        doc_id: int,
        fields: Mapping[str, Sequence[str]],
    ) -> None:
        """Accumulate one document's tokens before ``build``.

        Args:
            doc_id: Must equal the number of documents already added
                (sequential from 0), so doc id ``i`` always refers to the
                ``i``-th persisted chunk record.
            fields: Token sequences keyed by exactly ``FIELDS``. Empty
                sequences are allowed (a punctuation-only chunk).

        Raises:
            ValueError: If the index is already built, a field name is
                missing or extra, or ``doc_id`` breaks the sequence.
        """
        if self._built:
            raise ValueError("cannot add documents to a built index")
        missing = [field for field in FIELDS if field not in fields]
        if missing:
            raise ValueError(f"missing field(s): {', '.join(missing)}")
        extra = [key for key in fields if key not in FIELDS]
        if extra:
            raise ValueError(f"unknown field(s): {', '.join(extra)}")
        if doc_id != self._doc_count:
            raise ValueError(
                f"doc_id must equal doc_count {self._doc_count}, "
                f"got {doc_id}"
            )
        for field in FIELDS:
            tokens = list(fields[field])
            self._lengths[field].append(len(tokens))
            for token, count in Counter(tokens).items():
                self._postings[field].setdefault(token, {})[doc_id] = count
        self._doc_count += 1

    def build(self) -> None:
        """Freeze accumulators into sorted CSR arrays and compute idf/avgdl.

        Vocabulary becomes the lexicographically sorted list of tokens seen
        in any field (determinism), postings stay ascending by doc id, and
        the accumulation dicts are released.

        Raises:
            ValueError: If no document was added or ``build`` already ran.
        """
        if self._built:
            raise ValueError("index has already been built")
        if self._doc_count < 1:
            raise ValueError("cannot build an index with no documents")
        seen: set[str] = set()
        for field in FIELDS:
            seen.update(self._postings[field].keys())
        vocab = sorted(seen)
        doc_total = self._doc_count
        indptr = np.zeros(len(vocab) + 1, dtype=np.int64)
        docs: List[int] = []
        tf_lists: Dict[str, List[int]] = {field: [] for field in FIELDS}
        for row, token in enumerate(vocab):
            by_field = [
                self._postings[field].get(token, {}) for field in FIELDS
            ]
            ordered = sorted(set().union(*(map(dict.keys, by_field))))
            indptr[row + 1] = indptr[row] + len(ordered)
            for doc in ordered:
                docs.append(doc)
                for field, bucket in zip(FIELDS, by_field):
                    tf_lists[field].append(bucket.get(doc, 0))
        self._arrays["vocab"] = np.array(vocab, dtype=np.str_)
        self._arrays["indptr"] = indptr
        self._arrays["docs"] = np.array(docs, dtype=np.int32)
        for field in FIELDS:
            self._arrays[f"tf_{field}"] = np.array(
                tf_lists[field], dtype=np.int32
            )
            length_array = np.array(self._lengths[field], dtype=np.int64)
            self._arrays[f"len_{field}"] = length_array
            average = (
                float(length_array.sum()) / doc_total
                if doc_total
                else 0.0
            )
            self._avgdl[field] = average
        df = np.diff(indptr).astype(np.float64)
        self._arrays["idf"] = np.log(
            1.0 + (doc_total - df + 0.5) / (df + 0.5)
        )
        self._token_to_row = {
            token: row for row, token in enumerate(vocab)
        }
        _clear_accumulators(self._postings, self._lengths)
        self._built = True

    def _norms(
        self, field: str, docs: NDArray[Any]
    ) -> NDArray[Any]:
        """Return the per-document length norm of one field.

        Args:
            field: One of ``FIELDS``.
            docs: Document ids indexing into the length array.

        Returns:
            ``1 - b + b * len / avgdl`` per document, or ones when the
            average length is not positive.
        """
        average = self._avgdl.get(field, 0.0)
        if average <= 0.0:
            return np.ones(docs.shape, dtype=np.float64)
        lengths = np.asarray(
            self._arrays[f"len_{field}"][docs], dtype=np.float64
        )
        return 1.0 - self._b + self._b * lengths / average

    def search(self, query: str, k: int) -> List[ScoredChunk]:
        """Return the top-k documents sharing at least one query term.

        Args:
            query: Natural-language question or identifier.
            k: Maximum number of results; ``k <= 0`` returns ``[]``.

        Returns:
            Ranked hits, best first, ties broken by ascending doc id;
            only documents with a strictly positive score.

        Raises:
            ValueError: If the index has not been built.
        """
        if not self._built:
            raise ValueError("index has not been built")
        if k <= 0:
            return []
        seen: set[str] = set()
        unique_tokens: List[str] = []
        for token in tokenize(query):
            if token not in seen:
                seen.add(token)
                unique_tokens.append(token)
        if not unique_tokens:
            return []
        scores = np.zeros(self._doc_count, dtype=np.float64)
        for token in unique_tokens:
            row = self._token_to_row.get(token)
            if row is None:
                continue
            indptr = self._arrays["indptr"]
            start = int(indptr[row])
            stop = int(indptr[row + 1])
            if start == stop:
                continue
            docs = self._arrays["docs"][start:stop]
            tf_hat = np.zeros(docs.shape, dtype=np.float64)
            for field in FIELDS:
                term_frequency = (
                    self._arrays[f"tf_{field}"][start:stop].astype(
                        np.float64
                    )
                )
                tf_hat += (
                    self._weights[field]
                    * term_frequency
                    / self._norms(field, docs)
                )
            idf = float(self._arrays["idf"][row])
            contribution = (
                idf * tf_hat * (self._k1 + 1.0) / (self._k1 + tf_hat)
            )
            np.add.at(scores, docs, contribution)
        candidates = np.nonzero(scores > 0.0)[0]
        if candidates.size == 0:
            return []
        order = np.lexsort((candidates, -scores[candidates]))
        selected = candidates[order][:k]
        return [
            ScoredChunk(int(doc_id), float(scores[doc_id]))
            for doc_id in selected
        ]

    def to_bundle(self) -> Dict[str, NDArray[Any]]:
        """Export every array needed to persist the index.

        Raises:
            ValueError: If the index has not been built.
        """
        if not self._built:
            raise ValueError("index has not been built")
        return {
            "vocab": self._arrays["vocab"],
            "indptr": self._arrays["indptr"],
            "docs": self._arrays["docs"],
            "tf_body": self._arrays["tf_body"],
            "tf_header": self._arrays["tf_header"],
            "tf_path": self._arrays["tf_path"],
            "len_body": self._arrays["len_body"],
            "len_header": self._arrays["len_header"],
            "len_path": self._arrays["len_path"],
            "idf": self._arrays["idf"],
        }

    @classmethod
    def from_bundle(
        cls,
        bundle: Mapping[str, NDArray[Any]],
        avgdl: Mapping[str, float],
        config: IndexConfig,
    ) -> "BM25Index":
        """Reconstruct a built index from persisted arrays.

        Args:
            bundle: Arrays exactly as ``to_bundle`` produced them.
            avgdl: Average field lengths from the metadata file.
            config: Carries ``k1``, ``b`` and the field weights.

        Returns:
            A built index answering ``search`` immediately.

        Raises:
            ValueError: If a key of ``_BUNDLE_KEYS`` is missing, a length
                invariant is violated, ``docs`` is not an integer array,
                ``avgdl`` does not cover exactly ``FIELDS`` with
                non-negative values, or the index would be empty.
        """
        missing = [key for key in _BUNDLE_KEYS if key not in bundle]
        if missing:
            raise ValueError(f"missing bundle key(s): {', '.join(missing)}")
        arrays = {key: np.asarray(bundle[key]) for key in _BUNDLE_KEYS}
        vocab_size = arrays["vocab"].shape[0]
        if arrays["indptr"].shape[0] != vocab_size + 1:
            raise ValueError(
                "indptr must have vocab_size + 1 entries, got "
                f"{arrays['indptr'].shape[0]} for vocab {vocab_size}"
            )
        postings = int(arrays["indptr"][-1])
        for key in ("docs", "tf_body", "tf_header", "tf_path"):
            if arrays[key].shape[0] != postings:
                raise ValueError(
                    f"{key} must have indptr[-1]={postings} entries, "
                    f"got {arrays[key].shape[0]}"
                )
        if arrays["idf"].shape[0] != vocab_size:
            raise ValueError(
                f"idf must have vocab_size={vocab_size} entries, "
                f"got {arrays['idf'].shape[0]}"
            )
        doc_total = arrays["len_body"].shape[0]
        if doc_total < 1:
            raise ValueError("index must contain at least one document")
        for key in ("len_header", "len_path"):
            if arrays[key].shape[0] != doc_total:
                raise ValueError(
                    f"{key} must have {doc_total} entries, "
                    f"got {arrays[key].shape[0]}"
                )
        if not np.issubdtype(arrays["docs"].dtype, np.integer):
            raise ValueError("docs must be an integer array")
        if set(avgdl.keys()) != set(FIELDS):
            raise ValueError(
                f"avgdl must cover exactly {FIELDS}, got "
                f"{tuple(avgdl.keys())}"
            )
        if any(value < 0.0 for value in avgdl.values()):
            raise ValueError("avgdl values must be non-negative")
        index = cls(
            k1=config.k1,
            b=config.b,
            weight_body=config.weight_body,
            weight_header=config.weight_header,
            weight_path=config.weight_path,
        )
        index._arrays = arrays
        index._doc_count = doc_total
        index._avgdl = {field: float(avgdl[field]) for field in FIELDS}
        index._token_to_row = {
            str(token): row
            for row, token in enumerate(arrays["vocab"].tolist())
        }
        index._built = True
        return index


__all__ = ["BM25Index", "FIELDS", "ScoredChunk"]
