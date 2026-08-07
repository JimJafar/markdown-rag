"""Document-level hybrid retrieval: dense-primary, BM25-supporting.

Scores every chunk (dense cosine + BM25), aggregates to the document
level, and returns whole documents with their best-matching chunks as
citations — so an agent can answer "what do we know about X" from the
full notes, with the retrieved chunks proving where the answer came from.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from markdown_rag.index import Embedder, _tokenise

#: Weight of the BM25 (lexical) contribution vs. dense. Dense is the
#: primary signal: in a folder-organised vault the same term (e.g. the
#: folder name) recurs in nearly every document, so lexical frequency is a
#: weak discriminator. BM25 only rescues exact-term queries.
_BM25_WEIGHT = 0.3
#: Number of chunk citations to include per returned document.
_MAX_CITATIONS = 3


def _round6(value: Any) -> float:
    """Safely round a numpy scalar to 6dp for JSON output."""
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return 0.0


def _dense_scores(matrix: np.ndarray, query_vector: np.ndarray) -> np.ndarray:
    """Cosine similarity between a normalised query vector and rows of a
    normalised document matrix (dot product)."""
    query = np.asarray(query_vector, dtype=np.float32)
    norm = np.linalg.norm(query)
    if norm == 0:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    return matrix @ (query / norm)


def _aggregate_to_documents(
    chunk_scores: np.ndarray,
    documents: list[dict[str, Any]],
) -> np.ndarray:
    """Aggregate per-chunk scores to per-document scores (max over chunks)."""
    doc_scores = np.full(len(documents), -np.inf, dtype=np.float64)
    for doc_idx, doc in enumerate(documents):
        doc_scores[doc_idx] = np.max(chunk_scores[doc["chunk_idx"]])
    return doc_scores


def _bm25_doc_scores(index: dict[str, Any], query_tokens: list[str]) -> np.ndarray:
    """BM25 scores aggregated to documents, or zeros when nothing matches."""
    n_docs = len(index["documents"])
    if not query_tokens:
        return np.zeros(n_docs, dtype=np.float64)
    bm25 = BM25Okapi(index["bm25_corpus"])
    chunk_scores = bm25.get_scores(query_tokens)
    return _aggregate_to_documents(chunk_scores, index["documents"])


def retrieve(
    index: dict[str, Any],
    query: str,
    k: int = 5,
    embedder: Embedder | None = None,
) -> list[dict[str, Any]]:
    """Return the top-k documents for a query.

    Each result: {title, path, text, score, citations} where citations are
    the document's best-matching chunks [{chunk, score}]. Sorted by
    relevance, highest first. Empty query returns [].
    """
    if not query.strip():
        return []

    query_tokens = _tokenise(query)

    # Dense: embed the query with the same local model. Reuse a shared
    # embedder when provided (servers) to avoid reloading the model per call.
    model = embedder if embedder is not None else Embedder()
    query_vector = np.asarray(list(model.query_embed(query))[0], dtype=np.float32)

    dense = _dense_scores(index["embeddings"], query_vector)
    doc_dense = _aggregate_to_documents(dense, index["documents"])
    doc_lexical = _bm25_doc_scores(index, query_tokens)

    # Normalise lexical scores to [0, 1] so the weighting is scale-free.
    lex_max = np.max(doc_lexical) if len(doc_lexical) else 0.0
    if lex_max > 0:
        doc_lexical = doc_lexical / lex_max

    # Dense-primary fusion: dense + a supporting lexical term.
    fused = doc_dense + _BM25_WEIGHT * doc_lexical

    ranked = sorted(range(len(fused)), key=lambda i: fused[i], reverse=True)

    results: list[dict[str, Any]] = []
    for doc_idx in ranked[:k]:
        doc = index["documents"][doc_idx]
        # Citations: this document's chunks ranked by dense score.
        chunk_idxs = sorted(
            doc["chunk_idx"],
            key=lambda ci: dense[ci],
            reverse=True,
        )[:_MAX_CITATIONS]
        citations = [
            {"chunk": index["chunks"][ci]["text"], "score": _round6(dense[ci])}
            for ci in chunk_idxs
        ]
        results.append(
            {
                "title": doc["title"],
                "path": doc["path"],
                "text": doc["text"],
                "score": _round6(fused[doc_idx]),
                "citations": citations,
            }
        )
    return results
