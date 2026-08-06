"""Hybrid retrieval: BM25 (lexical) fused with dense cosine via RRF."""

from __future__ import annotations

from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from markdown_rag.index import Embedder, _tokenise

#: Number of candidates taken from each retriever before fusion.
_POOL_SIZE = 60
#: Reciprocal-rank fusion constant.
_RRF_K = 60


def _dense_scores(matrix: np.ndarray, query_vector: np.ndarray) -> np.ndarray:
    """Cosine similarity between a normalised query vector and rows of a
    normalised document matrix (dot product)."""
    query = np.asarray(query_vector, dtype=np.float32)
    norm = np.linalg.norm(query)
    if norm == 0:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    return matrix @ (query / norm)


def _bm25_candidates(index: dict[str, Any], query_tokens: list[str]) -> list[tuple[int, float]]:
    """Return [(doc_index, score)] for corpus tokens via BM25Okapi."""
    bm25 = BM25Okapi(index["bm25_corpus"])
    scores = bm25.get_scores(query_tokens)
    ordered = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [(i, scores[i]) for i in ordered if scores[i] > 0]


def _dense_candidates(index: dict[str, Any], query_vector: np.ndarray) -> list[tuple[int, float]]:
    scores = _dense_scores(index["embeddings"], query_vector)
    ordered = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [(i, scores[i]) for i in ordered]


def _rrf_fuse(
    *ranked_lists: list[tuple[int, float]],
) -> dict[int, float]:
    """Fuse ranked candidate lists by reciprocal-rank fusion."""
    fused: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (doc_idx, _score) in enumerate(ranked):
            fused[doc_idx] = fused.get(doc_idx, 0.0) + 1.0 / (_RRF_K + rank)
    return fused


def retrieve(
    index: dict[str, Any],
    query: str,
    k: int = 5,
    embedder: Embedder | None = None,
) -> list[dict[str, Any]]:
    """Return the top-k chunks for a query: [{path, chunk, score}] sorted by
    relevance (highest first). Empty query returns []"""
    if not query.strip():
        return []

    query_tokens = _tokenise(query)

    # Dense: embed the query with the same local model. Reuse a shared
    # embedder when provided (servers) to avoid reloading the model per call.
    model = embedder if embedder is not None else Embedder()
    query_vector = np.asarray(list(model.query_embed(query))[0], dtype=np.float32)

    dense = _dense_candidates(index, query_vector)[:_POOL_SIZE]
    lexical = _bm25_candidates(index, query_tokens)[:_POOL_SIZE]

    fused = _rrf_fuse(dense, lexical)
    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

    chunks = index["chunks"]
    results: list[dict[str, Any]] = []
    for doc_idx, score in ranked[:k]:
        results.append(
            {
                "path": chunks[doc_idx]["path"],
                "chunk": chunks[doc_idx]["text"],
                "score": round(float(score), 6),
            }
        )
    return results
