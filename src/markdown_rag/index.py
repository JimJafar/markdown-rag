"""In-memory index build: local embeddings + BM25 corpus.

Builds a hybrid retrieval index (dense matrix + tokenised corpus) from
markdown chunks using a bundled local ONNX model — fully offline, CPU/RAM.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from fastembed import TextEmbedding

from markdown_rag.chunking import chunk_vault

#: Fastembed model identifier — must match a model in fastembed's registry.
MODEL_NAME = "BAAI/bge-small-en-v1.5"
#: Bundled model data shipped as package data (offline, no download).
MODEL_DATA_DIR = Path(__file__).parent / "model_data"

#: Tokenise for BM25: lowercase, split on non-alphanumeric runs.
_TOKENISE_SPLIT = None


def _tokenise(text: str) -> list[str]:
    """Simple tokeniser shared by queries and the corpus."""
    out: list[str] = []
    for word in text.lower().split():
        cleaned = "".join(ch for ch in word if ch.isalnum())
        if cleaned:
            out.append(cleaned)
    return out


class Embedder:
    """Wraps fastembed with the bundled local model (offline)."""

    def __init__(self, model_name: str = MODEL_NAME, model_dir: Path | None = MODEL_DATA_DIR) -> None:
        # Point fastembed directly at the bundled files: no download, no cache.
        self._model = TextEmbedding(
            model_name=model_name,
            specific_model_path=str(model_dir) if model_dir is not None else None,
            threads=os.cpu_count() or 1,
        )

    def embed(self, texts: Iterable[str]) -> list[np.ndarray]:
        return list(self._model.passage_embed(texts))

    def query_embed(self, query: str) -> Iterable[np.ndarray]:
        return self._model.query_embed(query)


def build_index(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the in-memory hybrid index from chunk dicts.

    Returns:
        {"chunks": [...], "embeddings": np.ndarray, "bm25_corpus": [...], "dim": int}
    """
    texts = [c["text"] for c in chunks]
    embedder = Embedder()
    vectors = list(embedder.embed(texts))

    matrix = np.array(vectors, dtype=np.float32)
    # Normalise rows so cosine similarity == dot product.
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = np.nan_to_num(matrix)

    bm25_corpus = [_tokenise(t) for t in texts]

    return {
        "chunks": chunks,
        "embeddings": matrix,
        "bm25_corpus": bm25_corpus,
        "dim": matrix.shape[1],
    }


def build_index_from_vault(vault_dir: Path) -> dict[str, Any]:
    """Chunk a vault and build its index in one call."""
    return build_index(chunk_vault(vault_dir))
