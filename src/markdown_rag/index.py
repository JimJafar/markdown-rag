"""In-memory index build: local embeddings + BM25 corpus.

Builds a hybrid retrieval index (dense matrix + tokenised corpus) from
markdown chunks using a bundled local ONNX model — fully offline, CPU/RAM.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnxruntime
from fastembed import TextEmbedding

from markdown_rag.chunking import chunk_vault

#: Fastembed model identifier — must match a model in fastembed's registry.
MODEL_NAME = "BAAI/bge-small-en-v1.5"
#: Bundled model data shipped as package data (offline, no download).
MODEL_DATA_DIR = Path(__file__).parent / "model_data"

#: onnxruntime intra-op threads. Measured on a 20-core box: 20 threads reach
#: ~24 vec/s peak, 8 threads ~20 vec/s (82% of peak) while leaving the rest
#: of the machine usable for other work — the right default on a shared host.
DEFAULT_THREADS = 8

#: GPU providers, in preference order. The first one this onnxruntime build
#: exposes wins; a plain (CPU) `onnxruntime` exposes none of them.
_GPU_PROVIDERS = (
    "CUDAExecutionProvider",
    "TensorrtExecutionProvider",
    "ROCMExecutionProvider",
    "DmlExecutionProvider",
    "CoreMLExecutionProvider",
)


def preferred_providers() -> list[str]:
    """Best available onnxruntime providers: GPU first, CPU as fallback.

    GPU providers only appear when an onnxruntime build with GPU support is
    installed (e.g. the `fastembed-gpu` package, which brings onnxruntime-gpu);
    the plain build always yields ["CPUExecutionProvider"].
    """
    available = onnxruntime.get_available_providers()
    for gpu in _GPU_PROVIDERS:
        if gpu in available:
            return [gpu, "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]

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
    """Wraps fastembed with the bundled local model (offline).

    Providers default to auto-detected GPU-first/CPU-fallback. If the chosen
    GPU provider cannot actually run (e.g. missing CUDA/cuDNN runtime libs),
    a probe embed at construction fails and the embedder rebuilds on CPU,
    logging a warning — the server still starts.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        model_dir: Path | None = MODEL_DATA_DIR,
        threads: int | None = DEFAULT_THREADS,
        providers: list[str] | None = None,
    ) -> None:
        # Point fastembed directly at the bundled files: no download, no cache.
        # threads=None explicitly opts into all cores (os.cpu_count());
        # providers=None auto-detects GPU-then-CPU.
        self._model_name = model_name
        self._model_dir = model_dir
        self._threads = threads or os.cpu_count() or 1
        self.providers = list(providers if providers is not None else preferred_providers())
        self.active_provider = self.providers[0]
        self._model = self._build(self.providers)
        self._probe_and_fallback()

    def _build(self, providers: list[str]) -> TextEmbedding:
        return TextEmbedding(
            model_name=self._model_name,
            specific_model_path=str(self._model_dir) if self._model_dir is not None else None,
            threads=self._threads,
            providers=providers,
        )

    def _probe_and_fallback(self) -> None:
        """Run one tiny embed; on GPU failure rebuild the session on CPU."""
        if self.active_provider == "CPUExecutionProvider":
            return
        try:
            list(self._model.passage_embed(["onnxruntime provider probe"]))
        except Exception:
            logging.warning(
                "GPU provider %s failed to run (missing CUDA/cuDNN libraries?); "
                "falling back to CPU",
                self.active_provider,
            )
            self.providers = ["CPUExecutionProvider"]
            self.active_provider = "CPUExecutionProvider"
            self._model = self._build(self.providers)

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
    logging.info("embedding %d chunks on %s", len(texts), embedder.active_provider)
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
