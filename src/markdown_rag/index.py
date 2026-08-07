"""In-memory index build: local embeddings + BM25 corpus.

Builds a hybrid retrieval index (dense matrix + tokenised corpus) from
markdown chunks using a bundled local ONNX model — fully offline, CPU/RAM.
"""

from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnxruntime
from fastembed import TextEmbedding

from markdown_rag.chunking import chunk_vault


#: pip nvidia wheel names whose .so libs onnxruntime-gpu dlopens at session
#: creation (cuDNN, cuBLAS, cuFFT, NVRTC, NVJITLink).
_NVIDIA_WHEELS = (
    "nvidia-cudnn-cu12",
    "nvidia-cublas-cu12",
    "nvidia-cufft-cu12",
    "nvidia-cuda-nvrtc-cu12",
    "nvidia-nvjitlink-cu12",
)


def _preload_bundled_nvidia_libs() -> None:
    """dlopen the nvidia CUDA libs bundled in the same site-packages so the
    onnxruntime CUDA provider can find cuDNN/cuBLAS without the user setting
    LD_LIBRARY_PATH. Loading by absolute path registers the SONAMEs in the
    process, so onnxruntime's later dlopen of e.g. libcudnn.so.9 succeeds.
    No-op when no nvidia wheels are installed (plain CPU install)."""
    try:
        from importlib.metadata import PackageNotFoundError, distribution
    except ImportError:  # pragma: no cover - python <3.10
        return

    lib_paths: list[Path] = []
    for wheel in _NVIDIA_WHEELS:
        try:
            dist = distribution(wheel)
        except PackageNotFoundError:
            continue
        for f in dist.files or ():
            p = Path(str(dist.locate_file(f)))
            # Match lib*.so* (versioned SONAMEs like libcudnn.so.9 have
            # suffixes .9, not .so).
            if p.name.startswith("lib") and ".so" in p.name:
                lib_paths.append(p)

    if not lib_paths:
        return

    # Expose the dirs to the loader too (belt and braces for subprocesses).
    dirs = sorted({str(p.parent) for p in lib_paths})
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(dirs + ([existing] if existing else []))

    # Load in dependency order (nvjitlink first, cudnn last). Best effort:
    # a missing driver lib (libcuda) must not crash the import.
    order = {"nvidia-nvjitlink-cu12": 0, "nvidia-cublas-cu12": 1, "nvidia-cufft-cu12": 2, "nvidia-cuda-nvrtc-cu12": 3, "nvidia-cudnn-cu12": 4}
    lib_paths.sort(key=lambda p: next((order.get(w, 9) for w in _NVIDIA_WHEELS if w in str(p)), 9))
    for p in lib_paths:
        try:
            ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            # Missing transitive dep (e.g. libcuda.so.1 driver) — leave it;
            # the Embedder probe/fallback handles an unusable GPU provider.
            continue


# Preload before any session is created.
_preload_bundled_nvidia_libs()

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

#: Stopwords filtered from BM25 tokens. Query words like "what do we know
#: about the" occur in nearly every document; scoring them flattens BM25
#: scores to a useless plateau, so they are removed.
_STOPWORDS = frozenset(
    """
    a about an and are as at be been but by can could did do does for from
    had has have he her hers him his how i if in into is it its know me
    might more most my no nor not of on or our ours she should so some such
    than that the their theirs them then there these they this those to too
    up upon us was we were what when where which while who whom why will
    with would you your yours
    """.split()
)


def _tokenise(text: str) -> list[str]:
    """Tokeniser shared by queries and the corpus; filters stopwords and
    single-character tokens."""
    out: list[str] = []
    for word in text.lower().split():
        cleaned = "".join(ch for ch in word if ch.isalnum())
        if cleaned and len(cleaned) > 1 and cleaned not in _STOPWORDS:
            out.append(cleaned)
    return out


def _doc_title(chunk: dict[str, Any]) -> str:
    """Document title for a chunk: frontmatter title, else filename stem."""
    title = chunk.get("metadata", {}).get("title")
    if title:
        return str(title)
    return Path(chunk["path"]).stem


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

    Each chunk is embedded and BM25'd as ``<document title>. <chunk text>``
    so the document's identity (frontmatter title) is matchable — without
    this, a note titled "The Librarian" is invisible to its own name. The
    path is deliberately NOT included in the weighted text: folder names
    like ``Work/The Librarian/`` repeat across many documents and would
    flood the ranking with a ubiquitous token.

    Returns:
        {"chunks": [...], "embeddings": np.ndarray, "bm25_corpus": [...],
         "dim": int, "documents": [...]}
    """
    texts = [f"{_doc_title(c)}. {c['text']}" for c in chunks]
    embedder = Embedder()
    logging.info("embedding %d chunks on %s", len(texts), embedder.active_provider)
    vectors = list(embedder.embed(texts))

    matrix = np.array(vectors, dtype=np.float32)
    # Normalise rows so cosine similarity == dot product.
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = np.nan_to_num(matrix)

    bm25_corpus = [_tokenise(t) for t in texts]

    documents = _group_documents(chunks)

    return {
        "chunks": chunks,
        "embeddings": matrix,
        "bm25_corpus": bm25_corpus,
        "dim": matrix.shape[1],
        "documents": documents,
    }


def _group_documents(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group chunks by source file into document records.

    Returns a list of {"title", "path", "text", "chunk_idx"} — the full
    document text joined from its chunks, plus the indices of its chunks in
    the flat chunk list (for citation mapping).
    """
    by_path: dict[str, list[int]] = {}
    for i, c in enumerate(chunks):
        by_path.setdefault(c["path"], []).append(i)

    documents: list[dict[str, Any]] = []
    for path, idxs in by_path.items():
        doc_chunks = [chunks[i] for i in idxs]
        title = _doc_title(doc_chunks[0])
        text = "\n\n".join(c["text"] for c in doc_chunks)
        documents.append(
            {
                "title": title,
                "path": path,
                "text": text,
                "chunk_idx": idxs,
            }
        )
    return documents


def build_index_from_vault(vault_dir: Path) -> dict[str, Any]:
    """Chunk a vault and build its index in one call."""
    return build_index(chunk_vault(vault_dir))
