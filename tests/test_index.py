"""Unit tests for the in-memory index build and embeddings."""

from pathlib import Path

import numpy as np
import pytest

from markdown_rag.chunking import chunk_vault
from markdown_rag.index import Embedder, _tokenise, build_index

FIXTURES = Path(__file__).parent / "fixtures" / "vault"


def make_file(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture(scope="module")
def index(tmp_path_factory):
    """Build an index once over a small fixture vault; model load is slow."""
    d = tmp_path_factory.mktemp("vault")
    make_file(d, "one.md", "# Apples\n\nApples are round fruits that grow on trees.")
    make_file(d, "two.md", "# Cars\n\nCars have engines and wheels and run on roads.")
    chunks = chunk_vault(d)
    return build_index(chunks)


def test_embedder_dim(index):
    # 384-dim bge-small-en-v1.5
    assert index["dim"] == 384
    assert index["embeddings"].shape[1] == 384


def test_embedder_embeddings_match_chunk_count(index):
    assert index["embeddings"].shape[0] == len(index["chunks"])


def test_build_index_small_vault_is_fast(index):
    # Index build over the 2-chunk fixture must be fast (< a few seconds).
    assert index["embeddings"].shape[0] == 2


def test_embedder_returns_normalised_vectors(tmp_path):
    e = Embedder()
    vec = list(e.query_embed("apple tree"))[0]
    assert len(vec) == 384
    assert np.isclose(np.linalg.norm(np.array(vec)), 1.0, atol=1e-2)


def test_index_has_bm25_corpus(index):
    assert "bm25_corpus" in index
    assert len(index["bm25_corpus"]) == len(index["chunks"])


def test_tokenise_filters_stopwords():
    # Query words like "what do we know about the" are common to every doc;
    # scoring them flattens BM25. They must be filtered out.
    assert _tokenise("what do we know about the librarian") == ["librarian"]


def test_tokenise_keeps_meaningful_tokens():
    assert "embedding" in _tokenise("embedding and retrieval")


def test_index_injects_document_title(tmp_path):
    # A chunk's embedded/BM25 text must carry the document title so identity
    # (e.g. "The Librarian" in the frontmatter title) is matchable.
    p = tmp_path / "note.md"
    index = build_index(
        [
            {
                "path": str(p),
                "heading_path": "# Intro",
                "text": "Body text.",
                "metadata": {"title": "The Librarian Notes"},
            }
        ]
    )
    # BM25 corpus for the chunk is title + text tokens.
    assert "librarian" in index["bm25_corpus"][0]


def test_index_has_documents_grouping(index):
    # build_index must group chunks by source document for doc-level retrieval.
    assert "documents" in index
    paths = {c["path"] for c in index["chunks"]}
    assert {d["path"] for d in index["documents"]} == paths
    # each document carries its chunks and title
    d = index["documents"][0]
    assert "title" in d
    assert "text" in d
    assert "chunk_idx" in d


# --- provider auto-detection (GPU first, CPU fallback) ---


def test_preferred_providers_cpu_fallback(monkeypatch):
    monkeypatch.setattr(
        "onnxruntime.get_available_providers",
        lambda: ["AzureExecutionProvider", "CPUExecutionProvider"],
    )
    from markdown_rag.index import preferred_providers

    assert preferred_providers() == ["CPUExecutionProvider"]


def test_preferred_providers_prefers_cuda_over_tensorrt(monkeypatch):
    monkeypatch.setattr(
        "onnxruntime.get_available_providers",
        lambda: ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    from markdown_rag.index import preferred_providers

    assert preferred_providers() == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_preferred_providers_rocm_when_no_cuda(monkeypatch):
    monkeypatch.setattr(
        "onnxruntime.get_available_providers",
        lambda: ["ROCMExecutionProvider", "CPUExecutionProvider"],
    )
    from markdown_rag.index import preferred_providers

    assert preferred_providers() == ["ROCMExecutionProvider", "CPUExecutionProvider"]


# --- runtime GPU probe + CPU fallback ---


def test_embedder_keeps_gpu_when_probe_succeeds(monkeypatch):
    import markdown_rag.index as idx

    built: list = []

    class FakeModel:
        def __init__(self, **kwargs):
            self.providers = kwargs["providers"]
            built.append(kwargs["providers"])

        def passage_embed(self, texts):
            yield []

    monkeypatch.setattr(idx, "TextEmbedding", FakeModel)
    e = idx.Embedder(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert e.active_provider == "CUDAExecutionProvider"
    assert len(built) == 1


def test_embedder_falls_back_to_cpu_when_gpu_probe_fails(monkeypatch, caplog):
    import markdown_rag.index as idx

    built: list = []

    class FakeModel:
        def __init__(self, **kwargs):
            self.providers = kwargs["providers"]
            built.append(kwargs["providers"])

        def passage_embed(self, texts):
            if self.providers[0] != "CPUExecutionProvider":
                raise RuntimeError("simulated missing cuDNN")
            yield []

    monkeypatch.setattr(idx, "TextEmbedding", FakeModel)
    e = idx.Embedder(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert e.active_provider == "CPUExecutionProvider"
    assert e.providers == ["CPUExecutionProvider"]
    assert len(built) == 2  # rebuilt once on CPU
    assert "falling back to CPU" in caplog.text
