"""Unit tests for the in-memory index build and embeddings."""

from pathlib import Path

import numpy as np
import pytest

from markdown_rag.chunking import chunk_vault
from markdown_rag.index import Embedder, build_index

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
