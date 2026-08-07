"""Unit tests for document-level hybrid retrieval (dense-primary)."""

from pathlib import Path

import pytest

from markdown_rag.chunking import chunk_vault
from markdown_rag.index import build_index
from markdown_rag.retrieval import retrieve

FIXTURES = Path(__file__).parent / "fixtures" / "vault"


def make_file(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture(scope="module")
def index(tmp_path_factory):
    """A vault with clearly separable semantic + lexical topics."""
    d = tmp_path_factory.mktemp("vault")
    make_file(d, "fruit.md", "# Fruit\n\nApples are round fruits that grow on trees.")
    make_file(d, "cars.md", "# Cars\n\nCars have engines and wheels and run on roads.")
    make_file(d, "code.md", "# API\n\nEvery endpoint returns JSON. Use the HTTP method GET.")
    return build_index(chunk_vault(d))


def test_returns_documents_not_chunks(index):
    results = retrieve(index, "apples and fruit", k=2)
    assert len(results) == 2
    r = results[0]
    # document shape: title/path/text/score/citations
    assert set(r) == {"title", "path", "text", "score", "citations"}
    assert r["path"].endswith("fruit.md")
    # text is the full document, not a fragment
    assert "Apples are round" in r["text"]


def test_returns_top_k_sorted_by_score(index):
    results = retrieve(index, "apples and fruit", k=2)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_semantic_query_finds_fruit(index):
    results = retrieve(index, "what grows on trees and is round", k=3)
    assert results[0]["path"].endswith("fruit.md")


def test_lexical_query_still_finds_cars(index):
    # Exact-term query must still work via BM25 supporting pass.
    results = retrieve(index, "cars engines wheels", k=3)
    assert results[0]["path"].endswith("cars.md")


def test_citations_are_chunks_with_scores(index):
    r = retrieve(index, "apples and fruit", k=1)[0]
    assert len(r["citations"]) > 0
    c = r["citations"][0]
    assert set(c) == {"chunk", "score"}
    assert c["chunk"].startswith("Apples are round")
    assert c["score"] > 0


def test_title_from_frontmatter_or_filename(index):
    r = retrieve(index, "fruit", k=1)[0]
    # no frontmatter in fixture -> falls back to filename stem
    assert r["title"] == "fruit"


def test_k_larger_than_documents(index):
    results = retrieve(index, "fruit", k=99)
    assert len(results) == 3


def test_empty_query_returns_empty(index):
    assert retrieve(index, "", k=3) == []
