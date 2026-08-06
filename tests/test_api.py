"""Integration tests for the HTTP API."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from markdown_rag.chunking import chunk_vault
from markdown_rag.index import build_index
from markdown_rag.server import create_app


def make_file(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("vault")
    make_file(d, "fruit.md", "# Fruit\n\nApples are round fruits that grow on trees.")
    make_file(d, "cars.md", "# Cars\n\nCars have engines and wheels and run on roads.")
    make_file(d, "birds.md", "# Birds\n\nRobins and sparrows are small songbirds with feathers.")
    make_file(d, "ocean.md", "# Ocean\n\nThe ocean contains salt water, waves and many fish species.")
    make_file(d, "music.md", "# Music\n\nMusic uses melody, rhythm and harmony to sound pleasing.")
    make_file(d, "weather.md", "# Weather\n\nWeather includes rain, sunshine, wind and temperature.")
    index = build_index(chunk_vault(d))
    app = create_app(index)
    return TestClient(app)


def test_health_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_retrieve_returns_chunks_sorted(client):
    r = client.get("/retrieve", params={"q": "apples and fruit", "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    assert len(body) == 2
    scores = [item["score"] for item in body]
    assert scores == sorted(scores, reverse=True)
    assert set(body[0]) == {"path", "chunk", "score"}
    assert body[0]["path"].endswith("fruit.md")


def test_retrieve_post_accepts_json(client):
    r = client.post("/retrieve", json={"query": "cars engines", "k": 1})
    assert r.status_code == 200
    body = r.json()
    assert body[0]["path"].endswith("cars.md")


def test_retrieve_missing_query_returns_422(client):
    r = client.get("/retrieve")
    assert r.status_code == 422


def test_retrieve_empty_query_returns_422(client):
    # Empty query is the same error class as a missing query: 422.
    r = client.get("/retrieve", params={"q": ""})
    assert r.status_code == 422


def test_retrieve_default_k(client):
    r = client.get("/retrieve", params={"q": "fruit"})
    assert r.status_code == 200
    assert len(r.json()) == 5
