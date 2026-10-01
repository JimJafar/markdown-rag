"""Tests for incremental refresh: only added or changed notes are re-embedded."""

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from markdown_rag.refresh import LiveIndex
from markdown_rag.server import create_app


def make_file(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def touch_later(path: Path) -> None:
    """Move a file's mtime forward so a same-size rewrite still looks changed."""
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))


class FakeEmbedder:
    """Deterministic 8-dim vectors per text; records every text it embeds."""

    active_provider = "CPUExecutionProvider"

    def __init__(self) -> None:
        self.embedded: list[str] = []

    def embed(self, texts):
        texts = list(texts)
        self.embedded.extend(texts)
        vectors = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")
            vectors.append(np.random.default_rng(seed).normal(size=8).astype(np.float32))
        return vectors


@pytest.fixture
def vault(tmp_path):
    d = tmp_path / "vault"
    make_file(d, "apples.md", "# Apples\n\nApples are round fruits that grow on trees.")
    make_file(d, "cars.md", "# Cars\n\nCars have engines and wheels.")
    return d


def paths(index) -> set[str]:
    return {Path(doc["path"]).name for doc in index["documents"]}


def test_first_refresh_builds_whole_vault_in_one_batch(vault):
    embedder = FakeEmbedder()
    live = LiveIndex(vault, embedder)

    result = live.refresh()

    assert (result.added, result.changed, result.removed) == (2, 0, 0)
    assert result.chunks_embedded == 2
    assert paths(live.current) == {"apples.md", "cars.md"}
    assert live.current["embeddings"].shape == (2, 8)
    norms = np.linalg.norm(live.current["embeddings"], axis=1)
    assert np.allclose(norms, 1.0)


def test_unchanged_vault_embeds_nothing_and_keeps_the_same_index(vault):
    embedder = FakeEmbedder()
    live = LiveIndex(vault, embedder)
    live.refresh()
    before = live.current
    embedder.embedded.clear()

    result = live.refresh()

    assert (result.added, result.changed, result.removed, result.chunks_embedded) == (0, 0, 0, 0)
    assert embedder.embedded == []
    assert live.current is before


def test_added_note_is_embedded_alone(vault):
    embedder = FakeEmbedder()
    live = LiveIndex(vault, embedder)
    live.refresh()
    embedder.embedded.clear()
    make_file(vault, "sub/birds.md", "# Birds\n\nRobins are small songbirds.")

    result = live.refresh()

    assert (result.added, result.changed, result.removed) == (1, 0, 0)
    assert len(embedder.embedded) == 1 and "Robins" in embedder.embedded[0]
    assert paths(live.current) == {"apples.md", "cars.md", "birds.md"}
    assert live.current["embeddings"].shape[0] == len(live.current["chunks"]) == 3


def test_changed_note_replaces_its_old_chunks(vault):
    embedder = FakeEmbedder()
    live = LiveIndex(vault, embedder)
    live.refresh()
    embedder.embedded.clear()
    cars = vault / "cars.md"
    cars.write_text("# Cars\n\nCars have engines.\n\n## Electric\n\nBatteries power them.", encoding="utf-8")
    touch_later(cars)

    result = live.refresh()

    assert (result.added, result.changed, result.removed) == (0, 1, 0)
    assert all("Apples" not in text for text in embedder.embedded)
    texts = [c["text"] for c in live.current["chunks"]]
    assert "Cars have engines and wheels." not in texts
    assert "Batteries power them." in texts
    car_doc = next(d for d in live.current["documents"] if d["path"].endswith("cars.md"))
    assert [live.current["chunks"][i]["path"] for i in car_doc["chunk_idx"]] == [str(cars)] * 2


def test_deleted_note_is_removed_without_embedding(vault):
    embedder = FakeEmbedder()
    live = LiveIndex(vault, embedder)
    live.refresh()
    embedder.embedded.clear()
    (vault / "cars.md").unlink()

    result = live.refresh()

    assert (result.added, result.changed, result.removed) == (0, 0, 1)
    assert embedder.embedded == []
    assert paths(live.current) == {"apples.md"}
    assert live.current["embeddings"].shape[0] == 1


def test_unreadable_note_keeps_last_good_version_and_is_retried(vault):
    embedder = FakeEmbedder()
    live = LiveIndex(vault, embedder)
    live.refresh()
    cars = vault / "cars.md"
    cars.write_bytes(b"# Cars\n\n\xff\xfe not utf-8")
    touch_later(cars)

    first = live.refresh()

    assert first.failed == 1
    assert "Cars have engines and wheels." in [c["text"] for c in live.current["chunks"]]

    cars.write_text("# Cars\n\nCars are fixed now.", encoding="utf-8")
    touch_later(cars)
    second = live.refresh()

    assert (second.changed, second.failed) == (1, 0)
    assert "Cars are fixed now." in [c["text"] for c in live.current["chunks"]]


def test_vault_that_comes_up_empty_keeps_the_current_index(vault, caplog):
    # An unmounted network share looks like an empty directory; wiping the index
    # would force a full re-embed when it comes back.
    live = LiveIndex(vault, FakeEmbedder())
    live.refresh()
    before = live.current
    for note in vault.glob("*.md"):
        note.unlink()

    result = live.refresh()

    assert not result.updated
    assert live.current is before
    assert "no notes" in caplog.text


def test_empty_vault_at_start_serves_no_results(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    live = LiveIndex(empty, FakeEmbedder())

    live.refresh()

    assert live.current["documents"] == []
    assert live.current["embeddings"].shape[0] == 0


def test_missing_vault_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        LiveIndex(tmp_path / "absent", FakeEmbedder()).refresh()


def test_server_finds_a_note_added_after_start(tmp_path):
    """Integration: the real model, the HTTP API, and a refresh without restart."""
    d = tmp_path / "vault"
    make_file(d, "fruit.md", "# Fruit\n\nApples are round fruits that grow on trees.")
    make_file(d, "music.md", "# Music\n\nMusic uses melody, rhythm and harmony.")
    live = LiveIndex(d)
    live.refresh()
    client = TestClient(create_app(live, embedder=live.embedder))

    before = client.get("/retrieve", params={"q": "ocean waves and salt water", "k": 3}).json()
    make_file(d, "ocean.md", "# Ocean\n\nThe ocean contains salt water, waves and many fish.")
    live.refresh()
    after = client.get("/retrieve", params={"q": "ocean waves and salt water", "k": 3}).json()
    status = client.get("/status").json()

    assert all(not r["path"].endswith("ocean.md") for r in before)
    assert after[0]["path"].endswith("ocean.md")
    assert status["documents"] == 3
    assert status["chunks"] == 3
    assert status["last_refresh"] is not None
