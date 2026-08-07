"""Unit tests for heading-aware markdown chunking."""

from pathlib import Path

import pytest

from markdown_rag.chunking import chunk_file, chunk_vault


FIXTURES = Path(__file__).parent / "fixtures" / "vault"


def make_file(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


# --- chunk_file ---


def test_plain_paragraph_single_chunk(tmp_path):
    p = make_file(tmp_path, "note.md", "Some plain text body.\n\nMore text.")
    chunks = chunk_file(p)
    assert len(chunks) == 1
    assert chunks[0]["heading_path"] == ""
    assert "Some plain text body" in chunks[0]["text"]


def test_heading_opens_new_chunk(tmp_path):
    p = make_file(tmp_path, "note.md", "# Overview\n\nFirst para.\n\n## Details\n\nSecond para.")
    chunks = chunk_file(p)
    assert len(chunks) == 2
    assert chunks[0]["heading_path"] == "# Overview"
    assert chunks[1]["heading_path"] == "# Overview > ## Details"
    assert "First para" in chunks[0]["text"]
    assert "Second para" in chunks[1]["text"]


def test_heading_inherits_parent_path(tmp_path):
    p = make_file(tmp_path, "note.md", "# A\n\n## B\n\n### C\n\nBody under C.")
    chunks = chunk_file(p)
    assert len(chunks) == 1
    assert chunks[0]["heading_path"] == "# A > ## B > ### C"
    assert "Body under C" in chunks[0]["text"]


def test_frontmatter_parsed_as_metadata(tmp_path):
    p = make_file(
        tmp_path,
        "note.md",
        "---\ntitle: My Note\ntags:\n  - ai\n  - rag\ndate: 2026-01-01\n---\n\n# Heading\n\nBody.",
    )
    chunks = chunk_file(p)
    assert chunks[0]["metadata"]["title"] == "My Note"
    assert chunks[0]["metadata"]["tags"] == ["ai", "rag"]
    assert "title: My Note" not in chunks[0]["text"]


def test_relative_source_path(tmp_path):
    p = make_file(tmp_path, "sub/note.md", "# Heading\n\nBody.")
    chunks = chunk_file(p)
    assert chunks[0]["path"].endswith("sub/note.md")


# --- frontmatter shapes found in the real vault ---


def test_frontmatter_inline_list(tmp_path):
    p = make_file(
        tmp_path,
        "note.md",
        "---\ntags: [ai, memory, project-idea]\ndate: 2026-07-05\n---\n\n# H\n\nBody.",
    )
    chunks = chunk_file(p)
    assert chunks[0]["metadata"]["tags"] == ["ai", "memory", "project-idea"]


def test_frontmatter_list_under_named_key(tmp_path):
    p = make_file(
        tmp_path,
        "note.md",
        "---\nrelated:\n  - \"[[north-star]]\"\n  - \"[[memory-architecture-rethink]]\"\ntags:\n  - clippings\n---\n\n# H\n\nBody.",
    )
    chunks = chunk_file(p)
    assert chunks[0]["metadata"]["related"] == [
        "[[north-star]]",
        "[[memory-architecture-rethink]]",
    ]
    assert chunks[0]["metadata"]["tags"] == ["clippings"]


def test_frontmatter_empty_key_then_list(tmp_path):
    p = make_file(
        tmp_path,
        "note.md",
        "---\nauthor:\n  - \"[[Kartikey Chauhan]]\"\n---\n\n# H\n\nBody.",
    )
    chunks = chunk_file(p)
    assert chunks[0]["metadata"]["author"] == ["[[Kartikey Chauhan]]"]


def test_frontmatter_scalar_tags_then_list_item_no_crash(tmp_path):
    p = make_file(
        tmp_path,
        "note.md",
        "---\ntags: project-idea\nstatus: living\nrelated:\n  - \"[[north-star]]\"\n---\n\n# H\n\nBody.",
    )
    chunks = chunk_file(p)
    assert chunks[0]["metadata"]["related"] == ["[[north-star]]"]
    assert chunks[0]["metadata"]["tags"] == "project-idea"


# --- chunk_vault ---


def test_chunk_vault_skips_non_markdown(tmp_path):
    make_file(tmp_path, "note.md", "# H\n\nBody.")
    make_file(tmp_path, "image.png", "not markdown")
    make_file(tmp_path, "data.csv", "a,b")
    chunks = chunk_vault(tmp_path)
    assert len(chunks) == 1
    assert chunks[0]["path"].endswith("note.md")


def test_chunk_vault_recurses(tmp_path):
    make_file(tmp_path, "top.md", "# Top\n\nBody.")
    make_file(tmp_path, "sub/deep.md", "# Deep\n\nBody.")
    chunks = chunk_vault(tmp_path)
    assert len(chunks) == 2


def test_chunk_vault_empty_dir(tmp_path):
    assert chunk_vault(tmp_path) == []


def test_chunk_vault_missing_dir():
    with pytest.raises(FileNotFoundError):
        chunk_vault(Path("/definitely/not/here"))
