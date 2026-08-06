"""Heading-aware chunking for structured markdown vaults.

Reads a directory of markdown files and splits each into chunks whose
boundaries respect the heading structure. Each chunk carries its heading
path (e.g. ``# Overview > ## Details``) as context, plus frontmatter-derived
metadata.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from markdown_it import MarkdownIt

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*(?:\n|$)", re.DOTALL)


def _parse_frontmatter(block: str) -> dict[str, Any]:
    """Parse a minimal YAML-like frontmatter block. Returns {} when malformed."""
    metadata: dict[str, Any] = {}
    current_key: str | None = None
    list_mode = False
    for raw_line in block.strip().splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 0 and ": " in line:
            key, _, value = line.partition(": ")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            metadata[key] = value
            current_key = key
            list_mode = False
        elif line.startswith("  - ") or line.lstrip().startswith("- "):
            value = line.strip().lstrip("- ").strip().strip('"').strip("'")
            if list_mode and current_key is not None:
                existing = metadata[current_key]
                if isinstance(existing, list):
                    existing.append(value)
                else:
                    metadata[current_key] = [existing, value]
            else:
                metadata.setdefault("tags", []).append(value)
                current_key = "tags"
                list_mode = True
    return metadata


def _split_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Split leading YAML frontmatter from the body. Returns (metadata, body)."""
    match = _FRONTMATTER_RE.match(content)
    if match is None:
        return {}, content
    return _parse_frontmatter(match.group(1)), content[match.end() :]


_LEVELS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}


def _heading_level_from_tag(token) -> int:
    """Extract heading level (1-6) from a token's hN tag, else 0."""
    if token.type != "heading_open":
        return 0
    return _LEVELS.get(getattr(token, "tag", ""), 0)


def _inline_text(tokens: list[Any]) -> str:
    """Extract plain text from a block of inline tokens."""
    parts: list[str] = []
    for tok in tokens:
        if tok.type == "text" or tok.type.endswith("_open"):
            if getattr(tok, "content", ""):
                parts.append(tok.content)
        elif tok.type == "code_inline":
            parts.append(tok.content)
    return "".join(parts).strip()


def chunk_file(path: Path) -> list[dict[str, Any]]:
    """Chunk a single markdown file. Returns a list of chunk dicts."""
    content = path.read_text(encoding="utf-8")
    metadata, body = _split_frontmatter(content)

    md = MarkdownIt("commonmark", {"max_nesting": 20}).enable("table").enable("strikethrough")
    tokens = md.parse(body)

    chunks: list[dict[str, Any]] = []
    heading_stack: list[tuple[int, str]] = []  # (level, title)
    pending_text: list[Any] = []

    def flush() -> None:
        nonlocal pending_text
        if not pending_text:
            return
        text = _inline_text(pending_text)
        if text.strip():
            heading_path = " > ".join(title for _, title in heading_stack)
            chunks.append(
                {
                    "path": str(path),
                    "heading_path": heading_path,
                    "text": text,
                    "metadata": dict(metadata),
                }
            )
        pending_text = []

    i = 0
    n = len(tokens)
    while i < n:
        token = tokens[i]
        level = _heading_level_from_tag(token)
        if level:
            flush()
            # Consume heading_open -> inline (title) -> heading_close.
            title = ""
            if i + 1 < n and tokens[i + 1].type == "inline":
                title = tokens[i + 1].content.strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            if title:
                heading_stack.append((level, f"{'#' * level} {title}"))
            i += 1
            # Skip the inline title token and heading_close if present.
            while i < n and tokens[i].type in ("inline", "heading_close"):
                i += 1
            continue
        if token.type == "inline":
            pending_text.extend(token.children or [])
        elif token.type == "fence":
            pending_text.append(token)
        i += 1

    flush()
    return chunks


def chunk_vault(vault_dir: Path) -> list[dict[str, Any]]:
    """Chunk every markdown file under a vault directory (recursive)."""
    if not vault_dir.exists():
        raise FileNotFoundError(vault_dir)
    chunks: list[dict[str, Any]] = []
    for path in sorted(vault_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() == ".md":
            chunks.extend(chunk_file(path))
    return chunks
