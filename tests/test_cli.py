"""Tests for the CLI: markdown-rag serve <dir>."""

from pathlib import Path

from markdown_rag.cli import build_parser


def make_file(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def test_parser_requires_serve_and_dir():
    p = build_parser()
    # No subcommand -> error
    assert p.parse_args(["serve", "/some/vault"]).subcommand == "serve"
    assert p.parse_args(["serve", "/some/vault"]).vault == "/some/vault"


def test_parser_default_port():
    p = build_parser()
    args = p.parse_args(["serve", "/vault"])
    assert args.port == 8000


def test_parser_custom_port():
    p = build_parser()
    args = p.parse_args(["serve", "/vault", "--port", "8080"])
    assert args.port == 8080
