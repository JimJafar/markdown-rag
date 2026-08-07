"""Tests for the CLI: markdown-rag serve <dir>."""

import socket
from pathlib import Path

import pytest

from markdown_rag.cli import build_parser, main, port_available


def make_file(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def _busy_socket() -> tuple[socket.socket, int]:
    """Bind + listen on an ephemeral port; return (socket, port)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    return s, s.getsockname()[1]


# --- port availability ---


def test_port_available_true_when_free():
    s, port = _busy_socket()
    s.close()
    assert port_available("127.0.0.1", port)


def test_port_available_false_when_busy():
    s, port = _busy_socket()
    try:
        assert not port_available("127.0.0.1", port)
    finally:
        s.close()


def test_port_available_wildcard_conflicts_with_specific_listener():
    s, port = _busy_socket()
    try:
        assert not port_available("0.0.0.0", port)
    finally:
        s.close()


def test_main_fails_fast_when_port_busy(monkeypatch, caplog):
    import markdown_rag.cli as cli

    s, port = _busy_socket()
    try:
        called: list = []

        def fake_build(vault):
            called.append(vault)
            raise AssertionError("indexing must not start when the port is busy")

        monkeypatch.setattr(cli, "build_index_from_vault", fake_build)
        with pytest.raises(SystemExit) as exc:
            cli.main(["serve", "/some/vault", "--port", str(port)])
        assert exc.value.code == 1
        assert not called
        assert "already in use" in caplog.text
    finally:
        s.close()


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
