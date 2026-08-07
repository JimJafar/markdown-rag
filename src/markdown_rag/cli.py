"""Console entry point: markdown-rag serve <dir>."""

from __future__ import annotations

import argparse
import logging
import socket
from pathlib import Path

import uvicorn

from markdown_rag.index import build_index_from_vault
from markdown_rag.server import create_app


def port_available(host: str, port: int) -> bool:
    """True if host:port can be bound right now, mirroring uvicorn's bind.

    A wildcard host (0.0.0.0) conflicts with any specific-address listener
    on the same port, and vice versa, so the probe is accurate for both
    bind modes.
    """
    host_ip = socket.gethostbyname(host)  # IPv4; "localhost" -> 127.0.0.1
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host_ip, port))
        except OSError:
            return False
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="markdown-rag", description="RAG server for markdown vaults")
    sub = parser.add_subparsers(dest="subcommand")
    serve = sub.add_parser("serve", help="index a vault and serve retrieval over HTTP")
    serve.add_argument("vault", type=str, help="path to the markdown vault directory")
    serve.add_argument("--port", type=int, default=8000, help="port to listen on (default: 8000)")
    serve.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="interface to bind (default: localhost only)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.subcommand != "serve":
        build_parser().error("no command given; use `markdown-rag serve <dir>`")
        return

    vault = Path(args.vault).expanduser().resolve()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not port_available(args.host, args.port):
        logging.error(
            "port %d on %s is already in use; pick a free port with --port (e.g. --port 8123)",
            args.port,
            args.host,
        )
        raise SystemExit(1)

    logging.info("indexing vault %s ...", vault)
    index = build_index_from_vault(vault)
    logging.info("index built: %d chunks", len(index["chunks"]))

    app = create_app(index)
    logging.info("markdown-rag listening on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
