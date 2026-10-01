"""FastAPI app exposing retrieval + health endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

from markdown_rag.index import Embedder
from markdown_rag.refresh import LiveIndex
from markdown_rag.retrieval import retrieve


class RetrieveRequest(BaseModel):
    """JSON body for POST /retrieve."""

    query: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=100)


def create_app(index: dict[str, Any] | LiveIndex, embedder: Embedder | None = None) -> FastAPI:
    """Build the FastAPI app over a fixed index or a LiveIndex that refreshes itself."""
    app = FastAPI(title="markdown-rag", version="0.1.0")
    # One long-lived embedder shared across requests (model loaded once).
    embedder = embedder if embedder is not None else Embedder()
    live = index if isinstance(index, LiveIndex) else None

    def current() -> dict[str, Any]:
        # Read the reference once per request: a refresh swaps it, never mutates it.
        return live.current if live is not None else index

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/status")
    def status() -> dict[str, Any]:
        snapshot = current()
        last = live.last_refresh.isoformat() if live and live.last_refresh else None
        return {
            "documents": len(snapshot["documents"]),
            "chunks": len(snapshot["chunks"]),
            "last_refresh": last,
        }

    @app.get("/retrieve")
    def retrieve_get(
        q: str = Query(min_length=1),
        k: int = Query(default=5, ge=1, le=100),
    ) -> list[dict[str, Any]]:
        return retrieve(current(), q, k=k, embedder=embedder)

    @app.post("/retrieve")
    def retrieve_post(body: RetrieveRequest) -> list[dict[str, Any]]:
        return retrieve(current(), body.query, k=body.k, embedder=embedder)

    return app
