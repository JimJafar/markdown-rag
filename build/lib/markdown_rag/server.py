"""FastAPI app exposing retrieval + health endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

from markdown_rag.index import Embedder
from markdown_rag.retrieval import retrieve


class RetrieveRequest(BaseModel):
    """JSON body for POST /retrieve."""

    query: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=100)


def create_app(index: dict[str, Any]) -> FastAPI:
    """Build the FastAPI app over an already-built index."""
    app = FastAPI(title="markdown-rag", version="0.1.0")
    # One long-lived embedder shared across requests (model loaded once).
    embedder = Embedder()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/retrieve")
    def retrieve_get(
        q: str = Query(min_length=1),
        k: int = Query(default=5, ge=1, le=100),
    ) -> list[dict[str, Any]]:
        return retrieve(index, q, k=k, embedder=embedder)

    @app.post("/retrieve")
    def retrieve_post(body: RetrieveRequest) -> list[dict[str, Any]]:
        return retrieve(index, body.query, k=body.k, embedder=embedder)

    return app
