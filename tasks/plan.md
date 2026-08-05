# Plan — markdown-rag

Source: `tasks/spec.md`, `docs/ideas/markdown-rag.md`.

## Overview

A Python CLI server installed once via pipx, started per vault (`markdown-rag serve <dir>`). On start it reads the markdown tree, chunks heading-aware, embeds with a bundled local ONNX model (CPU/RAM, fully offline), and builds an in-memory hybrid BM25 + dense index. Agents call one retrieval endpoint and get back `[{path, chunk, score}]` — no vault-structure knowledge needed.

## Architecture decisions

1. **Packaging**: `src/` layout, `pyproject.toml`, console script `markdown-rag = markdown_rag.cli:main`. pipx-installable. The bundled ONNX model ships as package data so first run is fully offline (no download).
2. **Embeddings**: `fastembed` (Qdrant) — purpose-built local CPU ONNX embeddings, supports pointing at a local model dir via `model_dir`, so the bundled files are used and no network is needed at runtime. onnxruntime under the hood (C++, fast on CPU).
3. **Chunking**: `markdown-it-py` tokenizer. Split at heading boundaries; carry the heading path (e.g. `# Overview > ## Architecture`) as a context prefix on each chunk; parse leading `---` frontmatter (title/tags/date) as metadata. Boundaries at structure, never mid-sentence.
4. **Index**: in-memory only. BM25 via `rank-bm25`; dense via numpy matrix + cosine similarity. No persistence, no DB, no vector store — restart rebuilds in seconds.
5. **Retrieval**: reciprocal-rank fusion (RRF) of BM25 and dense top-N lists. RRF is parameter-light and robust to score scale differences between lexical and semantic — avoids fragile weighted-sum tuning.
6. **API**: single `GET /retrieve?q=&k=` endpoint (JSON POST also supported), returning `[{path, chunk, score}]` sorted by relevance. Plus a minimal `GET /health` so agents can check readiness. No filters, no pagination.
7. **CLI**: argparse with `serve <dir>` subcommand; port configurable via flag/env, default 8000.

## Dependency graph

CLI → Server (FastAPI app) → Index (BM25 + dense) → Chunking → Embedding (fastembed) → filesystem model data
Tasks ordered bottom-up from this graph.

## Phased task list

1. **Chunking** — read a markdown tree into heading-aware chunks with heading-path context and frontmatter metadata. *Foundation; nothing else works without it.*
2. **Embedding + index build** — embed chunks with the bundled local model and build the in-memory index from a vault. *Riskiest piece (model bundling + offline); done early to fail fast.*
3. **Hybrid retrieval** — BM25 + dense fused via RRF returns ranked chunks.
4. **HTTP API** — FastAPI `retrieve` + `health` endpoints with the agreed JSON schema; integration test against a fixture vault.
5. **CLI + packaging** — `markdown-rag serve <dir>`, pipx-installable, offline bundled model wired into package data.
6. **Docs + final verification** — README with install/spin-up instructions; full suite green; end-to-end smoke.

Checkpoints: after tasks 2 and 5 (core flow works end-to-end, human review).

## Risks / mitigations

- **Model bundling size / offline first-run** (high risk) — mitigate by committing the model files as package data; verify actual file size during task 2; if over-large, consider a smaller quantized model and note the trade-off.
- **fastembed API drift** (medium) — verify `model_dir` local-loading against current docs before relying on it.
- **BM25 tokenisation mismatch** (low) — tokenise queries and documents with the same tokenizer for comparable scores.

## Open questions

- Exact model file (size vs quality) — resolved empirically in task 2, not by fiat.
- Default `k` — proposed 5, confirmed by Jim in the approved spec.
- JSON schema — pinned in task 4 and validated against the retrieve contract.
