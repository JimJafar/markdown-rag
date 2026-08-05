# Spec — markdown-rag

Source: `docs/ideas/markdown-rag.md` (confirmed intent, direction B).

## Assumptions

1. **Runtime**: Python 3.11+. CLI packaged for `pipx` install; served by FastAPI via uvicorn.
2. **Embedding**: a small bundled ONNX model (CPU/RAM, fully offline). Model choice is an open question (size vs quality); default candidate is a MiniLM-style ~50–90MB sentence model.
3. **Retrieval index**: fully in-memory, rebuilt on start. No persistence, no DB, no vector store dependency. Dense matrix via numpy.
4. **Hybrid search**: BM25 (lexical) + dense vectors, fused via reciprocal-rank fusion (RRF) or weighted sum. Exact fusion is an open question, decided during implementation after validating against real queries.
5. **Chunking**: heading-aware — split at headings, carry heading path as context prefix, frontmatter → metadata. Boundaries at structure, never mid-sentence.
6. **API surface**: single retrieval endpoint, `GET /retrieve?q=` and/or JSON `POST` returning `[{path, chunk, score}]`. No filters, no pagination — agents can't use them.
7. **Scale target**: thousands of chunks; restart rebuilds in seconds.

→ Correct me now or I'll proceed with these.

## Project facts (stated once per project)

- **Structure**: source under `src/markdown_rag/`; tests under `tests/`; docs under `docs/`; spec/plan/tasks under `tasks/`.
- **Style**: standard PEP8; FastAPI idiomatic; UK English in docs and comments.
- **Testing**: pytest; test framework at `tests/`; unit tests for chunking/retrieval, integration test boots the server against a fixture vault.
- **Stack**: Python 3.11+, FastAPI, uvicorn, onnxruntime, numpy, `markdown-it-py` (or similar) for heading parsing, fastembed or raw ONNX for embeddings, and a small BM25 implementation (e.g. `rank-bm25` or hand-rolled).

## Objective

What: a generic per-collection RAG server — point it at a markdown tree, it indexes heading-aware, embeds locally, and serves one retrieval endpoint agents call for context.

Why: Jim has 672 notes today and vaults with thousands anticipated; manual search/re-reading is the current pain. Portability and trivial per-vault setup are hard requirements.

For whom: Jim primarily; open-sourcing later if portable/generic.

Success criteria (testable):

- `pipx install` the package once; `markdown-rag serve <dir>` starts a server from any vault.
- Fully offline: no API keys, no cloud, no model download at first run (model bundled).
- `GET /retrieve?q=<question>` returns `[{path, chunk, score}]` sorted by relevance.
- Chunk boundaries respect markdown structure (headings, no mid-sentence splits).
- Restart rebuilds the index; for a vault of thousands of chunks, rebuild completes in seconds.
- A real "what do we know about X" question retrieves the relevant chunks.

## Boundaries

**Always**: run tests before commits; validate inputs (bad query, empty vault, non-markdown files); UK English docs; keep offline/no-heavy-dep principle.
**Ask first**: schema changes; new dependencies beyond the listed stack; changing the API shape; adding filters/pagination/watch-mode/persistence.
**Never**: commit secrets; add cloud/API-key dependencies; add a heavy native dependency without approval; add out-of-scope features (watch mode, filters, persistence).

## Open questions

1. Which bundled ONNX embedding model (size vs quality trade-off)?
2. Fusion method: RRF vs weighted sum, and the blend weight?
3. Default `k` (top chunks returned) — propose 5, confirm.
4. Exact JSON schema — validate against a real agent.
