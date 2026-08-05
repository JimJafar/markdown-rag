# markdown-rag — confirmed intent

A generic, per-collection RAG server for large structured markdown collections, fed by LLM agents on demand.

## Outcome

A Python CLI server, installed once, started per vault. It indexes a markdown tree into an in-memory hybrid retrieval index and serves one retrieval endpoint that agents call for context.

## User

Jim (primary). Open to open-sourcing if portable/generic; likely several markdown vaults each getting their own server.

## Why now

672 notes today, vaults with thousands anticipated. Current workaround — manual search / re-reading — is a real, named pain.

## Success

- Spin-up for a new vault is `pipx install` once, then `markdown-rag serve <dir>` per vault.
- Fully offline, no cloud, no heavy native deps; small bundled ONNX model that runs on CPU in RAM.
- Agents call a single endpoint, need no vault-structure knowledge.
- Restart rebuilds the index in seconds.

## Constraint (adoption cost)

Trivial per-vault setup is a hard requirement. Reject heavy dependencies, multi-step installs, awkward UI.

## MVP scope — in

- One retrieve endpoint (`GET /retrieve?q=` or JSON POST), returns `[{path, chunk, score}]`.
- Heading-aware chunking (split at headings, carry heading path as context prefix, frontmatter → metadata).
- Hybrid BM25 + dense vectors, fused (RRF or weighted sum).
- Bundled small ONNX model, CPU/RAM, offline.
- pipx CLI, rebuild-on-start, no persistence.

## Out of scope (non-negotiable)

- Watch mode (seconds-restart is fine; solves no current pain).
- Structured filters (agents can't use them).
- Persistence (unneeded complexity).
- Debug views / rich API (no consumer).
- Docker/binary artifact.

## Decisions to pin in spec

- Model choice (size vs quality).
- Default `k`.
- Fusion weighting.
- JSON schema (validate against a real agent).

## Assumptions to validate

1. Agent-readable API — validate schema against a real agent before building more.
2. Retrieval quality — validate with real questions against a real vault before tuning fusion weights.
