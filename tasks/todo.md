# Tasks — markdown-rag

Slices are vertical, dependency-ordered bottom-up. Definition of done per task: full pytest suite green, import clean, no regressions. Checkpoints after tasks 2 and 5 (core flow end-to-end + review).

## Task 1 — User can index a vault into heading-aware chunks

- **Description**: A `chunking` module reads every `.md` file under a vault directory into structured chunks. Each chunk carries its heading path as context (e.g. `# Overview > ## Architecture`), the raw text, and frontmatter-derived metadata (title/tags/date). Boundaries respect markdown structure — never mid-sentence.
- **Acceptance criteria**:
  - [ ] `chunk_vault(dir)` returns chunks grouped per source file, each with `path`, `heading_path`, `text`, `metadata`
  - [ ] Headings open new chunks; a chunk inherits its parent heading path as context
  - [ ] Leading `---` frontmatter is parsed into `metadata` and excluded from body text
- **Verification**: `pytest tests/test_chunking.py`
- **Dependencies**: none
- **Files touched**: `src/markdown_rag/chunking.py`, `tests/test_chunking.py`
- **Size**: S

## Task 2 — User can build an in-memory index from a vault (embeddings)

- **Description**: An `index` module embeds chunks with a bundled local ONNX model via fastembed (CPU/RAM, offline) and builds the in-memory index: dense matrix + tokenised corpus for BM25. Verify the model data loads from the bundled local `model_data/` dir with no network.
- **Acceptance criteria**:
  - [ ] `build_index(chunks)` produces a dense embedding matrix and BM25 corpus; query embedding works
  - [ ] Model loads fully offline from bundled package data (`model_dir`), no download at runtime
  - [ ] Index build for a small fixture vault completes in well under a second
- **Verification**: `pytest tests/test_index.py`
- **Dependencies**: Task 1
- **Files touched**: `src/markdown_rag/index.py`, `src/markdown_rag/model_data/` (model files), `tests/test_index.py`, `pyproject.toml` (dependency additions)
- **Size**: M (model bundling risk)
- **Checkpoint after this task**

## Task 3 — User can retrieve ranked chunks (hybrid BM25 + dense)

- **Description**: A `retrieval` module fuses BM25 and dense cosine top-N via reciprocal-rank fusion, returning `[{path, chunk, score}]` sorted by relevance for a query. Same tokenizer for query and corpus.
- **Acceptance criteria**:
  - [ ] `retrieve(index, query, k)` returns top-k results with scores, sorted descending
  - [ ] Hybrid result contains genuinely relevant chunks for a semantic query and an exact-term query
  - [ ] RRF fusion implemented; weights sensible without fragile tuning
- **Verification**: `pytest tests/test_retrieval.py`
- **Dependencies**: Task 2
- **Files touched**: `src/markdown_rag/retrieval.py`, `tests/test_retrieval.py`
- **Size**: S

## Task 4 — User can query the server over HTTP

- **Description**: A FastAPI app exposing `GET /retrieve?q=&k=` (and JSON POST) returning `[{path, chunk, score}]`, plus `GET /health`. Inputs validated (empty query, missing k). Integration test boots the app against a fixture vault.
- **Acceptance criteria**:
  - [ ] `/retrieve` returns JSON `[{path, chunk, score}]` sorted by score
  - [ ] `/health` returns 200; invalid query returns 422; empty vault handled gracefully
  - [ ] Integration test boots the server against a small fixture vault and asserts response shape
- **Verification**: `pytest tests/test_api.py`
- **Dependencies**: Task 3
- **Files touched**: `src/markdown_rag/server.py`, `tests/test_api.py`
- **Size**: S

## Task 5 — User can install once and spin up a server per vault

- **Description**: Console script `markdown-rag serve <dir>` loads the index from the vault and starts uvicorn. Packaged so `pipx install` gives a global command; model bundled as package data; port via flag/env (default 8000).
- **Acceptance criteria**:
  - [ ] `markdown-rag serve <dir>` starts a server; `GET /retrieve` works end-to-end from a real vault
  - [ ] `pipx install` (from repo) yields the `markdown-rag` command with no per-vault setup
  - [ ] CLI flags for `--port` and vault dir; sensible defaults
- **Verification**: `pytest tests/test_cli.py` + `markdown-rag serve tests/fixtures/vault --port 0` smoke
- **Dependencies**: Task 4
- **Files touched**: `src/markdown_rag/cli.py`, `pyproject.toml`, `tests/test_cli.py`
- **Size**: S
- **Checkpoint after this task**

## Task 6 — User has install and spin-up instructions

- **Description**: README rewritten with accurate install (`pipx install`) and per-vault spin-up (`markdown-rag serve <dir>`) instructions; offline/no-deps guarantees stated; example curl. Full suite green; end-to-end smoke against a real vault.
- **Acceptance criteria**:
  - [ ] README documents install + spin-up + example query, verified against the real CLI
  - [ ] Full pytest suite green, package imports cleanly
  - [ ] End-to-end smoke: index a real vault, retrieve a "what do we know about X" question, top hits relevant
- **Verification**: `pytest` (full) + manual curl smoke
- **Dependencies**: Task 5
- **Files touched**: `README.md`, possibly `docs/`
- **Size**: S

## Not doing (approved out-of-scope)

Watch mode, structured filters, persistence, debug/rich API, Docker/binary artifact, cloud embeddings.
