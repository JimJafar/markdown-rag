# markdown-rag

A small RAG server for large structured markdown collections. Point it at a vault, it indexes the files on start, and serves one retrieval endpoint that LLM agents call for context.

Fully offline, runs on CPU in RAM — no cloud, no API keys, no heavy dependencies. Each vault gets its own server; nothing is shared between collections.

## Install (once)

```sh
pipx install git+https://github.com/JimJafar/markdown-rag.git
```

pipx gives you an isolated install and a global `markdown-rag` command — no per-vault setup needed. A small ONNX embedding model (~67 MB) is bundled, so there is no download at first run.

## Spin up a server (per vault)

```sh
markdown-rag serve /path/to/your/vault
```

The server indexes every `.md` file under the directory (recursively), builds an in-memory hybrid index, and listens on `127.0.0.1:8000`. The index is rebuilt from scratch on every start — a few seconds for a small vault, a few minutes for a thousand-note one — and there is no persistent store.

```sh
markdown-rag serve /path/to/vault --port 8080 --host 127.0.0.1
```

## Query it

```sh
curl -X GET "http://127.0.0.1:8000/retrieve" -G \
  --data-urlencode "q=what do we know about X" \
  --data-urlencode "k=5"
```

or as JSON:

```sh
curl -X POST http://127.0.0.1:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "what do we know about X", "k": 5}'
```

Both return the same shape — the top-k chunks sorted by relevance, with no vault-structure knowledge required:

```json
[
  {
    "path": "vault/overview.md",
    "chunk": "This vault documents the markdown-rag server, a tool for...",
    "score": 0.0167
  }
]
```

`GET /health` returns `{"status": "ok"}` so agents can check readiness.

## How it works

- **Chunking** is heading-aware: it splits at heading boundaries, carries the heading path (e.g. `# Intro > ## Architecture`) as context, and treats leading `---` frontmatter as metadata.
- **Embedding** uses a bundled `BAAI/bge-small-en-v1.5` ONNX model (384-dim, ~67 MB) via fastembed, loaded from package data — never downloaded.
- **Retrieval** fuses lexical search (BM25) and semantic search (dense cosine) with reciprocal-rank fusion, so both exact identifiers and paraphrase match well.
- **Index** is in-memory only: rebuild on start, no DB, no persistence.

## Development

```sh
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/python -m pytest
```
