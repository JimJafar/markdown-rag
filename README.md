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

Both return the same shape — the top-k **whole documents** sorted by relevance, each with its best-matching chunks as citations, so an agent can answer document-level questions and prove where the answer came from. No vault-structure knowledge required:

```json
[
  {
    "title": "The Librarian Proposal",
    "path": "vault/The Librarian Proposal.md",
    "text": "The goal is to implement a context-aware, trust-scored memory...",
    "score": 1.0024,
    "citations": [
      {"chunk": "The Librarian is an MCP (Model Context Protocol) server that manages...", "score": 0.8912}
    ]
  }
]
```

`GET /health` returns `{"status": "ok"}` so agents can check readiness.

## How it works

- **Chunking** is heading-aware: it splits at heading boundaries, carries the heading path (e.g. `# Intro > ## Architecture`) as context, and treats leading `---` frontmatter as metadata.
- **Embedding** uses a bundled `BAAI/bge-small-en-v1.5` ONNX model (384-dim, ~67 MB) via fastembed, loaded from package data — never downloaded.
- **Retrieval** is document-level: it ranks whole documents (dense semantic similarity as the primary signal, with a supporting BM25 lexical pass for exact-term queries) and returns each document's best-matching chunks as citations.
- **Index** is in-memory only: rebuild on start, no DB, no persistence.

## GPU (optional)

Embedding runs on CPU by default and needs no setup. When GPU libraries are present, markdown-rag auto-detects them — GPU first, CPU as fallback — and logs which provider it picked on start:

```text
INFO embedding 7193 chunks on CUDAExecutionProvider
```

If a GPU provider is advertised but can't actually run (e.g. missing cuDNN), a one-embed probe at startup falls back to CPU automatically with a warning — the server still starts.

To enable GPU on an existing install:

```sh
pipx install markdown-rag
pipx inject markdown-rag fastembed-gpu                # swaps onnxruntime for the CUDA build
pipx inject markdown-rag nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cufft-cu12
```

Install `fastembed-gpu` **after** the base install, as above — requesting both in one command resolves without error but silently leaves the CPU build in place; the startup log line is how you tell which you got.

No `LD_LIBRARY_PATH` needed: the server dlopens the bundled NVIDIA libs automatically at startup, so the CUDA provider finds cuDNN/cuBLAS out of the box. Embedding batches drop to 32 on GPU (fastembed's default 256 can overflow VRAM during attention on smaller cards); CPU keeps 256 for throughput.

The CUDA/cuDNN versions must match what onnxruntime-gpu expects (CUDA 12 + cuDNN 9 for onnxruntime 1.28). On a machine whose system CUDA is newer (e.g. CUDA 13 in `/opt/cuda`), the pip NVIDIA packages supply the exact CUDA 12 libs onnxruntime looks for. Measured on an RTX 5060 Ti: ~840 vec/s vs ~24 vec/s CPU — a thousand-note vault indexes in seconds.

## Development

```sh
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/python -m pytest
```
