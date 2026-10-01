"""Incremental refresh: keep the index in step with the vault without a full rebuild.

A ``LiveIndex`` remembers every note's chunks and embeddings, keyed by path and
stamped with the file's (mtime, size). ``refresh()`` re-scans the vault — metadata
only — and re-chunks and re-embeds just the notes that were added or changed, drops
deleted ones, then reassembles the index from the cached rows. Readers take
``live.current``, which is replaced in one assignment, so a query never sees a
half-built index.

Polling rather than file-system events: the vault is often a network mount (SMB,
synced drives) where inotify does not report changes made by other machines.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from markdown_rag.chunking import chunk_file, markdown_files
from markdown_rag.index import Embedder, assemble_index, chunk_texts, normalise

#: A file's change stamp: (st_mtime_ns, st_size).
Signature = tuple[int, int]


@dataclass(frozen=True)
class RefreshResult:
    """What one refresh pass did."""

    added: int = 0
    changed: int = 0
    removed: int = 0
    failed: int = 0
    chunks_embedded: int = 0
    seconds: float = 0.0

    @property
    def updated(self) -> bool:
        return bool(self.added or self.changed or self.removed)


@dataclass(frozen=True)
class _Note:
    signature: Signature
    chunks: list[dict[str, Any]]
    vectors: np.ndarray


class LiveIndex:
    """The current index for a vault, refreshed incrementally."""

    def __init__(self, vault: Path, embedder: Embedder | None = None) -> None:
        self.vault = vault
        self.embedder = embedder if embedder is not None else Embedder()
        self._notes: dict[str, _Note] = {}
        self._dim = 0
        self._lock = threading.Lock()  # one refresh at a time; readers never wait
        self.current: dict[str, Any] = assemble_index([], normalise([]))
        self.last_refresh: datetime | None = None
        self.last_result: RefreshResult | None = None

    def refresh(self) -> RefreshResult:
        """Bring the index up to date with the vault; embeds only added or changed notes."""
        with self._lock:
            started = time.monotonic()
            seen = markdown_files(self.vault)
            if not seen and self._notes:
                logging.warning(
                    "refresh: %s has no notes now; keeping the current index "
                    "(is the vault unmounted?)", self.vault)
                return RefreshResult(seconds=time.monotonic() - started)
            removed = [path for path in self._notes if path not in seen]
            stale = [path for path, sig in seen.items()
                     if path not in self._notes or self._notes[path].signature != sig]

            parsed: dict[str, list[dict[str, Any]]] = {}
            failed = 0
            for path in stale:
                try:
                    parsed[path] = chunk_file(Path(path))
                except (OSError, UnicodeDecodeError) as error:
                    # Keep the last good version (if any) and retry on the next pass:
                    # a note mid-sync or with a bad byte must not drop out of search.
                    failed += 1
                    logging.warning("refresh: skipped %s (%s); will retry", path, error)

            texts = [text for chunks in parsed.values() for text in chunk_texts(chunks)]
            vectors = normalise(list(self.embedder.embed(texts)), self._dim) if texts else None
            if vectors is not None:
                self._dim = vectors.shape[1]

            added = changed = offset = 0
            for path, chunks in parsed.items():
                rows = vectors[offset:offset + len(chunks)] if vectors is not None \
                    else np.zeros((0, self._dim), dtype=np.float32)
                offset += len(chunks)
                if path in self._notes:
                    changed += 1
                else:
                    added += 1
                self._notes[path] = _Note(seen[path], chunks, rows)
            for path in removed:
                del self._notes[path]

            result = RefreshResult(added, changed, len(removed), failed, len(texts),
                                   time.monotonic() - started)
            if result.updated:
                self.current = self._assemble()
            self.last_refresh = datetime.now(UTC)
            self.last_result = result
            return result

    def _assemble(self) -> dict[str, Any]:
        ordered = [self._notes[path] for path in sorted(self._notes)]
        chunks = [chunk for note in ordered for chunk in note.chunks]
        blocks = [note.vectors for note in ordered if len(note.vectors)]
        matrix = np.vstack(blocks) if blocks else np.zeros((0, self._dim), dtype=np.float32)
        return assemble_index(chunks, matrix)


def start_refresher(live: LiveIndex, interval: float) -> threading.Event:
    """Refresh every ``interval`` seconds on a daemon thread; set the returned event to stop."""
    stop = threading.Event()

    def loop() -> None:
        while not stop.wait(interval):
            try:
                result = live.refresh()
            except Exception:  # keep serving the last good index whatever went wrong
                logging.exception("refresh failed; keeping the current index")
                continue
            if result.updated or result.failed:
                logging.info(
                    "refresh: %d added, %d changed, %d removed, %d skipped; "
                    "%d chunks embedded in %.1fs",
                    result.added, result.changed, result.removed, result.failed,
                    result.chunks_embedded, result.seconds,
                )

    threading.Thread(target=loop, name="markdown-rag-refresh", daemon=True).start()
    return stop
