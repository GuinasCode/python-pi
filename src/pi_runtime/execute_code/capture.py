"""Bounded stream capture — the core of spec section 12 (Output Boundary).

`BoundedStreamCapture` is fed chunks as they arrive from a live process
stream. Memory usage is O(head_bytes + tail_bytes) regardless of how much
data actually flows through it — a 500 MB stdout never gets buffered
whole; the complete stream is written to `artifact_path` via genuine
incremental disk writes (one `write()` per chunk, no full-content
buffering anywhere), and only a small head+tail preview plus running
totals (bytes/lines/hash) are kept in memory.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


class BoundedStreamCapture:
    def __init__(self, artifact_path: Path, *, head_bytes: int = 4_000, tail_bytes: int = 4_000) -> None:
        self._artifact_path = artifact_path
        self._head_bytes = head_bytes
        self._tail_bytes = tail_bytes
        self._head = bytearray()
        self._tail = bytearray()
        self._total_bytes = 0
        self._total_lines = 0
        self._hasher = hashlib.sha256()
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = artifact_path.open("wb")

    def write(self, chunk: bytes) -> None:
        if not chunk:
            return
        self._file.write(chunk)  # real streaming write — never buffered whole
        self._hasher.update(chunk)
        self._total_bytes += len(chunk)
        self._total_lines += chunk.count(b"\n")

        if len(self._head) < self._head_bytes:
            remaining = self._head_bytes - len(self._head)
            self._head += chunk[:remaining]

        self._tail += chunk
        # Bound the tail buffer itself — never let it grow past 2x the
        # target, so a pathological caller streaming byte-by-byte can't
        # make this O(n) in the number of writes either.
        if len(self._tail) > self._tail_bytes * 2:
            self._tail = self._tail[-self._tail_bytes :]

    def finish(self) -> CaptureResult:
        self._file.close()
        # The tail buffer only ever retains data once its size exceeds
        # 2x tail_bytes (see write()) — that's the real threshold past
        # which `_tail` no longer holds the stream from byte zero, so
        # it's what "truncated" must actually key off, independent of
        # how head_bytes compares to tail_bytes.
        truncated = self._total_bytes > self._tail_bytes * 2
        if truncated:
            head = bytes(self._head)
            tail = bytes(self._tail[-self._tail_bytes :])
            preview = (
                head.decode("utf-8", errors="replace")
                + "\n...(truncated)...\n"
                + tail.decode("utf-8", errors="replace")
            )
        else:
            # total_bytes <= head_bytes + tail_bytes, so `_tail` (only
            # ever trimmed once it exceeds 2x tail_bytes) still holds
            # the entire stream from the start — no reconstruction
            # needed, no truncation marker.
            preview = bytes(self._tail).decode("utf-8", errors="replace")
        return CaptureResult(
            preview=preview,
            truncated=truncated,
            total_bytes=self._total_bytes,
            total_lines=self._total_lines,
            artifact_path=str(self._artifact_path),
            sha256=self._hasher.hexdigest(),
        )


class CaptureResult:
    """Plain data handed to OutputCapture's constructor — kept separate
    from result.OutputCapture itself so this module doesn't need to
    import the result module (no cycle, and capture.py stays usable
    standalone)."""

    def __init__(
        self, *, preview: str, truncated: bool, total_bytes: int, total_lines: int, artifact_path: str, sha256: str
    ) -> None:
        self.preview = preview
        self.truncated = truncated
        self.total_bytes = total_bytes
        self.total_lines = total_lines
        self.artifact_path = artifact_path
        self.sha256 = sha256


__all__ = ["BoundedStreamCapture", "CaptureResult"]
