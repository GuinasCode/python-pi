"""Session manager for Pi coding agent.

Simplified port of packages/coding-agent/src/core/session-manager.ts.
Handles session CRUD, JSONL file I/O, and tree navigation.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SessionEntry:
    """A single entry in a session tree."""

    seq: int
    parent_seq: int | None
    kind: str  # "message", "model_change", "thinking_level_change", "compaction", "label", "session_info"
    data: dict[str, Any]
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass
class SessionSearchResult:
    """One hit from SessionManager.search_sessions()."""

    info: SessionInfo
    score: int
    snippet: str


@dataclass
class SessionInfo:
    """Metadata about a session."""

    id: str
    name: str | None = None
    cwd: str = ""
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))
    updated_at: int = field(default_factory=lambda: int(time.time() * 1000))
    message_count: int = 0


class SessionManager:
    """Manages session files in JSONL format."""

    def __init__(self, session_dir: str | Path) -> None:
        self._session_dir = Path(session_dir)
        self._session_dir.mkdir(parents=True, exist_ok=True)

    def create_session(
        self,
        *,
        cwd: str | None = None,
        name: str | None = None,
    ) -> SessionInfo:
        """Create a new session and return its info."""
        session_id = uuid.uuid4().hex[:12]
        info = SessionInfo(
            id=session_id,
            name=name,
            cwd=cwd or str(Path.cwd()),
        )
        # Write header
        file_path = self._session_file(session_id)
        header = {
            "type": "session_info",
            "id": session_id,
            "name": name,
            "cwd": info.cwd,
            "created_at": info.created_at,
            "updated_at": info.updated_at,
        }
        file_path.write_text(json.dumps(header) + "\n", encoding="utf-8")
        return info

    def open_session(self, session_id: str) -> SessionInfo | None:
        """Open an existing session by ID."""
        file_path = self._session_file(session_id)
        if not file_path.exists():
            return None
        lines = file_path.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            return None
        header = json.loads(lines[0])
        if header.get("type") != "session_info":
            return None
        info = SessionInfo(
            id=session_id,
            name=header.get("name"),
            cwd=header.get("cwd", ""),
            created_at=header.get("created_at", 0),
            updated_at=header.get("updated_at", 0),
            message_count=max(0, len(lines) - 1),
        )
        return info

    def list_sessions(self) -> list[SessionInfo]:
        """List all sessions sorted by updated_at descending."""
        sessions: list[SessionInfo] = []
        for file_path in sorted(self._session_dir.glob("*.jsonl")):
            session_id = file_path.stem
            info = self.open_session(session_id)
            if info is not None:
                sessions.append(info)
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions

    def delete_session(self, session_id: str) -> bool:
        """Delete a session file."""
        file_path = self._session_file(session_id)
        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def append_entry(self, session_id: str, entry: SessionEntry) -> None:
        """Append an entry to a session."""
        file_path = self._session_file(session_id)
        line = json.dumps(
            {
                "seq": entry.seq,
                "parent_seq": entry.parent_seq,
                "kind": entry.kind,
                "data": entry.data,
                "timestamp": entry.timestamp,
            }
        )
        with file_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def get_entries(self, session_id: str) -> list[SessionEntry]:
        """Get all entries for a session."""
        file_path = self._session_file(session_id)
        if not file_path.exists():
            return []
        entries: list[SessionEntry] = []
        for line in file_path.read_text(encoding="utf-8").strip().splitlines()[1:]:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                entries.append(
                    SessionEntry(
                        seq=obj.get("seq", 0),
                        parent_seq=obj.get("parent_seq"),
                        kind=obj.get("kind", "message"),
                        data=obj.get("data", {}),
                        timestamp=obj.get("timestamp", 0),
                    )
                )
            except json.JSONDecodeError:
                continue
        return entries

    def search_sessions(self, query: str, *, limit: int = 20) -> list[SessionSearchResult]:
        """Deterministic full-text search over every stored session.

        No separate index/database — this re-reads each session's JSONL
        file directly (the same source of truth list_sessions() reads),
        so results can never drift out of sync with what's actually on
        disk. That's the right tradeoff at personal-CLI scale (tens to a
        few hundred sessions); it would need real indexing to stay fast
        at a much larger scale.

        A session matches when every whitespace-split, lowercased word in
        `query` appears as a substring somewhere in its searchable text
        (name, cwd, and every user/assistant message's text) — plain AND
        matching, not fuzzy/semantic. Ranked by total match count, ties
        broken by most-recently-updated first.
        """
        words = [w for w in query.lower().split() if w]
        if not words:
            return []

        results: list[SessionSearchResult] = []
        for file_path in sorted(self._session_dir.glob("*.jsonl")):
            session_id = file_path.stem
            info = self.open_session(session_id)
            if info is None:
                continue

            haystack_parts = [info.name or "", info.cwd]
            snippet = ""
            for entry in self.get_entries(session_id):
                if entry.kind != "message":
                    continue
                text = _extract_entry_text(entry.data)
                if not text:
                    continue
                haystack_parts.append(text)
                if not snippet and any(w in text.lower() for w in words):
                    snippet = text.strip()

            haystack = "\n".join(haystack_parts).lower()
            if not all(w in haystack for w in words):
                continue

            score = sum(haystack.count(w) for w in words)
            results.append(
                SessionSearchResult(
                    info=info,
                    score=score,
                    snippet=_truncate(snippet or info.name or info.cwd, 120),
                )
            )

        results.sort(key=lambda r: (r.score, r.info.updated_at), reverse=True)
        return results[:limit]

    def resolve_session_ref(self, ref: str) -> SessionInfo | None:
        """Resolve a session id or id-prefix to its SessionInfo.

        An exact id match always wins. Otherwise there must be exactly one
        session whose id starts with `ref` — zero or multiple prefix
        matches both return None (callers that need to tell "not found"
        apart from "ambiguous" for a better error message can re-scan
        list_sessions() themselves; that distinction isn't needed by the
        resolve-and-use call sites this exists for)."""
        exact = self.open_session(ref)
        if exact is not None:
            return exact
        matches = [s for s in self.list_sessions() if s.id.startswith(ref)]
        return matches[0] if len(matches) == 1 else None

    def continue_recent(self) -> SessionInfo | None:
        """Get the most recently updated session."""
        sessions = self.list_sessions()
        return sessions[0] if sessions else None

    def fork_session(self, session_id: str, *, name: str | None = None) -> SessionInfo | None:
        """Fork a session into a new one, copying all entries."""
        original = self.open_session(session_id)
        if original is None:
            return None
        new_info = self.create_session(cwd=original.cwd, name=name or f"fork-{session_id}")
        entries = self.get_entries(session_id)
        for entry in entries:
            self.append_entry(new_info.id, entry)
        return new_info

    def _session_file(self, session_id: str) -> Path:
        return self._session_dir / f"{session_id}.jsonl"


def _extract_entry_text(data: dict[str, Any]) -> str:
    """Plain-text content of one message entry's `data`, for search — same
    shape InteractiveSession._persist_message() writes: `content` is
    either a raw string (user messages) or a list of typed blocks
    (assistant messages: text/thinking/tool_use; tool results: text)."""
    content = data.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            if "text" in block:
                parts.append(str(block["text"]))
            elif "thinking" in block:
                parts.append(str(block["thinking"]))
    return " ".join(parts)


def _truncate(text: str, max_len: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= max_len else text[: max_len - 1] + "…"
