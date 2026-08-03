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
