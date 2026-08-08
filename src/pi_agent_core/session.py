"""Session management for the agent harness.

Mirrors ``packages/agent/src/harness/session/`` from the TypeScript Pi agent.

Defines the session tree entry types, ``Session`` class, ``SessionStorage``
protocol, and ``InMemoryRepository`` / ``JsonlRepository`` implementations.

The session tree is an append-only log of entries (messages, compactions,
branch summaries, etc.) forming a tree via ``parentId`` links. The ``Session``
class wraps a storage backend and provides methods to append entries, navigate
the tree, and build LLM context from the active branch.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable
from urllib.parse import quote

from pi_ai import Usage

from .messages import (
    create_branch_summary_message,
    create_compaction_summary_message,
    create_custom_message,
)

__all__ = [
    "ActiveToolsChangeEntry",
    "BranchSummaryEntry",
    "CompactionEntry",
    "CustomEntry",
    "CustomMessageEntry",
    "InMemoryRepository",
    "JsonlRepository",
    "JsonlSessionCreateOptions",
    "JsonlSessionMetadata",
    "LabelEntry",
    "LeafEntry",
    "MessageEntry",
    "ModelChangeEntry",
    "Session",
    "SessionBranchQuery",
    "SessionContext",
    "SessionCreateOptions",
    "SessionEntryCursorOptions",
    "SessionError",
    "SessionForkOptions",
    "SessionForkSelection",
    "SessionHead",
    "SessionInfoEntry",
    "SessionMetadata",
    "SessionRepository",
    "SessionStats",
    "SessionTreeEntry",
    "ThinkingLevelChangeEntry",
    "create_session_id",
    "create_timestamp",
]

# --- UUID v7 generation ---


def _uuid_v7() -> str:
    """Generate a UUID v7 string (time-ordered). Falls back to uuid4."""
    import uuid

    # Try to use uuid7 if available (Python 3.14+), otherwise construct manually
    try:
        return str(uuid.uuid7())  # type: ignore[attr-defined]
    except AttributeError:
        pass

    # Construct a UUID v7 manually
    timestamp_ms = int(time.time() * 1000)
    # 48-bit timestamp | 4-bit version (7) | 12-bit rand_a
    # 2-bit variant | 62-bit rand_b
    import secrets

    rand_a = secrets.randbits(12)
    rand_b = secrets.randbits(62)

    high = (timestamp_ms << 16) | (7 << 12) | rand_a
    low = (0b10 << 62) | rand_b

    # Pack as 128-bit integer and format. Version/variant bits are already
    # encoded above; passing version= here would additionally trigger stdlib
    # validation that rejects version 7 on Python < 3.14.
    value = (high << 64) | low
    return str(uuid.UUID(int=value))


def create_session_id() -> str:
    """Create a new session ID (UUID v7)."""
    return _uuid_v7()


def create_timestamp() -> str:
    """Create an ISO-8601 timestamp."""
    from datetime import datetime

    return datetime.now(UTC).isoformat()


# --- Error types ---


class SessionError(Exception):
    """Error from session storage, repositories, or tree operations."""

    def __init__(self, code: str, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.cause = cause


# --- Session tree entry types ---


@dataclass
class SessionTreeEntryBase:
    """Base fields for all session tree entries."""

    type: str
    id: str
    parent_id: str | None
    timestamp: str


@dataclass
class MessageEntry(SessionTreeEntryBase):
    """A message entry in the session tree."""

    type: str = field(default="message", init=False)
    message: Any = None  # AgentMessage


@dataclass
class ThinkingLevelChangeEntry(SessionTreeEntryBase):
    """A thinking level change entry."""

    type: str = field(default="thinking_level_change", init=False)
    thinking_level: str = "off"


@dataclass
class ModelChangeEntry(SessionTreeEntryBase):
    """A model change entry."""

    type: str = field(default="model_change", init=False)
    provider: str = ""
    model_id: str = ""


@dataclass
class ActiveToolsChangeEntry(SessionTreeEntryBase):
    """An active tools change entry."""

    type: str = field(default="active_tools_change", init=False)
    active_tool_names: list[str] = field(default_factory=list)


@dataclass
class CompactionEntry(SessionTreeEntryBase):
    """A compaction entry that replaces earlier history with a summary."""

    type: str = field(default="compaction", init=False)
    summary: str = ""
    first_kept_entry_id: str | None = None
    tokens_before: int = 0
    retained_tail: list[Any] | None = None  # list[AgentMessage]
    details: Any = None
    usage: Usage | None = None
    from_hook: bool | None = None


@dataclass
class BranchSummaryEntry(SessionTreeEntryBase):
    """A branch summary entry."""

    type: str = field(default="branch_summary", init=False)
    from_id: str = ""
    summary: str = ""
    details: Any = None
    usage: Usage | None = None
    from_hook: bool | None = None


@dataclass
class CustomEntry(SessionTreeEntryBase):
    """A custom (non-message) entry."""

    type: str = field(default="custom", init=False)
    custom_type: str = ""
    data: Any = None


@dataclass
class CustomMessageEntry(SessionTreeEntryBase):
    """A custom message entry that appears in context."""

    type: str = field(default="custom_message", init=False)
    custom_type: str = ""
    content: str | list[Any] = ""
    details: Any = None
    display: bool = True


@dataclass
class LabelEntry(SessionTreeEntryBase):
    """A label entry for an entry in the tree."""

    type: str = field(default="label", init=False)
    target_id: str = ""
    label: str | None = None


@dataclass
class SessionInfoEntry(SessionTreeEntryBase):
    """A session info entry (session name)."""

    type: str = field(default="session_info", init=False)
    name: str | None = None


@dataclass
class LeafEntry(SessionTreeEntryBase):
    """A leaf pointer entry."""

    type: str = field(default="leaf", init=False)
    target_id: str | None = None


SessionTreeEntry: TypeAlias = (
    MessageEntry
    | ThinkingLevelChangeEntry
    | ModelChangeEntry
    | ActiveToolsChangeEntry
    | CompactionEntry
    | BranchSummaryEntry
    | CustomEntry
    | CustomMessageEntry
    | LabelEntry
    | SessionInfoEntry
    | LeafEntry
)

# --- Metadata types ---


@dataclass
class SessionMetadata:
    """Base session metadata."""

    id: str
    created_at: str


@dataclass
class JsonlSessionMetadata(SessionMetadata):
    """Metadata for JSONL-backed sessions."""

    cwd: str = ""
    path: str = ""
    parent_session_path: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class SessionStats:
    """Aggregated session statistics."""

    message_count: int = 0
    cached_tokens: int = 0
    uncached_tokens: int = 0
    total_tokens: int = 0
    cost_total: float = 0.0


@dataclass
class SessionHead:
    """Session head pointer."""

    leaf_id: str | None = None


@dataclass
class SessionContext:
    """Context derived from session entries."""

    messages: list[Any] = field(default_factory=list)  # list[AgentMessage]
    thinking_level: str = "off"
    model: dict[str, str] | None = None  # {"provider": ..., "model_id": ...}
    active_tool_names: list[str] | None = None


@dataclass
class SessionEntryCursorOptions:
    """Options for reading entries with cursor-based pagination."""

    after_entry_seq: int = 0
    limit: int | None = None


@dataclass
class SessionBranchQuery:
    """Query for finding entries on a branch."""

    start: str | None = None
    stop_at_type: str | None = None
    stop_at_id: str | None = None
    type: str | None = None
    custom_type: str | None = None
    order: Literal["newestFirst", "oldestFirst"] | None = None
    limit: int | None = None


@dataclass
class SessionCreateOptions:
    """Options for creating a session."""

    id: str | None = None


@dataclass
class JsonlSessionCreateOptions(SessionCreateOptions):
    """Options for creating a JSONL session."""

    cwd: str = ""
    parent_session_path: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class SessionForkOptions:
    """Options for forking a session."""

    entry_id: str | None = None
    position: Literal["before", "at"] | None = None
    id: str | None = None


SessionForkSelection: TypeAlias = dict[str, str]
# {"kind": "all"} | {"kind": "through_entry", "entryId": ...} | {"kind": "before_user_message", "entryId": ...}


# --- Storage protocol ---


@runtime_checkable
class SessionStorage(Protocol):
    """Storage contract for one opened session."""

    @property
    def metadata(self) -> SessionMetadata: ...

    async def read_head(self) -> SessionHead: ...
    async def read_entry(self, id: str) -> SessionTreeEntry | None: ...
    async def read_entries(self, options: SessionEntryCursorOptions | None = None) -> list[SessionTreeEntry]: ...
    async def append_entry(self, entry: SessionTreeEntry) -> None: ...
    async def find_entries_on_branch(self, query: SessionBranchQuery) -> list[SessionTreeEntry]: ...
    async def read_path_to_root_or_compaction(self, leaf_id: str | None) -> list[SessionTreeEntry]: ...
    async def get_label(self, id: str) -> str | None: ...
    async def get_name(self) -> str | None: ...
    async def get_stats(self) -> SessionStats: ...


# --- Repository protocol ---


class SessionRepository(Protocol):
    """Protocol for session repositories."""

    async def create(self, options: SessionCreateOptions | None = None) -> Session: ...
    async def open(self, metadata: SessionMetadata) -> Session: ...
    async def list(self) -> list[SessionMetadata]: ...
    async def delete(self, metadata: SessionMetadata) -> None: ...
    async def fork(self, source: SessionMetadata, options: SessionForkOptions) -> Session: ...
    async def dispose(self) -> None: ...


# --- ArraySessionIndex ---


@dataclass
class _Projection:
    """Derived projections from session entries."""

    name: str | None = None
    labels_by_id: dict[str, str] = field(default_factory=dict)
    stats: SessionStats = field(default_factory=SessionStats)


def _create_projection() -> _Projection:
    return _Projection()


def _apply_projection(projection: _Projection, entry: SessionTreeEntry) -> None:
    """Update projection with a new entry."""
    if isinstance(entry, SessionInfoEntry):
        projection.name = entry.name.strip() if entry.name and entry.name.strip() else None
    elif isinstance(entry, LabelEntry):
        if entry.label and entry.label.strip():
            projection.labels_by_id[entry.target_id] = entry.label.strip()
        else:
            projection.labels_by_id.pop(entry.target_id, None)

    if isinstance(entry, MessageEntry):
        projection.stats.message_count += 1

    usage: Usage | None = None
    if isinstance(entry, MessageEntry):
        msg = entry.message
        if hasattr(msg, "role") and msg.role == "assistant":
            usage = getattr(msg, "usage", None)
    elif isinstance(entry, CompactionEntry | BranchSummaryEntry):
        usage = entry.usage

    if usage is None:
        return
    if not isinstance(usage.input, int) or not isinstance(usage.output, int):
        return
    if not isinstance(usage.cache_read, int) or not isinstance(usage.cache_write, int):
        return
    if not isinstance(usage.cost.total, int | float):
        return

    projection.stats.cached_tokens += usage.cache_read
    projection.stats.uncached_tokens += usage.input + usage.cache_write
    projection.stats.total_tokens += usage.input + usage.output + usage.cache_read + usage.cache_write
    projection.stats.cost_total += usage.cost.total


class ArraySessionIndex:
    """Ordered entries and derived projections for array-backed session storage.

    Mirrors ``array-session-index.ts``.
    """

    def __init__(self, entries: Sequence[SessionTreeEntry] | None = None) -> None:
        self._entries: list[SessionTreeEntry] = []
        self._by_id: dict[str, SessionTreeEntry] = {}
        self._leaf_id: str | None = None
        self._projection = _create_projection()
        if entries:
            self.replace(entries)

    def has(self, id: str) -> bool:
        return id in self._by_id

    def append(self, entry: SessionTreeEntry) -> None:
        if entry.id in self._by_id:
            raise SessionError("invalid_entry", f"Entry {entry.id} already exists")
        self._entries.append(entry)
        self._by_id[entry.id] = entry
        self._leaf_id = entry.target_id if isinstance(entry, LeafEntry) else entry.id
        _apply_projection(self._projection, entry)

    def replace(self, entries: Sequence[SessionTreeEntry]) -> None:
        next_entries: list[SessionTreeEntry] = []
        next_by_id: dict[str, SessionTreeEntry] = {}
        next_projection = _create_projection()
        next_leaf_id: str | None = None
        for entry in entries:
            if entry.id in next_by_id:
                raise SessionError("invalid_entry", f"Entry {entry.id} already exists")
            next_by_id[entry.id] = entry
            next_leaf_id = entry.target_id if isinstance(entry, LeafEntry) else entry.id
            _apply_projection(next_projection, entry)
            next_entries.append(entry)
        self._entries = next_entries
        self._by_id = next_by_id
        self._leaf_id = next_leaf_id
        self._projection = next_projection

    def read_head(self) -> SessionHead:
        if self._leaf_id is not None and self._leaf_id not in self._by_id:
            raise SessionError("invalid_session", f"Entry {self._leaf_id} not found")
        return SessionHead(leaf_id=self._leaf_id)

    def read_entry(self, id: str) -> SessionTreeEntry | None:
        return self._by_id.get(id)

    def read_entries(self, options: SessionEntryCursorOptions | None = None) -> list[SessionTreeEntry]:
        start = options.after_entry_seq if options and options.after_entry_seq else 0
        if options and options.limit is not None:
            end = start + options.limit
            return self._entries[start:end]
        return self._entries[start:]

    def find_entries_on_branch(self, query: SessionBranchQuery) -> list[SessionTreeEntry]:
        if query.limit is not None and (not isinstance(query.limit, int) or query.limit <= 0):
            raise ValueError("Session branch query limit must be a positive integer")
        if query.start is None:
            return []

        path_from_start: list[SessionTreeEntry] = []
        visited: set[str] = set()
        current = self._by_id.get(query.start)
        if current is None:
            raise SessionError("not_found", f"Entry {query.start} not found")

        while current is not None:
            if current.id in visited:
                raise SessionError("invalid_session", f"Session branch contains a cycle at {current.id}")
            visited.add(current.id)
            path_from_start.append(current)
            if query.order != "oldestFirst" and (current.id == query.stop_at_id or current.type == query.stop_at_type):
                break
            if current.parent_id is None:
                break
            parent = self._by_id.get(current.parent_id)
            if parent is None:
                raise SessionError("invalid_session", f"Entry {current.parent_id} not found")
            current = parent

        traversal = list(reversed(path_from_start)) if query.order == "oldestFirst" else path_from_start

        if query.order == "oldestFirst":
            stop_index = -1
            for i, entry in enumerate(traversal):
                if entry.id == query.stop_at_id or entry.type == query.stop_at_type:
                    stop_index = i
                    break
            bounded = traversal[: stop_index + 1] if stop_index >= 0 else traversal
        else:
            bounded = traversal

        entries = [
            entry
            for entry in bounded
            if (query.type is None or entry.type == query.type)
            and (
                query.custom_type is None or (isinstance(entry, CustomEntry) and entry.custom_type == query.custom_type)
            )
        ]

        return entries[: query.limit] if query.limit is not None else entries

    def get_label(self, id: str) -> str | None:
        return self._projection.labels_by_id.get(id)

    def get_name(self) -> str | None:
        return self._projection.name

    def get_stats(self) -> SessionStats:
        return SessionStats(
            message_count=self._projection.stats.message_count,
            cached_tokens=self._projection.stats.cached_tokens,
            uncached_tokens=self._projection.stats.uncached_tokens,
            total_tokens=self._projection.stats.total_tokens,
            cost_total=self._projection.stats.cost_total,
        )

    def read_path_to_root_or_compaction(self, requested_leaf_id: str | None) -> list[SessionTreeEntry]:
        if requested_leaf_id is None:
            return []
        path: list[SessionTreeEntry] = []
        stop_at_entry_id: str | None = None
        current = self._by_id.get(requested_leaf_id)
        if current is None:
            raise SessionError("not_found", f"Entry {requested_leaf_id} not found")

        while current is not None:
            path.append(current)
            if stop_at_entry_id is not None and current.id == stop_at_entry_id:
                break
            if isinstance(current, CompactionEntry):
                if current.retained_tail:
                    break
                stop_at_entry_id = current.first_kept_entry_id
            if current.parent_id is None:
                break
            parent = self._by_id.get(current.parent_id)
            if parent is None:
                raise SessionError("invalid_session", f"Entry {current.parent_id} not found")
            current = parent

        path.reverse()
        return path


# --- KeyedOperationQueue ---


class KeyedOperationQueue:
    """A keyed async operation queue that serializes per-key and limits concurrency.

    Mirrors ``keyed-operation-queue.ts``.
    """

    def __init__(self, *, max_concurrent_operations: int | None = None) -> None:
        if max_concurrent_operations is not None and (
            not isinstance(max_concurrent_operations, int) or max_concurrent_operations < 1
        ):
            raise ValueError("max_concurrent_operations must be a positive integer")
        self._max_concurrent = max_concurrent_operations
        self._tails: dict[str, asyncio.Future[None]] = {}
        self._active = 0
        self._permit_waiters: list[asyncio.Future[None]] = []
        self._barrier: asyncio.Future[None] = asyncio.get_event_loop().create_future()
        self._barrier.set_result(None)

    async def enqueue(self, key: str, operation: Callable[[], Any]) -> Any:
        """Enqueue an operation under ``key``, serialized per-key."""
        previous = self._tails.get(key)
        # Wait for barrier and previous key tail
        if previous is not None:
            await asyncio.gather(self._barrier, previous)
        else:
            await self._barrier

        result = await self._run_operation(operation)

        # Set up the tail for this key
        tail: asyncio.Future[None] = asyncio.get_event_loop().create_future()
        tail.set_result(None)
        self._tails[key] = tail
        # Clean up tail after a tick
        self._tails.pop(key, None)

        return result

    async def enqueue_barrier(self, operation: Callable[[], Any]) -> Any:
        """Enqueue a barrier operation that waits for all pending operations."""
        # Wait for all pending operations
        if self._tails:
            await asyncio.gather(*self._tails.values())
        await self._barrier
        result = await self._run_operation(operation)
        new_barrier: asyncio.Future[None] = asyncio.get_event_loop().create_future()
        new_barrier.set_result(None)
        self._barrier = new_barrier
        return result

    async def drain(self) -> None:
        if self._tails:
            await asyncio.gather(*self._tails.values())
        await self._barrier

    async def _run_operation(self, operation: Callable[[], Any]) -> Any:
        await self._acquire_permit()
        try:
            result = operation()
            if asyncio.iscoroutine(result):
                result = await result
            return result
        finally:
            self._release_permit()

    async def _acquire_permit(self) -> None:
        if self._max_concurrent is None:
            return
        if self._active < self._max_concurrent:
            self._active += 1
            return
        future: asyncio.Future[None] = asyncio.get_event_loop().create_future()
        self._permit_waiters.append(future)
        await future

    def _release_permit(self) -> None:
        if self._max_concurrent is None:
            return
        if self._permit_waiters:
            next_waiter = self._permit_waiters.pop(0)
            next_waiter.set_result(None)
        else:
            self._active -= 1


# --- Session context building ---


def _derive_session_context_state(path_entries: Sequence[SessionTreeEntry]) -> dict[str, Any]:
    thinking_level = "off"
    model: dict[str, str] | None = None
    active_tool_names: list[str] | None = None

    for entry in path_entries:
        if isinstance(entry, ThinkingLevelChangeEntry):
            thinking_level = entry.thinking_level
        elif isinstance(entry, ModelChangeEntry):
            model = {"provider": entry.provider, "model_id": entry.model_id}
        elif isinstance(entry, MessageEntry):
            msg = entry.message
            if hasattr(msg, "role") and msg.role == "assistant":
                model = {
                    "provider": getattr(msg, "provider", ""),
                    "model_id": getattr(msg, "model", ""),
                }
        elif isinstance(entry, ActiveToolsChangeEntry):
            active_tool_names = list(entry.active_tool_names)

    return {
        "thinking_level": thinking_level,
        "model": model,
        "active_tool_names": active_tool_names,
    }


def default_context_entry_transform(path_entries: Sequence[SessionTreeEntry]) -> list[SessionTreeEntry]:
    """Apply the default compaction transform to path entries."""
    compaction: CompactionEntry | None = None
    for entry in path_entries:
        if isinstance(entry, CompactionEntry):
            compaction = entry

    if compaction is None:
        return list(path_entries)

    entries: list[SessionTreeEntry] = [compaction]
    compaction_idx = next(
        i for i, e in enumerate(path_entries) if isinstance(e, CompactionEntry) and e.id == compaction.id
    )

    if compaction.retained_tail:
        for i in range(compaction_idx + 1, len(path_entries)):
            entries.append(path_entries[i])
        return entries

    if compaction.first_kept_entry_id:
        found_first_kept = False
        for i in range(compaction_idx):
            entry = path_entries[i]
            if entry.id == compaction.first_kept_entry_id:
                found_first_kept = True
            if found_first_kept:
                entries.append(entry)

    for i in range(compaction_idx + 1, len(path_entries)):
        entries.append(path_entries[i])

    return entries


def build_context_entries(
    path_entries: Sequence[SessionTreeEntry],
    entry_transforms: Sequence[Callable[[Sequence[SessionTreeEntry]], Sequence[SessionTreeEntry]]] | None = None,
) -> list[SessionTreeEntry]:
    """Build the context entries from path entries with optional transforms."""
    entries = default_context_entry_transform(path_entries)
    if entry_transforms:
        for transform in entry_transforms:
            entries = list(transform(entries))
    return entries


def session_entry_to_context_messages(entry: SessionTreeEntry) -> list[Any]:
    """Convert a session tree entry to context messages."""
    if isinstance(entry, MessageEntry):
        return [entry.message]
    if isinstance(entry, CustomMessageEntry):
        return [
            create_custom_message(
                entry.custom_type,
                entry.content,
                entry.display,
                entry.details,
                entry.timestamp,
            )
        ]
    if isinstance(entry, CompactionEntry):
        messages: list[Any] = [create_compaction_summary_message(entry.summary, entry.tokens_before, entry.timestamp)]
        if entry.retained_tail:
            messages.extend(entry.retained_tail)
        return messages
    if isinstance(entry, BranchSummaryEntry) and entry.summary:
        return [create_branch_summary_message(entry.summary, entry.from_id, entry.timestamp)]
    return []


def build_session_context(
    path_entries: Sequence[SessionTreeEntry],
    entry_transforms: Sequence[Callable[[Sequence[SessionTreeEntry]], Sequence[SessionTreeEntry]]] | None = None,
) -> SessionContext:
    """Build a :class:`SessionContext` from path entries."""
    state = _derive_session_context_state(path_entries)
    context_entries = build_context_entries(path_entries, entry_transforms)
    messages: list[Any] = []
    for entry in context_entries:
        messages.extend(session_entry_to_context_messages(entry))
    return SessionContext(
        messages=messages,
        thinking_level=state["thinking_level"],
        model=state["model"],
        active_tool_names=state["active_tool_names"],
    )


# --- Session class ---


class Session:
    """Wraps a session storage backend and provides tree navigation and context building.

    Mirrors ``session.ts``.
    """

    def __init__(
        self,
        storage: SessionStorage,
        leaf_id: str | None,
        entry_transforms: Sequence[Callable[[Sequence[SessionTreeEntry]], Sequence[SessionTreeEntry]]] | None = None,
    ) -> None:
        self._storage = storage
        self._leaf_id = leaf_id
        self._entry_transforms = entry_transforms or []
        self._append_tail: asyncio.Future[None] = asyncio.get_event_loop().create_future()
        self._append_tail.set_result(None)

    @property
    def metadata(self) -> SessionMetadata:
        return self._storage.metadata

    async def get_leaf_id(self) -> str | None:
        return self._leaf_id

    async def get_entry(self, id: str) -> SessionTreeEntry | None:
        return await self._storage.read_entry(id)

    async def get_entries(self, options: SessionEntryCursorOptions | None = None) -> list[SessionTreeEntry]:
        return list(await self._storage.read_entries(options))

    async def get_branch(self, from_id: str | None = None) -> list[SessionTreeEntry]:
        leaf = from_id if from_id is not None else self._leaf_id
        return list(await self._storage.read_path_to_root_or_compaction(leaf))

    async def find_entries_on_branch(self, query: SessionBranchQuery | None = None) -> list[SessionTreeEntry]:
        q = query or SessionBranchQuery()
        if q.start is None:
            q.start = self._leaf_id
        return list(await self._storage.find_entries_on_branch(q))

    async def find_entry_on_branch(self, query: SessionBranchQuery | None = None) -> SessionTreeEntry | None:
        q = query or SessionBranchQuery()
        q.limit = 1
        results = await self.find_entries_on_branch(q)
        return results[0] if results else None

    async def build_context_entries(
        self,
        entry_transforms: Sequence[Callable[[Sequence[SessionTreeEntry]], Sequence[SessionTreeEntry]]] | None = None,
    ) -> list[SessionTreeEntry]:
        transforms = list(self._entry_transforms) + list(entry_transforms or [])
        return build_context_entries(await self.get_branch(), transforms)

    async def build_context(
        self,
        entry_transforms: Sequence[Callable[[Sequence[SessionTreeEntry]], Sequence[SessionTreeEntry]]] | None = None,
    ) -> SessionContext:
        transforms = list(self._entry_transforms) + list(entry_transforms or [])
        return build_session_context(await self.get_branch(), transforms)

    async def get_label(self, id: str) -> str | None:
        return await self._storage.get_label(id)

    async def get_session_stats(self) -> SessionStats:
        return await self._storage.get_stats()

    async def get_session_name(self) -> str | None:
        return await self._storage.get_name()

    async def _create_entry_id(self) -> str:
        for _ in range(100):
            id = _uuid_v7()[-8:]
            if await self.get_entry(id) is None:
                return id
        return _uuid_v7()

    async def _enqueue_append(self, create_entry: Callable[[dict[str, Any]], Any]) -> Any:
        """Enqueue an append operation, serialized via append_tail."""
        # Wait for previous append
        if not self._append_tail.done():
            await self._append_tail

        loop = asyncio.get_event_loop()
        entry = create_entry(
            {
                "id": await self._create_entry_id(),
                "parent_id": self._leaf_id,
                "timestamp": create_timestamp(),
            }
        )
        await self._storage.append_entry(entry)

        # Update leaf_id
        self._leaf_id = entry.target_id if isinstance(entry, LeafEntry) else entry.id

        # Set append_tail for next operation
        new_tail: asyncio.Future[None] = loop.create_future()
        new_tail.set_result(None)
        self._append_tail = new_tail

        return entry

    async def _append_typed_entry(
        self,
        create_entry: Callable[[dict[str, Any]], SessionTreeEntry],
    ) -> str:
        entry: SessionTreeEntry = await self._enqueue_append(create_entry)
        return entry.id

    async def append_message(self, message: Any) -> str:
        """Append a message entry to the session tree."""
        return await self._append_typed_entry(
            lambda base: MessageEntry(
                id=base["id"], parent_id=base["parent_id"], timestamp=base["timestamp"], message=message
            )
        )

    async def append_thinking_level_change(self, thinking_level: str) -> str:
        return await self._append_typed_entry(
            lambda base: ThinkingLevelChangeEntry(
                id=base["id"], parent_id=base["parent_id"], timestamp=base["timestamp"], thinking_level=thinking_level
            )
        )

    async def append_model_change(self, provider: str, model_id: str) -> str:
        return await self._append_typed_entry(
            lambda base: ModelChangeEntry(
                id=base["id"],
                parent_id=base["parent_id"],
                timestamp=base["timestamp"],
                provider=provider,
                model_id=model_id,
            )
        )

    async def append_active_tools_change(self, active_tool_names: list[str]) -> str:
        return await self._append_typed_entry(
            lambda base: ActiveToolsChangeEntry(
                id=base["id"],
                parent_id=base["parent_id"],
                timestamp=base["timestamp"],
                active_tool_names=list(active_tool_names),
            )
        )

    async def append_compaction(
        self,
        summary: str,
        first_kept_entry_id: str | None,
        tokens_before: int,
        details: Any = None,
        from_hook: bool | None = None,
        usage: Usage | None = None,
        retained_tail: list[Any] | None = None,
    ) -> str:
        return await self._append_typed_entry(
            lambda base: CompactionEntry(
                id=base["id"],
                parent_id=base["parent_id"],
                timestamp=base["timestamp"],
                summary=summary,
                first_kept_entry_id=first_kept_entry_id,
                tokens_before=tokens_before,
                details=details,
                from_hook=from_hook,
                usage=usage,
                retained_tail=retained_tail,
            )
        )

    async def append_custom_entry(self, custom_type: str, data: Any = None) -> str:
        return await self._append_typed_entry(
            lambda base: CustomEntry(
                id=base["id"],
                parent_id=base["parent_id"],
                timestamp=base["timestamp"],
                custom_type=custom_type,
                data=data,
            )
        )

    async def append_custom_message_entry(
        self,
        custom_type: str,
        content: str | list[Any],
        display: bool,
        details: Any = None,
    ) -> str:
        return await self._append_typed_entry(
            lambda base: CustomMessageEntry(
                id=base["id"],
                parent_id=base["parent_id"],
                timestamp=base["timestamp"],
                custom_type=custom_type,
                content=content,
                display=display,
                details=details,
            )
        )

    async def append_label(self, target_id: str, label: str | None) -> str:
        if await self.get_entry(target_id) is None:
            raise SessionError("not_found", f"Entry {target_id} not found")
        return await self._append_typed_entry(
            lambda base: LabelEntry(
                id=base["id"],
                parent_id=base["parent_id"],
                timestamp=base["timestamp"],
                target_id=target_id,
                label=label,
            )
        )

    async def append_session_name(self, name: str) -> str:
        sanitized = re.sub(r"[\r\n]+", " ", name).strip()
        return await self._append_typed_entry(
            lambda base: SessionInfoEntry(
                id=base["id"],
                parent_id=base["parent_id"],
                timestamp=base["timestamp"],
                name=sanitized,
            )
        )

    async def move_to(
        self,
        entry_id: str | None,
        summary: dict[str, Any] | None = None,
    ) -> str | None:
        """Move the session leaf to a different entry, optionally appending a branch summary."""
        if entry_id is not None and await self.get_entry(entry_id) is None:
            raise SessionError("not_found", f"Entry {entry_id} not found")

        # Set leaf
        await self._enqueue_append(
            lambda base: LeafEntry(
                id=base["id"], parent_id=base["parent_id"], timestamp=base["timestamp"], target_id=entry_id
            )
        )

        if summary is None:
            return None

        return await self._append_typed_entry(
            lambda base: BranchSummaryEntry(
                id=base["id"],
                parent_id=base["parent_id"],
                timestamp=base["timestamp"],
                from_id=entry_id or "root",
                summary=summary["summary"],
                details=summary.get("details"),
                usage=summary.get("usage"),
                from_hook=summary.get("fromHook"),
            )
        )


async def create_session(
    storage: SessionStorage,
    entry_transforms: Sequence[Callable[[Sequence[SessionTreeEntry]], Sequence[SessionTreeEntry]]] | None = None,
) -> Session:
    """Create a Session from a storage backend, reading the current head."""
    head = await storage.read_head()
    return Session(storage, head.leaf_id, entry_transforms)


# --- Fork selection helpers ---


def create_session_fork_selection(options: SessionForkOptions) -> SessionForkSelection:
    """Determine fork selection from fork options."""
    if options.entry_id is None:
        return {"kind": "all"}
    if options.position == "at":
        return {"kind": "through_entry", "entryId": options.entry_id}
    return {"kind": "before_user_message", "entryId": options.entry_id}


async def read_session_entries_for_fork(
    source: Any,
    selection: SessionForkSelection,
) -> list[SessionTreeEntry]:
    """Read entries from source for forking based on selection."""
    if selection["kind"] == "all":
        entries = source.read_entries()
        if asyncio.iscoroutine(entries):
            entries = await entries
        return list(entries)

    entry_id = selection["entryId"]
    target = source.read_entry(entry_id)
    if asyncio.iscoroutine(target):
        target = await target
    if target is None:
        raise SessionError("invalid_fork_target", f"Entry {entry_id} not found")

    if selection["kind"] == "through_entry":
        path = source.read_path_to_root_or_compaction(target.id)
        if asyncio.iscoroutine(path):
            path = await path
        return list(path)

    # before_user_message
    if not isinstance(target, MessageEntry):
        raise SessionError("invalid_fork_target", f"Entry {entry_id} is not a user message")
    msg = target.message
    if not (hasattr(msg, "role") and msg.role == "user"):
        raise SessionError("invalid_fork_target", f"Entry {entry_id} is not a user message")

    path = source.read_path_to_root_or_compaction(target.parent_id)
    if asyncio.iscoroutine(path):
        path = await path
    return list(path)


# --- InMemoryRepository ---


class _InMemoryStorage:
    """In-memory session storage backend."""

    def __init__(self, metadata: SessionMetadata, entries: ArraySessionIndex) -> None:
        self._metadata = metadata
        self._entries = entries

    @property
    def metadata(self) -> SessionMetadata:
        return self._metadata

    async def read_head(self) -> SessionHead:
        return self._entries.read_head()

    async def read_entry(self, id: str) -> SessionTreeEntry | None:
        return self._entries.read_entry(id)

    async def read_entries(self, options: SessionEntryCursorOptions | None = None) -> list[SessionTreeEntry]:
        return self._entries.read_entries(options)

    async def append_entry(self, entry: SessionTreeEntry) -> None:
        self._entries.append(entry)

    async def find_entries_on_branch(self, query: SessionBranchQuery) -> list[SessionTreeEntry]:
        return self._entries.find_entries_on_branch(query)

    async def read_path_to_root_or_compaction(self, leaf_id: str | None) -> list[SessionTreeEntry]:
        return self._entries.read_path_to_root_or_compaction(leaf_id)

    async def get_label(self, id: str) -> str | None:
        return self._entries.get_label(id)

    async def get_name(self) -> str | None:
        return self._entries.get_name()

    async def get_stats(self) -> SessionStats:
        return self._entries.get_stats()


class InMemoryRepository:
    """In-memory session repository.

    Mirrors ``memory-repo.ts``. Uses :class:`ArraySessionIndex` for each session.
    """

    def __init__(
        self,
        entry_transforms: Sequence[Callable[[Sequence[SessionTreeEntry]], Sequence[SessionTreeEntry]]] | None = None,
    ) -> None:
        self._sessions: dict[str, _InMemoryStorage] = {}
        self._operations = KeyedOperationQueue()
        self._entry_transforms = entry_transforms or []
        self._disposed = False

    async def create(self, options: SessionCreateOptions | None = None) -> Session:
        self._assert_open()
        opts = options or SessionCreateOptions()
        session_id = opts.id or create_session_id()

        def _create() -> _InMemoryStorage:
            metadata = SessionMetadata(id=session_id, created_at=create_timestamp())
            index = ArraySessionIndex()
            storage = _InMemoryStorage(metadata, index)
            self._sessions[session_id] = storage
            return storage

        storage = await self._operations.enqueue(session_id, _create)
        return await create_session(storage, self._entry_transforms)

    async def open(self, metadata: SessionMetadata) -> Session:
        self._assert_open()
        storage = await self._operations.enqueue(metadata.id, lambda: self._get_state(metadata.id))
        return await create_session(storage, self._entry_transforms)

    async def list(self) -> list[SessionMetadata]:
        self._assert_open()
        result: list[SessionMetadata] = await self._operations.enqueue_barrier(
            lambda: [s.metadata for s in self._sessions.values()]
        )
        return result

    async def delete(self, metadata: SessionMetadata) -> None:
        self._assert_open()

        def _delete() -> None:
            self._sessions.pop(metadata.id, None)

        await self._operations.enqueue(metadata.id, _delete)

    async def fork(self, source: SessionMetadata, options: SessionForkOptions) -> Session:
        self._assert_open()
        opts = SessionCreateOptions(id=options.id)
        session_id = opts.id or create_session_id()
        selection = create_session_fork_selection(options)

        source_entries = await self._operations.enqueue(
            source.id,
            lambda: self._get_state(source.id)._entries,
        )
        fork_entries = await read_session_entries_for_fork(source_entries, selection)

        def _create_fork() -> _InMemoryStorage:
            metadata = SessionMetadata(id=session_id, created_at=create_timestamp())
            index = ArraySessionIndex(fork_entries)
            storage = _InMemoryStorage(metadata, index)
            self._sessions[session_id] = storage
            return storage

        storage = await self._operations.enqueue(session_id, _create_fork)
        return await create_session(storage, self._entry_transforms)

    async def dispose(self) -> None:
        self._disposed = True
        await self._operations.drain()

    def _assert_open(self) -> None:
        if self._disposed:
            raise SessionError("storage", "In-memory session repository is disposed")

    def _get_state(self, session_id: str) -> _InMemoryStorage:
        state = self._sessions.get(session_id)
        if state is None:
            raise SessionError("not_found", f"Session not found: {session_id}")
        return state


# --- JsonlRepository ---


def _encode_cwd(cwd: str) -> str:
    """Encode a cwd path for use in a directory name."""
    stripped = re.sub(r"^[/\\]", "", cwd)
    safe = re.sub(r"[/\\:]", "-", stripped)
    return f"--{safe}--"


def _parse_header(line: str, path: str) -> dict[str, Any]:
    """Parse a JSONL session header line."""
    try:
        value = json.loads(line)
    except json.JSONDecodeError as exc:
        raise SessionError("invalid_session", f"Invalid JSONL session file {path}: bad header", exc) from exc

    if not isinstance(value, dict):
        raise SessionError("invalid_session", f"Invalid JSONL session file {path}: header is not an object")

    if value.get("type") != "session" or value.get("version") != 3:
        if value.get("type") == "session":
            raise SessionError("invalid_session", f"Invalid JSONL session file {path}: unsupported version")
        raise SessionError("invalid_session", f"Invalid JSONL session file {path}: bad header")

    if not value.get("id"):
        raise SessionError("invalid_session", f"Invalid JSONL session file {path}: missing id")
    if not value.get("timestamp"):
        raise SessionError("invalid_session", f"Invalid JSONL session file {path}: missing timestamp")
    if not value.get("cwd"):
        raise SessionError("invalid_session", f"Invalid JSONL session file {path}: missing cwd")

    return value


def _parse_entry(line: str, path: str, line_number: int) -> SessionTreeEntry:
    """Parse a JSONL session entry line into a SessionTreeEntry."""
    try:
        value = json.loads(line)
    except json.JSONDecodeError as exc:
        raise SessionError(
            "invalid_entry",
            f"Invalid JSONL session file {path}: line {line_number} is not valid JSON",
            exc,
        ) from exc

    if not isinstance(value, dict):
        raise SessionError("invalid_entry", f"Invalid JSONL session file {path}: line {line_number} is not an object")

    entry_type = value.get("type")
    if not isinstance(entry_type, str):
        raise SessionError("invalid_entry", f"Invalid JSONL session file {path}: line {line_number} missing type")
    if not value.get("id"):
        raise SessionError("invalid_entry", f"Invalid JSONL session file {path}: line {line_number} missing id")
    if value.get("parentId") is not None and not isinstance(value.get("parentId"), str):
        raise SessionError("invalid_entry", f"Invalid JSONL session file {path}: line {line_number} bad parentId")
    if not value.get("timestamp"):
        raise SessionError("invalid_entry", f"Invalid JSONL session file {path}: line {line_number} missing timestamp")

    return _entry_from_dict(value)


def _entry_from_dict(value: dict[str, Any]) -> SessionTreeEntry:
    """Create a SessionTreeEntry from a parsed dict."""
    entry_type = value["type"]
    base = {
        "id": value["id"],
        "parent_id": value.get("parentId"),
        "timestamp": value["timestamp"],
    }

    if entry_type == "message":
        return MessageEntry(**base, message=value.get("message"))
    if entry_type == "thinking_level_change":
        return ThinkingLevelChangeEntry(**base, thinking_level=value.get("thinkingLevel", "off"))
    if entry_type == "model_change":
        return ModelChangeEntry(**base, provider=value.get("provider", ""), model_id=value.get("modelId", ""))
    if entry_type == "active_tools_change":
        return ActiveToolsChangeEntry(**base, active_tool_names=value.get("activeToolNames", []))
    if entry_type == "compaction":
        return CompactionEntry(
            **base,
            summary=value.get("summary", ""),
            first_kept_entry_id=value.get("firstKeptEntryId"),
            tokens_before=value.get("tokensBefore", 0),
            retained_tail=value.get("retainedTail"),
            details=value.get("details"),
            usage=value.get("usage"),
            from_hook=value.get("fromHook"),
        )
    if entry_type == "branch_summary":
        return BranchSummaryEntry(
            **base,
            from_id=value.get("fromId", ""),
            summary=value.get("summary", ""),
            details=value.get("details"),
            usage=value.get("usage"),
            from_hook=value.get("fromHook"),
        )
    if entry_type == "custom":
        return CustomEntry(**base, custom_type=value.get("customType", ""), data=value.get("data"))
    if entry_type == "custom_message":
        return CustomMessageEntry(
            **base,
            custom_type=value.get("customType", ""),
            content=value.get("content", ""),
            details=value.get("details"),
            display=value.get("display", True),
        )
    if entry_type == "label":
        return LabelEntry(**base, target_id=value.get("targetId", ""), label=value.get("label"))
    if entry_type == "session_info":
        return SessionInfoEntry(**base, name=value.get("name"))
    if entry_type == "leaf":
        return LeafEntry(**base, target_id=value.get("targetId"))

    raise SessionError("invalid_entry", f"Unknown entry type: {entry_type}")


def _entry_to_dict(entry: SessionTreeEntry) -> dict[str, Any]:
    """Serialize a SessionTreeEntry to a dict for JSONL storage."""
    result: dict[str, Any] = {
        "type": entry.type,
        "id": entry.id,
        "parentId": entry.parent_id,
        "timestamp": entry.timestamp,
    }

    if isinstance(entry, MessageEntry):
        result["message"] = entry.message
    elif isinstance(entry, ThinkingLevelChangeEntry):
        result["thinkingLevel"] = entry.thinking_level
    elif isinstance(entry, ModelChangeEntry):
        result["provider"] = entry.provider
        result["modelId"] = entry.model_id
    elif isinstance(entry, ActiveToolsChangeEntry):
        result["activeToolNames"] = entry.active_tool_names
    elif isinstance(entry, CompactionEntry):
        result["summary"] = entry.summary
        result["firstKeptEntryId"] = entry.first_kept_entry_id
        result["tokensBefore"] = entry.tokens_before
        result["retainedTail"] = entry.retained_tail
        result["details"] = entry.details
        result["usage"] = entry.usage
        result["fromHook"] = entry.from_hook
    elif isinstance(entry, BranchSummaryEntry):
        result["fromId"] = entry.from_id
        result["summary"] = entry.summary
        result["details"] = entry.details
        result["usage"] = entry.usage
        result["fromHook"] = entry.from_hook
    elif isinstance(entry, CustomEntry):
        result["customType"] = entry.custom_type
        result["data"] = entry.data
    elif isinstance(entry, CustomMessageEntry):
        result["customType"] = entry.custom_type
        result["content"] = entry.content
        result["details"] = entry.details
        result["display"] = entry.display
    elif isinstance(entry, LabelEntry):
        result["targetId"] = entry.target_id
        result["label"] = entry.label
    elif isinstance(entry, SessionInfoEntry):
        result["name"] = entry.name
    elif isinstance(entry, LeafEntry):
        result["targetId"] = entry.target_id

    return result


def _metadata_from_header(header: dict[str, Any], path: str) -> JsonlSessionMetadata:
    return JsonlSessionMetadata(
        id=header["id"],
        created_at=header["timestamp"],
        cwd=header["cwd"],
        path=path,
        parent_session_path=header.get("parentSession"),
        metadata=header.get("metadata"),
    )


def _load_jsonl_session_metadata(path: str) -> JsonlSessionMetadata:
    """Load just the metadata (first line) from a JSONL session file."""
    with open(path, encoding="utf-8") as f:
        first_line = f.readline()
    if not first_line.strip():
        raise SessionError("invalid_session", f"Invalid JSONL session file {path}: missing header")
    header = _parse_header(first_line.strip(), path)
    return _metadata_from_header(header, path)


def _load_jsonl_session(path: str) -> tuple[JsonlSessionMetadata, list[SessionTreeEntry]]:
    """Load a full JSONL session file."""
    content = Path(path).read_text(encoding="utf-8")
    lines = [line for line in content.split("\n") if line.strip()]
    if not lines:
        raise SessionError("invalid_session", f"Invalid JSONL session file {path}: missing header")

    header = _parse_header(lines[0], path)
    entries: list[SessionTreeEntry] = []
    entry_ids: set[str] = set()
    for i, line in enumerate(lines[1:], start=2):
        entry = _parse_entry(line, path, i)
        if entry.id in entry_ids:
            raise SessionError("invalid_session", f"Invalid JSONL session file {path}: duplicate entry id {entry.id}")
        entry_ids.add(entry.id)
        entries.append(entry)

    return _metadata_from_header(header, path), entries


class _JsonlStorage:
    """JSONL-backed session storage."""

    def __init__(self, metadata: JsonlSessionMetadata, entries: ArraySessionIndex, path: str) -> None:
        self._metadata = metadata
        self._entries = entries
        self._path = path

    @property
    def metadata(self) -> JsonlSessionMetadata:
        return self._metadata

    async def read_head(self) -> SessionHead:
        return self._entries.read_head()

    async def read_entry(self, id: str) -> SessionTreeEntry | None:
        return self._entries.read_entry(id)

    async def read_entries(self, options: SessionEntryCursorOptions | None = None) -> list[SessionTreeEntry]:
        return self._entries.read_entries(options)

    async def append_entry(self, entry: SessionTreeEntry) -> None:
        if self._entries.has(entry.id):
            raise SessionError("invalid_entry", f"Entry {entry.id} already exists")
        line = json.dumps(_entry_to_dict(entry)) + "\n"
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(line)
        self._entries.append(entry)

    async def find_entries_on_branch(self, query: SessionBranchQuery) -> list[SessionTreeEntry]:
        return self._entries.find_entries_on_branch(query)

    async def read_path_to_root_or_compaction(self, leaf_id: str | None) -> list[SessionTreeEntry]:
        return self._entries.read_path_to_root_or_compaction(leaf_id)

    async def get_label(self, id: str) -> str | None:
        return self._entries.get_label(id)

    async def get_name(self) -> str | None:
        return self._entries.get_name()

    async def get_stats(self) -> SessionStats:
        return self._entries.get_stats()


class JsonlRepository:
    """JSONL file-based session repository.

    Mirrors ``jsonl-repo.ts``. Stores sessions as JSONL files in a directory tree
    organized by encoded cwd.
    """

    def __init__(
        self,
        sessions_root: str,
        *,
        entry_transforms: Sequence[Callable[[Sequence[SessionTreeEntry]], Sequence[SessionTreeEntry]]] | None = None,
    ) -> None:
        self._sessions_root = Path(sessions_root).resolve()
        self._entry_transforms = entry_transforms or []
        self._disposed = False
        self._entry_indexes: dict[str, ArraySessionIndex] = {}

    async def create(self, options: JsonlSessionCreateOptions | None = None) -> Session:
        self._assert_open()
        opts = options or JsonlSessionCreateOptions(cwd=os.getcwd())
        session_id = opts.id or create_session_id()

        dir_path = self._get_session_dir(opts.cwd)
        dir_path.mkdir(parents=True, exist_ok=True)

        timestamp = create_timestamp()
        file_name = f"{timestamp.replace(':', '-')}-{quote(session_id)}.jsonl"
        file_path = dir_path / file_name

        if file_path.exists():
            raise SessionError("invalid_session", f"Session already exists: {file_path}")

        header = {
            "type": "session",
            "version": 3,
            "id": session_id,
            "timestamp": timestamp,
            "cwd": opts.cwd,
            "parentSession": opts.parent_session_path,
            "metadata": opts.metadata,
        }

        content = json.dumps(header) + "\n"
        file_path.write_text(content, encoding="utf-8")

        metadata = _metadata_from_header(header, str(file_path))
        index = ArraySessionIndex()
        self._entry_indexes[str(file_path)] = index
        storage = _JsonlStorage(metadata, index, str(file_path))
        return await create_session(storage, self._entry_transforms)

    async def open(self, metadata: JsonlSessionMetadata) -> Session:
        self._assert_open()
        path = Path(metadata.path)
        if not path.exists():
            raise SessionError("not_found", f"Session not found: {metadata.path}")

        meta, entries_data = _load_jsonl_session(str(path))
        index = ArraySessionIndex(entries_data)
        self._entry_indexes[str(path)] = index
        storage = _JsonlStorage(meta, index, str(path))
        return await create_session(storage, self._entry_transforms)

    async def list(self, cwd: str | None = None) -> list[JsonlSessionMetadata]:
        self._assert_open()
        if not self._sessions_root.exists():
            return []

        dirs = [self._get_session_dir(cwd)] if cwd else [d for d in self._sessions_root.iterdir() if d.is_dir()]

        sessions: list[JsonlSessionMetadata] = []
        for dir_path in dirs:
            if not dir_path.exists():
                continue
            for file_path in sorted(dir_path.glob("*.jsonl")):
                try:
                    sessions.append(_load_jsonl_session_metadata(str(file_path)))
                except SessionError:
                    continue

        sessions.sort(key=lambda s: s.created_at, reverse=True)
        return sessions

    async def delete(self, metadata: JsonlSessionMetadata) -> None:
        self._assert_open()
        path = Path(metadata.path)
        if path.exists():
            path.unlink()
        self._entry_indexes.pop(metadata.path, None)

    async def fork(self, source: JsonlSessionMetadata, options: SessionForkOptions) -> Session:
        self._assert_open()
        source_path = Path(source.path)
        if not source_path.exists():
            raise SessionError("not_found", f"Session not found: {source.path}")

        _meta, source_entries_data = _load_jsonl_session(source.path)
        source_index = ArraySessionIndex(source_entries_data)
        selection = create_session_fork_selection(options)
        fork_entries = await read_session_entries_for_fork(source_index, selection)

        # Create new session and replay the selected entries into its storage
        opts = JsonlSessionCreateOptions(
            id=options.id,
            cwd=source.cwd,
            parent_session_path=source.path,
            metadata=source.metadata,
        )
        new_session = await self.create(opts)
        storage = new_session._storage
        for entry in fork_entries:
            await storage.append_entry(entry)
        return await create_session(storage, self._entry_transforms)

    async def dispose(self) -> None:
        self._disposed = True

    def _assert_open(self) -> None:
        if self._disposed:
            raise SessionError("storage", "JSONL session repository is disposed")

    def _get_session_dir(self, cwd: str) -> Path:
        return self._sessions_root / _encode_cwd(cwd)
