"""Session fork utilities.

A fork creates a new session linked to its parent via parentSessionId.
The child starts with zero harness entries and an isolated context.

The child session ID is derived deterministically so that replay
re-attaches to the same child instead of spawning a duplicate:

    child_id = f(parent_session_id, tool_call_id)

If the process crashes and pi re-executes with replay, the same
invocation resolves to the same child session.
"""

from __future__ import annotations

import hashlib


def derive_child_session_id(parent_session_id: str, tool_call_id: str) -> str:
    """Return a deterministic child session ID.

    SHA-256 of ``"{parent_session_id}:{tool_call_id}"`` truncated to 24 hex
    chars.  Determinism ensures that replay with the same inputs re-attaches
    to the existing child rather than creating a duplicate.
    """
    raw = f"{parent_session_id}:{tool_call_id}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


__all__ = ["derive_child_session_id"]
