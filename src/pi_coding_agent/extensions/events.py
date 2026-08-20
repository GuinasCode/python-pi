"""Lifecycle event types for the extension system.

Phase D slice of packages/coding-agent/src/core/extensions/types.ts: the
original defines 30+ event types (agent/turn/message/session/input/
provider/tool events, one ToolCallEvent variant per built-in tool, ...).
This port covers the handful with the most real-world leverage — tool
call/result (mutate arguments, block execution, override results) and
agent/turn/session lifecycle notifications — porting the rest
opportunistically as concrete use cases justify them, rather than
speculatively translating the entire original surface up front.

Unlike the original (one ``ToolCallEvent`` variant per built-in tool, e.g.
``BashToolCallEvent``/``ReadToolCallEvent``/...), this port uses a single
generic :class:`ToolCallEvent` for every tool name — narrowing by
``tool_name`` is a plain string check here rather than a discriminated
union, since Python's structural typing gets little from replicating that
split.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pi_ai import ImageContent, TextContent

__all__ = [
    "AgentEndEvent",
    "AgentStartEvent",
    "ExtensionContext",
    "ExtensionHandler",
    "SessionShutdownEvent",
    "SessionStartEvent",
    "ToolCallEvent",
    "ToolCallEventResult",
    "ToolResultEvent",
    "ToolResultEventResult",
    "TurnEndEvent",
    "TurnStartEvent",
]


@dataclass
class ExtensionContext:
    """Passed to every event handler alongside the event itself.

    Deliberately minimal for now — the original's ExtensionContext also
    exposes actions (sendMessage, appendEntry, setLabel, ...); those are
    added if/when a handler use case actually needs them.
    """

    cwd: str


@dataclass
class ToolCallEvent:
    """Fired before a tool executes. ``arguments`` is the *same* dict the
    tool call will actually run with — mutate it in place to patch
    arguments before execution; later handlers see earlier mutations."""

    type: str = field(default="tool_call", init=False)
    tool_call_id: str = ""
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallEventResult:
    """Return from a ``tool_call`` handler to block execution or note intent."""

    block: bool = False
    reason: str | None = None
    terminate: bool | None = None


@dataclass
class ToolResultEvent:
    """Fired after a tool executes (or was blocked)."""

    type: str = field(default="tool_result", init=False)
    tool_call_id: str = ""
    tool_name: str = ""
    content: list[TextContent | ImageContent] = field(default_factory=list)
    is_error: bool = False


@dataclass
class ToolResultEventResult:
    """Return from a ``tool_result`` handler to override the finalized result.

    Only set fields take effect; unset (``None``) fields keep whatever the
    previous handler (or the real execution) produced — handlers chain,
    they don't each replace the whole event.
    """

    content: list[TextContent | ImageContent] | None = None
    is_error: bool | None = None


@dataclass
class AgentStartEvent:
    type: str = field(default="agent_start", init=False)


@dataclass
class AgentEndEvent:
    type: str = field(default="agent_end", init=False)


@dataclass
class TurnStartEvent:
    type: str = field(default="turn_start", init=False)
    turn: int = 0


@dataclass
class TurnEndEvent:
    type: str = field(default="turn_end", init=False)
    turn: int = 0


@dataclass
class SessionStartEvent:
    type: str = field(default="session_start", init=False)


@dataclass
class SessionShutdownEvent:
    type: str = field(default="session_shutdown", init=False)


ExtensionHandler = Callable[[Any, ExtensionContext], "Any | Awaitable[Any]"]
