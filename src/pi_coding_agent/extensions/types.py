"""Core types for the Pi extension system.

Python port of packages/coding-agent/src/core/extensions/types.ts. Ported
incrementally — this module starts with tool registration and error
reporting (Phase A/B: enough for AgentSession.reload()/get_extensions());
event subscription, command/shortcut/flag registration, and rendering
hooks are added onto :class:`ExtensionAPI` by later phases
(pi_coding_agent.extensions.runner and friends), not reinvented here.

Deliberate simplification vs. the TS original: extension entry points here
must be synchronous (``def extension(pi) -> None``, not ``async def``).
The original allows `Promise<void>` because its session construction is
already async top-to-bottom; this port's ``AgentSession.__init__`` is
synchronous, and unifying that split just for extension loading isn't
worth the complexity. An extension needing async setup can do it lazily
inside a registered tool's ``execute()``, which is already async.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pi_agent_core.types import AgentTool
from pi_coding_agent.extensions.events import ExtensionHandler

ExtensionFactory = Callable[["ExtensionAPI"], Any]


@dataclass
class ExtensionError:
    """One extension that failed to load or register cleanly."""

    path: str
    error: str


@dataclass
class LoadedExtension:
    """One successfully loaded extension: its source path, the tools it
    registered, and the event handlers it subscribed. ``tool_names`` is a
    convenience view for display/reporting (mirrors the original's
    ``tools.keys()``); ``tools`` holds the actual :class:`AgentTool`
    objects, the source of truth callers register into an
    ``AgentSession``."""

    path: str
    tools: list[AgentTool] = field(default_factory=list)
    handlers: dict[str, list[ExtensionHandler]] = field(default_factory=dict)

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self.tools]


@dataclass
class LoadExtensionsResult:
    extensions: list[LoadedExtension] = field(default_factory=list)
    errors: list[ExtensionError] = field(default_factory=list)


class ExtensionAPI:
    """The ``pi`` object passed to an extension's entry point.

    Phase A/B/D surface: tool registration and event subscription.
    Command/shortcut/flag registration, rendering hooks, and provider
    registration are added directly onto this class by later phases.
    """

    def __init__(self) -> None:
        self._tools: list[AgentTool] = []
        self._handlers: dict[str, list[ExtensionHandler]] = defaultdict(list)

    def register_tool(self, tool: AgentTool) -> None:
        """Register a tool the LLM can call."""
        self._tools.append(tool)

    @property
    def tools(self) -> list[AgentTool]:
        return list(self._tools)

    def on(self, event_name: str, handler: ExtensionHandler) -> None:
        """Subscribe to a lifecycle event. See pi_coding_agent.extensions.events
        for the currently-supported event names and their (event, result) shapes:
        "tool_call", "tool_result", "agent_start", "agent_end", "turn_start",
        "turn_end", "session_start", "session_shutdown"."""
        self._handlers[event_name].append(handler)

    @property
    def handlers(self) -> dict[str, list[ExtensionHandler]]:
        return dict(self._handlers)
