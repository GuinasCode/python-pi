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

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pi_agent_core.types import AgentTool

ExtensionFactory = Callable[["ExtensionAPI"], Any]


@dataclass
class ExtensionError:
    """One extension that failed to load or register cleanly."""

    path: str
    error: str


@dataclass
class LoadedExtension:
    """One successfully loaded extension: its source path and the tools it
    registered. ``tool_names`` is a convenience view for display/reporting
    (mirrors the original's ``tools.keys()``); ``tools`` holds the actual
    :class:`AgentTool` objects, the source of truth callers register into
    an ``AgentSession``."""

    path: str
    tools: list[AgentTool] = field(default_factory=list)

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self.tools]


@dataclass
class LoadExtensionsResult:
    extensions: list[LoadedExtension] = field(default_factory=list)
    errors: list[ExtensionError] = field(default_factory=list)


class ExtensionAPI:
    """The ``pi`` object passed to an extension's entry point.

    Phase A/B surface: tool registration only. Later phases add event
    subscription (``on``), command/shortcut/flag registration, rendering
    hooks, and provider registration directly onto this class.
    """

    def __init__(self) -> None:
        self._tools: list[AgentTool] = []

    def register_tool(self, tool: AgentTool) -> None:
        """Register a tool the LLM can call."""
        self._tools.append(tool)

    @property
    def tools(self) -> list[AgentTool]:
        return list(self._tools)
