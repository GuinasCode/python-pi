"""ExtensionRunner: loads and holds every extension for one session, and
dispatches lifecycle events to whichever of them subscribed.

Phase B/D slice of packages/coding-agent/src/core/extensions/runner.ts —
tool registration, error reporting, and event emission for the handful of
events events.py currently defines. Command/shortcut/flag registries and
provider registration that the full ``ExtensionRunner`` also owns are
added by later phases directly onto this class.
"""

from __future__ import annotations

import inspect
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

from pi_agent_core.types import AgentTool
from pi_ai.models import MutableModels
from pi_coding_agent.extensions.events import (
    ExtensionContext,
    ExtensionHandler,
    ToolCallEvent,
    ToolCallEventResult,
    ToolResultEvent,
    ToolResultEventResult,
)
from pi_coding_agent.extensions.loader import discover_extension_paths, load_extensions
from pi_coding_agent.extensions.types import (
    ExtensionError,
    ExtensionFlag,
    LoadExtensionsResult,
    RegisteredCommand,
)

__all__ = ["ExtensionRunner"]


async def _call_handler(handler: ExtensionHandler, event: Any, ctx: ExtensionContext) -> Any:
    result = handler(event, ctx)
    if inspect.isawaitable(result):
        result = await result
    return result


class ExtensionRunner:
    """Discovers, loads, and holds every extension for a session's cwd;
    dispatches lifecycle events to their subscribed handlers."""

    def __init__(
        self,
        cwd: str | Path,
        agent_dir: str | Path | None = None,
        configured_paths: list[str] | None = None,
        models: MutableModels | None = None,
    ) -> None:
        self._cwd = cwd
        self._agent_dir = agent_dir
        self._configured_paths = configured_paths or []
        self._models = models
        self._result = LoadExtensionsResult()
        self._handlers: dict[str, list[tuple[str, ExtensionHandler]]] = defaultdict(list)
        # Shared with every extension's ExtensionAPI (see load_extensions) so
        # set_flag_value() takes effect for pi.get_flag() calls made from
        # inside a tool's execute(), long after load() itself has returned.
        self._flag_values: dict[str, bool | str] = {}

    def load(self) -> LoadExtensionsResult:
        """(Re)discover and (re)load every extension, replacing any
        previous result. Safe to call again (e.g. from a session reload)
        after project files on disk have changed."""
        paths = discover_extension_paths(self._cwd, self._agent_dir, self._configured_paths)
        self._result = load_extensions(paths, flag_values=self._flag_values, models=self._models)

        self._handlers = defaultdict(list)
        for ext in self._result.extensions:
            for event_name, handlers in ext.handlers.items():
                for handler in handlers:
                    self._handlers[event_name].append((ext.path, handler))

        return self._result

    def get_commands(self) -> list[RegisteredCommand]:
        """Every slash command registered by every successfully loaded extension."""
        return [command for ext in self._result.extensions for command in ext.commands]

    def get_flags(self) -> dict[str, ExtensionFlag]:
        """Every CLI flag declared by every successfully loaded extension,
        keyed by name (a later extension's declaration wins on a name clash)."""
        flags: dict[str, ExtensionFlag] = {}
        for ext in self._result.extensions:
            flags.update(ext.flags)
        return flags

    def set_flag_value(self, name: str, value: bool | str) -> None:
        """Set a flag's current value — visible to every loaded extension's
        pi.get_flag(name) from this point on."""
        self._flag_values[name] = value

    def get_extensions(self) -> LoadExtensionsResult:
        """The result of the most recent load() — empty (no extensions,
        no errors) if load() hasn't been called yet."""
        return self._result

    def get_tools(self) -> list[AgentTool]:
        """Every tool registered by every successfully loaded extension."""
        return [tool for ext in self._result.extensions for tool in ext.tools]

    def get_extension_paths(self) -> list[str]:
        """Source paths of every successfully loaded extension."""
        return [ext.path for ext in self._result.extensions]

    def has_handlers(self, event_name: str) -> bool:
        return bool(self._handlers.get(event_name))

    def _context(self) -> ExtensionContext:
        return ExtensionContext(cwd=str(self._cwd))

    def _record_runtime_error(self, extension_path: str, event_name: str, exc: Exception) -> None:
        """A handler raising must never take down the emit — record it
        alongside load errors instead (visible via get_extensions().errors)."""
        self._result.errors.append(ExtensionError(path=extension_path, error=f"[{event_name}] {exc}"))

    async def emit(self, event_name: str, event: Any) -> None:
        """Fire a pure-notification event (agent_start/agent_end/turn_start/
        turn_end/session_start/session_shutdown): every handler runs for
        side effects, in registration order; return values are ignored."""
        ctx = self._context()
        for extension_path, handler in self._handlers.get(event_name, []):
            try:
                await _call_handler(handler, event, ctx)
            except Exception as exc:
                self._record_runtime_error(extension_path, event_name, exc)

    async def emit_tool_call(self, event: ToolCallEvent) -> ToolCallEventResult | None:
        """Fire "tool_call". ``event.arguments`` is mutated in place by
        handlers that want to patch arguments; a handler returning
        ``block=True`` short-circuits immediately, skipping any remaining
        handlers, mirroring the original's "Early termination" contract.
        """
        ctx = self._context()
        result: ToolCallEventResult | None = None
        for extension_path, handler in self._handlers.get("tool_call", []):
            try:
                handler_result = await _call_handler(handler, event, ctx)
            except Exception as exc:
                self._record_runtime_error(extension_path, "tool_call", exc)
                continue
            if handler_result:
                result = handler_result
                if result.block:
                    return result
        return result

    async def emit_tool_result(self, event: ToolResultEvent) -> ToolResultEventResult | None:
        """Fire "tool_result". Handlers chain: each sees the previous
        handler's mutations already applied, and only the fields it sets
        on its returned :class:`ToolResultEventResult` take effect. Returns
        None if no handler actually changed anything."""
        ctx = self._context()
        current = replace(event)
        modified = False
        for extension_path, handler in self._handlers.get("tool_result", []):
            try:
                handler_result = await _call_handler(handler, current, ctx)
            except Exception as exc:
                self._record_runtime_error(extension_path, "tool_result", exc)
                continue
            if handler_result is None:
                continue
            if handler_result.content is not None:
                current.content = handler_result.content
                modified = True
            if handler_result.is_error is not None:
                current.is_error = handler_result.is_error
                modified = True
        if not modified:
            return None
        return ToolResultEventResult(content=current.content, is_error=current.is_error)
