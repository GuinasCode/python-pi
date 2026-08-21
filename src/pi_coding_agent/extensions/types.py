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
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pi_agent_core.types import AgentTool
from pi_ai.models import MutableModels, Provider
from pi_coding_agent.extensions.events import ExtensionContext, ExtensionHandler

ExtensionFactory = Callable[["ExtensionAPI"], Any]

CommandHandler = Callable[[str, ExtensionContext], "Any | Awaitable[Any]"]


@dataclass
class RegisteredCommand:
    """A slash command an extension registered via ``pi.register_command``."""

    name: str
    handler: CommandHandler
    description: str | None = None


@dataclass
class ExtensionFlag:
    """A CLI flag declaration an extension registered via ``pi.register_flag``."""

    name: str
    type: Literal["boolean", "string"]
    default: bool | str | None = None
    description: str | None = None


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
    commands: list[RegisteredCommand] = field(default_factory=list)
    flags: dict[str, ExtensionFlag] = field(default_factory=dict)

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self.tools]


@dataclass
class LoadExtensionsResult:
    extensions: list[LoadedExtension] = field(default_factory=list)
    errors: list[ExtensionError] = field(default_factory=list)


class ExtensionAPI:
    """The ``pi`` object passed to an extension's entry point.

    Phase A/B/D/E/F surface: tool registration, event subscription, command
    registration, flag declaration/reading, and provider registration.
    Shortcut registration and rendering hooks are not implemented — see
    ARCHITECTURE.md's extension-system status note for why (they need UI
    infrastructure — a keybinding dispatcher, a widget/dialog system —
    this port doesn't have yet).
    """

    def __init__(
        self,
        flag_values: dict[str, bool | str] | None = None,
        models: MutableModels | None = None,
    ) -> None:
        self._tools: list[AgentTool] = []
        self._handlers: dict[str, list[ExtensionHandler]] = defaultdict(list)
        self._commands: list[RegisteredCommand] = []
        self._flags: dict[str, ExtensionFlag] = {}
        # Shared with the owning ExtensionRunner (same dict object, not a
        # copy) so pi.get_flag() sees values the runner sets *after* this
        # extension's factory already ran and returned — e.g. from a tool's
        # execute() called later — without the runner needing to hold onto
        # this ExtensionAPI instance itself.
        self._flag_values: dict[str, bool | str] = flag_values if flag_values is not None else {}
        # Also shared (not copied): the same MutableModels the AgentSession
        # actually streams against, so register_provider/unregister_provider
        # take effect immediately — during initial load and from any later
        # command/event handler alike — unlike the original's queue-then-
        # apply-once-bound dance, which our construction order doesn't need.
        self._models = models

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

    def register_command(
        self,
        name: str,
        handler: CommandHandler,
        description: str | None = None,
    ) -> None:
        """Register a slash command. ``handler(args_text, ctx)`` runs when
        the user types ``/name ...`` in interactive mode."""
        self._commands.append(RegisteredCommand(name=name, handler=handler, description=description))

    @property
    def commands(self) -> list[RegisteredCommand]:
        return list(self._commands)

    def register_flag(
        self,
        name: str,
        *,
        type: Literal["boolean", "string"] = "string",
        default: bool | str | None = None,
        description: str | None = None,
    ) -> None:
        """Declare a CLI flag. Not yet wired to argv parsing (see
        ExtensionRunner.set_flag_value for how a value actually reaches
        get_flag) — declaring it here makes it discoverable/documented and
        gives it a programmatic path to a value regardless."""
        self._flags[name] = ExtensionFlag(name=name, type=type, default=default, description=description)

    @property
    def flags(self) -> dict[str, ExtensionFlag]:
        return dict(self._flags)

    def get_flag(self, name: str) -> bool | str | None:
        """Current value of a registered flag, or its declared default if
        nothing set one yet."""
        if name in self._flag_values:
            return self._flag_values[name]
        declared = self._flags.get(name)
        return declared.default if declared else None

    def register_provider(self, provider: Provider[Any]) -> None:
        """Register or replace a model provider. A no-op if this session
        wasn't given a MutableModels to register into (e.g. a harness that
        doesn't pass one). Simplified vs. the original: only the full
        ``Provider`` object form is supported, not the partial
        ``ProviderConfig`` (baseUrl-only override, OAuth login flow) —
        this port has no equivalent config-merge/OAuth machinery to plug
        that into yet.
        """
        if self._models is not None:
            self._models.set_provider(provider)

    def unregister_provider(self, name: str) -> None:
        """Remove a previously registered provider. No-op if it isn't registered."""
        if self._models is not None:
            self._models.delete_provider(name)
