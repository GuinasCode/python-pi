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

from rich.console import RenderableType

from pi_agent_core.types import AgentTool
from pi_ai.models import MutableModels, Provider
from pi_coding_agent.extensions.events import ExtensionContext, ExtensionHandler

ExtensionFactory = Callable[["ExtensionAPI"], Any]

CommandHandler = Callable[[str, ExtensionContext], "Any | Awaitable[Any]"]
ShortcutHandler = Callable[[ExtensionContext], "Any | Awaitable[Any]"]

# Phase G rendering hooks are deliberately synchronous-only (unlike
# command/shortcut handlers, which may be async): they run inline while a
# message is already being printed/streamed, not from a point where
# awaiting one makes sense.
MarkdownTransformer = Callable[[str, ExtensionContext], str]
MessageRenderer = Callable[[str, ExtensionContext], "RenderableType | str | None"]
EntryRenderer = Callable[[str, dict[str, Any], ExtensionContext], "RenderableType | str | None"]


@dataclass
class RegisteredCommand:
    """A slash command an extension registered via ``pi.register_command``."""

    name: str
    handler: CommandHandler
    description: str | None = None


@dataclass
class RegisteredShortcut:
    """A keybinding an extension registered via ``pi.register_shortcut``.

    ``key`` uses Textual's key-name syntax (e.g. ``"ctrl+g"``, ``"f2"``) —
    the same strings ``textual.binding.Binding`` accepts — since the T3
    dispatcher that fires this handler is a thin layer over Textual's own
    ``on_key``, not a separate parser.
    """

    key: str
    handler: ShortcutHandler
    description: str | None = None


@dataclass
class RegisteredTheme:
    """A color theme an extension registered via ``pi.register_theme``.

    Field names/semantics mirror ``textual.theme.Theme`` directly (the T5
    dispatcher in tui_app.py constructs one of those from this dataclass
    via ``dataclasses.asdict`` minus ``name``) rather than inventing a
    parallel color-scheme shape — only ``primary`` is required, matching
    Theme's own construction contract.
    """

    name: str
    primary: str
    secondary: str | None = None
    warning: str | None = None
    error: str | None = None
    success: str | None = None
    accent: str | None = None
    foreground: str | None = None
    background: str | None = None
    surface: str | None = None
    panel: str | None = None
    dark: bool = True


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
    shortcuts: list[RegisteredShortcut] = field(default_factory=list)
    themes: list[RegisteredTheme] = field(default_factory=list)
    markdown_transformers: list[MarkdownTransformer] = field(default_factory=list)
    message_renderers: dict[str, MessageRenderer] = field(default_factory=dict)
    entry_renderers: dict[str, EntryRenderer] = field(default_factory=dict)

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self.tools]


@dataclass
class LoadExtensionsResult:
    extensions: list[LoadedExtension] = field(default_factory=list)
    errors: list[ExtensionError] = field(default_factory=list)


class ExtensionAPI:
    """The ``pi`` object passed to an extension's entry point.

    Phase A/B/D/E/F/T3/G surface: tool registration, event subscription,
    command registration, shortcut/theme registration, rendering hooks
    (markdown transformers, message/entry renderers), flag
    declaration/reading, and provider registration. Dialogs, widgets, and
    autocomplete *providers* (the extension-facing API — the popup
    mechanism itself is done, Phase T4) are not implemented yet — see
    ARCHITECTURE.md's extension-system status note (Phase H).
    """

    def __init__(
        self,
        flag_values: dict[str, bool | str] | None = None,
        models: MutableModels | None = None,
    ) -> None:
        self._tools: list[AgentTool] = []
        self._handlers: dict[str, list[ExtensionHandler]] = defaultdict(list)
        self._commands: list[RegisteredCommand] = []
        self._shortcuts: list[RegisteredShortcut] = []
        self._themes: list[RegisteredTheme] = []
        self._markdown_transformers: list[MarkdownTransformer] = []
        self._message_renderers: dict[str, MessageRenderer] = {}
        self._entry_renderers: dict[str, EntryRenderer] = {}
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

    def register_shortcut(
        self,
        key: str,
        handler: ShortcutHandler,
        description: str | None = None,
    ) -> None:
        """Register a global keybinding. ``handler(ctx)`` runs when the
        user presses ``key`` (a Textual key-name, e.g. ``"ctrl+g"``) in the
        Textual app (``--ui-mode fullscreen``) — a no-op in the classic
        REPL, which has no keybinding dispatcher to fire it from."""
        self._shortcuts.append(RegisteredShortcut(key=key, handler=handler, description=description))

    @property
    def shortcuts(self) -> list[RegisteredShortcut]:
        return list(self._shortcuts)

    def register_theme(
        self,
        name: str,
        primary: str,
        *,
        secondary: str | None = None,
        warning: str | None = None,
        error: str | None = None,
        success: str | None = None,
        accent: str | None = None,
        foreground: str | None = None,
        background: str | None = None,
        surface: str | None = None,
        panel: str | None = None,
        dark: bool = True,
    ) -> None:
        """Register a color theme, selectable as ``app.theme = name`` in the
        Textual app (``--ui-mode fullscreen``) — a no-op in the classic
        REPL, which has no theming system to register into. Colors are hex
        strings (e.g. ``"#1e1e2e"``), matching ``textual.theme.Theme``."""
        self._themes.append(
            RegisteredTheme(
                name=name,
                primary=primary,
                secondary=secondary,
                warning=warning,
                error=error,
                success=success,
                accent=accent,
                foreground=foreground,
                background=background,
                surface=surface,
                panel=panel,
                dark=dark,
            )
        )

    @property
    def themes(self) -> list[RegisteredTheme]:
        return list(self._themes)

    def register_markdown_transformer(self, transformer: MarkdownTransformer) -> None:
        """Register a text transform run on assistant text right before it's
        rendered as Markdown. ``transformer(text, ctx) -> text``. Chained in
        registration order across every extension — each sees the previous
        transformer's output, not the original text."""
        self._markdown_transformers.append(transformer)

    @property
    def markdown_transformers(self) -> list[MarkdownTransformer]:
        return list(self._markdown_transformers)

    def register_message_renderer(self, role: str, renderer: MessageRenderer) -> None:
        """Register a custom renderer for a message role (``"assistant"`` is
        the only role actually flushed through one today — see
        InteractiveSession._flush_text_block). ``renderer(text, ctx)``
        returning a Rich renderable or string replaces the default Markdown
        rendering entirely; returning ``None`` falls through to it. A later
        registration for the same role replaces an earlier one."""
        self._message_renderers[role] = renderer

    @property
    def message_renderers(self) -> dict[str, MessageRenderer]:
        return dict(self._message_renderers)

    def register_entry_renderer(self, tool_name: str, renderer: EntryRenderer) -> None:
        """Register a custom renderer for a specific tool's transcript
        entries. ``renderer(phase, event, ctx)`` — ``phase`` is
        ``"start"``/``"end"``, ``event`` is a plain dict (``args`` for
        start; ``result_text``/``is_error`` for end) — returning a Rich
        renderable or string replaces the default line for that phase;
        returning ``None`` falls through to it. A later registration for
        the same tool name replaces an earlier one."""
        self._entry_renderers[tool_name] = renderer

    @property
    def entry_renderers(self) -> dict[str, EntryRenderer]:
        return dict(self._entry_renderers)

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
