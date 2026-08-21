"""Textual-based interactive UI (``--ui-mode fullscreen`` / ``--alt``).

Phase T0 of the extension-system UI foundation (see ARCHITECTURE.md's
extension-system status note for the full phase plan, T0-T6 then G/H).
A real alt-screen Textual app that reuses InteractiveSession's event and
slash-command handling — via the OutputSink it already renders through —
instead of duplicating that logic for a second front-end. The classic
REPL (``interactive_mode.repl_loop``) is untouched and stays the default;
this is opt-in via ``--ui-mode fullscreen``/``--alt`` while it matures.

Scope for T0-T6 so far: the app shell (transcript + input + footer),
streaming turn rendering, slash commands (including extension-registered
ones), a real multi-line prompt editor, permission-mode confirmation via a
modal dialog (Phase T2), a keybinding dispatcher for extension-registered
shortcuts (Phase T3), a slash-command autocomplete popup (Phase T4), and
extension-registered color themes (Phase T5 — registered onto Textual's
own theme system; switchable today via Textual's built-in command palette,
Ctrl+P), and a live-updating footer (Phase T6 — InteractiveSession calls
back into the footer refresh after every streamed event, so status
transitions like "thinking..." -> "running: bash" -> "ready" show up
during a turn, not only right before/after it), and (H, in progress)
ExtensionUIContext's select()/confirm()/input()/notify() — TextualExtensionUIContext,
backed by SelectDialog/ConfirmDialog/InputDialog. NOT wired yet: the rest
of ExtensionUIContext (setWidget/setFooter/setHeader/setTitle, editor
manipulation, addAutocompleteProvider, theme get/set,
getToolsExpanded/setToolsExpanded) and the extension-management screens.
"""

from __future__ import annotations

from typing import Any, ClassVar

from rich.console import RenderableType
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import VerticalScroll
from textual.theme import Theme
from textual.widgets import OptionList, Static

from pi_ai import StopReason
from pi_coding_agent.autocomplete import command_suggestions
from pi_coding_agent.dialogs import ConfirmDialog, InputDialog, SelectDialog
from pi_coding_agent.extension_ui import AutocompleteProvider
from pi_coding_agent.interactive_mode import InteractiveSession, _fmt_args
from pi_coding_agent.permission_mode import permission_mode_label
from pi_coding_agent.prompt_editor import PromptTextArea

__all__ = ["PiApp"]


class _TranscriptSink:
    """OutputSink that appends into the transcript ScrollView instead of
    printing to stdout.

    InteractiveSession's event handler calls ``print(delta, end="")``
    repeatedly while a response streams in (one call per token) — mounting
    a fresh widget per call would be both wasteful and visually wrong (a
    wall of one-token widgets instead of a paragraph). Instead, calls with
    ``end=""`` accumulate into the *same* trailing Static via ``.update()``;
    a call with the default ``end="\\n"`` (or a full renderable, e.g. a
    Markdown block or diff) finalizes that line and starts a fresh one.
    """

    def __init__(self, transcript: VerticalScroll) -> None:
        self._transcript = transcript
        self._current: Static | None = None
        self._current_text = ""

    def print(self, markup: str = "", *, end: str = "\n") -> None:
        if self._current is None:
            self._current = Static("", markup=True)
            self._transcript.mount(self._current)
        self._current_text += markup
        self._current.update(self._current_text)
        self._transcript.scroll_end(animate=False)
        if end == "\n":
            self._current = None
            self._current_text = ""

    def print_renderable(self, renderable: RenderableType) -> None:
        self._current = None
        self._current_text = ""
        self._transcript.mount(Static(renderable))
        self._transcript.scroll_end(animate=False)


class TextualExtensionUIContext:
    """Phase H: the interactive-prompt half of ExtensionUIContext, backed
    by the T2/H dialog screens (dialogs.py) — select()/confirm()/input()
    each push a modal and await its result; notify() uses Textual's own
    toast mechanism directly, no dialog needed."""

    def __init__(self, app: PiApp) -> None:
        self._app = app
        self._tools_expanded = True

    async def select(self, message: str, choices: list[str]) -> str | None:
        return await self._app.push_screen_wait(SelectDialog(message, choices))

    async def confirm(self, message: str) -> bool:
        result = await self._app.push_screen_wait(ConfirmDialog(message))
        return bool(result)

    async def input(self, message: str, default: str = "") -> str | None:
        return await self._app.push_screen_wait(InputDialog(message, default))

    def notify(self, message: str, *, severity: str = "information") -> None:
        self._app.notify(message, severity=severity)  # type: ignore[arg-type]

    def set_header(self, content: RenderableType | str | None) -> None:
        self._app._set_slot("#ext-header", content)

    def set_footer(self, content: RenderableType | str | None) -> None:
        self._app._set_slot("#ext-footer", content)

    def set_title(self, title: str | None) -> None:
        self._app.title = title or ""

    def set_widget(self, content: RenderableType | str | None) -> None:
        self._app._set_slot("#ext-widget", content)

    def get_theme(self) -> str | None:
        return self._app.theme

    def set_theme(self, name: str) -> None:
        self._app.theme = name

    def get_tools_expanded(self) -> bool:
        return self._tools_expanded

    def set_tools_expanded(self, expanded: bool) -> None:
        self._tools_expanded = expanded

    def add_autocomplete_provider(self, provider: AutocompleteProvider) -> None:
        self._app._autocomplete_providers.append(provider)

    def get_editor_text(self) -> str:
        return self._app.query_one("#prompt-input", PromptTextArea).text

    def set_editor_text(self, text: str) -> None:
        self._app.query_one("#prompt-input", PromptTextArea).text = text

    def paste_to_editor(self, text: str) -> None:
        self._app.query_one("#prompt-input", PromptTextArea).insert(text)


class PiApp(App[None]):
    """Alt-screen Textual front-end for an InteractiveSession."""

    CSS = """
    #transcript {
        height: 1fr;
        padding: 0 1;
    }
    #ext-header {
        dock: top;
        height: auto;
        padding: 0 1;
        background: $panel;
        display: none;
    }
    #status-footer {
        dock: bottom;
        height: 2;
        padding: 0 1;
        background: $panel;
    }
    #ext-footer {
        dock: bottom;
        height: auto;
        padding: 0 1;
        background: $panel;
        display: none;
    }
    #ext-widget {
        dock: bottom;
        height: auto;
        padding: 0 1;
        display: none;
    }
    #suggestions {
        dock: bottom;
        height: auto;
        max-height: 6;
        border: solid $accent;
        display: none;
    }
    #prompt-input {
        dock: bottom;
        height: 3;
        border: solid $accent;
    }
    """

    # priority=True: Textual's default Screen already binds shift+tab to
    # focus_previous, which — being a closer DOM ancestor of the focused
    # Input than the App — would otherwise win the normal walk-up-from-
    # focused-widget resolution and this binding would never fire. Priority
    # bindings are checked before that walk, so ours wins instead.
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("shift+tab", "cycle_permission_mode", "Cycle permission mode", show=False, priority=True),
    ]

    def __init__(self, session: InteractiveSession) -> None:
        super().__init__()
        self._session = session
        self._autocomplete_providers: list[AutocompleteProvider] = []

    def compose(self) -> ComposeResult:
        yield Static(id="ext-header")
        yield VerticalScroll(id="transcript")
        yield Static(id="status-footer")
        yield Static(id="ext-footer")
        yield Static(id="ext-widget")
        yield OptionList(id="suggestions")
        yield PromptTextArea(id="prompt-input")

    def _set_slot(self, selector: str, content: RenderableType | str | None) -> None:
        """Phase H: shared show/hide logic for the ext-header/ext-footer/
        ext-widget slots — ``content=None`` hides the slot again."""
        widget = self.query_one(selector, Static)
        if content is None:
            widget.display = False
            return
        widget.update(content)
        widget.display = True

    def on_mount(self) -> None:
        transcript = self.query_one("#transcript", VerticalScroll)
        self._session._output = _TranscriptSink(transcript)
        self._session._confirm_tool_fn = self._confirm_tool_via_modal
        self._session._on_status_change = self._update_footer
        self._session._ui_context = TextualExtensionUIContext(self)
        for theme in self._session._agent_session.get_extension_themes():
            self.register_theme(
                Theme(
                    name=theme.name,
                    primary=theme.primary,
                    secondary=theme.secondary,
                    warning=theme.warning,
                    error=theme.error,
                    success=theme.success,
                    accent=theme.accent,
                    foreground=theme.foreground,
                    background=theme.background,
                    surface=theme.surface,
                    panel=theme.panel,
                    dark=theme.dark,
                )
            )
        self._update_footer()
        self.query_one("#prompt-input", PromptTextArea).focus()

    async def _confirm_tool_via_modal(self, tool_name: str, args: dict[str, Any]) -> bool:
        """InteractiveSession._confirm_tool_fn: replaces the REPL's
        blocking y/N input() with an async modal dialog — the blocking
        version would freeze this app's event loop entirely."""
        args_str = _fmt_args(args)
        question = f"Allow [bold]{tool_name}[/bold]({args_str})?"
        result = await self.push_screen_wait(ConfirmDialog(question))
        return bool(result)

    def _update_footer(self) -> None:
        mode_label = permission_mode_label(self._session._permission_mode)
        footer = self.query_one("#status-footer", Static)
        footer.update(f"{self._session._state_line()}\n{mode_label} [dim](shift+tab to cycle)[/dim]")

    def action_cycle_permission_mode(self) -> None:
        self._session._cycle_permission_mode()
        self._update_footer()

    def on_text_area_changed(self, event: PromptTextArea.Changed) -> None:
        """Phase T4/H: recompute the suggestion popup on every keystroke in
        the prompt editor — built-in slash-command matches, plus (Phase H)
        whatever extra suggestions any ctx.ui.add_autocomplete_provider()
        callback offers for the current text, merged in and
        de-duplicated."""
        if event.text_area.id != "prompt-input":
            return
        text = event.text_area.text
        extension_names = [c.name for c in self._session._agent_session.get_extension_commands()]
        matches = command_suggestions(text, extension_names)
        for provider in self._autocomplete_providers:
            for suggestion in provider(text):
                if suggestion not in matches:
                    matches.append(suggestion)
        suggestions = self.query_one("#suggestions", OptionList)
        if not matches:
            suggestions.display = False
            return
        suggestions.clear_options()
        suggestions.add_options(matches)
        suggestions.highlighted = 0
        suggestions.display = True

    def _hide_suggestions(self) -> None:
        self.query_one("#suggestions", OptionList).display = False

    def on_key(self, event: events.Key) -> None:
        """Phase T3/T4 keyboard dispatch, checked before the key reaches
        whatever widget is focused:

        1. Autocomplete popup controls (Tab accepts, Escape dismisses, Up/
           Down move the highlight) when the popup (Phase T4) is visible —
           these must win over the prompt editor's own handling of those
           keys (e.g. Tab would otherwise move focus).
        2. A matching extension-registered shortcut (Phase T3) — e.g. so a
           shortcut key doesn't also get typed as a literal character into
           the prompt editor.

        A no-op on every other keystroke, which is the common case (no
        popup open, no extension shortcuts registered).
        """
        suggestions = self.query_one("#suggestions", OptionList)
        if suggestions.display:
            if event.key == "tab" and suggestions.highlighted is not None:
                event.stop()
                event.prevent_default()
                option = suggestions.get_option_at_index(suggestions.highlighted)
                self._accept_suggestion(str(option.prompt))
                return
            if event.key == "escape":
                event.stop()
                event.prevent_default()
                self._hide_suggestions()
                return
            if event.key in ("up", "down") and suggestions.option_count:
                event.stop()
                event.prevent_default()
                current = suggestions.highlighted or 0
                step = -1 if event.key == "up" else 1
                suggestions.highlighted = (current + step) % suggestions.option_count
                return

        if not any(s.key == event.key for s in self._session._agent_session.get_extension_shortcuts()):
            return
        event.stop()
        event.prevent_default()
        self._dispatch_shortcut(event.key)

    def _accept_suggestion(self, command: str) -> None:
        input_widget = self.query_one("#prompt-input", PromptTextArea)
        input_widget.text = f"{command} "
        input_widget.move_cursor(input_widget.document.end)
        self._hide_suggestions()

    @work(exclusive=True)
    async def _dispatch_shortcut(self, key: str) -> None:
        """Runs as a worker for the same reason _handle_submission does:
        the handler may end up awaiting push_screen_wait several calls
        down (e.g. a future confirm/select dialog), which only works from
        inside a worker context."""
        await self._session._handle_extension_shortcut(key)
        self._update_footer()

    def on_prompt_text_area_submitted(self, event: PromptTextArea.Submitted) -> None:
        text = event.value.strip()
        event.text_area.text = ""
        if not text:
            return
        self._handle_submission(text)

    @work(exclusive=True)
    async def _handle_submission(self, text: str) -> None:
        """Runs as a Textual worker (not a plain message-handler coroutine)
        because it may end up calling push_screen_wait (via
        _confirm_tool_via_modal, several calls down through run_turn's tool
        loop) — push_screen_wait only works from inside a worker context.
        """
        if text.startswith("/"):
            should_continue = await self._session._handle_command(text)
            self._update_footer()
            if not should_continue:
                self.exit()
            return

        transcript = self.query_one("#transcript", VerticalScroll)
        transcript.mount(Static(f"[bold]> {text}[/bold]", markup=True))
        transcript.scroll_end(animate=False)

        result = await self._session.run_turn(text)
        self._update_footer()
        if result is not None and result.stop_reason == StopReason.ERROR:
            error_msg = result.error_message or "Unknown error"
            transcript.mount(Static(f"[red]error:[/red] {error_msg}", markup=True))
            transcript.scroll_end(animate=False)
