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

import asyncio
from typing import Any, ClassVar

from rich.console import Group, RenderableType
from rich.markup import escape
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import VerticalGroup, VerticalScroll
from textual.theme import Theme
from textual.widgets import OptionList, Static

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
        self._app.query_one("#app-footer", AppFooter).set_extension(content)

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


class AppFooter(Static):
    """The single structural footer region — always mounted, always
    visible, never toggled or torn down. Replaces what used to be two
    independent docked widgets (``#status-footer`` for live session
    status, ``#ext-footer`` for extension-supplied content) competing for
    the same screen real estate: one could be showing while the other
    was hidden, both could end up stacked with their own separate
    borders/backgrounds, and nothing about their layout expressed that
    they were really one region. Live status (``set_status``, called by
    ``PiApp._update_footer``) and extension content (``set_extension``,
    called by ``TextualExtensionUIContext.set_footer``) are two inputs
    into ONE rendered widget instead — never two boxes.

    ``height: auto`` plus a CSS ``border`` (not manually printed ``─``
    characters — Textual's own box model draws all four edges, sized to
    the actual content) is what makes the height intrinsic: 1 status
    line renders as a 3-row box (line + top/bottom border), 2 lines as
    4 rows, and so on, with no fixed ``height: N`` anywhere to fight
    with real content — the fixed ``height: 2`` this replaces is exactly
    what made a 3-line status get silently clipped."""

    DEFAULT_CSS = """
    AppFooter {
        width: 100%;
        height: auto;
        border: solid $accent;
        padding: 0 1;
        background: $panel;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("", markup=True, **kwargs)
        self._status_content: RenderableType = ""
        self._extension_content: RenderableType | None = None

    # set_status/set_extension each end with their own direct call to
    # self.update(...) — deliberately not both routed through one shared
    # "recompute and update" helper. That refactor was tried first and
    # silently broke real rendering in this Textual version:
    # Static.render_line() came back blank on every row (verified with a
    # minimal repro outside this app entirely — a bare Static subclass
    # with the same border CSS) even though .content/.visual still
    # showed the right text, purely from moving the same self.update(...)
    # call one indirection layer away from the setter that decided the
    # new content. Some interaction with Static's internal render/style
    # cache, not anything specific to Group or to this widget's own
    # state — a small amount of duplication between these two methods is
    # the trade for not depending on that fragile behavior.

    def set_status(self, content: RenderableType) -> None:
        """Live session status — provider/model, session id, permission
        mode, etc. (``PiApp._update_footer``, wired as
        ``InteractiveSession._on_status_change``). Re-renders in place:
        no widget is added, removed, or remounted."""
        self._status_content = content
        if self._extension_content is None:
            self.update(self._status_content)
        else:
            self.update(Group(self._status_content, self._extension_content))

    def set_extension(self, content: RenderableType | None) -> None:
        """Extension-supplied footer content (``ctx.ui.set_footer`` ->
        ``TextualExtensionUIContext.set_footer``). ``None`` removes it —
        the footer itself never disappears, only this piece of its
        content does, and the status content above it is unaffected."""
        self._extension_content = content
        if content is None:
            self.update(self._status_content)
        else:
            # rich.console.Group stacks arbitrary renderables (markup
            # strings included — Static's own str/RenderableType duality
            # doesn't extend to combining two of them, but Group does)
            # top to bottom with no manual line-counting needed here; the
            # footer's height: auto CSS picks up however tall that
            # combined render ends up being.
            self.update(Group(self._status_content, content))


class TransientFooterContent(OptionList):
    """Self-contained transient overlay for short-lived interactive
    picks that belong neither to the persistent AppFooter nor to the
    transcript — currently just the bare ``/model`` command's inline
    model list (``PiApp._select_model_transient``). Lives inside
    #bottom-bar, directly above AppFooter (see ``PiApp.compose``), so it
    visually sits between the transcript and the persistent footer —
    never replacing, hiding, or reusing AppFooter itself, and never
    stored as footer state (persistent footer state != transient
    interaction state).

    ``show_options``/``hide`` are the only two ways in or out, and
    ``hide`` is unconditionally called from ``_select_model_transient``'s
    ``finally`` — cleanup never depends on a caller remembering to clear
    it after a real pick, a cancel (Escape), or an early return (e.g. no
    models configured). ``display`` starts False and ``hide`` always
    restores that, so at rest this widget takes no layout space at all."""

    DEFAULT_CSS = """
    TransientFooterContent {
        width: 100%;
        height: auto;
        max-height: 10;
        border: solid $accent;
        display: none;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.display = False

    def show_options(self, labels: list[str]) -> None:
        self.clear_options()
        self.add_options(labels)
        self.highlighted = 0
        self.display = True
        self.focus()

    def hide(self) -> None:
        self.display = False
        self.clear_options()


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
    #bottom-bar {
        dock: bottom;
        width: 100%;
        height: auto;
    }
    #ext-widget {
        height: auto;
        padding: 0 1;
        display: none;
    }
    #suggestions {
        height: auto;
        max-height: 6;
        border: solid $accent;
        display: none;
    }
    #prompt-input {
        height: auto;
        min-height: 3;
        max-height: 12;
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
        # Set for the duration of a run_turn() call — the prompt input
        # stays focused/usable while this is True (Textual never disables
        # it), but a submission arriving during that window is routed to
        # _handle_mid_turn_submission instead of starting a second,
        # concurrent run_turn — /steer and /stop instead of a conflicting
        # second turn.
        self._turn_in_progress = False
        # Resolved by on_option_list_option_selected (a real pick) or
        # on_key (Escape) while the bare `/model` picker is open — see
        # _select_model_transient. None whenever no picker is open, so
        # both handlers can check "is anyone waiting on this" instead of
        # blindly resolving a stale/already-done future.
        self._transient_selection_future: asyncio.Future[str | None] | None = None

    def compose(self) -> ComposeResult:
        yield Static(id="ext-header")
        yield VerticalScroll(id="transcript")
        # A single widget docked to the bottom edge, everything below the
        # transcript stacked *inside* it via normal top-to-bottom flow
        # layout — not each docked to the edge individually. Textual's
        # dock positions every docked sibling independently flush against
        # its edge (see _arrange_dock_widgets in textual/_arrange.py: each
        # one gets `height - widget_height`, with no accumulated offset
        # between same-edge siblings) — so the *previous* layout, where
        # app-footer/ext-widget/suggestions/prompt-input were each
        # `dock: bottom` in their own right, had them all physically
        # overlapping at the screen's bottom edge, with only DOM paint
        # order (prompt-input last, so drawn on top) making that
        # invisible day to day. One dock + real flow layout inside it is
        # what actually guarantees no overlap, not a fragile paint-order
        # coincidence.
        with VerticalGroup(id="bottom-bar"):
            # Sits directly *above* AppFooter — see TransientFooterContent's
            # docstring — so the model picker (the only thing that shows
            # it today) appears between the transcript and the persistent
            # footer, never inside or instead of it.
            yield TransientFooterContent(id="transient-footer")
            yield AppFooter(id="app-footer")
            yield Static(id="ext-widget")
            yield OptionList(id="suggestions")
            yield PromptTextArea(id="prompt-input")

    def _set_slot(self, selector: str, content: RenderableType | str | None) -> None:
        """Phase H: shared show/hide logic for the ext-header/ext-widget
        slots — ``content=None`` hides the slot again. NOT used for the
        footer: AppFooter is a single structural region that's always
        visible and never display-toggled — see set_footer above, which
        calls AppFooter.set_extension instead."""
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
        status = f"{self._session._state_line()}\n{mode_label} [dim](shift+tab to cycle)[/dim]"
        self.query_one("#app-footer", AppFooter).set_status(status)

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
        transient = self.query_one("#transient-footer", TransientFooterContent)
        if transient.display and event.key == "escape":
            event.stop()
            event.prevent_default()
            if self._transient_selection_future is not None and not self._transient_selection_future.done():
                self._transient_selection_future.set_result(None)
            return

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

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Resolves _select_model_transient's pending future on a real
        pick from the transient model list — the #suggestions popup
        (also an OptionList) never actually holds focus (its up/down/Tab
        handling is done manually in on_key while focus stays on the
        prompt editor — see there), so this only ever fires for
        #transient-footer in practice; the id check is a defensive
        no-op guard against that changing later, not dead code."""
        if event.option_list.id != "transient-footer":
            return
        event.stop()
        if self._transient_selection_future is not None and not self._transient_selection_future.done():
            self._transient_selection_future.set_result(str(event.option.prompt))

    async def _select_model_transient(self) -> None:
        """Bare `/model` (no arguments) in the fullscreen app: show the
        model list as a transient overlay directly above the persistent
        footer (TransientFooterContent, never AppFooter itself) instead
        of a centered modal dialog. `finally` guarantees the panel is
        hidden and the prompt refocused no matter how this ends — a real
        pick, Escape, or an early return below — so the caller can never
        leave stale model-list content on screen. `/model <id>` (with an
        argument) is unaffected: that still goes through
        InteractiveSession._handle_command's direct-switch path, which
        never opens any picker."""
        providers = self._session._models.get_providers()
        choices = [model for provider in providers for model in provider.get_models()]
        if not choices:
            self._append_transcript_note("[dim]no providers configured[/dim]")
            return
        if len(choices) == 1:
            only = choices[0]
            self._append_transcript_note(
                f"[dim]only model available:[/dim] [bold]{escape(only.provider)}/{escape(only.id)}[/bold]"
            )
            return

        labels = [f"{m.provider}/{m.id}" for m in choices]
        panel = self.query_one("#transient-footer", TransientFooterContent)
        panel.show_options(labels)

        future: asyncio.Future[str | None] = asyncio.get_running_loop().create_future()
        self._transient_selection_future = future
        try:
            selected_label = await future
        finally:
            self._transient_selection_future = None
            panel.hide()
            self.query_one("#prompt-input", PromptTextArea).focus()

        if selected_label is None:
            self._append_transcript_note("[dim]cancelled[/dim]")
            return

        selected = next((m for m, label in zip(choices, labels, strict=True) if label == selected_label), None)
        if selected is None:
            return
        self._session._model = selected
        self._session._agent_session.set_model(selected)
        self._append_transcript_note(
            f"[green]switched to[/green] [bold]{escape(selected.provider)}/{escape(selected.id)}[/bold]"
        )
        self._update_footer()

    def on_prompt_text_area_submitted(self, event: PromptTextArea.Submitted) -> None:
        text = event.value.strip()
        event.text_area.text = ""
        if not text:
            return
        if self._turn_in_progress:
            self._handle_mid_turn_submission(text)
            return
        self._handle_submission(text)

    def _append_transcript_note(self, markup: str) -> None:
        transcript = self.query_one("#transcript", VerticalScroll)
        transcript.mount(Static(markup, markup=True))
        transcript.scroll_end(animate=False)

    def _handle_mid_turn_submission(self, text: str) -> None:
        """A message submitted while run_turn() is still in flight — never
        starts a second, concurrent turn (AgentSession.prompt() isn't
        reentrant). "/stop" requests the current turn abort at the next
        tool-call boundary; anything else (with or without an explicit
        "/steer" prefix — the input box doesn't need the prefix to know a
        turn is already running) is queued via queue_steer_message and
        delivered as an ordinary follow-up UserMessage at that same
        boundary, exactly like a normal next prompt would be, just without
        waiting for this one to finish first."""
        if text == "/stop":
            self._session._agent_session.request_stop()
            self._append_transcript_note("[dim]stop requested — finishing at the next turn boundary[/dim]")
            return
        steer_text = text[len("/steer") :].strip() if text.startswith("/steer") else text
        if not steer_text:
            self._append_transcript_note("[red]usage:[/red] /steer <text>")
            return
        self._session._agent_session.queue_steer_message(steer_text)
        self._append_transcript_note(f"[dim]queued for the next turn boundary:[/dim] {escape(steer_text)}")

    @work(exclusive=True)
    async def _handle_submission(self, text: str) -> None:
        """Runs as a Textual worker (not a plain message-handler coroutine)
        because it may end up calling push_screen_wait (via
        _confirm_tool_via_modal, several calls down through run_turn's tool
        loop) — push_screen_wait only works from inside a worker context.
        exclusive=True cancels a *previous, still-running* call to this
        same worker if a new one starts — harmless here since
        on_prompt_text_area_submitted only ever calls this when
        self._turn_in_progress is already False, i.e. no previous call
        could still be running.
        """
        if text.startswith("/"):
            # Bare `/model` (no id argument) is intercepted here rather
            # than left to InteractiveSession._handle_command: that
            # shared handler's own no-argument branch opens a centered
            # SelectDialog modal via ctx.ui.select() (fine for
            # extensions calling the same generic API, but not the
            # transient-footer-above-the-persistent-footer UX this app
            # wants specifically for its own model picker) — see
            # _select_model_transient. `/model <id>` is untouched: that
            # still flows through _handle_command's direct-switch path.
            if text.lower() == "/model":
                await self._select_model_transient()
                return
            should_continue = await self._session._handle_command(text)
            self._update_footer()
            if not should_continue:
                self.exit()
            return

        transcript = self.query_one("#transcript", VerticalScroll)
        transcript.mount(Static(f"[bold]> {text}[/bold]", markup=True))
        transcript.scroll_end(animate=False)

        self._turn_in_progress = True
        try:
            # No separate error print here: InteractiveSession._handle_event
            # already appends an "error: ..." line to the transcript via
            # the "error" event (see AgentSession._consume_stream, which
            # both emits that event *and* returns it as run_turn's
            # result) — printing it again here off result.stop_reason
            # duplicated every single error, back to back.
            await self._session.run_turn(text)
        finally:
            self._turn_in_progress = False
        self._update_footer()
