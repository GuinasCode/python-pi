"""ExtensionUIContext: interactive prompts an extension can show to the user.

Phase H of the extension-system UI foundation (see ARCHITECTURE.md).
Ported subset of the original TS ``ExtensionUIContext``: ``select``/
``confirm``/``input`` (interactive prompts), ``notify`` (a toast/status
message), ``set_header``/``set_footer``/``set_title``/``set_widget``
(custom UI regions — Textual widget slots reserved in ``tui_app.py``'s
``PiApp.compose()`` for exactly this), ``get_theme``/``set_theme`` (on top
of T5's theme registration), ``get_tools_expanded``/``set_tools_expanded``
(whether tool-call transcript entries show their full result preview —
consulted directly from InteractiveSession._handle_event, so unlike the
chrome methods above this one actually works in the classic REPL too),
and ``add_autocomplete_provider`` (extra suggestions merged into T4's
popup). The wider surface the original also exposes —
``pasteToEditor``/``setEditorText``/``getEditorText``/
``setEditorComponent`` and the interactive TUI's own extension management
screens — isn't ported yet; each needs its own concrete Textual mechanism
decided first, the same way this slice needed SelectDialog/InputDialog
(dialogs.py) and the widget slots before it could exist.

Two implementations: :class:`NoopExtensionUIContext` (the classic REPL,
and any future non-interactive mode like print/RPC — nothing to show a
prompt on, so prompts resolve to "no answer" rather than blocking or
raising) and the Textual-backed one in ``tui_app.py`` (kept there, not
here, since it needs ``PiApp``/``push_screen_wait`` and importing those
here would be circular — ``tui_app.py`` already imports this module).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from rich.console import RenderableType

__all__ = ["ExtensionUIContext", "NoopExtensionUIContext"]

AutocompleteProvider = Callable[[str], "list[str]"]


class ExtensionUIContext(Protocol):
    async def select(self, message: str, choices: list[str]) -> str | None:
        """Ask the user to pick one of `choices`. None if cancelled."""
        ...

    async def confirm(self, message: str) -> bool:
        """Ask the user a yes/no question."""
        ...

    async def input(self, message: str, default: str = "") -> str | None:
        """Ask the user for a line of text. None if cancelled."""
        ...

    def notify(self, message: str, *, severity: str = "information") -> None:
        """Show a transient status message. Fire-and-forget, not awaited."""
        ...

    def set_header(self, content: RenderableType | str | None) -> None:
        """Show (or, with None, hide) a header line above the transcript."""
        ...

    def set_footer(self, content: RenderableType | str | None) -> None:
        """Show (or, with None, hide) an extra footer line, alongside — not
        replacing — the built-in permission-mode/status footer."""
        ...

    def set_title(self, title: str | None) -> None:
        """Set (or, with None, reset) the app's title."""
        ...

    def set_widget(self, content: RenderableType | str | None) -> None:
        """Show (or, with None, hide) a custom widget region just above the
        prompt editor."""
        ...

    def get_theme(self) -> str | None:
        """The name of the currently active theme, if any."""
        ...

    def set_theme(self, name: str) -> None:
        """Switch to a theme by name (built-in or extension-registered via
        ``pi.register_theme``)."""
        ...

    def get_tools_expanded(self) -> bool:
        """Whether tool-call transcript entries currently show their full
        result preview (diff/output text) or just the summary line."""
        ...

    def set_tools_expanded(self, expanded: bool) -> None:
        """Set whether future tool-call transcript entries show their full
        result preview. Doesn't retroactively change entries already
        printed — this only affects what gets printed from this point on,
        the same way a terminal's scrollback isn't redrawn."""
        ...

    def add_autocomplete_provider(self, provider: AutocompleteProvider) -> None:
        """Register `provider(current_text) -> suggestions`, consulted
        alongside the built-in slash-command matches (T4) every time the
        prompt editor's text changes; its suggestions are merged into the
        same popup."""
        ...


class NoopExtensionUIContext:
    """No interactive surface to prompt on: select/input report "cancelled"
    (None) and confirm reports "declined" (False) rather than blocking
    forever or silently proceeding with a destructive default; notify,
    the widget-slot setters, theming, and autocomplete are simply dropped
    — the classic REPL has none of that chrome. ``tools_expanded`` is the
    one exception: it's real, in-memory state here too (not dropped),
    since InteractiveSession consults it directly regardless of which
    front-end is running.
    """

    def __init__(self) -> None:
        self._tools_expanded = True

    async def select(self, message: str, choices: list[str]) -> str | None:
        return None

    async def confirm(self, message: str) -> bool:
        return False

    async def input(self, message: str, default: str = "") -> str | None:
        return None

    def notify(self, message: str, *, severity: str = "information") -> None:
        pass

    def set_header(self, content: RenderableType | str | None) -> None:
        pass

    def set_footer(self, content: RenderableType | str | None) -> None:
        pass

    def set_title(self, title: str | None) -> None:
        pass

    def set_widget(self, content: RenderableType | str | None) -> None:
        pass

    def get_theme(self) -> str | None:
        return None

    def set_theme(self, name: str) -> None:
        pass

    def get_tools_expanded(self) -> bool:
        return self._tools_expanded

    def set_tools_expanded(self, expanded: bool) -> None:
        self._tools_expanded = expanded

    def add_autocomplete_provider(self, provider: AutocompleteProvider) -> None:
        pass
