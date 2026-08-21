"""ExtensionUIContext: interactive prompts an extension can show to the user.

Phase H of the extension-system UI foundation (see ARCHITECTURE.md).
Ported subset of the original TS ``ExtensionUIContext``: ``select``/
``confirm``/``input`` (interactive prompts), ``notify`` (a toast/status
message), and ``set_header``/``set_footer``/``set_title``/``set_widget``
(custom UI regions — Textual widget slots reserved in ``tui_app.py``'s
``PiApp.compose()`` for exactly this). The wider surface the original also
exposes — ``pasteToEditor``/``setEditorText``/``getEditorText``/
``setEditorComponent``, ``addAutocompleteProvider``, theme getters/setters,
and ``getToolsExpanded``/``setToolsExpanded`` — isn't ported yet; each
needs its own concrete Textual mechanism decided first, the same way this
slice needed SelectDialog/InputDialog (dialogs.py) and the widget slots
before it could exist.

Two implementations: :class:`NoopExtensionUIContext` (the classic REPL,
and any future non-interactive mode like print/RPC — nothing to show a
prompt on, so prompts resolve to "no answer" rather than blocking or
raising) and the Textual-backed one in ``tui_app.py`` (kept there, not
here, since it needs ``PiApp``/``push_screen_wait`` and importing those
here would be circular — ``tui_app.py`` already imports this module).
"""

from __future__ import annotations

from typing import Protocol

from rich.console import RenderableType

__all__ = ["ExtensionUIContext", "NoopExtensionUIContext"]


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


class NoopExtensionUIContext:
    """No interactive surface to prompt on: select/input report "cancelled"
    (None) and confirm reports "declined" (False) rather than blocking
    forever or silently proceeding with a destructive default; notify and
    the widget-slot setters are simply dropped — the classic REPL has no
    header/footer/title/widget chrome to put any of them into."""

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
