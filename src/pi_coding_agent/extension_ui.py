"""ExtensionUIContext: interactive prompts an extension can show to the user.

Phase H of the extension-system UI foundation (see ARCHITECTURE.md).
Ported subset of the original TS ``ExtensionUIContext``: ``select``/
``confirm``/``input`` (interactive prompts) and ``notify`` (a toast/status
message). The wider surface the original also exposes —
``setWidget``/``setFooter``/``setHeader``/``setTitle`` (custom UI regions),
``pasteToEditor``/``setEditorText``/``getEditorText``/``setEditorComponent``,
``addAutocompleteProvider``, theme getters/setters, and
``getToolsExpanded``/``setToolsExpanded`` — isn't ported yet; each needs its
own concrete Textual mechanism decided first, the same way this slice
needed SelectDialog/InputDialog (dialogs.py) before it could exist.

Two implementations: :class:`NoopExtensionUIContext` (the classic REPL,
and any future non-interactive mode like print/RPC — nothing to show a
prompt on, so prompts resolve to "no answer" rather than blocking or
raising) and the Textual-backed one in ``tui_app.py`` (kept there, not
here, since it needs ``PiApp``/``push_screen_wait`` and importing those
here would be circular — ``tui_app.py`` already imports this module).
"""

from __future__ import annotations

from typing import Protocol

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


class NoopExtensionUIContext:
    """No interactive surface to prompt on: select/input report "cancelled"
    (None) and confirm reports "declined" (False) rather than blocking
    forever or silently proceeding with a destructive default; notify is
    simply dropped."""

    async def select(self, message: str, choices: list[str]) -> str | None:
        return None

    async def confirm(self, message: str) -> bool:
        return False

    async def input(self, message: str, default: str = "") -> str | None:
        return None

    def notify(self, message: str, *, severity: str = "information") -> None:
        pass
