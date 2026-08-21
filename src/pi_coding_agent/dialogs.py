"""Modal dialogs for the Textual app.

Phase T2 of the extension-system UI foundation (see ARCHITECTURE.md).
Textual's push_screen/ModalScreen gives an overlay/dialog system natively
— the original TS ExtensionUIContext.select/confirm/input/editor hooks
each hand-built their own overlay handling; here they're one small
ModalScreen subclass apiece. This module starts with ConfirmDialog (what
Phase T2 actually needed: replacing the REPL's blocking y/N input() so
permission-mode confirmation works in the Textual app without freezing
its event loop) — select/input/editor dialogs are added the same way
once something needs them (Phase H's ExtensionUIContext).
"""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

__all__ = ["ConfirmDialog"]


class ConfirmDialog(ModalScreen[bool | None]):
    """A yes/no confirmation dialog. ``await app.push_screen_wait(ConfirmDialog(question))``
    returns True/False for yes/no, and False for Escape (matching the
    REPL's "default to no" behavior on Ctrl+C/EOF)."""

    CSS = """
    ConfirmDialog {
        align: center middle;
    }
    ConfirmDialog > Vertical {
        width: auto;
        max-width: 80%;
        height: auto;
        border: solid $accent;
        padding: 1 2;
        background: $surface;
    }
    ConfirmDialog #hint {
        color: $text-muted;
        margin-top: 1;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("y", "confirm_yes", "Yes", show=False),
        Binding("n,escape", "confirm_no", "No", show=False),
    ]

    def __init__(self, question: str) -> None:
        super().__init__()
        self._question = question

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(self._question, markup=True)
            yield Static("(y)es / (n)o", id="hint", markup=True)

    def action_confirm_yes(self) -> None:
        self.dismiss(True)

    def action_confirm_no(self) -> None:
        self.dismiss(False)
