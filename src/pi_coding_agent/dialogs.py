"""Modal dialogs for the Textual app.

Phases T2/H of the extension-system UI foundation (see ARCHITECTURE.md).
Textual's push_screen/ModalScreen gives an overlay/dialog system natively
— the original TS ExtensionUIContext.select/confirm/input/editor hooks
each hand-built their own overlay handling; here they're one small
ModalScreen subclass apiece. ConfirmDialog is from Phase T2 (replacing
the REPL's blocking y/N input() so permission-mode confirmation works in
the Textual app without freezing its event loop); SelectDialog and
InputDialog back Phase H's ExtensionUIContext.select()/input(). An
"editor" dialog (a multi-line variant) isn't added yet — nothing calls
for one beyond what InputDialog already covers.
"""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

__all__ = ["ConfirmDialog", "InputDialog", "SelectDialog"]


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


class SelectDialog(ModalScreen[str | None]):
    """A single-choice picker: ``await app.push_screen_wait(SelectDialog(question, choices))``
    returns the chosen string, or ``None`` on Escape/no choices."""

    CSS = """
    SelectDialog {
        align: center middle;
    }
    SelectDialog > Vertical {
        width: auto;
        max-width: 80%;
        height: auto;
        border: solid $accent;
        padding: 1 2;
        background: $surface;
    }
    SelectDialog OptionList {
        width: auto;
        height: auto;
        max-height: 12;
        margin-top: 1;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(self, question: str, choices: list[str]) -> None:
        super().__init__()
        self._question = question
        self._choices = choices

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(self._question, markup=True)
            yield OptionList(*(Option(choice, id=choice) for choice in self._choices))

    def on_mount(self) -> None:
        self.query_one(OptionList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(str(event.option.prompt))

    def action_cancel(self) -> None:
        self.dismiss(None)


class InputDialog(ModalScreen[str | None]):
    """A single-line text prompt: ``await app.push_screen_wait(InputDialog(question))``
    returns the entered text, or ``None`` on Escape."""

    CSS = """
    InputDialog {
        align: center middle;
    }
    InputDialog > Vertical {
        width: 60;
        height: auto;
        border: solid $accent;
        padding: 1 2;
        background: $surface;
    }
    InputDialog Input {
        margin-top: 1;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(self, question: str, default: str = "") -> None:
        super().__init__()
        self._question = question
        self._default = default

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(self._question, markup=True)
            yield Input(value=self._default)

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    def action_cancel(self) -> None:
        self.dismiss(None)
