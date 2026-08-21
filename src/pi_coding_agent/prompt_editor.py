"""Multi-line prompt editor widget for the Textual app.

Phase T1 of the extension-system UI foundation (see ARCHITECTURE.md).
Wraps Textual's ``TextArea`` — which already gives cursor movement,
selection, and multi-line editing for free — instead of hand-rolling an
editor the way the original TS ``editor.ts`` (~80KB) does. This is most
of what that file exists for, gotten essentially for free by adopting
Textual.

The one behavior TextArea doesn't have out of the box: submitting on
Enter. TextArea treats Enter as "insert a newline" (reasonable for a
generic text editor, wrong for a chat-style prompt box), so this
subclass intercepts it. Shift+Enter is deliberately *not* used for the
"insert newline instead" binding — many terminals send the identical
byte sequence for plain Enter and Shift+Enter, so it can't be relied on
portably. Ctrl+J (a distinct control byte) is used instead.
"""

from __future__ import annotations

from textual import events
from textual.message import Message
from textual.widgets import TextArea

__all__ = ["PromptTextArea"]


class PromptTextArea(TextArea):
    """Enter submits (posts ``Submitted``); Ctrl+J inserts a literal newline."""

    class Submitted(Message):
        """Posted when the user presses Enter. ``value`` is the text at
        that moment — the widget's own content is left untouched; the
        handler is responsible for clearing it once it's done with the
        value (mirrors Textual's own Input.Submitted contract)."""

        def __init__(self, text_area: PromptTextArea, value: str) -> None:
            self.text_area = text_area
            self.value = value
            super().__init__()

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self, self.text))
            return
        if event.key == "ctrl+j":
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        await super()._on_key(event)
