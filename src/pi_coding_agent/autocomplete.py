"""Slash-command autocomplete matching for the Textual prompt editor.

Phase T4 of the extension-system UI foundation (see ARCHITECTURE.md).
Textual's ``Suggester`` (inline ghost-text completion) only attaches to
``Input``, not ``TextArea`` — the prompt editor is a ``TextArea`` (Phase
T1, for multi-line editing), so this instead drives a small popup
(``OptionList``, wired in tui_app.py) from plain text matching. This
module owns just the matching logic — command names, in order, that
complete what's currently typed — so it's testable without a running
Textual app; tui_app.py owns showing/hiding the popup and applying a
selection back into the editor.

Built-in slash commands only get listed once a working extension-command
source is threaded through (mirrors register_command's coupling to
InteractiveSession) — this list is intentionally kept in sync with
InteractiveSession._handle_command's built-ins by hand, the same way the
REPL's own ``/help`` text already is.
"""

from __future__ import annotations

BUILTIN_COMMANDS = ["/help", "/exit", "/quit", "/model", "/clear", "/tools", "/session", "/extensions"]

__all__ = ["BUILTIN_COMMANDS", "command_suggestions"]


def command_suggestions(text: str, extension_command_names: list[str]) -> list[str]:
    """Command names (built-ins first, then extension-registered ones,
    each de-duplicated and in a stable order) that are completions of
    `text`, when `text` looks like a slash command still being typed —
    starts with "/" and has no space yet (a space means the user has
    moved on to typing arguments, nothing left to complete). Returns []
    otherwise, including when `text` already exactly matches a command
    (nothing left to offer)."""
    if not text.startswith("/") or " " in text:
        return []
    names = list(dict.fromkeys([*BUILTIN_COMMANDS, *(f"/{n}" for n in extension_command_names)]))
    return [n for n in names if n.startswith(text) and n != text]
