"""Tests for pi_tui.raw_input's platform-independent edit loop.

Exercises _edit_loop directly with a fake key source, so these tests never
touch a real TTY / termios / msvcrt.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from pi_tui.raw_input import _BACKTAB, _edit_loop


def _keys(*sequence: str) -> Callable[[], str]:
    it = iter(sequence)
    return lambda: next(it)


class TestEditLoop:
    def test_returns_typed_text_on_enter(self) -> None:
        result = _edit_loop(_keys("h", "i", "\n"), on_cycle=lambda: None)
        assert result == "hi"

    def test_backspace_removes_last_char(self) -> None:
        result = _edit_loop(_keys("h", "i", "\x7f", "\n"), on_cycle=lambda: None)
        assert result == "h"

    def test_backspace_on_empty_buffer_is_noop(self) -> None:
        result = _edit_loop(_keys("\x7f", "h", "\n"), on_cycle=lambda: None)
        assert result == "h"

    def test_ctrl_c_raises_keyboard_interrupt(self) -> None:
        with pytest.raises(KeyboardInterrupt):
            _edit_loop(_keys("h", "\x03"), on_cycle=lambda: None)

    def test_ctrl_d_on_empty_buffer_raises_eof(self) -> None:
        with pytest.raises(EOFError):
            _edit_loop(_keys("\x04"), on_cycle=lambda: None)

    def test_ctrl_d_on_nonempty_buffer_is_noop(self) -> None:
        result = _edit_loop(_keys("h", "\x04", "i", "\n"), on_cycle=lambda: None)
        assert result == "hi"

    def test_backtab_invokes_on_cycle_without_submitting(self) -> None:
        calls = []
        result = _edit_loop(_keys("h", _BACKTAB, "i", "\n"), on_cycle=lambda: calls.append(1))
        assert result == "hi"
        assert calls == [1]

    def test_backtab_can_be_pressed_multiple_times(self) -> None:
        calls = []
        result = _edit_loop(_keys(_BACKTAB, _BACKTAB, _BACKTAB, "x", "\n"), on_cycle=lambda: calls.append(1))
        assert result == "x"
        assert len(calls) == 3

    def test_plain_tab_and_unhandled_escape_are_ignored(self) -> None:
        result = _edit_loop(_keys("\t", "\x1b", "\x1bOP", "x", "\n"), on_cycle=lambda: None)
        assert result == "x"

    def test_empty_key_from_discarded_extended_key_is_ignored(self) -> None:
        result = _edit_loop(_keys("", "x", "\n"), on_cycle=lambda: None)
        assert result == "x"
