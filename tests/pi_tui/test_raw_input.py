"""Tests for pi_tui.raw_input's platform-independent edit loop, plus (on
Windows) the real key-reading primitives.

The _edit_loop tests exercise it directly with a fake key source, so they
never touch a real TTY / termios / the Windows console API. The
TestWindowsReadKey class below is different: it calls the *real*
_read_key(), but with its two low-level primitives (_read_console_char,
_console_char_ready) monkeypatched — this is the seam that lets the
escape-sequence assembly logic (does ESC + '[' + 'Z' really become
_BACKTAB?) get verified without a real console, closing the "zero test
coverage" gap on this platform's key detection.
"""

from __future__ import annotations

import sys
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


@pytest.mark.skipif(sys.platform != "win32", reason="exercises the Windows-only ReadConsoleW-based _read_key")
class TestWindowsReadKey:
    """These call the real pi_tui.raw_input._read_key — only its two
    low-level char-source primitives are faked — to verify the fix for
    Shift+Tab not being detected: the Windows console only reports it as
    the same ESC [ Z sequence POSIX terminals send once
    ENABLE_VIRTUAL_TERMINAL_INPUT is on, read char-by-char via ReadConsoleW.
    msvcrt.getwch()'s legacy two-byte extended-key scheme (the previous
    implementation) bypasses that translation and can't reliably tell
    Shift+Tab apart from plain Tab on modern console hosts.
    """

    def test_escape_then_bracket_then_z_is_backtab(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import pi_tui.raw_input as raw_input

        chars = iter(["\x1b", "[", "Z"])
        monkeypatch.setattr(raw_input, "_read_console_char", lambda: next(chars))
        monkeypatch.setattr(raw_input, "_console_char_ready", lambda _timeout: True)
        assert raw_input._read_key() == _BACKTAB

    def test_plain_char_is_returned_immediately_without_peeking(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import pi_tui.raw_input as raw_input

        monkeypatch.setattr(raw_input, "_read_console_char", lambda: "x")

        def _fail_ready(_timeout: float) -> bool:
            raise AssertionError("a plain char must not trigger the escape-sequence peek")

        monkeypatch.setattr(raw_input, "_console_char_ready", _fail_ready)
        assert raw_input._read_key() == "x"

    def test_lone_escape_with_nothing_following_is_returned_as_is(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import pi_tui.raw_input as raw_input

        monkeypatch.setattr(raw_input, "_read_console_char", lambda: "\x1b")
        monkeypatch.setattr(raw_input, "_console_char_ready", lambda _timeout: False)
        assert raw_input._read_key() == "\x1b"
