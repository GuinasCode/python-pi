"""Tests for pi_tui.raw_input's platform-independent edit loop, plus (on
Windows) the real key-reading primitives.

The _edit_loop tests exercise it directly with a fake key source, so they
never touch a real TTY / termios / the Windows console API. The
TestWindowsReadKey class below is different: it calls the *real*
_read_key(), but with its two low-level primitives (_read_console_char,
_console_char_ready) monkeypatched — this is the seam that lets the
escape-sequence assembly logic (does ESC + '[' + 'Z' really become
_BACKTAB? does a 4-byte sequence like Delete still assemble correctly?)
get verified without a real console, closing the "zero test coverage"
gap on this platform's key detection.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

import pytest

from pi_tui.raw_input import _BACKTAB, _DELETE, _END, _HOME, _LEFT, _RIGHT, _edit_loop


def _keys(*sequence: str) -> Callable[[], str]:
    it = iter(sequence)
    return lambda: next(it)


def _recorder() -> tuple[Callable[[str, int], None], list[tuple[str, int]]]:
    """A fake on_render that just remembers what it was called with, in
    order — enough to check _edit_loop calls it exactly when the buffer
    (or cursor) actually changes, without needing a real terminal."""
    calls: list[tuple[str, int]] = []

    def _record(text: str, cursor: int) -> None:
        calls.append((text, cursor))

    return _record, calls


class TestEditLoop:
    def test_returns_typed_text_on_enter(self) -> None:
        render, _ = _recorder()
        result = _edit_loop(_keys("h", "i", "\n"), render, on_cycle=lambda: None)
        assert result == "hi"

    def test_render_is_called_after_every_character_with_cursor_at_end(self) -> None:
        render, calls = _recorder()
        _edit_loop(_keys("h", "i", "\n"), render, on_cycle=lambda: None)
        assert calls == [("h", 1), ("hi", 2)]

    def test_backspace_removes_char_before_cursor(self) -> None:
        render, calls = _recorder()
        result = _edit_loop(_keys("h", "i", "\x7f", "\n"), render, on_cycle=lambda: None)
        assert result == "h"
        assert calls == [("h", 1), ("hi", 2), ("h", 1)]

    def test_backspace_on_empty_buffer_is_noop(self) -> None:
        render, calls = _recorder()
        result = _edit_loop(_keys("\x7f", "h", "\n"), render, on_cycle=lambda: None)
        assert result == "h"
        # the empty-buffer backspace above must not have triggered a render
        assert calls == [("h", 1)]

    def test_ctrl_c_raises_keyboard_interrupt(self) -> None:
        render, _ = _recorder()
        with pytest.raises(KeyboardInterrupt):
            _edit_loop(_keys("h", "\x03"), render, on_cycle=lambda: None)

    def test_ctrl_d_on_empty_buffer_raises_eof(self) -> None:
        render, _ = _recorder()
        with pytest.raises(EOFError):
            _edit_loop(_keys("\x04"), render, on_cycle=lambda: None)

    def test_ctrl_d_on_nonempty_buffer_is_noop(self) -> None:
        render, _ = _recorder()
        result = _edit_loop(_keys("h", "\x04", "i", "\n"), render, on_cycle=lambda: None)
        assert result == "hi"

    def test_backtab_invokes_on_cycle_without_submitting(self) -> None:
        render, _ = _recorder()
        calls = []
        result = _edit_loop(_keys("h", _BACKTAB, "i", "\n"), render, on_cycle=lambda: calls.append(1))
        assert result == "hi"
        assert calls == [1]

    def test_backtab_also_triggers_a_render(self) -> None:
        render, calls = _recorder()
        _edit_loop(_keys("h", _BACKTAB, "\n"), render, on_cycle=lambda: None)
        # once for "h" being typed, once again for the backtab itself (the
        # caller's on_cycle mutated state — e.g. permission mode — that the
        # next render needs to reflect)
        assert calls == [("h", 1), ("h", 1)]

    def test_backtab_can_be_pressed_multiple_times(self) -> None:
        render, _ = _recorder()
        calls = []
        result = _edit_loop(_keys(_BACKTAB, _BACKTAB, _BACKTAB, "x", "\n"), render, on_cycle=lambda: calls.append(1))
        assert result == "x"
        assert len(calls) == 3

    def test_plain_tab_and_unhandled_escape_are_ignored(self) -> None:
        render, calls = _recorder()
        result = _edit_loop(_keys("\t", "\x1b", "\x1bOP", "x", "\n"), render, on_cycle=lambda: None)
        assert result == "x"
        assert calls == [("x", 1)]

    def test_empty_key_from_discarded_extended_key_is_ignored(self) -> None:
        render, calls = _recorder()
        result = _edit_loop(_keys("", "x", "\n"), render, on_cycle=lambda: None)
        assert result == "x"
        assert calls == [("x", 1)]


class TestEditLoopCursorNavigation:
    """Left/Right/Home/End move the cursor without touching the buffer;
    typing and Backspace/Delete then act at the cursor's position, not
    always the end — this is the fix for not being able to navigate back
    into already-typed text to edit it."""

    def test_left_then_typing_inserts_before_the_last_char(self) -> None:
        render, _ = _recorder()
        result = _edit_loop(_keys("h", "i", _LEFT, "X", "\n"), render, on_cycle=lambda: None)
        assert result == "hXi"

    def test_left_moves_cursor_left_and_reports_it(self) -> None:
        render, calls = _recorder()
        _edit_loop(_keys("h", "i", _LEFT, "\n"), render, on_cycle=lambda: None)
        assert calls[-1] == ("hi", 1)

    def test_left_at_start_of_buffer_is_noop(self) -> None:
        render, calls = _recorder()
        result = _edit_loop(_keys("h", _LEFT, _LEFT, _LEFT, "\n"), render, on_cycle=lambda: None)
        assert result == "h"
        # one render for typing "h", one for the single leftward move that
        # actually did something (cursor 1 -> 0); the next two presses at
        # cursor 0 are no-ops and must not re-render
        assert calls == [("h", 1), ("h", 0)]

    def test_right_moves_cursor_right_after_a_left(self) -> None:
        render, calls = _recorder()
        _edit_loop(_keys("h", "i", _LEFT, _RIGHT, "\n"), render, on_cycle=lambda: None)
        assert calls[-1] == ("hi", 2)

    def test_right_at_end_of_buffer_is_noop(self) -> None:
        render, calls = _recorder()
        result = _edit_loop(_keys("h", _RIGHT, _RIGHT, "\n"), render, on_cycle=lambda: None)
        assert result == "h"
        assert calls == [("h", 1)]

    def test_home_moves_cursor_to_start(self) -> None:
        render, calls = _recorder()
        result = _edit_loop(_keys("h", "i", _HOME, "X", "\n"), render, on_cycle=lambda: None)
        assert result == "Xhi"
        assert calls[-2] == ("hi", 0)  # the Home move itself

    def test_home_at_start_is_noop(self) -> None:
        render, calls = _recorder()
        _edit_loop(_keys(_HOME, "\n"), render, on_cycle=lambda: None)
        assert calls == []

    def test_end_moves_cursor_to_end(self) -> None:
        render, _ = _recorder()
        result = _edit_loop(_keys("h", "i", _HOME, _END, "X", "\n"), render, on_cycle=lambda: None)
        assert result == "hiX"

    def test_end_at_end_is_noop(self) -> None:
        render, calls = _recorder()
        _edit_loop(_keys("h", _END, "\n"), render, on_cycle=lambda: None)
        assert calls == [("h", 1)]

    def test_delete_removes_char_at_cursor(self) -> None:
        render, _ = _recorder()
        result = _edit_loop(_keys("h", "i", _HOME, _DELETE, "\n"), render, on_cycle=lambda: None)
        assert result == "i"

    def test_delete_at_end_of_buffer_is_noop(self) -> None:
        render, calls = _recorder()
        result = _edit_loop(_keys("h", _DELETE, "\n"), render, on_cycle=lambda: None)
        assert result == "h"
        assert calls == [("h", 1)]

    def test_backspace_at_start_of_buffer_is_noop_even_with_text_after_cursor(self) -> None:
        render, calls = _recorder()
        result = _edit_loop(_keys("h", "i", _HOME, "\x7f", "\n"), render, on_cycle=lambda: None)
        assert result == "hi"
        assert calls == [("h", 1), ("hi", 2), ("hi", 0)]


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

    def test_four_byte_delete_sequence_assembles_correctly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import pi_tui.raw_input as raw_input

        chars = iter(["\x1b", "[", "3", "~"])
        monkeypatch.setattr(raw_input, "_read_console_char", lambda: next(chars))
        monkeypatch.setattr(raw_input, "_console_char_ready", lambda _timeout: True)
        assert raw_input._read_key() == _DELETE

    def test_three_byte_sequence_stops_without_peeking_a_fourth_time(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A 3-byte sequence (e.g. Left, ESC [ D) must not wait out the
        timeout a 3rd time once its final byte (D) has arrived — that
        would add a needless ~50ms of input lag to every arrow keypress."""
        import pi_tui.raw_input as raw_input

        chars = iter(["\x1b", "[", "D"])
        ready_calls = 0

        def _ready(_timeout: float) -> bool:
            nonlocal ready_calls
            ready_calls += 1
            if ready_calls > 2:
                raise AssertionError("stopped early after the final byte, should not peek again")
            return True

        monkeypatch.setattr(raw_input, "_read_console_char", lambda: next(chars))
        monkeypatch.setattr(raw_input, "_console_char_ready", _ready)
        assert raw_input._read_key() == _LEFT
        assert ready_calls == 2
