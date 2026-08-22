"""Tests for pi_tui.raw_input's platform-independent edit loop, plus (on
Windows) the real key-reading primitives.

The _edit_loop tests exercise it directly with a fake key source, so they
never touch a real TTY / termios / the Windows console API. The
TestWindowsReadKey class below is different: it calls the *real*
_read_key(), but with its two low-level primitives (_read_console_char,
_console_char_ready) monkeypatched — this is the seam that lets the
escape-sequence assembly logic (does ESC + '[' + 'Z' really become
_BACKTAB? do the longer modifyOtherKeys sequences still assemble
correctly?) get verified without a real console, closing the "zero test
coverage" gap on this platform's key detection.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

import pytest

from pi_tui.raw_input import (
    _BACKTAB,
    _CTRL_BACKSPACE,
    _CTRL_LEFT,
    _CTRL_RIGHT,
    _DELETE,
    _END,
    _HOME,
    _LEFT,
    _RIGHT,
    _SHIFT_BACKSPACE,
    _SHIFT_LEFT,
    _SHIFT_RIGHT,
    _edit_loop,
)

Render = tuple[str, int, tuple[int, int] | None]


def _keys(*sequence: str) -> Callable[[], str]:
    it = iter(sequence)
    return lambda: next(it)


def _recorder() -> tuple[Callable[[str, int, tuple[int, int] | None], None], list[Render]]:
    """A fake on_render that just remembers what it was called with, in
    order — enough to check _edit_loop calls it exactly when the buffer,
    cursor, or selection actually changes, without needing a real
    terminal."""
    calls: list[Render] = []

    def _record(text: str, cursor: int, selection: tuple[int, int] | None) -> None:
        calls.append((text, cursor, selection))

    return _record, calls


class TestEditLoop:
    def test_returns_typed_text_on_enter(self) -> None:
        render, _ = _recorder()
        result = _edit_loop(_keys("h", "i", "\n"), render, on_cycle=lambda: None)
        assert result == "hi"

    def test_render_is_called_after_every_character_with_cursor_at_end(self) -> None:
        render, calls = _recorder()
        _edit_loop(_keys("h", "i", "\n"), render, on_cycle=lambda: None)
        assert calls == [("h", 1, None), ("hi", 2, None)]

    def test_backspace_removes_char_before_cursor(self) -> None:
        render, calls = _recorder()
        result = _edit_loop(_keys("h", "i", "\x7f", "\n"), render, on_cycle=lambda: None)
        assert result == "h"
        assert calls == [("h", 1, None), ("hi", 2, None), ("h", 1, None)]

    def test_backspace_on_empty_buffer_is_noop(self) -> None:
        render, calls = _recorder()
        result = _edit_loop(_keys("\x7f", "h", "\n"), render, on_cycle=lambda: None)
        assert result == "h"
        # the empty-buffer backspace above must not have triggered a render
        assert calls == [("h", 1, None)]

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
        assert calls == [("h", 1, None), ("h", 1, None)]

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
        assert calls == [("x", 1, None)]

    def test_empty_key_from_discarded_extended_key_is_ignored(self) -> None:
        render, calls = _recorder()
        result = _edit_loop(_keys("", "x", "\n"), render, on_cycle=lambda: None)
        assert result == "x"
        assert calls == [("x", 1, None)]


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
        assert calls[-1] == ("hi", 1, None)

    def test_left_at_start_of_buffer_is_noop(self) -> None:
        render, calls = _recorder()
        result = _edit_loop(_keys("h", _LEFT, _LEFT, _LEFT, "\n"), render, on_cycle=lambda: None)
        assert result == "h"
        # one render for typing "h", one for the single leftward move that
        # actually did something (cursor 1 -> 0); the next two presses at
        # cursor 0 are no-ops and must not re-render
        assert calls == [("h", 1, None), ("h", 0, None)]

    def test_right_moves_cursor_right_after_a_left(self) -> None:
        render, calls = _recorder()
        _edit_loop(_keys("h", "i", _LEFT, _RIGHT, "\n"), render, on_cycle=lambda: None)
        assert calls[-1] == ("hi", 2, None)

    def test_right_at_end_of_buffer_is_noop(self) -> None:
        render, calls = _recorder()
        result = _edit_loop(_keys("h", _RIGHT, _RIGHT, "\n"), render, on_cycle=lambda: None)
        assert result == "h"
        assert calls == [("h", 1, None)]

    def test_home_moves_cursor_to_start(self) -> None:
        render, calls = _recorder()
        result = _edit_loop(_keys("h", "i", _HOME, "X", "\n"), render, on_cycle=lambda: None)
        assert result == "Xhi"
        assert calls[-2] == ("hi", 0, None)  # the Home move itself

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
        assert calls == [("h", 1, None)]

    def test_delete_removes_char_at_cursor(self) -> None:
        render, _ = _recorder()
        result = _edit_loop(_keys("h", "i", _HOME, _DELETE, "\n"), render, on_cycle=lambda: None)
        assert result == "i"

    def test_delete_at_end_of_buffer_is_noop(self) -> None:
        render, calls = _recorder()
        result = _edit_loop(_keys("h", _DELETE, "\n"), render, on_cycle=lambda: None)
        assert result == "h"
        assert calls == [("h", 1, None)]

    def test_backspace_at_start_of_buffer_is_noop_even_with_text_after_cursor(self) -> None:
        render, calls = _recorder()
        result = _edit_loop(_keys("h", "i", _HOME, "\x7f", "\n"), render, on_cycle=lambda: None)
        assert result == "hi"
        assert calls == [("h", 1, None), ("hi", 2, None), ("hi", 0, None)]


class TestEditLoopWordNavigation:
    """Ctrl+Left/Right jump by word; Ctrl+Backspace (or its recognized
    equivalents) deletes the word behind the cursor."""

    def test_ctrl_left_jumps_to_start_of_previous_word(self) -> None:
        render, calls = _recorder()
        _edit_loop(_keys(*"foo bar", _CTRL_LEFT, "\n"), render, on_cycle=lambda: None)
        assert calls[-1] == ("foo bar", 4, None)

    def test_ctrl_left_twice_jumps_to_start_of_buffer(self) -> None:
        render, calls = _recorder()
        _edit_loop(_keys(*"foo bar", _CTRL_LEFT, _CTRL_LEFT, "\n"), render, on_cycle=lambda: None)
        assert calls[-1] == ("foo bar", 0, None)

    def test_ctrl_left_at_start_is_noop(self) -> None:
        render, calls = _recorder()
        result = _edit_loop(_keys(_CTRL_LEFT, "\n"), render, on_cycle=lambda: None)
        assert result == ""
        assert calls == []

    def test_ctrl_right_jumps_to_end_of_next_word(self) -> None:
        render, calls = _recorder()
        _edit_loop(_keys(*"foo bar", _HOME, _CTRL_RIGHT, "\n"), render, on_cycle=lambda: None)
        assert calls[-1] == ("foo bar", 3, None)

    def test_ctrl_right_at_end_is_noop(self) -> None:
        render, calls = _recorder()
        result = _edit_loop(_keys(*"foo", _CTRL_RIGHT, "\n"), render, on_cycle=lambda: None)
        assert result == "foo"
        assert calls == [("f", 1, None), ("fo", 2, None), ("foo", 3, None)]

    @pytest.mark.parametrize("ctrl_backspace_key", list(_CTRL_BACKSPACE))
    def test_ctrl_backspace_deletes_the_word_behind_the_cursor(self, ctrl_backspace_key: str) -> None:
        render, _ = _recorder()
        result = _edit_loop(_keys(*"foo bar", ctrl_backspace_key, "\n"), render, on_cycle=lambda: None)
        assert result == "foo "

    def test_ctrl_backspace_from_start_of_buffer_is_noop(self) -> None:
        render, calls = _recorder()
        result = _edit_loop(_keys(*"foo", _HOME, _CTRL_BACKSPACE[0], "\n"), render, on_cycle=lambda: None)
        assert result == "foo"
        assert calls[-1] == ("foo", 0, None)  # Home's render; the ctrl-backspace after it didn't add one

    def test_ctrl_backspace_mid_word_deletes_back_to_word_start(self) -> None:
        render, _ = _recorder()
        keys = _keys(*"foobar", _LEFT, _LEFT, _LEFT, _CTRL_BACKSPACE[0], "\n")
        result = _edit_loop(keys, render, on_cycle=lambda: None)
        assert result == "bar"


class TestEditLoopSelection:
    """Shift+Left/Right start and extend a selection; any unshifted
    movement collapses it; typing or any delete-family key with a
    selection active replaces/removes the whole thing instead of acting
    at just the cursor."""

    def test_shift_left_starts_a_selection(self) -> None:
        render, calls = _recorder()
        _edit_loop(_keys("h", "i", _SHIFT_LEFT, "\n"), render, on_cycle=lambda: None)
        assert calls[-1] == ("hi", 1, (1, 2))

    def test_shift_left_twice_extends_the_selection(self) -> None:
        render, calls = _recorder()
        _edit_loop(_keys("h", "i", _SHIFT_LEFT, _SHIFT_LEFT, "\n"), render, on_cycle=lambda: None)
        assert calls[-1] == ("hi", 0, (0, 2))

    def test_shift_right_selects_forward(self) -> None:
        render, calls = _recorder()
        _edit_loop(_keys("h", "i", _HOME, _SHIFT_RIGHT, "\n"), render, on_cycle=lambda: None)
        assert calls[-1] == ("hi", 1, (0, 1))

    def test_shift_left_then_shift_right_shrinks_back_to_no_selection(self) -> None:
        render, calls = _recorder()
        _edit_loop(_keys("h", "i", _SHIFT_LEFT, _SHIFT_RIGHT, "\n"), render, on_cycle=lambda: None)
        assert calls[-1] == ("hi", 2, None)

    def test_plain_arrow_after_shift_arrow_collapses_the_selection(self) -> None:
        render, calls = _recorder()
        _edit_loop(_keys("h", "i", _SHIFT_LEFT, _LEFT, "\n"), render, on_cycle=lambda: None)
        assert calls[-1] == ("hi", 0, None)

    def test_typing_with_an_active_selection_replaces_it(self) -> None:
        render, _ = _recorder()
        result = _edit_loop(_keys(*"hello", _SHIFT_LEFT, _SHIFT_LEFT, "X", "\n"), render, on_cycle=lambda: None)
        assert result == "helX"

    def test_backspace_with_an_active_selection_removes_it_not_one_char(self) -> None:
        render, _ = _recorder()
        result = _edit_loop(_keys(*"hello", _SHIFT_LEFT, _SHIFT_LEFT, "\x7f", "\n"), render, on_cycle=lambda: None)
        assert result == "hel"

    def test_delete_with_an_active_selection_removes_it(self) -> None:
        render, _ = _recorder()
        result = _edit_loop(_keys(*"hello", _SHIFT_LEFT, _SHIFT_LEFT, _DELETE, "\n"), render, on_cycle=lambda: None)
        assert result == "hel"

    def test_shift_backspace_with_an_active_selection_removes_it(self) -> None:
        render, _ = _recorder()
        result = _edit_loop(
            _keys(*"hello", _SHIFT_LEFT, _SHIFT_LEFT, _SHIFT_BACKSPACE[0], "\n"), render, on_cycle=lambda: None
        )
        assert result == "hel"


class TestEditLoopShiftBackspace:
    """Interpreted as "delete from the cursor back to the start of the
    buffer" — there's no concept of multiple real lines in this editor
    (Enter always submits), only visually wrapped ones, so "delete the
    current line" collapses to that."""

    @pytest.mark.parametrize("shift_backspace_key", list(_SHIFT_BACKSPACE))
    def test_deletes_from_cursor_back_to_start(self, shift_backspace_key: str) -> None:
        render, _ = _recorder()
        result = _edit_loop(_keys(*"hello world", shift_backspace_key, "\n"), render, on_cycle=lambda: None)
        assert result == ""

    def test_only_deletes_up_to_the_cursor_not_past_it(self) -> None:
        # "hello world" is 11 chars; 5x Left from the end (cursor 11) lands
        # the cursor at 6, right before "world" — Shift+Backspace there
        # should delete "hello " and leave "world" untouched.
        render, _ = _recorder()
        result = _edit_loop(
            _keys(*"hello world", _LEFT, _LEFT, _LEFT, _LEFT, _LEFT, _SHIFT_BACKSPACE[0], "\n"),
            render,
            on_cycle=lambda: None,
        )
        assert result == "world"

    def test_at_start_of_buffer_is_noop(self) -> None:
        render, calls = _recorder()
        result = _edit_loop(_keys(*"hi", _HOME, _SHIFT_BACKSPACE[0], "\n"), render, on_cycle=lambda: None)
        assert result == "hi"
        assert calls[-1] == ("hi", 0, None)  # Home's render; nothing added after it


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

    def test_long_modify_other_keys_sequence_assembles_correctly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Shift+Backspace's modifyOtherKeys form is 10 bytes after ESC —
        well past the old 3-byte cap, needs the wider accumulation window."""
        import pi_tui.raw_input as raw_input

        target = _SHIFT_BACKSPACE[1]  # "\x1b[27;2;127~"
        chars = iter(target)  # includes the leading ESC, read first
        monkeypatch.setattr(raw_input, "_read_console_char", lambda: next(chars))
        monkeypatch.setattr(raw_input, "_console_char_ready", lambda _timeout: True)
        assert raw_input._read_key() == target
