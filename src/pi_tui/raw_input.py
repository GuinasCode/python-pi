"""Minimal raw-mode line reader that recognizes Shift+Tab as a "cycle" event.

The interactive REPL previously read each line with the builtin ``input()``,
which has no way to distinguish Shift+Tab from a normal keypress (readline
just treats it as Tab). To let Shift+Tab drive the permission-mode toggle
(mirroring Claude Code's "shift+tab to cycle" footer) while still typing a
message, we read raw keys ourselves.

This is deliberately minimal — just enough line editing (printable
chars, cursor movement, word jump/delete, selection, enter, Ctrl+C,
Ctrl+D, Up/Down history recall) to replace ``input()``, plus the one
extra control key.
When stdin isn't a real TTY (piped input, tests), it falls back to a
plain ``readline()`` with no cycle detection, matching how ``input()``
degrades in that situation.

Unlike a typical line editor, this one doesn't echo characters itself —
every buffer mutation (character typed, cursor moved, backspace/delete,
selection changed) and every Shift+Tab calls the caller's
``on_render(current_text, cursor_index, selection)``, handing it full
control of what gets drawn. That split exists because the caller
(interactive mode) shows a live status footer right below the input, and
the input itself can wrap across multiple terminal rows once it's long
enough — patching the screen incrementally in place while tracking
exactly how many rows the wrapped input currently occupies (and where
within them the cursor/selection now sit, since they can be anywhere in
the buffer) is fragile row-arithmetic that isn't this module's job to get
right; letting the caller redraw everything from a fixed anchor on every
keystroke sidesteps needing that arithmetic at all, at a cost (a full
repaint per keystroke) a human's typing rate makes unnoticeable.

Special-key detection relies on two different xterm-family CSI
extensions, both requested via ``\\x1b[>4;2m`` ("modifyOtherKeys" mode 2)
on entering raw mode:
  - Shift/Ctrl+Arrow already gets a distinct sequence in *every* commonly
    used terminal without needing modifyOtherKeys at all — the older,
    universally-supported ``CSI 1 ; modifier {ABCD}`` form (e.g. Shift+Left
    is ``ESC [ 1 ; 2 D``).
  - Shift/Ctrl+Backspace does not: an unmodified Backspace already has an
    ASCII value (DEL/BS) regardless of which modifier is held, so telling
    them apart needs the terminal to opt in to reporting *every* key
    (including already-ASCII ones) with an explicit modifier, which is
    exactly what modifyOtherKeys mode 2 does — it reports Backspace as
    ``ESC [ 27 ; modifier ; keycode ~``. Terminals that don't implement
    it (rare among actively maintained ones, but not universal) simply
    ignore the enabling sequence and keep sending plain Backspace, so
    Shift/Ctrl+Backspace degrade to acting like plain Backspace there —
    not a crash, just back to today's behavior for that one combination.
  - Shift+Tab itself doesn't need any of this — CSI "backtab" (ESC [ Z) is
    its own long-standing, independent convention (see below).
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator

_BACKTAB = "\x1b[Z"
_UP = "\x1b[A"
_DOWN = "\x1b[B"
_LEFT = "\x1b[D"
_RIGHT = "\x1b[C"
_HOME = "\x1b[H"
_END = "\x1b[F"
_DELETE = "\x1b[3~"
_SHIFT_LEFT = "\x1b[1;2D"
_SHIFT_RIGHT = "\x1b[1;2C"
_CTRL_LEFT = "\x1b[1;5D"
_CTRL_RIGHT = "\x1b[1;5C"
# modifyOtherKeys reports the same key with more than one keycode
# depending on the terminal (BS's ASCII value 8, or DEL's 127) — accept
# both rather than betting on one.
_SHIFT_BACKSPACE = ("\x1b[27;2;8~", "\x1b[27;2;127~")
_CTRL_BACKSPACE = ("\x1b[27;5;8~", "\x1b[27;5;127~", "\x1b\x7f", "\x17")

# Sent on entering raw mode to opt every key (including already-ASCII
# ones like Backspace) into being reported with an explicit modifier —
# see the module docstring. Reset on exit so it doesn't leak into
# whatever runs next in this terminal.
_ENABLE_MODIFY_OTHER_KEYS = "\x1b[>4;2m"
_DISABLE_MODIFY_OTHER_KEYS = "\x1b[>4m"


def _is_csi_final_byte(ch: str) -> bool:
    """True for the byte that ends a CSI escape sequence (ECMA-48: a
    letter or ``~``, following ESC ``[`` and any digit/``;`` parameter
    bytes) — e.g. the ``D``/``C``/``Z`` in ``ESC[D``/``ESC[C``/``ESC[Z``,
    or the ``~`` in ``ESC[3~``/``ESC[27;2;127~``. Both platforms'
    ``_read_key`` stop accumulating a sequence as soon as they see one,
    instead of always waiting out the full per-byte timeout even for the
    common short sequences (arrows, Shift+Tab) — only the longer
    modifyOtherKeys ones actually need that extra wait."""
    return ch.isalpha() or ch == "~"


def _word_left(buf: list[str], cursor: int) -> int:
    """Index of the start of the word behind `cursor` — skip any
    whitespace immediately before it, then the run of non-whitespace
    before that."""
    i = cursor
    while i > 0 and buf[i - 1].isspace():
        i -= 1
    while i > 0 and not buf[i - 1].isspace():
        i -= 1
    return i


def _word_right(buf: list[str], cursor: int) -> int:
    """Index just past the end of the word ahead of `cursor` — skip any
    whitespace at/after it, then the run of non-whitespace after that."""
    i, n = cursor, len(buf)
    while i < n and buf[i].isspace():
        i += 1
    while i < n and not buf[i].isspace():
        i += 1
    return i


def _edit_loop(
    read_key: Callable[[], str],
    on_render: Callable[[str, int, tuple[int, int] | None], None],
    on_cycle: Callable[[], None],
    history: list[str] | None = None,
) -> str:
    """Core line-editing loop, decoupled from the platform key source so it
    can be unit-tested with a fake ``read_key``. Terminal writes here are
    limited to what submitting/interrupting needs — everything about
    what's currently on screen while the user is still typing is the
    caller's ``on_render``, not this loop's.

    ``cursor`` is a plain index into ``buf`` (0..len(buf)) — typed
    characters insert there (not just append), Backspace/Delete remove
    the character behind/ahead of it (or Ctrl+Backspace/Ctrl+Right/Left
    a whole word), and Left/Right/Home/End move it without touching the
    buffer.

    ``anchor`` tracks an in-progress Shift+arrow selection: not set (no
    selection) until Shift+Left/Right is pressed, at which point it's
    pinned to the cursor's position *before* that move; any further
    Shift+arrow extends the selection by moving the cursor again, any
    *unshifted* movement or edit clears it. Whenever a selection is
    active, typing/Backspace/Delete/Ctrl+Backspace/Shift+Backspace all
    replace/remove the whole selection rather than acting at just the
    cursor — see ``_delete_selection``.

    ``history`` is prior submitted lines, oldest first (the caller's to
    own/persist — this loop only ever reads it, appending is the
    caller's job once a line is actually submitted). Up/Down walk
    backward/forward through it, same convention as a shell: the first
    Up saves whatever's currently in ``buf`` as an in-progress draft, so
    Down-ing back past the newest history entry restores it rather than
    leaving an empty line; Down when not currently browsing history (or
    with no history at all) does nothing, matching how bash's Down key
    is a no-op at the bottom rather than clearing the line.
    """
    buf: list[str] = []
    cursor = 0
    anchor: int | None = None
    history = history if history is not None else []
    history_index: int | None = None  # None = not browsing; else an index into history
    draft: str = ""  # buf's content just before the first Up, restored by Down past the end

    def _selection() -> tuple[int, int] | None:
        if anchor is None or anchor == cursor:
            return None
        return (anchor, cursor) if anchor < cursor else (cursor, anchor)

    def _delete_selection() -> bool:
        """If a selection is active, delete it (cursor lands at its start)
        and report True. Callers with their own single-char/word delete
        logic check this first and skip their own logic when it fires —
        deleting *something* always takes priority over deleting nothing
        when a selection exists, matching every mainstream editor."""
        nonlocal cursor, anchor
        sel = _selection()
        if sel is None:
            return False
        lo, hi = sel
        del buf[lo:hi]
        cursor = lo
        anchor = None
        return True

    while True:
        key = read_key()

        if key in ("\r", "\n"):
            return "".join(buf)

        if key == "\x03":  # Ctrl+C
            raise KeyboardInterrupt

        if key == "\x04":  # Ctrl+D
            if not buf:
                raise EOFError
            continue

        if key in _SHIFT_BACKSPACE:
            if not _delete_selection():
                if cursor > 0:
                    del buf[0:cursor]
                    cursor = 0
                else:
                    continue
            on_render("".join(buf), cursor, _selection())
            continue

        if key in _CTRL_BACKSPACE:
            if not _delete_selection():
                start = _word_left(buf, cursor)
                if start == cursor:
                    continue
                del buf[start:cursor]
                cursor = start
            on_render("".join(buf), cursor, _selection())
            continue

        if key in ("\x7f", "\x08"):  # Backspace
            if not _delete_selection():
                if cursor == 0:
                    continue
                del buf[cursor - 1]
                cursor -= 1
            on_render("".join(buf), cursor, _selection())
            continue

        if key == _DELETE:
            if not _delete_selection():
                if cursor == len(buf):
                    continue
                del buf[cursor]
            on_render("".join(buf), cursor, _selection())
            continue

        if key == _SHIFT_LEFT:
            if cursor > 0:
                if anchor is None:
                    anchor = cursor
                cursor -= 1
                on_render("".join(buf), cursor, _selection())
            continue

        if key == _SHIFT_RIGHT:
            if cursor < len(buf):
                if anchor is None:
                    anchor = cursor
                cursor += 1
                on_render("".join(buf), cursor, _selection())
            continue

        if key == _CTRL_LEFT:
            anchor = None
            new_cursor = _word_left(buf, cursor)
            if new_cursor != cursor:
                cursor = new_cursor
                on_render("".join(buf), cursor, None)
            continue

        if key == _CTRL_RIGHT:
            anchor = None
            new_cursor = _word_right(buf, cursor)
            if new_cursor != cursor:
                cursor = new_cursor
                on_render("".join(buf), cursor, None)
            continue

        if key == _LEFT:
            had_selection = anchor is not None
            anchor = None
            if cursor > 0:
                cursor -= 1
                on_render("".join(buf), cursor, None)
            elif had_selection:
                on_render("".join(buf), cursor, None)
            continue

        if key == _RIGHT:
            had_selection = anchor is not None
            anchor = None
            if cursor < len(buf):
                cursor += 1
                on_render("".join(buf), cursor, None)
            elif had_selection:
                on_render("".join(buf), cursor, None)
            continue

        if key == _HOME:
            had_selection = anchor is not None
            anchor = None
            if cursor != 0 or had_selection:
                cursor = 0
                on_render("".join(buf), cursor, None)
            continue

        if key == _END:
            had_selection = anchor is not None
            anchor = None
            if cursor != len(buf) or had_selection:
                cursor = len(buf)
                on_render("".join(buf), cursor, None)
            continue

        if key == _BACKTAB:
            on_cycle()
            on_render("".join(buf), cursor, _selection())
            continue

        if key == _UP:
            if not history:
                continue
            if history_index is None:
                draft = "".join(buf)
                history_index = len(history) - 1
            elif history_index > 0:
                history_index -= 1
            else:
                continue
            buf = list(history[history_index])
            cursor = len(buf)
            anchor = None
            on_render("".join(buf), cursor, None)
            continue

        if key == _DOWN:
            if history_index is None:
                continue
            if history_index < len(history) - 1:
                history_index += 1
                buf = list(history[history_index])
            else:
                history_index = None
                buf = list(draft)
            cursor = len(buf)
            anchor = None
            on_render("".join(buf), cursor, None)
            continue

        if not key or key == "\t" or key.startswith("\x1b"):
            # Unhandled control/escape sequence (an unmapped arrow/function
            # key, a lone Escape press, ...) — swallow rather than
            # inserting garbage.
            continue

        _delete_selection()
        buf.insert(cursor, key)
        cursor += 1
        on_render("".join(buf), cursor, None)


if sys.platform == "win32":
    import contextlib
    import ctypes
    from ctypes import wintypes

    # (DWORD)-10, i.e. -10 as an unsigned 32-bit value — passing the
    # literal negative int to GetStdHandle's DWORD argtype raises
    # OverflowError in ctypes.
    _STD_INPUT_HANDLE = 0xFFFFFFF6
    _ENABLE_VIRTUAL_TERMINAL_INPUT = 0x0200
    _WAIT_OBJECT_0 = 0

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.GetStdHandle.argtypes = [wintypes.DWORD]
    _kernel32.GetStdHandle.restype = wintypes.HANDLE
    _kernel32.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    _kernel32.GetConsoleMode.restype = wintypes.BOOL
    _kernel32.SetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    _kernel32.SetConsoleMode.restype = wintypes.BOOL
    _kernel32.ReadConsoleW.argtypes = [
        wintypes.HANDLE,
        wintypes.LPWSTR,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    ]
    _kernel32.ReadConsoleW.restype = wintypes.BOOL
    _kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    _kernel32.WaitForSingleObject.restype = wintypes.DWORD

    def _console_input_handle() -> wintypes.HANDLE:
        return _kernel32.GetStdHandle(_STD_INPUT_HANDLE)  # type: ignore[no-any-return]

    @contextlib.contextmanager
    def _raw_mode() -> Iterator[None]:
        handle = _console_input_handle()
        old_mode = wintypes.DWORD()
        _kernel32.GetConsoleMode(handle, ctypes.byref(old_mode))
        # ENABLE_VIRTUAL_TERMINAL_INPUT and nothing else — every line/echo/
        # signal-processing bit off, the Windows analog of tty.setraw's
        # ~(ICANON|ECHO|ISIG) below. Turning off "processed input" also
        # means Ctrl+C arrives as a literal \x03 byte instead of the console
        # generating its own break signal, matching _edit_loop's explicit
        # \x03 handling and the POSIX raw-mode behavior it mirrors.
        _kernel32.SetConsoleMode(handle, wintypes.DWORD(_ENABLE_VIRTUAL_TERMINAL_INPUT))
        sys.stdout.write(_ENABLE_MODIFY_OTHER_KEYS)
        sys.stdout.flush()
        try:
            yield
        finally:
            sys.stdout.write(_DISABLE_MODIFY_OTHER_KEYS)
            sys.stdout.flush()
            _kernel32.SetConsoleMode(handle, old_mode)

    def _read_console_char() -> str:
        buf = ctypes.create_unicode_buffer(1)
        n_read = wintypes.DWORD()
        ok = _kernel32.ReadConsoleW(_console_input_handle(), buf, 1, ctypes.byref(n_read), None)
        if not ok or n_read.value == 0:
            return ""
        return buf.value

    def _console_char_ready(timeout: float) -> bool:
        millis = max(0, int(timeout * 1000))
        result: int = _kernel32.WaitForSingleObject(_console_input_handle(), wintypes.DWORD(millis))
        return result == _WAIT_OBJECT_0

    def _read_key() -> str:
        ch = _read_console_char()
        if ch != "\x1b":
            return ch
        seq = ch
        # Up to 12 more bytes: the longest sequences _edit_loop recognizes
        # are the modifyOtherKeys ones (e.g. ESC[27;2;127~, 9 bytes after
        # ESC) — but stop as soon as a CSI final byte arrives rather than
        # always waiting out the timeout, so the common short sequences
        # aren't delayed by it.
        for _ in range(12):
            if not _console_char_ready(0.05):
                break
            nxt = _read_console_char()
            seq += nxt
            if _is_csi_final_byte(nxt):
                break
        return seq

else:
    import contextlib
    import select
    import termios
    import tty

    @contextlib.contextmanager
    def _raw_mode() -> Iterator[None]:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            # setraw (not setcbreak): _edit_loop already echoes every key it
            # accepts itself, and handles Ctrl+C/Ctrl+D as plain bytes rather
            # than relying on the tty driver's signal generation — setcbreak
            # would leave ECHO/ISIG on, producing doubled characters and a
            # real SIGINT racing our own \x03 handling.
            tty.setraw(fd)
            sys.stdout.write(_ENABLE_MODIFY_OTHER_KEYS)
            sys.stdout.flush()
            yield
        finally:
            sys.stdout.write(_DISABLE_MODIFY_OTHER_KEYS)
            sys.stdout.flush()
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def _read_key() -> str:
        ch = sys.stdin.read(1)
        if ch != "\x1b":
            return ch
        seq = ch
        # Up to 12 more bytes: the longest sequences _edit_loop recognizes
        # are the modifyOtherKeys ones (e.g. ESC[27;2;127~, 9 bytes after
        # ESC) — but stop as soon as a CSI final byte arrives rather than
        # always waiting out the timeout, so the common short sequences
        # aren't delayed by it.
        for _ in range(12):
            ready, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not ready:
                break
            nxt = sys.stdin.read(1)
            seq += nxt
            if _is_csi_final_byte(nxt):
                break
        return seq


def read_line_with_cycle(
    prompt: str,
    *,
    on_render: Callable[[str, int, tuple[int, int] | None], None],
    on_cycle: Callable[[], None],
    history: list[str] | None = None,
) -> str:
    """Read one line of input, calling ``on_render(current_text,
    cursor_index, selection)`` after every keystroke (initially with
    ``("", 0, None)``, before the user has typed anything) so the caller
    can draw the prompt/text/cursor/selection/any live status around it,
    and ``on_cycle()`` each time Shift+Tab is pressed (without
    submitting). ``selection`` is a ``(start, end)`` index pair into the
    text when a Shift+arrow selection is active, ``None`` otherwise.
    ``history`` (oldest first) is what Up/Down walk through — see
    ``_edit_loop``'s docstring. Raises ``KeyboardInterrupt`` on Ctrl+C
    and ``EOFError`` on Ctrl+D at an empty line — same contract as
    ``input()``.
    """
    if not sys.stdin.isatty():
        # No live rendering possible for piped input — same plain prompt
        # input() itself would write in this situation.
        sys.stdout.write(prompt)
        sys.stdout.flush()
        line = sys.stdin.readline()
        if line == "":
            raise EOFError
        return line.rstrip("\n")

    on_render("", 0, None)
    with _raw_mode():
        return _edit_loop(_read_key, on_render, on_cycle, history=history)


def _select_loop(read_key: Callable[[], str], count: int, on_render: Callable[[int], None], index: int) -> int | None:
    """Core selection loop, decoupled from the platform key source for the
    same testability reason ``_edit_loop`` is. Up/Down move a highlighted
    index within ``[0, count)``, clamped rather than wrapping (matches
    ``SelectDialog``, the Textual equivalent); Enter confirms it; Escape,
    Ctrl+C and Ctrl+D all cancel (return None) rather than only Escape —
    there's no partially-typed text here worth distinguishing an
    interrupt from a deliberate cancel over, unlike ``_edit_loop``."""
    while True:
        key = read_key()

        if key in ("\r", "\n"):
            return index

        if key in ("\x1b", "\x03", "\x04"):
            return None

        if key == _UP:
            if index > 0:
                index -= 1
                on_render(index)
            continue

        if key == _DOWN:
            if index < count - 1:
                index += 1
                on_render(index)
            continue


def select_from_list(count: int, *, on_render: Callable[[int], None], initial: int = 0) -> int | None:
    """Let the user pick one of ``count`` items with Up/Down + Enter,
    calling ``on_render(highlighted_index)`` once up front and again after
    every move so the caller can (re)draw the list — same split of
    responsibility as ``read_line_with_cycle``: this function only reads
    keys and tracks which index is highlighted, the caller owns every
    byte written to the screen. Returns the confirmed index, or ``None``
    if cancelled (Escape/Ctrl+C/Ctrl+D) or if stdin isn't a real TTY
    (piped input has no way to navigate a menu at all).
    """
    if count <= 0 or not sys.stdin.isatty():
        return None

    index = max(0, min(initial, count - 1))
    on_render(index)
    with _raw_mode():
        return _select_loop(_read_key, count, on_render, index)
