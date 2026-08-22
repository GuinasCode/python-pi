"""Minimal raw-mode line reader that recognizes Shift+Tab as a "cycle" event.

The interactive REPL previously read each line with the builtin ``input()``,
which has no way to distinguish Shift+Tab from a normal keypress (readline
just treats it as Tab). To let Shift+Tab drive the permission-mode toggle
(mirroring Claude Code's "shift+tab to cycle" footer) while still typing a
message, we read raw keys ourselves.

This is deliberately minimal — no history — just enough line editing
(printable chars, cursor movement, backspace/delete, enter, Ctrl+C,
Ctrl+D) to replace ``input()``, plus the one extra control key. When
stdin isn't a real TTY (piped input, tests), it falls back to a plain
``readline()`` with no cycle detection, matching how ``input()`` degrades
in that situation.

Unlike a typical line editor, this one doesn't echo characters itself —
every buffer mutation (character typed, cursor moved, backspace/delete)
and every Shift+Tab calls the caller's ``on_render(current_text,
cursor_index)``, handing it full control of what gets drawn. That split
exists because the caller (interactive mode) shows a live status footer
right below the input, and the input itself can wrap across multiple
terminal rows once it's long enough — patching the screen incrementally
in place while tracking exactly how many rows the wrapped input
currently occupies (and where within them the cursor now sits, once it
can move anywhere in the buffer rather than always the end) is fragile
row-arithmetic that isn't this module's job to get right; letting the
caller redraw everything from a fixed anchor on every keystroke
sidesteps needing that arithmetic at all, at a cost (a full repaint per
keystroke) a human's typing rate makes unnoticeable.

Shift+Tab is reported as the CSI "backtab" sequence ESC [ Z on both
platforms, normalized to the same internal sentinel below:
  - POSIX terminals (xterm and derivatives, most Linux/macOS terminals) send
    it natively.
  - Windows only sends it once the console is put in "virtual terminal
    input" mode (``ENABLE_VIRTUAL_TERMINAL_INPUT``), which makes it translate
    special keys to the same ANSI sequences POSIX terminals use, read via
    ``ReadConsoleW``. This deliberately does *not* use ``msvcrt.getwch()``:
    that call bypasses VT translation entirely and reports Tab-family keys
    through the legacy two-byte "extended key" scheme (a NUL/0xE0 lead byte
    then a BIOS scan code) — a scheme that, on modern console hosts
    (Windows Terminal/ConPTY), frequently can't tell Shift+Tab apart from
    plain Tab at all, since Tab already has its own ASCII value and isn't
    routed through that extended-key path regardless of Shift.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator

_BACKTAB = "\x1b[Z"
_LEFT = "\x1b[D"
_RIGHT = "\x1b[C"
_HOME = "\x1b[H"
_END = "\x1b[F"
_DELETE = "\x1b[3~"


def _is_csi_final_byte(ch: str) -> bool:
    """True for the byte that ends a CSI escape sequence (ECMA-48: a
    letter or ``~``, following ESC ``[`` and any digit/``;`` parameter
    bytes) — e.g. the ``D``/``C``/``Z`` in ``ESC[D``/``ESC[C``/``ESC[Z``,
    or the ``~`` in ``ESC[3~``. Both platforms' ``_read_key`` stop
    accumulating a sequence as soon as they see one, instead of always
    waiting out the full per-byte timeout even for the common 3-byte
    sequences (arrows, Shift+Tab) — only the rarer 4-byte ones like
    Delete actually need that extra wait."""
    return ch.isalpha() or ch == "~"


def _edit_loop(
    read_key: Callable[[], str],
    on_render: Callable[[str, int], None],
    on_cycle: Callable[[], None],
) -> str:
    """Core line-editing loop, decoupled from the platform key source so it
    can be unit-tested with a fake ``read_key``. Terminal writes here are
    limited to what submitting/interrupting needs — everything about
    what's currently on screen while the user is still typing is the
    caller's ``on_render``, not this loop's.

    ``cursor`` is a plain index into ``buf`` (0..len(buf)) — typed
    characters insert there (not just append), Backspace/Delete remove
    the character behind/ahead of it, and Left/Right/Home/End move it
    without touching the buffer.
    """
    buf: list[str] = []
    cursor = 0
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

        if key in ("\x7f", "\x08"):  # Backspace
            if cursor > 0:
                del buf[cursor - 1]
                cursor -= 1
                on_render("".join(buf), cursor)
            continue

        if key == _DELETE:
            if cursor < len(buf):
                del buf[cursor]
                on_render("".join(buf), cursor)
            continue

        if key == _LEFT:
            if cursor > 0:
                cursor -= 1
                on_render("".join(buf), cursor)
            continue

        if key == _RIGHT:
            if cursor < len(buf):
                cursor += 1
                on_render("".join(buf), cursor)
            continue

        if key == _HOME:
            if cursor != 0:
                cursor = 0
                on_render("".join(buf), cursor)
            continue

        if key == _END:
            if cursor != len(buf):
                cursor = len(buf)
                on_render("".join(buf), cursor)
            continue

        if key == _BACKTAB:
            on_cycle()
            on_render("".join(buf), cursor)
            continue

        if not key or key == "\t" or key.startswith("\x1b"):
            # Unhandled control/escape sequence (an unmapped arrow/function
            # key, a lone Escape press, ...) — swallow rather than
            # inserting garbage.
            continue

        buf.insert(cursor, key)
        cursor += 1
        on_render("".join(buf), cursor)


if sys.platform == "win32":
    import contextlib
    import ctypes
    from ctypes import wintypes

    # (DWORD)-10, i.e. -10 as an unsigned 32-bit value — passing the literal
    # -10 to GetStdHandle's DWORD argtype raises OverflowError in ctypes.
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
        try:
            yield
        finally:
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
        # Up to 3 more bytes: covers every sequence _edit_loop recognizes,
        # including the 4-byte ones (Delete is ESC [ 3 ~) — but stop as
        # soon as a CSI final byte arrives rather than always waiting out
        # the timeout, so the common 3-byte sequences aren't delayed by it.
        for _ in range(3):
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
            yield
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def _read_key() -> str:
        ch = sys.stdin.read(1)
        if ch != "\x1b":
            return ch
        seq = ch
        # Up to 3 more bytes: covers every sequence _edit_loop recognizes,
        # including the 4-byte ones (Delete is ESC [ 3 ~) — but stop as
        # soon as a CSI final byte arrives rather than always waiting out
        # the timeout, so the common 3-byte sequences aren't delayed by it.
        for _ in range(3):
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
    on_render: Callable[[str, int], None],
    on_cycle: Callable[[], None],
) -> str:
    """Read one line of input, calling ``on_render(current_text,
    cursor_index)`` after every keystroke (initially with ``("", 0)``,
    before the user has typed anything) so the caller can draw the
    prompt/text/cursor/any live status around it, and ``on_cycle()`` each
    time Shift+Tab is pressed (without submitting). Raises
    ``KeyboardInterrupt`` on Ctrl+C and ``EOFError`` on Ctrl+D at an empty
    line — same contract as ``input()``.
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

    on_render("", 0)
    with _raw_mode():
        return _edit_loop(_read_key, on_render, on_cycle)
