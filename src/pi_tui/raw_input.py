"""Minimal raw-mode line reader that recognizes Shift+Tab as a "cycle" event.

The interactive REPL previously read each line with the builtin ``input()``,
which has no way to distinguish Shift+Tab from a normal keypress (readline
just treats it as Tab). To let Shift+Tab drive the permission-mode toggle
(mirroring Claude Code's "shift+tab to cycle" footer) while still typing a
message, we read raw keys ourselves.

This is deliberately minimal — no history, no cursor movement/left-right
editing, no multi-line wrapping awareness — just enough line editing
(printable chars, backspace, enter, Ctrl+C, Ctrl+D) to replace ``input()``,
plus the one extra control key. When stdin isn't a real TTY (piped input,
tests), it falls back to a plain ``readline()`` with no cycle detection,
matching how ``input()`` degrades in that situation.

Shift+Tab is reported by terminals in two different shapes depending on
platform:
  - POSIX terminals (xterm and derivatives, most Linux/macOS terminals, and
    Windows Terminal running a POSIX shell) send the CSI "backtab" sequence
    ESC [ Z.
  - The Windows console API (read via ``msvcrt``) never surfaces that ANSI
    sequence — it reports Tab-family keys through the legacy two-byte
    extended-key scheme (a NUL or 0xE0 lead byte followed by a scan code),
    where Shift+Tab's scan code is 0x0F.
Both are normalized to the same internal sentinel below.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator

_BACKTAB = "\x1b[Z"


def _edit_loop(read_key: Callable[[], str], on_cycle: Callable[[], None]) -> str:
    """Core line-editing loop, decoupled from the platform key source so it
    can be unit-tested with a fake ``read_key``."""
    buf: list[str] = []
    while True:
        key = read_key()

        if key in ("\r", "\n"):
            # Explicit \r\n, not just \n: correct in raw mode (see _raw_mode
            # below, POSIX uses tty.setraw which disables OPOST/ONLCR, so a
            # bare \n would move the cursor down without returning to
            # column 0) and harmless in cooked mode (an extra \r is a no-op).
            sys.stdout.write("\r\n")
            sys.stdout.flush()
            return "".join(buf)

        if key == "\x03":  # Ctrl+C
            sys.stdout.write("\r\n")
            sys.stdout.flush()
            raise KeyboardInterrupt

        if key == "\x04":  # Ctrl+D
            if not buf:
                sys.stdout.write("\r\n")
                sys.stdout.flush()
                raise EOFError
            continue

        if key in ("\x7f", "\x08"):  # Backspace
            if buf:
                buf.pop()
                sys.stdout.write("\b \b")
                sys.stdout.flush()
            continue

        if key == _BACKTAB:
            on_cycle()
            continue

        if not key or key == "\t" or key.startswith("\x1b"):
            # Unhandled control/escape sequence (arrows, plain Tab, a lone
            # Escape press, ...) — swallow rather than inserting garbage.
            continue

        buf.append(key)
        sys.stdout.write(key)
        sys.stdout.flush()


if sys.platform == "win32":
    import contextlib
    import msvcrt

    @contextlib.contextmanager
    def _raw_mode() -> Iterator[None]:
        yield

    def _read_key() -> str:
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            code = msvcrt.getwch()
            if code == "\x0f":  # Shift+Tab scan code
                return _BACKTAB
            return ""  # unhandled extended key (arrows, F-keys, ...)
        return ch

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
        for _ in range(2):
            ready, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not ready:
                break
            seq += sys.stdin.read(1)
        return seq


def read_line_with_cycle(prompt: str, *, on_cycle: Callable[[], None]) -> str:
    """Read one line of input, calling ``on_cycle()`` each time Shift+Tab is
    pressed (without submitting). Raises ``KeyboardInterrupt`` on Ctrl+C and
    ``EOFError`` on Ctrl+D at an empty line — same contract as ``input()``.
    """
    sys.stdout.write(prompt)
    sys.stdout.flush()

    if not sys.stdin.isatty():
        line = sys.stdin.readline()
        if line == "":
            raise EOFError
        return line.rstrip("\n")

    with _raw_mode():
        return _edit_loop(_read_key, on_cycle)
