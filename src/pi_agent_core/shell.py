"""Cross-platform resolution of a POSIX-compatible shell for the bash tool.

The bash tool is documented and used as a POSIX shell (commands like
``true``, ``sleep``, ``&&`` chains). ``subprocess.run(cmd, shell=True)``
launches ``cmd.exe`` on Windows, which does not understand that syntax.
Resolve an actual ``sh``-compatible shell (Git Bash) explicitly so command
semantics stay consistent across platforms.
"""

from __future__ import annotations

import os
import shutil
from functools import lru_cache


@lru_cache(maxsize=1)
def resolve_posix_shell() -> str | None:
    """Return the path to a POSIX shell, or ``None`` to use the OS default.

    On non-Windows platforms, ``None`` is returned since ``shell=True``
    already invokes ``/bin/sh``. On Windows, ``bash`` (Git Bash) or ``sh``
    is located on ``PATH``; if neither is found, ``None`` is returned and
    callers fall back to the OS default shell.
    """
    if os.name != "nt":
        return None
    for candidate in ("bash", "sh"):
        found = shutil.which(candidate)
        if found:
            return found
    return None


def build_subprocess_args(command: str) -> tuple[list[str] | str, bool]:
    """Return ``(args, shell)`` for ``subprocess.run``/``Popen`` given a shell command.

    Uses an explicit POSIX shell when one can be resolved (required on
    Windows for bash semantics), otherwise falls back to ``shell=True``
    with the OS default.
    """
    shell_path = resolve_posix_shell()
    if shell_path:
        return [shell_path, "-c", command], False
    return command, True
