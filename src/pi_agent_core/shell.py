"""Cross-platform resolution of a POSIX-compatible shell for the bash tool.

The bash tool is documented and used as a POSIX shell (commands like
``true``, ``sleep``, ``&&`` chains). ``subprocess.run(cmd, shell=True)``
launches ``cmd.exe`` on Windows, which does not understand that syntax.
Resolve an actual ``sh``-compatible shell (Git Bash) explicitly so command
semantics stay consistent across platforms.
"""

from __future__ import annotations

import ntpath
import os
import shutil
from functools import lru_cache


def _is_wsl_launcher_stub(path: str, system_root: str) -> bool:
    """Whether *path* is Windows' own bash.exe/sh.exe shim under SystemRoot.

    Windows ships a ``bash.exe`` in ``%SystemRoot%\\System32`` that launches
    WSL (not a real POSIX shell). If no WSL distro is installed it fails
    immediately with "no distributions installed" — and since it's a valid
    PATH hit, ``shutil.which("bash")`` happily returns it ahead of (or
    instead of) Git Bash. Reject anything living under SystemRoot.

    *path*/*system_root* are always Windows-style strings (this only ever
    runs for real from resolve_posix_shell's ``os.name == "nt"`` branch) —
    normalized with ``ntpath`` explicitly rather than ``os.path``, so the
    logic (and its tests) are correct regardless of the host OS this
    actually executes on, not just when the host happens to be Windows.
    """
    try:
        resolved = ntpath.normcase(ntpath.abspath(path))
        root = ntpath.normcase(ntpath.abspath(system_root))
    except OSError:
        return False
    return resolved == root or resolved.startswith(root + ntpath.sep)


@lru_cache(maxsize=1)
def resolve_posix_shell() -> str | None:
    """Return the path to a POSIX shell, or ``None`` to use the OS default.

    On non-Windows platforms, ``None`` is returned since ``shell=True``
    already invokes ``/bin/sh``. On Windows, ``bash`` (Git Bash) or ``sh``
    is located on ``PATH`` — skipping Windows' own WSL-launcher stub — with
    a fallback search of common Git-for-Windows install locations. If none
    is found, ``None`` is returned and callers fall back to the OS default
    shell.
    """
    if os.name != "nt":
        return None

    system_root = os.environ.get("SYSTEMROOT", r"C:\Windows")

    candidates: list[str] = []
    for name in ("bash", "sh"):
        found = shutil.which(name)
        if found:
            candidates.append(found)
    for env_var in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432"):
        base = os.environ.get(env_var)
        if base:
            candidates.append(os.path.join(base, "Git", "bin", "bash.exe"))

    for candidate in candidates:
        if not os.path.isfile(candidate):
            continue
        if _is_wsl_launcher_stub(candidate, system_root):
            continue
        return candidate
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
