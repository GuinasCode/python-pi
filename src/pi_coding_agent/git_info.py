"""Best-effort git repo-name/branch lookup for the interactive status line."""

from __future__ import annotations

import subprocess
from pathlib import Path

_GIT_TIMEOUT = 1.0


def _run_git(cwd: str, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", cwd, *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def get_git_repo_line(cwd: str) -> str | None:
    """Return "(repo-name:branch)" for the git repo containing *cwd*.

    Returns None when *cwd* isn't inside a git repo, or git isn't
    installed/reachable — callers should just omit the line in that case.
    """
    toplevel = _run_git(cwd, "rev-parse", "--show-toplevel")
    if not toplevel:
        return None
    repo_name = Path(toplevel).name

    branch = _run_git(cwd, "branch", "--show-current")
    if not branch:
        short_sha = _run_git(cwd, "rev-parse", "--short", "HEAD")
        branch = f"detached:{short_sha}" if short_sha else "unknown"

    return f"({repo_name}:{branch})"
