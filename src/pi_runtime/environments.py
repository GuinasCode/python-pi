"""Execution Backends — Fase 11 of the research-first-runtime plan.

ExecutionBackend is the contract every tool should be able to run
against: cwd, command execution, file read/write, environment variables,
process lifecycle (timeout/kill), artifact access (reading a file back
after a command produced it).

LocalExecutionBackend wraps the existing, already-tested execute_bash/
read_file/write_file tools (pi_coding_agent.tools) — a real local
executor, not a reimplementation of one.

DockerExecutionBackend and SshExecutionBackend shell out to the real
`docker`/`ssh` CLI via subprocess — genuine execution when the binary is
on PATH, an explicit "not available" CommandResult when it isn't (same
pattern as pi_coding_agent.tools.browser_fetch_url reporting a missing
Playwright install), never a fake/simulated result (Regra 1.3). No new
dependency (the paramiko/docker-sdk kind) is added for this vertical
slice — CLI subprocess is exactly what execute_bash already uses for
local commands.

SandboxExecutionBackend is a registered TODO (Regra 1.5), not faked: a
real OS-level sandbox (seccomp/firejail/Windows job objects) is a
substantial, platform-specific undertaking with no existing infra in
this repo to build on, and "sandbox" implies a security boundary — a
fake one would be actively misleading, worse than simply not having it
yet.
"""

from __future__ import annotations

import base64
import posixpath
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from pi_coding_agent.tools import execute_bash, read_file, write_file


def normalize_path(path: str) -> str:
    """Fase 11 acceptance criterion: "paths são normalizados". Uses
    posixpath explicitly (not pathlib.Path, which is OS-dependent) so a
    path behaves identically whether the backend is local (any host OS)
    or a remote POSIX shell (docker/ssh) — collapses `.`/`..` segments
    and normalizes separators without resolving against the local
    filesystem (a remote path may not exist locally at all)."""
    return posixpath.normpath(path.replace("\\", "/"))


@dataclass
class CommandResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


class ExecutionBackend(Protocol):
    name: str

    def cwd(self) -> str: ...
    def run(self, command: str, *, timeout: float = 120.0, env: dict[str, str] | None = None) -> CommandResult: ...
    def read_file(self, path: str) -> str: ...
    def write_file(self, path: str, content: str) -> None: ...


@dataclass
class LocalExecutionBackend:
    """Runs directly on this process's own host — wraps execute_bash/
    read_file/write_file, unchanged."""

    name: str = "local"
    _cwd: str = field(default_factory=lambda: str(Path.cwd()))

    def cwd(self) -> str:
        return self._cwd

    def run(self, command: str, *, timeout: float = 120.0, env: dict[str, str] | None = None) -> CommandResult:
        result = execute_bash(command, cwd=self._cwd, timeout=int(timeout), env=env)
        text = result.content[0].get("text", "") if result.content else ""
        details = result.details or {}
        if not result.is_error:
            return CommandResult(stdout=text, stderr="", exit_code=int(details.get("exitCode", 0)))
        timed_out = "timed out" in text.lower()
        if timed_out:
            return CommandResult(stdout="", stderr=text, exit_code=-1, timed_out=True)
        return CommandResult(stdout="", stderr=text, exit_code=int(details.get("exitCode", 1) or 1))

    def read_file(self, path: str) -> str:
        result = read_file(normalize_path(path))
        if result.is_error:
            raise FileNotFoundError(result.content[0].get("text", path) if result.content else path)
        return result.content[0].get("text", "") if result.content else ""

    def write_file(self, path: str, content: str) -> None:
        result = write_file(normalize_path(path), content)
        if result.is_error:
            raise OSError(result.content[0].get("text", path) if result.content else path)


def _shell_out(args: list[str], *, timeout: float) -> CommandResult:
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return CommandResult(stdout="", stderr=f"timed out after {timeout:.0f}s", exit_code=-1, timed_out=True)
    except Exception as exc:
        return CommandResult(stdout="", stderr=str(exc), exit_code=1)
    return CommandResult(stdout=proc.stdout, stderr=proc.stderr, exit_code=proc.returncode)


@dataclass
class DockerExecutionBackend:
    """Runs commands inside a named, already-running container via
    `docker exec` — does not create/manage the container's lifecycle
    itself (a separate, larger concern), only executes within one that
    already exists."""

    container: str
    name: str = "docker"
    _cwd: str = "/"

    def _binary_available(self) -> bool:
        return shutil.which("docker") is not None

    def cwd(self) -> str:
        return self._cwd

    def run(self, command: str, *, timeout: float = 120.0, env: dict[str, str] | None = None) -> CommandResult:
        if not self._binary_available():
            return CommandResult(stdout="", stderr="docker is not installed or not on PATH", exit_code=127)
        args = ["docker", "exec", "-w", self._cwd]
        for key, value in (env or {}).items():
            args += ["-e", f"{key}={value}"]
        args += [self.container, "sh", "-c", command]
        return _shell_out(args, timeout=timeout)

    def read_file(self, path: str) -> str:
        result = self.run(f"cat -- {normalize_path(path)!r}")
        if not result.ok:
            raise FileNotFoundError(result.stderr or path)
        return result.stdout

    def write_file(self, path: str, content: str) -> None:
        encoded = base64.b64encode(content.encode()).decode()
        result = self.run(f"echo {encoded} | base64 -d > {normalize_path(path)!r}")
        if not result.ok:
            raise OSError(result.stderr or path)


@dataclass
class SshExecutionBackend:
    """Runs commands on a remote host via the `ssh` CLI. Uses the host's
    own configured auth (keys/agent/ssh config) rather than accepting
    credentials directly — this backend never touches or stores a
    password/key itself."""

    host: str
    name: str = "ssh"
    _cwd: str = "~"

    def _binary_available(self) -> bool:
        return shutil.which("ssh") is not None

    def cwd(self) -> str:
        return self._cwd

    def run(self, command: str, *, timeout: float = 120.0, env: dict[str, str] | None = None) -> CommandResult:
        if not self._binary_available():
            return CommandResult(stdout="", stderr="ssh is not installed or not on PATH", exit_code=127)
        env_prefix = "".join(f"{key}={value} " for key, value in (env or {}).items())
        remote_command = f"cd {self._cwd} && {env_prefix}{command}"
        return _shell_out(["ssh", self.host, remote_command], timeout=timeout)

    def read_file(self, path: str) -> str:
        result = self.run(f"cat -- {normalize_path(path)!r}")
        if not result.ok:
            raise FileNotFoundError(result.stderr or path)
        return result.stdout

    def write_file(self, path: str, content: str) -> None:
        encoded = base64.b64encode(content.encode()).decode()
        result = self.run(f"echo {encoded} | base64 -d > {normalize_path(path)!r}")
        if not result.ok:
            raise OSError(result.stderr or path)


class SandboxExecutionBackend:
    """TODO (Regra 1.5): a real OS-level sandbox is a substantial,
    platform-specific undertaking (seccomp on Linux, App Sandbox on
    macOS, job objects on Windows) with no existing infra in this repo
    to build on. Explicitly unimplemented rather than faked — "sandbox"
    implies a security boundary, and a fake one would be actively
    misleading."""

    name = "sandbox"

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise NotImplementedError(
            "SandboxExecutionBackend is not implemented yet (Regra 1.5 — registered as a "
            "TODO, not faked). Use LocalExecutionBackend, DockerExecutionBackend, or "
            "SshExecutionBackend for real isolation today."
        )


__all__ = [
    "CommandResult",
    "DockerExecutionBackend",
    "ExecutionBackend",
    "LocalExecutionBackend",
    "SandboxExecutionBackend",
    "SshExecutionBackend",
    "normalize_path",
]
