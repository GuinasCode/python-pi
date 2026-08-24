"""CodeExecutor — code -> subprocess -> bounded stdout -> result.

Runs a Python snippet as a real child process (this interpreter,
`sys.executable`, matching the pattern pi_coding_agent.subagent.runner
already uses for subagent children — same interpreter, same venv, no
`pi`-on-PATH requirement).

Slice A4 wires this into the shared runtime's PolicyEngine/Budget rather
than inventing execute_code-local versions of either (see
pi_runtime.execute_code.security for why, and for the honest limits of
the `mode="strict"`/`mode="project"` filesystem story — this is
best-effort exposure reduction, not a sandbox)."""

from __future__ import annotations

import asyncio
import contextlib
import tempfile
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from pi_runtime.execute_code.capture import BoundedStreamCapture
from pi_runtime.execute_code.result import ExecuteCodeResult, ExecutionStatus, OutputCapture
from pi_runtime.execute_code.rpc import RpcHandler, RpcServer
from pi_runtime.execute_code.security import minimal_environment, wrap_handlers_with_policy
from pi_runtime.tools import PolicyEngine, PolicyViolation

if TYPE_CHECKING:
    from pi_runtime.state import Budget

_DEFAULT_TIMEOUT_SECONDS = 60.0
_EXECUTE_CODE_POLICY_NAME = "execute_code"


class InvalidCodeError(ValueError):
    """Raised for code that fails to even parse as Python — caught
    before a subprocess is ever spawned (spec: distinguish invalid_code
    from a runtime failure)."""


def _validate_syntax(code: str) -> None:
    try:
        compile(code, "<execute_code>", "exec")
    except SyntaxError as exc:
        raise InvalidCodeError(str(exc)) from exc


class CodeExecutor:
    def __init__(self, *, artifacts_root: Path | None = None) -> None:
        self._artifacts_root = artifacts_root or Path(tempfile.gettempdir()) / "pi-execute-code"

    async def execute(
        self,
        code: str,
        *,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        mode: str = "strict",
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        rpc_handlers: dict[str, RpcHandler] | None = None,
        max_rpc_calls: int | None = None,
        policy_engine: PolicyEngine | None = None,
        budget: Budget | None = None,
    ) -> ExecuteCodeResult:
        execution_id = uuid.uuid4().hex[:12]
        artifacts_dir = self._artifacts_root / execution_id
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "script.py").write_text(code, encoding="utf-8")

        empty = OutputCapture(preview="", truncated=False, total_bytes=0, total_lines=0, artifact_path=None, sha256="")

        if policy_engine is not None:
            try:
                policy_engine.evaluate(_EXECUTE_CODE_POLICY_NAME)
            except PolicyViolation as exc:
                return ExecuteCodeResult(
                    status=ExecutionStatus.POLICY_DENIED,
                    exit_code=None,
                    duration_ms=0.0,
                    stdout=empty,
                    stderr=empty,
                    mode=mode,
                    error_message=str(exc),
                    artifacts_dir=str(artifacts_dir),
                )

        if budget is not None:
            reason = budget.exceeded()
            if reason is not None:
                return ExecuteCodeResult(
                    status=ExecutionStatus.RESOURCE_LIMIT,
                    exit_code=None,
                    duration_ms=0.0,
                    stdout=empty,
                    stderr=empty,
                    mode=mode,
                    error_message=f"budget exceeded before execution started: {reason}",
                    artifacts_dir=str(artifacts_dir),
                )
            budget.record_usage(tool_calls=1)

        try:
            _validate_syntax(code)
        except InvalidCodeError as exc:
            return ExecuteCodeResult(
                status=ExecutionStatus.INVALID_CODE,
                exit_code=None,
                duration_ms=0.0,
                stdout=empty,
                stderr=empty,
                mode=mode,
                error_message=str(exc),
                artifacts_dir=str(artifacts_dir),
            )

        effective_handlers = (
            wrap_handlers_with_policy(rpc_handlers, policy_engine=policy_engine, budget=budget)
            if rpc_handlers
            else rpc_handlers
        )

        return await self._run_subprocess(
            code,
            timeout=timeout,
            mode=mode,
            cwd=cwd,
            env=env,
            artifacts_dir=artifacts_dir,
            rpc_handlers=effective_handlers,
            max_rpc_calls=max_rpc_calls,
        )

    async def _run_subprocess(
        self,
        code: str,
        *,
        timeout: float,
        mode: str,
        cwd: str | None,
        env: dict[str, str] | None,
        artifacts_dir: Path,
        rpc_handlers: dict[str, RpcHandler] | None,
        max_rpc_calls: int | None,
    ) -> ExecuteCodeResult:
        import os
        import sys

        rpc_server: RpcServer | None = None
        # mode="strict" (default): minimal env, throwaway cwd inside this
        # execution's own artifacts dir — reduces default exposure but is
        # NOT a sandbox (see pi_runtime.execute_code.security's module
        # docstring: absolute paths and os.chdir still escape it).
        # mode="project": explicit opt-in to the real project cwd/env.
        # An explicitly-passed cwd/env always wins over either default.
        if env is not None:
            child_env = dict(env)
        elif mode == "project":
            child_env = dict(os.environ)
        else:
            child_env = minimal_environment()

        effective_cwd = cwd
        if effective_cwd is None and mode != "project":
            sandbox_dir = artifacts_dir / "sandbox"
            sandbox_dir.mkdir(parents=True, exist_ok=True)
            effective_cwd = str(sandbox_dir)

        if rpc_handlers:
            rpc_server = RpcServer(handlers=rpc_handlers, max_calls=max_rpc_calls)
            port = await rpc_server.start()
            child_env["PI_RPC_HOST"] = "127.0.0.1"
            child_env["PI_RPC_PORT"] = str(port)
            child_env["PI_RPC_TOKEN"] = rpc_server.token

        started = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(artifacts_dir / "script.py"),
            cwd=effective_cwd,
            env=child_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout_capture = BoundedStreamCapture(artifacts_dir / "stdout.log")
        stderr_capture = BoundedStreamCapture(artifacts_dir / "stderr.log")

        async def _pump(reader: asyncio.StreamReader, capture: BoundedStreamCapture) -> None:
            while True:
                chunk = await reader.read(65536)
                if not chunk:
                    return
                capture.write(chunk)

        assert proc.stdout is not None
        assert proc.stderr is not None

        status = ExecutionStatus.SUCCESS
        exit_code: int | None = None
        error_message: str | None = None
        try:
            try:
                await asyncio.wait_for(
                    asyncio.gather(_pump(proc.stdout, stdout_capture), _pump(proc.stderr, stderr_capture), proc.wait()),
                    timeout=timeout,
                )
                exit_code = proc.returncode
                status = ExecutionStatus.SUCCESS if exit_code == 0 else ExecutionStatus.NONZERO_EXIT
            except TimeoutError:
                status = ExecutionStatus.TIMEOUT
                error_message = f"execution exceeded {timeout:.0f}s timeout"
                proc.kill()
                with contextlib.suppress(Exception):
                    await proc.wait()
                exit_code = proc.returncode
            except asyncio.CancelledError:
                status = ExecutionStatus.CANCELLED
                proc.kill()
                with contextlib.suppress(Exception):
                    await proc.wait()
                raise
        finally:
            duration_ms = (time.monotonic() - started) * 1000
            if rpc_server is not None:
                await rpc_server.close()

        rpc_call_count = len(rpc_server.call_log) if rpc_server is not None else 0
        stdout_result = stdout_capture.finish()
        stderr_result = stderr_capture.finish()

        return ExecuteCodeResult(
            status=status,
            exit_code=exit_code,
            duration_ms=duration_ms,
            rpc_call_count=rpc_call_count,
            stdout=OutputCapture(
                preview=stdout_result.preview,
                truncated=stdout_result.truncated,
                total_bytes=stdout_result.total_bytes,
                total_lines=stdout_result.total_lines,
                artifact_path=stdout_result.artifact_path,
                sha256=stdout_result.sha256,
            ),
            stderr=OutputCapture(
                preview=stderr_result.preview,
                truncated=stderr_result.truncated,
                total_bytes=stderr_result.total_bytes,
                total_lines=stderr_result.total_lines,
                artifact_path=stderr_result.artifact_path,
                sha256=stderr_result.sha256,
            ),
            mode=mode,
            error_message=error_message,
            artifacts_dir=str(artifacts_dir),
        )


__all__ = ["CodeExecutor", "InvalidCodeError"]
