"""CodeExecutor — Slice A1: code -> subprocess -> bounded stdout -> result.

Runs a Python snippet as a real child process (this interpreter,
`sys.executable`, matching the pattern pi_coding_agent.subagent.runner
already uses for subagent children — same interpreter, same venv, no
`pi`-on-PATH requirement). No RPC yet (Slice A2); no policy/budget
enforcement yet (Slice A4) — this slice's own job is exactly "code runs,
output never explodes the parent's memory or the model's context,
timeout/cancellation work, exit code is captured", nothing more.
"""

from __future__ import annotations

import asyncio
import contextlib
import tempfile
import time
import uuid
from pathlib import Path

from pi_runtime.execute_code.capture import BoundedStreamCapture
from pi_runtime.execute_code.result import ExecuteCodeResult, ExecutionStatus, OutputCapture

_DEFAULT_TIMEOUT_SECONDS = 60.0


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
    ) -> ExecuteCodeResult:
        execution_id = uuid.uuid4().hex[:12]
        artifacts_dir = self._artifacts_root / execution_id
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "script.py").write_text(code, encoding="utf-8")

        try:
            _validate_syntax(code)
        except InvalidCodeError as exc:
            empty = OutputCapture(
                preview="", truncated=False, total_bytes=0, total_lines=0, artifact_path=None, sha256=""
            )
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

        return await self._run_subprocess(
            code, timeout=timeout, mode=mode, cwd=cwd, env=env, artifacts_dir=artifacts_dir
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
    ) -> ExecuteCodeResult:
        import sys

        started = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(artifacts_dir / "script.py"),
            cwd=cwd,
            env=env,
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

        stdout_result = stdout_capture.finish()
        stderr_result = stderr_capture.finish()

        return ExecuteCodeResult(
            status=status,
            exit_code=exit_code,
            duration_ms=duration_ms,
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
