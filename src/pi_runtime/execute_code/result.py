"""Result contract for execute_code — Slice A1.

Distinguishes success/nonzero_exit/timeout/cancelled/policy_denied/
rpc_error/invalid_code/resource_limit explicitly (spec section 15: "não
transformar tudo em string genérica") so a caller (or AgentRuntime's own
Verifier, see pi_runtime.loop) can decide retry/repair/replan/ask/stop
from the status alone, not by parsing text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ExecutionStatus(str, Enum):
    SUCCESS = "success"
    NONZERO_EXIT = "nonzero_exit"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    POLICY_DENIED = "policy_denied"
    RPC_ERROR = "rpc_error"
    INVALID_CODE = "invalid_code"
    RESOURCE_LIMIT = "resource_limit"


@dataclass
class OutputCapture:
    """Bounded capture of one stream (stdout or stderr) — spec section
    12: never collect indefinitely in memory. `preview` holds only the
    first `head_bytes` + last `tail_bytes` actually seen (a fixed,
    bounded amount regardless of how large the real stream was); the
    complete stream is always written to `artifact_path` via true
    streaming disk writes, never buffered whole in memory first."""

    preview: str
    truncated: bool
    total_bytes: int
    total_lines: int
    artifact_path: str | None
    sha256: str


@dataclass
class ExecuteCodeResult:
    status: ExecutionStatus
    exit_code: int | None
    duration_ms: float
    stdout: OutputCapture
    stderr: OutputCapture
    mode: str
    error_message: str | None = None
    rpc_call_count: int = 0
    artifacts_dir: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == ExecutionStatus.SUCCESS


__all__ = ["ExecuteCodeResult", "ExecutionStatus", "OutputCapture"]
