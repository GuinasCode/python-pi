"""execute_code — GAP A of GAPS_EXECUTE_CODE_BROWSER_PROMPT.md.

Lets a program running as a real child process use an explicit,
allowlisted subset of Pi's tools programmatically (via RPC to the
parent's ToolRegistry — Slice A2+), so large intermediate results
(logs, test output, git history) can be processed/filtered in Python
before only a small final result reaches the model's context.

Slice A1 (this module's current state): code -> subprocess -> bounded
stdout/stderr -> typed ExecuteCodeResult. No RPC yet.
"""

from __future__ import annotations

from pi_runtime.execute_code.result import ExecuteCodeResult, ExecutionStatus, OutputCapture
from pi_runtime.execute_code.runner import CodeExecutor, InvalidCodeError

__all__ = ["CodeExecutor", "ExecuteCodeResult", "ExecutionStatus", "InvalidCodeError", "OutputCapture"]
