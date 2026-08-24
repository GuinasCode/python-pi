"""Slice A4 — policy/budget enforcement and mode=strict/project semantics."""

from __future__ import annotations

import asyncio
from pathlib import Path

from pi_runtime.execute_code.handlers import DEFAULT_HANDLERS
from pi_runtime.execute_code.result import ExecutionStatus
from pi_runtime.execute_code.runner import CodeExecutor
from pi_runtime.state import Budget
from pi_runtime.tools import PolicyEngine, PolicyMode, default_registry


def _executor(tmp_path: Path) -> CodeExecutor:
    return CodeExecutor(artifacts_root=tmp_path / "artifacts")


class TestExecuteCodePolicyGate:
    def test_permissive_policy_allows_execution(self, tmp_path: Path) -> None:
        engine = PolicyEngine(default_registry(), mode=PolicyMode.PERMISSIVE)
        result = asyncio.run(_executor(tmp_path).execute("print('hi')", timeout=15, policy_engine=engine))
        assert result.status == ExecutionStatus.SUCCESS

    def test_default_policy_with_no_confirm_callback_denies_high_risk_execute_code(self, tmp_path: Path) -> None:
        """execute_code is registered Risk.HIGH — PolicyMode.DEFAULT asks
        on HIGH, and with no confirm callback ASK fails closed to DENY."""
        engine = PolicyEngine(default_registry(), mode=PolicyMode.DEFAULT)
        result = asyncio.run(_executor(tmp_path).execute("print('should never run')", timeout=15, policy_engine=engine))
        assert result.status == ExecutionStatus.POLICY_DENIED
        # no subprocess should have run at all
        assert result.stdout.total_bytes == 0

    def test_confirm_callback_approving_lets_execution_proceed(self, tmp_path: Path) -> None:
        engine = PolicyEngine(default_registry(), mode=PolicyMode.DEFAULT, confirm=lambda spec: True)
        result = asyncio.run(_executor(tmp_path).execute("print('approved')", timeout=15, policy_engine=engine))
        assert result.status == ExecutionStatus.SUCCESS

    def test_denial_is_recorded_in_the_shared_audit_log(self, tmp_path: Path) -> None:
        engine = PolicyEngine(default_registry(), mode=PolicyMode.DEFAULT)
        asyncio.run(_executor(tmp_path).execute("print('x')", timeout=15, policy_engine=engine))
        assert any(entry.tool_name == "execute_code" for entry in engine.audit_log)


class TestRpcCallsRespectPolicy:
    def test_rpc_tool_denied_by_policy_is_rejected_but_execution_still_succeeds(self, tmp_path: Path) -> None:
        """The script itself is allowed to run (permissive execute_code
        gate), but each individual RPC call inside is still gated by the
        same PolicyEngine — a STRICT mode should ask (and fail closed to
        deny) on grep, which the RPC name 'search_files' maps to."""
        engine = PolicyEngine(
            default_registry(), mode=PolicyMode.STRICT, confirm=lambda spec: spec.name == "execute_code"
        )
        target = tmp_path / "scoped"
        target.mkdir()
        (target / "note.txt").write_text("anything here")
        code = """
from pi_tools import terminal, RpcCallError
try:
    terminal("echo denied-if-this-runs")
    print("should not reach here")
except RpcCallError as exc:
    print("caught:", exc.error_type)
"""
        result = asyncio.run(
            _executor(tmp_path).execute(code, rpc_handlers=DEFAULT_HANDLERS, timeout=15, policy_engine=engine)
        )
        assert result.status == ExecutionStatus.SUCCESS
        assert "caught: policy_denied" in result.stdout.preview


class TestBudget:
    def test_exhausted_budget_denies_execution_before_spawning(self, tmp_path: Path) -> None:
        budget = Budget(max_tool_calls=0)
        result = asyncio.run(_executor(tmp_path).execute("print('nope')", timeout=15, budget=budget))
        assert result.status == ExecutionStatus.RESOURCE_LIMIT
        assert result.stdout.total_bytes == 0

    def test_execute_code_call_itself_consumes_one_unit_of_budget(self, tmp_path: Path) -> None:
        budget = Budget(max_tool_calls=5)
        asyncio.run(_executor(tmp_path).execute("print('x')", timeout=15, budget=budget))
        assert budget.consumed_tool_calls == 1

    def test_rpc_calls_inside_the_script_also_consume_the_shared_budget(self, tmp_path: Path) -> None:
        budget = Budget(max_tool_calls=5)
        target = tmp_path / "f.txt"
        target.write_text("hi")
        code = f"""
from pi_tools import read_file
read_file({str(target)!r})
"""
        asyncio.run(_executor(tmp_path).execute(code, rpc_handlers=DEFAULT_HANDLERS, timeout=15, budget=budget))
        # 1 for the execute_code call itself + 1 for the read_file RPC call
        assert budget.consumed_tool_calls == 2

    def test_rpc_call_denied_once_budget_runs_out_mid_script(self, tmp_path: Path) -> None:
        budget = Budget(max_tool_calls=1)  # consumed entirely by the execute_code call itself
        target = tmp_path / "f.txt"
        target.write_text("hi")
        code = f"""
from pi_tools import read_file, RpcCallError
try:
    read_file({str(target)!r})
    print("should not reach here")
except RpcCallError as exc:
    print("caught:", exc.error_type)
"""
        result = asyncio.run(
            _executor(tmp_path).execute(code, rpc_handlers=DEFAULT_HANDLERS, timeout=15, budget=budget)
        )
        assert result.status == ExecutionStatus.SUCCESS
        assert "caught: resource_limit" in result.stdout.preview


class TestModeIsolation:
    def test_strict_mode_does_not_forward_the_parents_full_environment(self, tmp_path: Path) -> None:
        code = "import os\nprint('marker present:', 'PI_TEST_SECRET_MARKER' in os.environ)"
        import os as _os

        _os.environ["PI_TEST_SECRET_MARKER"] = "shh"
        try:
            result = asyncio.run(_executor(tmp_path).execute(code, timeout=15, mode="strict"))
        finally:
            del _os.environ["PI_TEST_SECRET_MARKER"]
        assert result.status == ExecutionStatus.SUCCESS
        assert "marker present: False" in result.stdout.preview

    def test_project_mode_forwards_the_parents_environment(self, tmp_path: Path) -> None:
        code = "import os\nprint('marker present:', 'PI_TEST_SECRET_MARKER' in os.environ)"
        import os as _os

        _os.environ["PI_TEST_SECRET_MARKER"] = "shh"
        try:
            result = asyncio.run(_executor(tmp_path).execute(code, timeout=15, mode="project"))
        finally:
            del _os.environ["PI_TEST_SECRET_MARKER"]
        assert result.status == ExecutionStatus.SUCCESS
        assert "marker present: True" in result.stdout.preview

    def test_strict_mode_cwd_is_not_the_real_current_directory(self, tmp_path: Path) -> None:
        real_cwd = str(Path.cwd())
        code = "import os\nprint(os.getcwd())"
        result = asyncio.run(_executor(tmp_path).execute(code, timeout=15, mode="strict"))
        assert result.status == ExecutionStatus.SUCCESS
        assert real_cwd not in result.stdout.preview

    def test_explicit_cwd_overrides_the_mode_default(self, tmp_path: Path) -> None:
        explicit = tmp_path / "explicit-cwd"
        explicit.mkdir()
        code = "import os\nprint(os.getcwd())"
        result = asyncio.run(_executor(tmp_path).execute(code, timeout=15, mode="strict", cwd=str(explicit)))
        assert result.status == ExecutionStatus.SUCCESS
        assert str(explicit) in result.stdout.preview
