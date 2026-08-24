"""End-to-end integration: CodeExecutor spawning a real child process
that imports the real `pi_tools` library and calls `read_file` via RPC
back to the parent's real `read_file_handler`. This is Slice A2's
literal acceptance test — a script running as a genuine subprocess uses
a tool programmatically without the intermediate result ever reaching
this test's own process except as the script's own filtered summary.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from pi_runtime.execute_code.handlers import DEFAULT_HANDLERS
from pi_runtime.execute_code.result import ExecutionStatus
from pi_runtime.execute_code.runner import CodeExecutor


def _executor(tmp_path: Path) -> CodeExecutor:
    return CodeExecutor(artifacts_root=tmp_path / "artifacts")


class TestReadFileOverRpc:
    def test_script_reads_a_file_via_pi_tools_and_filters_it(self, tmp_path: Path) -> None:
        log_path = tmp_path / "app.log"
        log_path.write_text("ok\nERROR: disk full\nok\nERROR: timeout\nok\n")

        code = f"""
from pi_tools import read_file
text = read_file({str(log_path)!r})
errors = [line for line in text.splitlines() if "ERROR" in line]
print("count:", len(errors))
"""
        result = asyncio.run(_executor(tmp_path).execute(code, rpc_handlers=DEFAULT_HANDLERS, timeout=15))

        assert result.status == ExecutionStatus.SUCCESS
        assert "count: 2" in result.stdout.preview
        assert result.rpc_call_count == 1

    def test_multiple_rpc_calls_in_one_script_are_all_counted(self, tmp_path: Path) -> None:
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("alpha")
        b.write_text("beta")

        code = f"""
from pi_tools import read_file
x = read_file({str(a)!r})
y = read_file({str(b)!r})
print(x.strip().split("|")[-1].strip() + y.strip().split("|")[-1].strip())
"""
        result = asyncio.run(_executor(tmp_path).execute(code, rpc_handlers=DEFAULT_HANDLERS, timeout=15))
        assert result.status == ExecutionStatus.SUCCESS
        assert result.rpc_call_count == 2

    def test_rpc_error_from_a_missing_file_reaches_the_script(self, tmp_path: Path) -> None:
        code = f"""
from pi_tools import read_file, RpcCallError
try:
    read_file({str(tmp_path / "missing.txt")!r})
    print("should not reach here")
except RpcCallError as exc:
    print("caught:", exc.error_type)
"""
        result = asyncio.run(_executor(tmp_path).execute(code, rpc_handlers=DEFAULT_HANDLERS, timeout=15))
        assert result.status == ExecutionStatus.SUCCESS
        assert "caught: tool_error" in result.stdout.preview

    def test_calling_a_tool_not_in_the_handler_set_is_rejected(self, tmp_path: Path) -> None:
        """DEFAULT_HANDLERS only allowlists read_file — pi_tools.call_tool
        for anything else must fail explicitly, not silently succeed."""
        code = """
from pi_tools import call_tool, RpcCallError
try:
    call_tool("delete_everything")
    print("should not reach here")
except RpcCallError as exc:
    print("caught:", exc.error_type)
"""
        result = asyncio.run(_executor(tmp_path).execute(code, rpc_handlers=DEFAULT_HANDLERS, timeout=15))
        assert result.status == ExecutionStatus.SUCCESS
        assert "caught: unknown_tool" in result.stdout.preview

    def test_pi_tools_outside_the_rpc_harness_fails_loudly(self, tmp_path: Path) -> None:
        """No rpc_handlers passed -> no PI_RPC_* env vars set -> pi_tools
        must refuse rather than hang trying to connect to nothing."""
        code = """
from pi_tools import read_file, PiToolsUnavailable
try:
    read_file("whatever")
    print("should not reach here")
except PiToolsUnavailable:
    print("correctly unavailable")
"""
        result = asyncio.run(_executor(tmp_path).execute(code, timeout=15))
        assert result.status == ExecutionStatus.SUCCESS
        assert "correctly unavailable" in result.stdout.preview

    def test_script_uses_search_files_and_terminal_over_rpc(self, tmp_path: Path) -> None:
        (tmp_path / "mod.py").write_text("def target_function():\n    pass\n")

        code = f"""
from pi_tools import search_files, terminal
matches = search_files("def target_function", path={str(tmp_path)!r})
echoed = terminal("echo integration-ok")
print("matches:", "mod.py" in matches)
print("echoed:", "integration-ok" in echoed)
"""
        result = asyncio.run(_executor(tmp_path).execute(code, rpc_handlers=DEFAULT_HANDLERS, timeout=15))
        assert result.status == ExecutionStatus.SUCCESS
        assert "matches: True" in result.stdout.preview
        assert "echoed: True" in result.stdout.preview
        assert result.rpc_call_count == 2

    def test_only_the_final_print_reaches_the_result_not_intermediate_file_contents(self, tmp_path: Path) -> None:
        """The literal architectural requirement (spec section 7): full
        result -> Python -> filtering -> small result -> model context,
        never the raw file content itself."""
        big_file = tmp_path / "big.txt"
        big_file.write_text("\n".join(f"line {i}" for i in range(500)))

        code = f"""
from pi_tools import read_file
text = read_file({str(big_file)!r})
print("total lines:", len(text.splitlines()))
"""
        result = asyncio.run(_executor(tmp_path).execute(code, rpc_handlers=DEFAULT_HANDLERS, timeout=15))
        assert result.status == ExecutionStatus.SUCCESS
        # the raw per-line content never appears in the final stdout — only the count does
        assert "line 250" not in result.stdout.preview
        assert "total lines: 500" in result.stdout.preview
