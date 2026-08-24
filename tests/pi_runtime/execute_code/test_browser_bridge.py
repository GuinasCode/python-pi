"""Slice B7 — execute_code <-> browser bridge, end-to-end: a real child
process calls pi_tools.browser_snapshot()/browser_evaluate() over RPC,
which the parent dispatches into a real, already-open Playwright
session — not enabled unless explicitly wired in, per spec section 40.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from pi_runtime.browser import BrowserManager
from pi_runtime.execute_code.browser_bridge import build_browser_bridge_handlers
from pi_runtime.execute_code.handlers import DEFAULT_HANDLERS
from pi_runtime.execute_code.result import ExecutionStatus
from pi_runtime.execute_code.runner import CodeExecutor

sys.path.insert(0, str(Path(__file__).parent.parent / "browser"))
from fixtures import fixture_server


def _executor(tmp_path: Path) -> CodeExecutor:
    return CodeExecutor(artifacts_root=tmp_path / "artifacts")


class TestBrowserBridgeEndToEnd:
    def test_script_reads_a_real_page_snapshot_via_rpc(self, tmp_path: Path) -> None:
        async def _run() -> None:
            with fixture_server() as base_url:
                async with BrowserManager() as browser_manager:
                    session = await browser_manager.open_session()
                    await browser_manager.navigate(session.session_id, f"{base_url}/form")

                    handlers = {
                        **DEFAULT_HANDLERS,
                        **build_browser_bridge_handlers(browser_manager, session.session_id),
                    }
                    code = """
from pi_tools import browser_snapshot
snap = browser_snapshot()
print("has button:", "button" in snap)
print("has textbox:", "textbox" in snap)
"""
                    result = await _executor(tmp_path).execute(code, rpc_handlers=handlers, timeout=15)
                    assert result.status == ExecutionStatus.SUCCESS
                    assert "has button: True" in result.stdout.preview
                    assert "has textbox: True" in result.stdout.preview

        asyncio.run(_run())

    def test_script_filters_evaluate_results_small_output_from_a_big_page(self, tmp_path: Path) -> None:
        """The literal spec section 40 example: browser -> execute_code
        -> process -> small output -> model context."""

        async def _run() -> None:
            with fixture_server() as base_url:
                async with BrowserManager() as browser_manager:
                    session = await browser_manager.open_session()
                    await browser_manager.navigate(session.session_id, f"{base_url}/")

                    handlers = {
                        **DEFAULT_HANDLERS,
                        **build_browser_bridge_handlers(browser_manager, session.session_id),
                    }
                    code = """
import json
from pi_tools import browser_evaluate

result_json = browser_evaluate(
    "Array.from({length: 400}, (_, i) => ({id: i, tag: i % 7 === 0 ? 'match' : 'skip'}))"
)
items = json.loads(result_json)
matches = [x for x in items if x["tag"] == "match"]
print("total:", len(items))
print("matches:", len(matches))
"""
                    result = await _executor(tmp_path).execute(code, rpc_handlers=handlers, timeout=15)
                    assert result.status == ExecutionStatus.SUCCESS
                    assert "total: 400" in result.stdout.preview
                    assert "matches: 58" in result.stdout.preview  # i in 0..399 with i % 7 == 0
                    # the raw per-item text never appears in the final output, only the counts do
                    assert '"id": 500' not in result.stdout.preview

        asyncio.run(_run())

    def test_bridge_is_not_enabled_unless_explicitly_wired(self, tmp_path: Path) -> None:
        """Without build_browser_bridge_handlers merged in, browser_snapshot
        must fail exactly like calling any other non-allowlisted tool —
        the bridge is never on by default (spec section 40)."""

        async def _run() -> None:
            code = """
from pi_tools import browser_snapshot, RpcCallError
try:
    browser_snapshot()
    print("should not reach here")
except RpcCallError as exc:
    print("caught:", exc.error_type)
"""
            result = await _executor(tmp_path).execute(code, rpc_handlers=DEFAULT_HANDLERS, timeout=15)
            assert result.status == ExecutionStatus.SUCCESS
            assert "caught: unknown_tool" in result.stdout.preview

        asyncio.run(_run())

    def test_browser_bridge_errors_are_reported_not_crashed(self, tmp_path: Path) -> None:
        async def _run() -> None:
            with fixture_server() as base_url:
                async with BrowserManager() as browser_manager:
                    session = await browser_manager.open_session()
                    await browser_manager.navigate(session.session_id, f"{base_url}/")

                    handlers = {
                        **DEFAULT_HANDLERS,
                        **build_browser_bridge_handlers(browser_manager, session.session_id),
                    }
                    code = """
from pi_tools import browser_evaluate, RpcCallError
try:
    browser_evaluate("this is not valid js (((")
    print("should not reach here")
except RpcCallError as exc:
    print("caught:", exc.error_type)
"""
                    result = await _executor(tmp_path).execute(code, rpc_handlers=handlers, timeout=15)
                    assert result.status == ExecutionStatus.SUCCESS
                    assert "caught: tool_error" in result.stdout.preview

        asyncio.run(_run())
