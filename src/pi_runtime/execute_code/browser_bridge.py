"""Slice B7 — the optional execute_code <-> browser bridge (spec section
40). NOT part of DEFAULT_HANDLERS and never auto-enabled: a caller who
wants a script to read browser state must explicitly build these
handlers and pass them into `CodeExecutor.execute(rpc_handlers=...)`,
merged with (or instead of) `DEFAULT_HANDLERS`.

```python
from pi_runtime.browser import BrowserManager
from pi_runtime.execute_code.browser_bridge import build_browser_bridge_handlers
from pi_runtime.execute_code.handlers import DEFAULT_HANDLERS
from pi_runtime.execute_code.runner import CodeExecutor

async with BrowserManager() as browser_manager:
    session = await browser_manager.open_session()
    await browser_manager.navigate(session.session_id, "https://example.com")

    handlers = {**DEFAULT_HANDLERS, **build_browser_bridge_handlers(browser_manager, session.session_id)}
    result = await CodeExecutor().execute(
        "from pi_tools import browser_snapshot\nprint(browser_snapshot())",
        rpc_handlers=handlers,
    )
```

Deliberately scoped to exactly one pre-opened session, chosen by whoever
wires the bridge — not something the script itself can pick or create.
A script can read/act on that one session's current page (snapshot,
evaluate) but can never call `open_session` itself: bridging in
"any browser action" would let a script open new sessions with whatever
backend/cdp_url it wants, defeating the "explicit, controlled" mandate
this whole RPC layer exists for. The same PolicyEngine/Budget wrapping
`pi_runtime.execute_code.security.wrap_handlers_with_policy` already
applies to these handlers exactly like any other — bridging them in
doesn't bypass that layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pi_runtime.execute_code.rpc import RpcError, RpcHandler

if TYPE_CHECKING:
    from pi_runtime.browser import BrowserManager


def build_browser_bridge_handlers(browser_manager: BrowserManager, session_id: str) -> dict[str, RpcHandler]:
    async def browser_snapshot_handler(_tool: str, _arguments: dict[str, Any]) -> str:
        try:
            page_snapshot = await browser_manager.snapshot(session_id)
        except Exception as exc:
            raise RpcError(str(exc), error_type="tool_error") from exc
        return page_snapshot.text

    async def browser_evaluate_handler(_tool: str, arguments: dict[str, Any]) -> str:
        script = arguments.get("script")
        if not isinstance(script, str):
            raise RpcError("missing or invalid 'script' argument", error_type="malformed_request")
        result = await browser_manager.evaluate(session_id, script)
        if not result.ok:
            raise RpcError(result.error or f"browser_evaluate failed: {result.status.value}", error_type="tool_error")
        return result.preview

    return {
        "browser_snapshot": browser_snapshot_handler,
        "browser_evaluate": browser_evaluate_handler,
    }


__all__ = ["build_browser_bridge_handlers"]
