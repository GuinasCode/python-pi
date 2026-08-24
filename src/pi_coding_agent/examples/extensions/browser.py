"""Example Pi extension: browser automation tools backed by a real,
persistent Playwright session — registered via the official extension
mechanism (pi.register_tool), same as
pi_coding_agent/examples/extensions/execute_code.py.

Copy this into <project>/.pi/extensions/ to try it. See docs/browser.md
for what the harness does and doesn't guarantee, and
docs/security/browser.md for its security model.

One BrowserManager (and, per session_id, one persistent
BrowserContext/Page) is shared across every tool call this extension
registers — a browser_navigate followed later by a browser_click
against the same session_id acts on the same open page, never a fresh
browser per call.
"""

from __future__ import annotations

from typing import Any

from pi_agent_core.types import AgentTool, AgentToolResult
from pi_ai import TextContent
from pi_runtime.browser import BrowserManager

_manager = BrowserManager()


def _text_result(text: str) -> AgentToolResult:
    return AgentToolResult(content=[TextContent(text=text)])


async def _browser_open(_tool_call_id: str, _args: dict[str, Any], _context: Any, _on_update: Any) -> AgentToolResult:
    session = await _manager.open_session()
    return _text_result(f"session_id: {session.session_id}")


async def _browser_navigate(
    _tool_call_id: str, args: dict[str, Any], _context: Any, _on_update: Any
) -> AgentToolResult:
    result = await _manager.navigate(args["session_id"], args["url"])
    if not result.ok:
        return _text_result(f"error: {result.error}")
    excerpt = result.evidence.excerpt if result.evidence else ""
    return _text_result(f"navigated to {result.url}\n{excerpt}")


async def _browser_snapshot(
    _tool_call_id: str, args: dict[str, Any], _context: Any, _on_update: Any
) -> AgentToolResult:
    page_snapshot = await _manager.snapshot(args["session_id"])
    return _text_result(page_snapshot.text)


async def _browser_click(_tool_call_id: str, args: dict[str, Any], _context: Any, _on_update: Any) -> AgentToolResult:
    result = await _manager.click(args["session_id"], args["ref"])
    return _text_result(f"status: {result.status.value}" + (f"\nerror: {result.error}" if result.error else ""))


async def _browser_fill(_tool_call_id: str, args: dict[str, Any], _context: Any, _on_update: Any) -> AgentToolResult:
    result = await _manager.fill(args["session_id"], args["ref"], args["text"])
    return _text_result(f"status: {result.status.value}" + (f"\nerror: {result.error}" if result.error else ""))


async def _browser_close(_tool_call_id: str, args: dict[str, Any], _context: Any, _on_update: Any) -> AgentToolResult:
    closed = await _manager.close_session(args["session_id"])
    return _text_result("closed" if closed else "no such open session")


def extension(pi: Any) -> None:
    pi.register_tool(
        AgentTool(
            name="browser_open",
            description="Open a new persistent browser session and return its session_id.",
            parameters={"type": "object", "properties": {}},
            label="Open Browser",
            execute=_browser_open,
        )
    )
    pi.register_tool(
        AgentTool(
            name="browser_navigate",
            description="Navigate an open browser session's active page to a URL.",
            parameters={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "url": {"type": "string"},
                },
                "required": ["session_id", "url"],
            },
            label="Browser Navigate",
            execute=_browser_navigate,
        )
    )
    pi.register_tool(
        AgentTool(
            name="browser_snapshot",
            description="Capture a bounded accessibility-tree snapshot of the active page, with element refs.",
            parameters={"type": "object", "properties": {"session_id": {"type": "string"}}, "required": ["session_id"]},
            label="Browser Snapshot",
            execute=_browser_snapshot,
        )
    )
    pi.register_tool(
        AgentTool(
            name="browser_click",
            description="Click an element by the ref from the most recent browser_snapshot.",
            parameters={
                "type": "object",
                "properties": {"session_id": {"type": "string"}, "ref": {"type": "string"}},
                "required": ["session_id", "ref"],
            },
            label="Browser Click",
            execute=_browser_click,
        )
    )
    pi.register_tool(
        AgentTool(
            name="browser_fill",
            description="Set an input/textarea's value by ref.",
            parameters={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "ref": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["session_id", "ref", "text"],
            },
            label="Browser Fill",
            execute=_browser_fill,
        )
    )
    pi.register_tool(
        AgentTool(
            name="browser_close",
            description="Close a browser session and release its resources.",
            parameters={"type": "object", "properties": {"session_id": {"type": "string"}}, "required": ["session_id"]},
            label="Browser Close",
            execute=_browser_close,
        )
    )
