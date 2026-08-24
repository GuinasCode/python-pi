"""Slice B7 — the official example extension exposing browser
automation through the real extension mechanism (pi.register_tool),
proven against a real Chromium instance via the local fixture server."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "pi_runtime" / "browser"))
from fixtures import fixture_server

from pi_coding_agent.examples.extensions.browser import extension
from pi_coding_agent.extensions.types import ExtensionAPI


def _tool(pi: ExtensionAPI, name: str):  # type: ignore[no-untyped-def]
    return next(t for t in pi.tools if t.name == name)


class TestBrowserExtensionRegistration:
    def test_registers_all_expected_tools(self) -> None:
        pi = ExtensionAPI()
        extension(pi)
        names = {t.name for t in pi.tools}
        assert names == {
            "browser_open",
            "browser_navigate",
            "browser_snapshot",
            "browser_click",
            "browser_fill",
            "browser_close",
        }


class TestBrowserExtensionEndToEnd:
    def test_open_navigate_fill_click_snapshot_close_composes(self) -> None:
        async def _run() -> None:
            with fixture_server() as base_url:
                pi = ExtensionAPI()
                extension(pi)

                open_result = await _tool(pi, "browser_open").execute("c1", {}, None, lambda _u: None)
                session_id = open_result.content[0].text.split("session_id: ")[1].strip()

                nav_result = await _tool(pi, "browser_navigate").execute(
                    "c2", {"session_id": session_id, "url": f"{base_url}/form"}, None, lambda _u: None
                )
                assert "navigated to" in nav_result.content[0].text

                snap_result = await _tool(pi, "browser_snapshot").execute(
                    "c3", {"session_id": session_id}, None, lambda _u: None
                )
                snapshot_text = snap_result.content[0].text
                assert "@e" in snapshot_text

                textbox_ref = next(
                    line.split("@")[1].split("]")[0] for line in snapshot_text.splitlines() if "textbox" in line
                )
                fill_args = {"session_id": session_id, "ref": textbox_ref, "text": "extension-test"}
                fill_result = await _tool(pi, "browser_fill").execute("c4", fill_args, None, lambda _u: None)
                assert "status: success" in fill_result.content[0].text

                button_ref = next(
                    line.split("@")[1].split("]")[0]
                    for line in snapshot_text.splitlines()
                    if "button" in line and "Submit" in line
                )
                click_result = await _tool(pi, "browser_click").execute(
                    "c5", {"session_id": session_id, "ref": button_ref}, None, lambda _u: None
                )
                assert "status: success" in click_result.content[0].text

                close_result = await _tool(pi, "browser_close").execute(
                    "c6", {"session_id": session_id}, None, lambda _u: None
                )
                assert close_result.content[0].text == "closed"

        asyncio.run(_run())
