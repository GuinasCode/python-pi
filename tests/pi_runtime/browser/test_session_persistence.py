"""Slice B7 — spec section 39's mandatory test: prove that a sequence of
separate tool calls (navigate -> click -> type -> snapshot) shares the
same browser session, context, and page throughout, with DOM state
remaining coherent across the whole sequence — not a fresh browser
reset between calls."""

from __future__ import annotations

import asyncio

from fixtures import fixture_server

from pi_runtime.browser import BrowserManager


class TestMandatorySessionPersistence:
    def test_four_sequential_calls_share_the_same_session_context_and_page(self) -> None:
        async def _run() -> None:
            with fixture_server() as base_url:
                async with BrowserManager() as manager:
                    session = await manager.open_session()
                    context_before = session.context
                    page_before = session.get_page()

                    # tool call 1: browser_navigate(url)
                    nav_result = await manager.navigate(session.session_id, f"{base_url}/form")
                    assert nav_result.ok

                    # tool call 2: browser_click(ref) — fill the field first via ref
                    page_snapshot = await manager.snapshot(session.session_id)
                    text_ref = next(n.ref for n in page_snapshot.nodes if n.role == "textbox")
                    fill_result = await manager.fill(session.session_id, text_ref, "session-persistence-proof")
                    assert fill_result.ok

                    # tool call 3: browser_click(ref) on the submit button
                    button_ref = next(n.ref for n in page_snapshot.nodes if n.role == "button" and n.name == "Submit")
                    click_result = await manager.click(session.session_id, button_ref)
                    assert click_result.ok

                    # tool call 4: browser_snapshot() — DOM state from calls 2-3 must be visible
                    final_snapshot = await manager.snapshot(session.session_id)
                    assert final_snapshot.nodes  # the page after two mutations still snapshots cleanly

                    # same session/context/page object identity throughout — never recreated
                    assert manager.get_session(session.session_id) is session
                    assert session.context is context_before
                    assert session.get_page() is page_before

                    # DOM state produced by the earlier calls (fill + click) is coherent:
                    # the click handler read the field's value that the fill call set,
                    # on the very same page instance, three tool calls later.
                    page = session.get_page()
                    assert page is not None
                    result_text = await page.locator("#result").text_content()
                    assert result_text == "clicked: session-persistence-proof"

        asyncio.run(_run())
