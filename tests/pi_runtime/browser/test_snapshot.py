"""Slice B2 — browser_snapshot and the ElementRef model."""

from __future__ import annotations

import asyncio

import pytest
from fixtures import fixture_server

from pi_runtime.browser import BrowserManager, StaleRefError


class TestSnapshotIsBoundedAndTextual:
    def test_snapshot_returns_interactive_elements_with_refs(self) -> None:
        async def _run() -> None:
            with fixture_server() as base_url:
                async with BrowserManager() as manager:
                    session = await manager.open_session()
                    await manager.navigate(session.session_id, f"{base_url}/form")

                    page_snapshot = await manager.snapshot(session.session_id)

                    roles = [node.role for node in page_snapshot.nodes]
                    assert "textbox" in roles
                    assert "button" in roles
                    assert "combobox" in roles
                    assert all(node.ref.startswith("e") for node in page_snapshot.nodes)
                    assert "@e" in page_snapshot.text

        asyncio.run(_run())

    def test_snapshot_never_contains_raw_html(self) -> None:
        async def _run() -> None:
            with fixture_server() as base_url:
                async with BrowserManager() as manager:
                    session = await manager.open_session()
                    await manager.navigate(session.session_id, f"{base_url}/form")
                    page_snapshot = await manager.snapshot(session.session_id)
                    assert "<button" not in page_snapshot.text
                    assert "<input" not in page_snapshot.text

        asyncio.run(_run())

    def test_snapshot_includes_url_and_title(self) -> None:
        async def _run() -> None:
            with fixture_server() as base_url:
                async with BrowserManager() as manager:
                    session = await manager.open_session()
                    await manager.navigate(session.session_id, f"{base_url}/")
                    page_snapshot = await manager.snapshot(session.session_id)
                    assert page_snapshot.url == f"{base_url}/"

        asyncio.run(_run())


class TestRefResolution:
    def test_ref_resolves_to_a_locator_that_matches_one_real_element(self) -> None:
        async def _run() -> None:
            with fixture_server() as base_url:
                async with BrowserManager() as manager:
                    session = await manager.open_session()
                    await manager.navigate(session.session_id, f"{base_url}/form")
                    page_snapshot = await manager.snapshot(session.session_id)

                    button_ref = next(n.ref for n in page_snapshot.nodes if n.role == "button" and n.name == "Submit")
                    locator = manager.resolve_ref(session.session_id, button_ref)
                    assert await locator.count() == 1
                    assert await locator.text_content() == "Submit"

        asyncio.run(_run())

    def test_unknown_ref_raises_stale_ref_error(self) -> None:
        async def _run() -> None:
            with fixture_server() as base_url:
                async with BrowserManager() as manager:
                    session = await manager.open_session()
                    await manager.navigate(session.session_id, f"{base_url}/form")
                    await manager.snapshot(session.session_id)

                    with pytest.raises(StaleRefError):
                        manager.resolve_ref(session.session_id, "e999")

        asyncio.run(_run())

    def test_ref_from_a_previous_snapshot_is_stale_after_a_new_one(self) -> None:
        async def _run() -> None:
            with fixture_server() as base_url:
                async with BrowserManager() as manager:
                    session = await manager.open_session()
                    await manager.navigate(session.session_id, f"{base_url}/form")
                    first_snapshot = await manager.snapshot(session.session_id)
                    button_ref = next(n.ref for n in first_snapshot.nodes if n.role == "button")

                    # navigating away invalidates every ref from the old page
                    await manager.navigate(session.session_id, f"{base_url}/")

                    with pytest.raises(StaleRefError):
                        manager.resolve_ref(session.session_id, button_ref)

        asyncio.run(_run())

    def test_disambiguates_elements_sharing_role_and_name(self) -> None:
        """Two elements with the identical role+accessible name must
        resolve to two different, correct elements (nth-based
        disambiguation), not both collapsing onto the first match."""

        async def _run() -> None:
            with fixture_server() as base_url:
                async with BrowserManager() as manager:
                    session = await manager.open_session()
                    await manager.navigate(session.session_id, f"{base_url}/dynamic")
                    page = session.get_page()
                    assert page is not None
                    await page.set_content('<button id="a">Click me</button><button id="b">Click me</button>')
                    page_snapshot = await manager.snapshot(session.session_id)
                    button_refs = [n.ref for n in page_snapshot.nodes if n.role == "button"]
                    assert len(button_refs) == 2

                    locator_a = manager.resolve_ref(session.session_id, button_refs[0])
                    locator_b = manager.resolve_ref(session.session_id, button_refs[1])
                    id_a = await locator_a.get_attribute("id")
                    id_b = await locator_b.get_attribute("id")
                    assert {id_a, id_b} == {"a", "b"}

        asyncio.run(_run())


class TestSnapshotErrorSemantics:
    def test_snapshot_on_unknown_session_raises_value_error(self) -> None:
        async def _run() -> None:
            async with BrowserManager() as manager:
                with pytest.raises(ValueError, match="no such"):
                    await manager.snapshot("nope")

        asyncio.run(_run())

    def test_snapshot_on_closed_session_raises_value_error(self) -> None:
        async def _run() -> None:
            async with BrowserManager() as manager:
                session = await manager.open_session()
                await manager.close_session(session.session_id)
                with pytest.raises(ValueError, match="closed"):
                    await manager.snapshot(session.session_id)

        asyncio.run(_run())
