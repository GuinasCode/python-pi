"""Slice B3 — click/type/fill/press/select/scroll/wait."""

from __future__ import annotations

import asyncio

from fixtures import fixture_server

from pi_runtime.browser import BrowserManager, InteractionStatus


async def _ref_for(manager: BrowserManager, session_id: str, *, role: str, name: str | None = None) -> str:
    page_snapshot = await manager.snapshot(session_id)
    return next(n.ref for n in page_snapshot.nodes if n.role == role and (name is None or n.name == name))


class TestClick:
    def test_click_runs_the_elements_handler(self) -> None:
        async def _run() -> None:
            with fixture_server() as base_url:
                async with BrowserManager() as manager:
                    session = await manager.open_session()
                    await manager.navigate(session.session_id, f"{base_url}/form")
                    ref = await _ref_for(manager, session.session_id, role="button", name="Submit")

                    result = await manager.click(session.session_id, ref)

                    assert result.ok
                    page = session.get_page()
                    assert page is not None
                    assert "clicked" in await page.locator("#result").text_content()

        asyncio.run(_run())

    def test_click_on_stale_ref_is_reported_not_silently_misclicked(self) -> None:
        async def _run() -> None:
            with fixture_server() as base_url:
                async with BrowserManager() as manager:
                    session = await manager.open_session()
                    await manager.navigate(session.session_id, f"{base_url}/form")
                    ref = await _ref_for(manager, session.session_id, role="button", name="Submit")
                    await manager.navigate(session.session_id, f"{base_url}/")  # invalidates ref

                    result = await manager.click(session.session_id, ref)

                    assert not result.ok
                    assert result.status == InteractionStatus.STALE_REF

        asyncio.run(_run())


class TestFillAndType:
    def test_fill_sets_the_input_value(self) -> None:
        async def _run() -> None:
            with fixture_server() as base_url:
                async with BrowserManager() as manager:
                    session = await manager.open_session()
                    await manager.navigate(session.session_id, f"{base_url}/form")
                    ref = await _ref_for(manager, session.session_id, role="textbox")

                    result = await manager.fill(session.session_id, ref, "hello world")

                    assert result.ok
                    page = session.get_page()
                    assert page is not None
                    assert await page.locator("#username").input_value() == "hello world"

        asyncio.run(_run())

    def test_type_dispatches_real_key_events(self) -> None:
        async def _run() -> None:
            with fixture_server() as base_url:
                async with BrowserManager() as manager:
                    session = await manager.open_session()
                    await manager.navigate(session.session_id, f"{base_url}/form")
                    ref = await _ref_for(manager, session.session_id, role="textbox")

                    result = await manager.type_text(session.session_id, ref, "abc")

                    assert result.ok
                    page = session.get_page()
                    assert page is not None
                    assert await page.locator("#username").input_value() == "abc"

        asyncio.run(_run())

    def test_fill_and_then_click_composes_end_to_end(self) -> None:
        async def _run() -> None:
            with fixture_server() as base_url:
                async with BrowserManager() as manager:
                    session = await manager.open_session()
                    await manager.navigate(session.session_id, f"{base_url}/form")

                    text_ref = await _ref_for(manager, session.session_id, role="textbox")
                    await manager.fill(session.session_id, text_ref, "guilherme")

                    button_ref = await _ref_for(manager, session.session_id, role="button", name="Submit")
                    await manager.click(session.session_id, button_ref)

                    page = session.get_page()
                    assert page is not None
                    assert await page.locator("#result").text_content() == "clicked: guilherme"

        asyncio.run(_run())


class TestSelect:
    def test_select_by_value(self) -> None:
        async def _run() -> None:
            with fixture_server() as base_url:
                async with BrowserManager() as manager:
                    session = await manager.open_session()
                    await manager.navigate(session.session_id, f"{base_url}/form")
                    ref = await _ref_for(manager, session.session_id, role="combobox")

                    result = await manager.select_option(session.session_id, ref, value="blue")

                    assert result.ok
                    page = session.get_page()
                    assert page is not None
                    assert await page.locator("#color").input_value() == "blue"

        asyncio.run(_run())

    def test_select_by_label(self) -> None:
        async def _run() -> None:
            with fixture_server() as base_url:
                async with BrowserManager() as manager:
                    session = await manager.open_session()
                    await manager.navigate(session.session_id, f"{base_url}/form")
                    ref = await _ref_for(manager, session.session_id, role="combobox")

                    result = await manager.select_option(session.session_id, ref, label="Blue")

                    assert result.ok

        asyncio.run(_run())

    def test_select_without_value_or_label_is_a_clear_error(self) -> None:
        async def _run() -> None:
            with fixture_server() as base_url:
                async with BrowserManager() as manager:
                    session = await manager.open_session()
                    await manager.navigate(session.session_id, f"{base_url}/form")
                    ref = await _ref_for(manager, session.session_id, role="combobox")

                    result = await manager.select_option(session.session_id, ref)

                    assert not result.ok
                    assert result.status == InteractionStatus.ERROR

        asyncio.run(_run())


class TestPress:
    def test_press_enter_triggers_keydown_handling(self) -> None:
        async def _run() -> None:
            with fixture_server() as base_url:
                async with BrowserManager() as manager:
                    session = await manager.open_session()
                    await manager.navigate(session.session_id, f"{base_url}/form")
                    ref = await _ref_for(manager, session.session_id, role="textbox")

                    result = await manager.press(session.session_id, ref, "A")

                    assert result.ok
                    page = session.get_page()
                    assert page is not None
                    assert await page.locator("#username").input_value() == "A"

        asyncio.run(_run())


class TestScroll:
    def test_scroll_into_view_succeeds_on_a_real_element(self) -> None:
        async def _run() -> None:
            with fixture_server() as base_url:
                async with BrowserManager() as manager:
                    session = await manager.open_session()
                    await manager.navigate(session.session_id, f"{base_url}/form")
                    ref = await _ref_for(manager, session.session_id, role="button", name="Submit")

                    result = await manager.scroll_into_view(session.session_id, ref)

                    assert result.ok

        asyncio.run(_run())


class TestWait:
    def test_wait_for_text_that_appears_after_a_delay(self) -> None:
        async def _run() -> None:
            with fixture_server() as base_url:
                async with BrowserManager() as manager:
                    session = await manager.open_session()
                    await manager.navigate(session.session_id, f"{base_url}/dynamic")

                    result = await manager.wait_for(session.session_id, text="loaded", timeout=5.0)

                    assert result.ok

        asyncio.run(_run())

    def test_wait_for_load_state(self) -> None:
        async def _run() -> None:
            with fixture_server() as base_url:
                async with BrowserManager() as manager:
                    session = await manager.open_session()
                    await manager.navigate(session.session_id, f"{base_url}/")

                    result = await manager.wait_for(session.session_id, load_state="load", timeout=5.0)

                    assert result.ok

        asyncio.run(_run())

    def test_wait_with_no_condition_is_a_clear_error(self) -> None:
        async def _run() -> None:
            with fixture_server() as base_url:
                async with BrowserManager() as manager:
                    session = await manager.open_session()
                    await manager.navigate(session.session_id, f"{base_url}/")

                    result = await manager.wait_for(session.session_id)

                    assert not result.ok
                    assert result.status == InteractionStatus.ERROR

        asyncio.run(_run())

    def test_wait_for_text_that_never_appears_times_out(self) -> None:
        async def _run() -> None:
            with fixture_server() as base_url:
                async with BrowserManager() as manager:
                    session = await manager.open_session()
                    await manager.navigate(session.session_id, f"{base_url}/")

                    result = await manager.wait_for(session.session_id, text="never-shows-up", timeout=0.5)

                    assert not result.ok
                    assert result.status == InteractionStatus.TIMEOUT

        asyncio.run(_run())
