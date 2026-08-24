"""Slice B1 — BrowserManager/BrowserSession: real, persistent Playwright
sessions (not a browser launched fresh per call)."""

from __future__ import annotations

import asyncio

import pytest
from fixtures import fixture_server

from pi_runtime.browser import BrowserManager
from pi_runtime.tools import PolicyEngine, PolicyViolation, ToolRegistry


class TestSessionLifecycle:
    def test_open_session_returns_a_live_session(self) -> None:
        async def _run() -> None:
            async with BrowserManager() as manager:
                session = await manager.open_session()
                assert not session.closed
                assert manager.get_session(session.session_id) is session

        asyncio.run(_run())

    def test_close_session_marks_it_closed(self) -> None:
        async def _run() -> None:
            async with BrowserManager() as manager:
                session = await manager.open_session()
                assert await manager.close_session(session.session_id) is True
                assert session.closed

        asyncio.run(_run())

    def test_closing_twice_is_a_noop_returning_false(self) -> None:
        async def _run() -> None:
            async with BrowserManager() as manager:
                session = await manager.open_session()
                await manager.close_session(session.session_id)
                assert await manager.close_session(session.session_id) is False

        asyncio.run(_run())

    def test_closing_unknown_session_returns_false(self) -> None:
        async def _run() -> None:
            async with BrowserManager() as manager:
                assert await manager.close_session("nope") is False

        asyncio.run(_run())

    def test_open_session_goes_through_policy_when_configured(self) -> None:
        async def _run() -> None:
            registry = ToolRegistry()  # "browser" deliberately not registered
            policy = PolicyEngine(registry)
            async with BrowserManager(policy_engine=policy) as manager:
                with pytest.raises(PolicyViolation):
                    await manager.open_session()

        asyncio.run(_run())

    def test_session_expires_after_its_timeout(self) -> None:
        async def _run() -> None:
            async with BrowserManager() as manager:
                session = await manager.open_session(timeout_seconds=0.01)
                await asyncio.sleep(0.05)
                assert session.is_expired()

        asyncio.run(_run())


class TestAttachDetach:
    def test_attach_returns_the_same_session_and_touches_it(self) -> None:
        async def _run() -> None:
            async with BrowserManager() as manager:
                session = await manager.open_session()
                first_touch = session.last_used_at
                await asyncio.sleep(0.01)
                attached = manager.attach_session(session.session_id)
                assert attached is session
                assert attached.last_used_at > first_touch

        asyncio.run(_run())

    def test_attach_unknown_session_returns_none(self) -> None:
        async def _run() -> None:
            async with BrowserManager() as manager:
                assert manager.attach_session("nope") is None

        asyncio.run(_run())

    def test_attach_closed_session_returns_none(self) -> None:
        async def _run() -> None:
            async with BrowserManager() as manager:
                session = await manager.open_session()
                await manager.close_session(session.session_id)
                assert manager.attach_session(session.session_id) is None

        asyncio.run(_run())

    def test_detach_removes_from_registry_without_closing_resources(self) -> None:
        async def _run() -> None:
            async with BrowserManager() as manager:
                session = await manager.open_session()
                detached = manager.detach_session(session.session_id)
                assert detached is session
                assert manager.get_session(session.session_id) is None
                assert not session.closed  # resources still alive — caller owns cleanup now
                await session.close()

        asyncio.run(_run())


class TestCleanup:
    def test_cleanup_expired_closes_only_expired_sessions(self) -> None:
        async def _run() -> None:
            async with BrowserManager() as manager:
                short = await manager.open_session(timeout_seconds=0.01)
                long_lived = await manager.open_session(timeout_seconds=300)
                await asyncio.sleep(0.05)

                expired_ids = await manager.cleanup_expired()

                assert expired_ids == [short.session_id]
                assert short.closed
                assert not long_lived.closed

        asyncio.run(_run())

    def test_close_all_closes_every_session_and_tears_down_the_browser(self) -> None:
        async def _run() -> None:
            manager = BrowserManager()
            session_a = await manager.open_session()
            session_b = await manager.open_session()
            await manager.close_all()
            assert session_a.closed
            assert session_b.closed
            assert manager._browser is None

        asyncio.run(_run())


class TestRealPersistentNavigation:
    def test_navigate_reuses_the_same_page_across_calls(self) -> None:
        """The core B1 claim: this is NOT a browser launched fresh per
        call. The same Page object serves every navigate() on a session."""

        async def _run() -> None:
            with fixture_server() as base_url:
                async with BrowserManager() as manager:
                    session = await manager.open_session()
                    page_before = session.get_page()

                    result1 = await manager.navigate(session.session_id, f"{base_url}/")
                    result2 = await manager.navigate(session.session_id, f"{base_url}/form")

                    assert result1.ok
                    assert result2.ok
                    assert session.get_page() is page_before
                    assert session.navigations == [f"{base_url}/", f"{base_url}/form"]

        asyncio.run(_run())

    def test_cookies_persist_across_navigations_in_the_same_session(self) -> None:
        """Real proof of session persistence (spec section 39's spirit,
        ahead of click/type existing yet): a cookie set by one navigation
        is still sent by the browser on the next navigation in the same
        session, because it's the same BrowserContext throughout."""

        async def _run() -> None:
            with fixture_server() as base_url:
                async with BrowserManager() as manager:
                    session = await manager.open_session()

                    await manager.navigate(session.session_id, f"{base_url}/set-cookie")
                    result = await manager.navigate(session.session_id, f"{base_url}/echo-cookie")

                    assert result.ok
                    assert result.evidence is not None
                    assert "fixture_session=abc123" in result.evidence.excerpt

        asyncio.run(_run())

    def test_two_sessions_do_not_share_cookies(self) -> None:
        """BrowserContext isolation (spec section 32): session A's
        cookie must not leak into session B."""

        async def _run() -> None:
            with fixture_server() as base_url:
                async with BrowserManager() as manager:
                    session_a = await manager.open_session()
                    session_b = await manager.open_session()

                    await manager.navigate(session_a.session_id, f"{base_url}/set-cookie")
                    result_b = await manager.navigate(session_b.session_id, f"{base_url}/echo-cookie")

                    assert result_b.ok
                    assert result_b.evidence is not None
                    assert "fixture_session" not in result_b.evidence.excerpt

        asyncio.run(_run())


class TestNavigationFailureIsolation:
    def test_navigating_unknown_session_fails_explicitly(self) -> None:
        async def _run() -> None:
            async with BrowserManager() as manager:
                result = await manager.navigate("nope", "https://example.com")
                assert not result.ok
                assert "no such" in (result.error or "")

        asyncio.run(_run())

    def test_navigating_a_closed_session_fails_explicitly(self) -> None:
        async def _run() -> None:
            async with BrowserManager() as manager:
                session = await manager.open_session()
                await manager.close_session(session.session_id)
                result = await manager.navigate(session.session_id, "https://example.com")
                assert not result.ok
                assert "closed" in (result.error or "")

        asyncio.run(_run())

    def test_navigating_an_expired_session_fails_explicitly_and_closes_it(self) -> None:
        async def _run() -> None:
            async with BrowserManager() as manager:
                session = await manager.open_session(timeout_seconds=0.01)
                await asyncio.sleep(0.05)
                result = await manager.navigate(session.session_id, "https://example.com")
                assert not result.ok
                assert "timed out" in (result.error or "")
                assert session.closed

        asyncio.run(_run())

    def test_navigating_a_bad_url_becomes_a_result_not_a_crash(self) -> None:
        async def _run() -> None:
            async with BrowserManager() as manager:
                session = await manager.open_session()
                result = await manager.navigate(session.session_id, "http://127.0.0.1:1/unreachable", timeout=3.0)
                assert not result.ok
                assert result.error

        asyncio.run(_run())
