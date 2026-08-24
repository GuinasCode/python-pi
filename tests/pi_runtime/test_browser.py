"""Tests for pi_runtime.browser.BrowserManager. Covers Fase 5's
acceptance criteria from plan.md section 9:

- browser pode ser criado/encerrado
- navegação é observável
- páginas podem virar Evidence
- falhas não derrubam todo o agente

browser_fetch_url itself is stubbed here (its own Playwright-level
behavior is already covered by tests/pi_coding_agent/test_tools.py's
TestBrowserFetchUrlScreenshot) — these tests are about BrowserManager's
own session/timeout/failure-isolation logic, not Playwright.
"""

from __future__ import annotations

import time

import pytest

from pi_coding_agent.tools import ToolResult
from pi_runtime.browser import BrowserManager
from pi_runtime.tools import PolicyEngine, PolicyViolation, ToolRegistry


def _ok_result(text: str = "page text", *, with_image: bool = False) -> ToolResult:
    content: list[dict[str, str]] = [{"type": "text", "text": text}]
    if with_image:
        content.append({"type": "image", "data": "Zm9v", "mime_type": "image/png"})
    return ToolResult(content=content, details={"url": "https://example.com", "title": "Example"})


def _error_result(message: str = "boom") -> ToolResult:
    return ToolResult(content=[{"type": "text", "text": message}], is_error=True)


class TestSessionLifecycle:
    def test_open_session_returns_a_live_session(self) -> None:
        manager = BrowserManager()
        session = manager.open_session()
        assert not session.closed
        assert manager.get_session(session.session_id) is session

    def test_close_session_marks_it_closed(self) -> None:
        manager = BrowserManager()
        session = manager.open_session()
        assert manager.close_session(session.session_id) is True
        assert session.closed

    def test_closing_twice_is_a_noop_returning_false(self) -> None:
        manager = BrowserManager()
        session = manager.open_session()
        manager.close_session(session.session_id)
        assert manager.close_session(session.session_id) is False

    def test_closing_unknown_session_returns_false(self) -> None:
        manager = BrowserManager()
        assert manager.close_session("nope") is False

    def test_open_session_goes_through_policy_when_configured(self) -> None:
        registry = ToolRegistry()  # "browser" deliberately not registered
        policy = PolicyEngine(registry)
        manager = BrowserManager(policy_engine=policy)
        with pytest.raises(PolicyViolation):
            manager.open_session()

    def test_session_expires_after_its_timeout(self) -> None:
        manager = BrowserManager()
        session = manager.open_session(timeout_seconds=0.01)
        time.sleep(0.02)
        assert session.is_expired()


class TestNavigationIsObservable:
    def test_successful_navigation_is_recorded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import pi_runtime.browser as browser_module

        monkeypatch.setattr(browser_module, "browser_fetch_url", lambda url, **kw: _ok_result())
        manager = BrowserManager()
        session = manager.open_session()

        manager.navigate(session.session_id, "https://example.com")

        assert session.navigations == ["https://example.com"]

    def test_failed_navigation_is_still_recorded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import pi_runtime.browser as browser_module

        monkeypatch.setattr(browser_module, "browser_fetch_url", lambda url, **kw: _error_result())
        manager = BrowserManager()
        session = manager.open_session()

        manager.navigate(session.session_id, "https://broken.example")

        assert session.navigations == ["https://broken.example"]


class TestPagesBecomeEvidence:
    def test_successful_page_becomes_evidence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import pi_runtime.browser as browser_module

        monkeypatch.setattr(browser_module, "browser_fetch_url", lambda url, **kw: _ok_result("hello page"))
        manager = BrowserManager()
        session = manager.open_session()

        result = manager.navigate(session.session_id, "https://example.com")

        assert result.ok
        assert result.evidence is not None
        assert result.evidence.url == "https://example.com"
        assert "hello page" in result.evidence.excerpt
        assert result.evidence.extraction_method == "browser"

    def test_screenshot_block_is_reflected_in_the_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import pi_runtime.browser as browser_module

        monkeypatch.setattr(browser_module, "browser_fetch_url", lambda url, **kw: _ok_result(with_image=True))
        manager = BrowserManager()
        session = manager.open_session()

        result = manager.navigate(session.session_id, "https://example.com", screenshot=True)
        assert result.screenshot_captured is True


class TestFailuresDoNotCrash:
    def test_tool_reported_error_becomes_a_result_not_an_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import pi_runtime.browser as browser_module

        monkeypatch.setattr(browser_module, "browser_fetch_url", lambda url, **kw: _error_result("network down"))
        manager = BrowserManager()
        session = manager.open_session()

        result = manager.navigate(session.session_id, "https://example.com")
        assert not result.ok
        assert "network down" in (result.error or "")

    def test_raised_exception_becomes_a_result_not_a_crash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import pi_runtime.browser as browser_module

        def _boom(url: str, **kw: object) -> ToolResult:
            raise RuntimeError("playwright crashed")

        monkeypatch.setattr(browser_module, "browser_fetch_url", _boom)
        manager = BrowserManager()
        session = manager.open_session()

        result = manager.navigate(session.session_id, "https://example.com")
        assert not result.ok
        assert "playwright crashed" in (result.error or "")

    def test_navigating_a_closed_session_fails_explicitly(self) -> None:
        manager = BrowserManager()
        session = manager.open_session()
        manager.close_session(session.session_id)

        result = manager.navigate(session.session_id, "https://example.com")
        assert not result.ok
        assert "closed" in (result.error or "")

    def test_navigating_an_expired_session_fails_explicitly(self) -> None:
        manager = BrowserManager()
        session = manager.open_session(timeout_seconds=0.01)
        time.sleep(0.02)

        result = manager.navigate(session.session_id, "https://example.com")
        assert not result.ok
        assert "timed out" in (result.error or "")
        assert session.closed  # expiry closes the session as a side effect

    def test_navigating_an_unknown_session_fails_explicitly(self) -> None:
        manager = BrowserManager()
        result = manager.navigate("nope", "https://example.com")
        assert not result.ok
        assert "no such" in (result.error or "")
