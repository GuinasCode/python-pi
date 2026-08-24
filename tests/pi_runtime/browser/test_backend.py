"""Slice B6 — pluggable backend (playwright_local default, optional CDP)."""

from __future__ import annotations

import asyncio
import socket

import pytest
from fixtures import fixture_server

from pi_runtime.browser import BrowserBackend, BrowserManager
from pi_runtime.browser.backend import BrowserBackendConfigError, launch_browser


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class TestPlaywrightLocalIsDefault:
    def test_default_backend_launches_its_own_browser(self) -> None:
        async def _run() -> None:
            async with BrowserManager() as manager:
                session = await manager.open_session()
                assert not session.closed

        asyncio.run(_run())


class TestCdpBackendRequiresUrl:
    def test_cdp_backend_without_url_raises_config_error(self) -> None:
        async def _run() -> None:
            async with BrowserManager(backend=BrowserBackend.CDP) as manager:
                with pytest.raises(BrowserBackendConfigError):
                    await manager.open_session()

        asyncio.run(_run())


class TestCdpBackendConnectsToARealBrowser:
    def test_connect_over_cdp_to_a_real_chromium_process(self) -> None:
        """Not a mock: launches a real Chromium with remote debugging
        enabled, then proves a second, independent BrowserManager can
        drive it over CDP instead of launching its own browser."""

        async def _run() -> None:
            port = _free_port()
            from playwright.async_api import async_playwright

            async with async_playwright() as pw:
                launcher_browser = await pw.chromium.launch(headless=True, args=[f"--remote-debugging-port={port}"])
                try:
                    cdp_url = f"http://127.0.0.1:{port}"
                    async with BrowserManager(backend=BrowserBackend.CDP, cdp_url=cdp_url) as manager:
                        session = await manager.open_session()
                        with fixture_server() as base_url:
                            result = await manager.navigate(session.session_id, f"{base_url}/")
                            assert result.ok
                finally:
                    await launcher_browser.close()

        asyncio.run(_run())


class TestLaunchBrowserHelper:
    def test_cdp_without_url_raises_before_touching_playwright(self) -> None:
        async def _run() -> None:
            from playwright.async_api import async_playwright

            async with async_playwright() as pw:
                with pytest.raises(BrowserBackendConfigError):
                    await launch_browser(
                        pw, backend=BrowserBackend.CDP, headless=True, cdp_url=None, executable_path=None
                    )

        asyncio.run(_run())
