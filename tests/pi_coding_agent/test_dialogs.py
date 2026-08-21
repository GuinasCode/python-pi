"""Tests for pi_coding_agent.dialogs.ConfirmDialog."""

from __future__ import annotations

import pytest
from textual import work
from textual.app import App, ComposeResult

from pi_coding_agent.dialogs import ConfirmDialog


class _HostApp(App[None]):
    """push_screen_wait only works from inside a Textual worker, so the
    test host asks via a @work-decorated method, matching how PiApp itself
    calls it (from a worker) rather than directly from a message handler.
    """

    def compose(self) -> ComposeResult:
        yield from ()

    def __init__(self) -> None:
        super().__init__()
        self.result: bool | None = None

    @work
    async def ask(self, question: str) -> None:
        self.result = await self.push_screen_wait(ConfirmDialog(question))


class TestConfirmDialog:
    @pytest.mark.asyncio
    async def test_pressing_y_dismisses_with_true(self) -> None:
        app = _HostApp()
        async with app.run_test() as pilot:
            app.ask("Allow bash(ls)?")
            await pilot.pause()
            await pilot.press("y")
            await pilot.pause()
            assert app.result is True

    @pytest.mark.asyncio
    async def test_pressing_n_dismisses_with_false(self) -> None:
        app = _HostApp()
        async with app.run_test() as pilot:
            app.ask("Allow bash(ls)?")
            await pilot.pause()
            await pilot.press("n")
            await pilot.pause()
            assert app.result is False

    @pytest.mark.asyncio
    async def test_pressing_escape_dismisses_with_false(self) -> None:
        app = _HostApp()
        async with app.run_test() as pilot:
            app.ask("Allow bash(ls)?")
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert app.result is False
