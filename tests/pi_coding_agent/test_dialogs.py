"""Tests for pi_coding_agent.dialogs."""

from __future__ import annotations

from typing import Any

import pytest
from textual import work
from textual.app import App, ComposeResult
from textual.widgets import Input, OptionList

from pi_coding_agent.dialogs import ConfirmDialog, InputDialog, SelectDialog


class _HostApp(App[None]):
    """push_screen_wait only works from inside a Textual worker, so the
    test host asks via a @work-decorated method, matching how PiApp itself
    calls it (from a worker) rather than directly from a message handler.
    """

    def compose(self) -> ComposeResult:
        yield from ()

    def __init__(self) -> None:
        super().__init__()
        self.result: Any = None

    @work
    async def ask(self, question: str) -> None:
        self.result = await self.push_screen_wait(ConfirmDialog(question))

    @work
    async def select(self, question: str, choices: list[str]) -> None:
        self.result = await self.push_screen_wait(SelectDialog(question, choices))

    @work
    async def ask_input(self, question: str, default: str = "") -> None:
        self.result = await self.push_screen_wait(InputDialog(question, default))


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


class TestSelectDialog:
    @pytest.mark.asyncio
    async def test_selecting_an_option_dismisses_with_its_text(self) -> None:
        app = _HostApp()
        async with app.run_test() as pilot:
            app.select("Pick a color", ["red", "green", "blue"])
            await pilot.pause()
            assert app.screen.query_one(OptionList).highlighted == 0
            await pilot.press("down", "enter")
            await pilot.pause()
            assert app.result == "green"

    @pytest.mark.asyncio
    async def test_escape_dismisses_with_none(self) -> None:
        app = _HostApp()
        async with app.run_test() as pilot:
            app.select("Pick a color", ["red", "green", "blue"])
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert app.result is None


class TestInputDialog:
    @pytest.mark.asyncio
    async def test_typing_and_enter_dismisses_with_the_text(self) -> None:
        app = _HostApp()
        async with app.run_test() as pilot:
            app.ask_input("What's your name?")
            await pilot.pause()
            app.screen.query_one(Input).value = "Bob"
            await pilot.press("enter")
            await pilot.pause()
            assert app.result == "Bob"

    @pytest.mark.asyncio
    async def test_default_value_is_prefilled(self) -> None:
        app = _HostApp()
        async with app.run_test() as pilot:
            app.ask_input("What's your name?", "Alice")
            await pilot.pause()
            assert app.screen.query_one(Input).value == "Alice"

    @pytest.mark.asyncio
    async def test_escape_dismisses_with_none(self) -> None:
        app = _HostApp()
        async with app.run_test() as pilot:
            app.ask_input("What's your name?")
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert app.result is None
