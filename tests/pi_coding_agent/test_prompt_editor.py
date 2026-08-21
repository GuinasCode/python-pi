"""Tests for pi_coding_agent.prompt_editor.PromptTextArea."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult

from pi_coding_agent.prompt_editor import PromptTextArea


class _HostApp(App[None]):
    def compose(self) -> ComposeResult:
        yield PromptTextArea(id="editor")


class TestPromptTextArea:
    @pytest.mark.asyncio
    async def test_enter_posts_submitted_without_inserting_a_newline(self) -> None:
        app = _HostApp()
        async with app.run_test() as pilot:
            editor = app.query_one("#editor", PromptTextArea)
            editor.text = "hello"
            editor.focus()
            await pilot.press("enter")
            await pilot.pause()
            # Submitted fires with the text as it was; the widget's own
            # content is left untouched by PromptTextArea itself — the
            # host app (PiApp) is what clears it, per the docstring.
            assert editor.text == "hello"

    @pytest.mark.asyncio
    async def test_submitted_message_carries_the_current_text(self) -> None:
        received: list[str] = []

        class _CapturingApp(App[None]):
            def compose(self) -> ComposeResult:
                yield PromptTextArea(id="editor")

            def on_prompt_text_area_submitted(self, event: PromptTextArea.Submitted) -> None:
                received.append(event.value)

        app = _CapturingApp()
        async with app.run_test() as pilot:
            editor = app.query_one("#editor", PromptTextArea)
            editor.text = "what is 2+2?"
            editor.focus()
            await pilot.press("enter")
            await pilot.pause()
            assert received == ["what is 2+2?"]

    @pytest.mark.asyncio
    async def test_ctrl_j_inserts_a_literal_newline_instead_of_submitting(self) -> None:
        received: list[str] = []

        class _CapturingApp(App[None]):
            def compose(self) -> ComposeResult:
                yield PromptTextArea(id="editor")

            def on_prompt_text_area_submitted(self, event: PromptTextArea.Submitted) -> None:
                received.append(event.value)

        app = _CapturingApp()
        async with app.run_test() as pilot:
            editor = app.query_one("#editor", PromptTextArea)
            editor.focus()
            for ch in "line one":
                await pilot.press(ch)
            await pilot.press("ctrl+j")
            for ch in "line two":
                await pilot.press(ch)
            await pilot.pause()

            assert editor.text == "line one\nline two"
            assert received == []  # never submitted
