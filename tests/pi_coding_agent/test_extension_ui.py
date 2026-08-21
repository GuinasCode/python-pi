"""Tests for pi_coding_agent.extension_ui.NoopExtensionUIContext."""

from __future__ import annotations

import asyncio

from pi_coding_agent.extension_ui import NoopExtensionUIContext


class TestNoopExtensionUIContext:
    def test_select_resolves_to_none(self) -> None:
        ctx = NoopExtensionUIContext()
        assert asyncio.run(ctx.select("pick one", ["a", "b"])) is None

    def test_confirm_resolves_to_false(self) -> None:
        ctx = NoopExtensionUIContext()
        assert asyncio.run(ctx.confirm("are you sure?")) is False

    def test_input_resolves_to_none(self) -> None:
        ctx = NoopExtensionUIContext()
        assert asyncio.run(ctx.input("what's your name?")) is None

    def test_notify_does_not_raise(self) -> None:
        NoopExtensionUIContext().notify("hello")

    def test_widget_slot_setters_do_not_raise(self) -> None:
        ctx = NoopExtensionUIContext()
        ctx.set_header("hi")
        ctx.set_footer("hi")
        ctx.set_title("hi")
        ctx.set_widget("hi")
        ctx.set_header(None)
        ctx.set_footer(None)
        ctx.set_title(None)
        ctx.set_widget(None)
