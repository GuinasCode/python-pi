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

    def test_theme_getter_setter_do_not_raise_and_get_returns_none(self) -> None:
        ctx = NoopExtensionUIContext()
        ctx.set_theme("midnight")
        assert ctx.get_theme() is None

    def test_tools_expanded_is_real_in_memory_state(self) -> None:
        ctx = NoopExtensionUIContext()
        assert ctx.get_tools_expanded() is True
        ctx.set_tools_expanded(False)
        assert ctx.get_tools_expanded() is False

    def test_add_autocomplete_provider_does_not_raise(self) -> None:
        NoopExtensionUIContext().add_autocomplete_provider(lambda text: ["x"])

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
