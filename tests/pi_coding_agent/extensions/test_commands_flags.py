"""Tests for command/flag registration: ExtensionAPI, loader, and ExtensionRunner."""

from __future__ import annotations

import asyncio
from pathlib import Path

from pi_ai import TextContent
from pi_coding_agent.extensions.events import ExtensionContext
from pi_coding_agent.extensions.loader import load_extension_from_path, load_extensions
from pi_coding_agent.extensions.runner import ExtensionRunner
from pi_coding_agent.extensions.types import ExtensionAPI

_COMMAND_EXTENSION = """
def _greet(args_text, ctx):
    return f"hello {args_text} from {ctx.cwd}"

def extension(pi):
    pi.register_command("greet", _greet, description="Greets someone")
"""

_FLAG_EXTENSION = """
def extension(pi):
    pi.register_flag("verbose", type="boolean", default=False, description="Be verbose")
"""

_SHORTCUT_EXTENSION = """
def _on_hotkey(ctx):
    return f"hotkey fired from {ctx.cwd}"

def extension(pi):
    pi.register_shortcut("ctrl+g", _on_hotkey, description="Fire the hotkey")
"""

_THEME_EXTENSION = """
def extension(pi):
    pi.register_theme("midnight", primary="#1e1e2e", dark=True)
"""

_RENDERING_EXTENSION = """
def _shout(text, ctx):
    return text.upper()

def _render_assistant(text, ctx):
    return f"custom: {text}"

def _render_greet(phase, event, ctx):
    if phase == "start":
        return f"starting greet with {event['args']}"
    return f"finished greet: {event['result_text']}"

def extension(pi):
    pi.register_markdown_transformer(_shout)
    pi.register_message_renderer("assistant", _render_assistant)
    pi.register_entry_renderer("greet", _render_greet)
"""


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


class TestExtensionAPICommands:
    def test_register_command_is_collected(self) -> None:
        api = ExtensionAPI()

        def handler(_args: str, _ctx: ExtensionContext) -> str:
            return "ok"

        api.register_command("mycmd", handler, description="does a thing")
        assert len(api.commands) == 1
        assert api.commands[0].name == "mycmd"
        assert api.commands[0].description == "does a thing"


class TestExtensionAPIShortcuts:
    def test_register_shortcut_is_collected(self) -> None:
        api = ExtensionAPI()

        def handler(_ctx: ExtensionContext) -> str:
            return "ok"

        api.register_shortcut("ctrl+g", handler, description="does a thing")
        assert len(api.shortcuts) == 1
        assert api.shortcuts[0].key == "ctrl+g"
        assert api.shortcuts[0].description == "does a thing"


class TestExtensionAPIThemes:
    def test_register_theme_is_collected(self) -> None:
        api = ExtensionAPI()
        api.register_theme("midnight", primary="#1e1e2e", dark=True)
        assert len(api.themes) == 1
        assert api.themes[0].name == "midnight"
        assert api.themes[0].primary == "#1e1e2e"
        assert api.themes[0].dark is True


class TestExtensionAPIRenderingHooks:
    def test_register_markdown_transformer_is_collected(self) -> None:
        api = ExtensionAPI()

        def transform(text: str, _ctx: ExtensionContext) -> str:
            return text.upper()

        api.register_markdown_transformer(transform)
        assert len(api.markdown_transformers) == 1
        assert api.markdown_transformers[0]("hi", ExtensionContext(cwd=".")) == "HI"

    def test_register_message_renderer_is_collected(self) -> None:
        api = ExtensionAPI()

        def render(text: str, _ctx: ExtensionContext) -> str:
            return f"custom: {text}"

        api.register_message_renderer("assistant", render)
        assert "assistant" in api.message_renderers

    def test_register_entry_renderer_is_collected(self) -> None:
        api = ExtensionAPI()

        def render(_phase: str, _event: dict[str, object], _ctx: ExtensionContext) -> str:
            return "handled"

        api.register_entry_renderer("greet", render)
        assert "greet" in api.entry_renderers


class TestExtensionAPIFlags:
    def test_get_flag_returns_declared_default(self) -> None:
        api = ExtensionAPI()
        api.register_flag("verbose", type="boolean", default=False)
        assert api.get_flag("verbose") is False

    def test_get_flag_undeclared_returns_none(self) -> None:
        api = ExtensionAPI()
        assert api.get_flag("nope") is None

    def test_get_flag_reflects_shared_flag_values_dict(self) -> None:
        shared: dict[str, bool | str] = {}
        api = ExtensionAPI(flag_values=shared)
        api.register_flag("mode", type="string", default="a")
        assert api.get_flag("mode") == "a"
        shared["mode"] = "b"  # simulates the runner setting a value later
        assert api.get_flag("mode") == "b"


class TestLoaderCapturesCommandsAndFlags:
    def test_command_extension_is_captured(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "cmd.py", _COMMAND_EXTENSION)
        api = ExtensionAPI()
        error = load_extension_from_path(path, api)
        assert error is None
        assert len(api.commands) == 1
        assert api.commands[0].name == "greet"

    def test_flag_extension_is_captured(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "flag.py", _FLAG_EXTENSION)
        api = ExtensionAPI()
        error = load_extension_from_path(path, api)
        assert error is None
        assert "verbose" in api.flags
        assert api.flags["verbose"].default is False

    def test_load_extensions_aggregates_into_loaded_extension(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "cmd.py", _COMMAND_EXTENSION)
        result = load_extensions([path])
        assert len(result.extensions) == 1
        assert result.extensions[0].commands[0].name == "greet"


class TestLoaderCapturesShortcuts:
    def test_shortcut_extension_is_captured(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "shortcut.py", _SHORTCUT_EXTENSION)
        api = ExtensionAPI()
        error = load_extension_from_path(path, api)
        assert error is None
        assert len(api.shortcuts) == 1
        assert api.shortcuts[0].key == "ctrl+g"

    def test_load_extensions_aggregates_shortcuts_into_loaded_extension(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "shortcut.py", _SHORTCUT_EXTENSION)
        result = load_extensions([path])
        assert len(result.extensions) == 1
        assert result.extensions[0].shortcuts[0].key == "ctrl+g"


class TestExtensionRunnerShortcuts:
    def test_get_shortcuts_aggregates_across_extensions(self, tmp_path: Path) -> None:
        _write(tmp_path / ".pi" / "extensions" / "shortcut.py", _SHORTCUT_EXTENSION)
        runner = ExtensionRunner(tmp_path, tmp_path / "agent")
        runner.load()
        shortcuts = runner.get_shortcuts()
        assert len(shortcuts) == 1
        assert shortcuts[0].key == "ctrl+g"

    def test_shortcut_handler_can_be_invoked(self, tmp_path: Path) -> None:
        _write(tmp_path / ".pi" / "extensions" / "shortcut.py", _SHORTCUT_EXTENSION)
        runner = ExtensionRunner(tmp_path, tmp_path / "agent")
        runner.load()
        shortcut = runner.get_shortcuts()[0]
        result = shortcut.handler(ExtensionContext(cwd=str(tmp_path)))
        assert result == f"hotkey fired from {tmp_path}"

    def test_no_shortcuts_when_nothing_loaded(self, tmp_path: Path) -> None:
        runner = ExtensionRunner(tmp_path, tmp_path / "agent")
        runner.load()
        assert runner.get_shortcuts() == []


class TestLoaderCapturesThemes:
    def test_theme_extension_is_captured(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "theme.py", _THEME_EXTENSION)
        api = ExtensionAPI()
        error = load_extension_from_path(path, api)
        assert error is None
        assert len(api.themes) == 1
        assert api.themes[0].name == "midnight"


class TestExtensionRunnerThemes:
    def test_get_themes_aggregates_across_extensions(self, tmp_path: Path) -> None:
        _write(tmp_path / ".pi" / "extensions" / "theme.py", _THEME_EXTENSION)
        runner = ExtensionRunner(tmp_path, tmp_path / "agent")
        runner.load()
        themes = runner.get_themes()
        assert len(themes) == 1
        assert themes[0].name == "midnight"
        assert themes[0].primary == "#1e1e2e"

    def test_no_themes_when_nothing_loaded(self, tmp_path: Path) -> None:
        runner = ExtensionRunner(tmp_path, tmp_path / "agent")
        runner.load()
        assert runner.get_themes() == []


class TestLoaderCapturesRenderingHooks:
    def test_rendering_extension_is_captured(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "rendering.py", _RENDERING_EXTENSION)
        api = ExtensionAPI()
        error = load_extension_from_path(path, api)
        assert error is None
        assert len(api.markdown_transformers) == 1
        assert "assistant" in api.message_renderers
        assert "greet" in api.entry_renderers


class TestExtensionRunnerRenderingHooks:
    def test_get_markdown_transformers_aggregates_in_load_order(self, tmp_path: Path) -> None:
        _write(tmp_path / ".pi" / "extensions" / "rendering.py", _RENDERING_EXTENSION)
        runner = ExtensionRunner(tmp_path, tmp_path / "agent")
        runner.load()
        transformers = runner.get_markdown_transformers()
        assert len(transformers) == 1
        assert transformers[0]("hi", ExtensionContext(cwd=str(tmp_path))) == "HI"

    def test_get_message_renderer_returns_registered_renderer(self, tmp_path: Path) -> None:
        _write(tmp_path / ".pi" / "extensions" / "rendering.py", _RENDERING_EXTENSION)
        runner = ExtensionRunner(tmp_path, tmp_path / "agent")
        runner.load()
        renderer = runner.get_message_renderer("assistant")
        assert renderer is not None
        assert renderer("hi", ExtensionContext(cwd=str(tmp_path))) == "custom: hi"

    def test_get_message_renderer_returns_none_when_unregistered(self, tmp_path: Path) -> None:
        runner = ExtensionRunner(tmp_path, tmp_path / "agent")
        runner.load()
        assert runner.get_message_renderer("assistant") is None

    def test_get_entry_renderer_returns_registered_renderer(self, tmp_path: Path) -> None:
        _write(tmp_path / ".pi" / "extensions" / "rendering.py", _RENDERING_EXTENSION)
        runner = ExtensionRunner(tmp_path, tmp_path / "agent")
        runner.load()
        renderer = runner.get_entry_renderer("greet")
        assert renderer is not None
        ctx = ExtensionContext(cwd=str(tmp_path))
        assert renderer("start", {"args": {"name": "Bob"}}, ctx) == "starting greet with {'name': 'Bob'}"

    def test_get_entry_renderer_returns_none_for_other_tools(self, tmp_path: Path) -> None:
        _write(tmp_path / ".pi" / "extensions" / "rendering.py", _RENDERING_EXTENSION)
        runner = ExtensionRunner(tmp_path, tmp_path / "agent")
        runner.load()
        assert runner.get_entry_renderer("bash") is None


class TestExtensionRunnerCommandsAndFlags:
    def test_get_commands_aggregates_across_extensions(self, tmp_path: Path) -> None:
        _write(tmp_path / ".pi" / "extensions" / "cmd.py", _COMMAND_EXTENSION)
        runner = ExtensionRunner(tmp_path, tmp_path / "agent")
        runner.load()
        commands = runner.get_commands()
        assert len(commands) == 1
        assert commands[0].name == "greet"

    def test_command_handler_can_be_invoked(self, tmp_path: Path) -> None:
        _write(tmp_path / ".pi" / "extensions" / "cmd.py", _COMMAND_EXTENSION)
        runner = ExtensionRunner(tmp_path, tmp_path / "agent")
        runner.load()
        command = runner.get_commands()[0]
        result = command.handler("Bob", ExtensionContext(cwd=str(tmp_path)))
        assert result == f"hello Bob from {tmp_path}"

    def test_get_flags_aggregates_across_extensions(self, tmp_path: Path) -> None:
        _write(tmp_path / ".pi" / "extensions" / "flag.py", _FLAG_EXTENSION)
        runner = ExtensionRunner(tmp_path, tmp_path / "agent")
        runner.load()
        flags = runner.get_flags()
        assert "verbose" in flags
        assert flags["verbose"].type == "boolean"

    def test_set_flag_value_is_visible_from_a_tool_registered_by_the_same_extension(self, tmp_path: Path) -> None:
        # The flag's value must reach pi.get_flag() called later (e.g. from
        # a tool's execute()), even though the ExtensionAPI instance that
        # ran the factory is long gone by the time set_flag_value() runs.
        source = """
from pi_agent_core.types import AgentTool, AgentToolResult
from pi_ai import TextContent

def extension(pi):
    pi.register_flag("verbose", type="boolean", default=False)

    async def _run(_id, _args, _ctx, _cb):
        return AgentToolResult(content=[TextContent(text=str(pi.get_flag("verbose")))])

    pi.register_tool(AgentTool(name="report", description="d", parameters={}, execute=_run))
"""
        _write(tmp_path / ".pi" / "extensions" / "flagtool.py", source)
        runner = ExtensionRunner(tmp_path, tmp_path / "agent")
        runner.load()

        tool = runner.get_tools()[0]
        assert tool.execute is not None
        execute = tool.execute

        async def _run() -> str:
            result = await execute("call-1", {}, None, None)
            text = result.content[0]
            assert isinstance(text, TextContent)
            return text.text

        assert asyncio.run(_run()) == "False"

        runner.set_flag_value("verbose", True)
        assert asyncio.run(_run()) == "True"

    def test_no_commands_or_flags_when_nothing_loaded(self, tmp_path: Path) -> None:
        runner = ExtensionRunner(tmp_path, tmp_path / "agent")
        runner.load()
        assert runner.get_commands() == []
        assert runner.get_flags() == {}
