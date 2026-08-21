"""Tests for pi.register_provider()/unregister_provider()."""

from __future__ import annotations

from pathlib import Path

from pi_ai.models import MutableModels, Provider
from pi_coding_agent.extensions.runner import ExtensionRunner
from pi_coding_agent.extensions.types import ExtensionAPI


def _fake_provider(provider_id: str) -> Provider[str]:
    return Provider(id=provider_id, name=provider_id, models=[], stream_fn=lambda *_a, **_k: None)


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


class TestExtensionAPIProviderRegistration:
    def test_register_provider_sets_it_on_models(self) -> None:
        models = MutableModels()
        api = ExtensionAPI(models=models)
        api.register_provider(_fake_provider("my-proxy"))
        assert models.get_provider("my-proxy") is not None

    def test_unregister_provider_removes_it(self) -> None:
        models = MutableModels()
        models.set_provider(_fake_provider("my-proxy"))
        api = ExtensionAPI(models=models)
        api.unregister_provider("my-proxy")
        assert models.get_provider("my-proxy") is None

    def test_no_models_configured_is_a_noop(self) -> None:
        api = ExtensionAPI()  # models=None
        api.register_provider(_fake_provider("my-proxy"))  # does not raise
        api.unregister_provider("my-proxy")  # does not raise


class TestExtensionRunnerProviderRegistration:
    def test_extension_can_register_a_provider_at_load_time(self, tmp_path: Path) -> None:
        source = """
from pi_ai.models import Provider

def _stream(*_a, **_k):
    return None

def extension(pi):
    pi.register_provider(Provider(id="my-proxy", name="My Proxy", models=[], stream_fn=_stream))
"""
        _write(tmp_path / ".pi" / "extensions" / "provider.py", source)
        models = MutableModels()
        runner = ExtensionRunner(tmp_path, tmp_path / "agent", models=models)
        runner.load()

        assert models.get_provider("my-proxy") is not None
        assert runner.get_extensions().errors == []

    def test_extension_can_register_a_provider_from_a_tool_call_later(self, tmp_path: Path) -> None:
        # Mirrors the original's "safe to call from command handlers or
        # event callbacks without requiring a reload" contract.
        import asyncio

        source = """
from pi_agent_core.types import AgentTool, AgentToolResult
from pi_ai import TextContent
from pi_ai.models import Provider

def _stream(*_a, **_k):
    return None

def extension(pi):
    async def _run(_id, _args, _ctx, _cb):
        pi.register_provider(Provider(id="late-proxy", name="Late", models=[], stream_fn=_stream))
        return AgentToolResult(content=[TextContent(text="registered")])

    pi.register_tool(AgentTool(name="setup", description="d", parameters={}, execute=_run))
"""
        _write(tmp_path / ".pi" / "extensions" / "provider.py", source)
        models = MutableModels()
        runner = ExtensionRunner(tmp_path, tmp_path / "agent", models=models)
        runner.load()
        assert models.get_provider("late-proxy") is None

        tool = runner.get_tools()[0]
        assert tool.execute is not None
        execute = tool.execute

        async def _run() -> None:
            await execute("call-1", {}, None, None)

        asyncio.run(_run())
        assert models.get_provider("late-proxy") is not None
