"""Tests for pi_coding_agent.extensions.runner.ExtensionRunner."""

from __future__ import annotations

from pathlib import Path

from pi_coding_agent.extensions.runner import ExtensionRunner

_HELLO_EXTENSION = """
from pi_agent_core.types import AgentTool

def extension(pi):
    pi.register_tool(AgentTool(name="hello", description="Greets someone", parameters={}))
"""


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


class TestExtensionRunner:
    def test_no_extensions_before_load(self, tmp_path: Path) -> None:
        runner = ExtensionRunner(tmp_path / "project", tmp_path / "agent")
        assert runner.get_extensions().extensions == []
        assert runner.get_tools() == []
        assert runner.get_extension_paths() == []

    def test_load_discovers_and_registers_tools(self, tmp_path: Path) -> None:
        cwd = tmp_path / "project"
        agent_dir = tmp_path / "agent"
        _write(cwd / ".pi" / "extensions" / "hello.py", _HELLO_EXTENSION)

        runner = ExtensionRunner(cwd, agent_dir)
        result = runner.load()

        assert len(result.extensions) == 1
        assert [t.name for t in runner.get_tools()] == ["hello"]
        assert len(runner.get_extension_paths()) == 1

    def test_reload_picks_up_a_newly_created_extension(self, tmp_path: Path) -> None:
        cwd = tmp_path / "project"
        agent_dir = tmp_path / "agent"
        runner = ExtensionRunner(cwd, agent_dir)
        runner.load()
        assert runner.get_tools() == []

        _write(cwd / ".pi" / "extensions" / "hello.py", _HELLO_EXTENSION)
        runner.load()
        assert [t.name for t in runner.get_tools()] == ["hello"]

    def test_broken_extension_is_reported_as_error(self, tmp_path: Path) -> None:
        cwd = tmp_path / "project"
        agent_dir = tmp_path / "agent"
        _write(cwd / ".pi" / "extensions" / "broken.py", "raise RuntimeError('boom')\n")

        runner = ExtensionRunner(cwd, agent_dir)
        result = runner.load()
        assert result.extensions == []
        assert len(result.errors) == 1
        assert "boom" in result.errors[0].error
