"""Tests for pi_coding_agent.subagent."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from pi_coding_agent.subagent.agent_def import discover_agents


class TestAgentDef:
    def test_load_from_project_agents_dir(self, tmp_path: Path) -> None:
        # discover_agents looks in <cwd>/.pi/agents/
        agents_dir = tmp_path / ".pi" / "agents"
        agents_dir.mkdir(parents=True)
        (agents_dir / "scout.md").write_text(
            textwrap.dedent("""\
            ---
            name: scout
            description: Fast recon
            tools: read, grep, bash
            model: gpt-4o
            ---
            You are a scout.
            """)
        )
        agents = discover_agents(tmp_path)
        assert "scout" in agents
        a = agents["scout"]
        assert a.description == "Fast recon"
        assert a.tools == ["read", "grep", "bash"]
        assert a.model == "gpt-4o"
        assert "scout" in a.system_prompt
        assert a.trust_level == "project"

    def test_name_defaults_to_stem(self, tmp_path: Path) -> None:
        agents_dir = tmp_path / ".pi" / "agents"
        agents_dir.mkdir(parents=True)
        (agents_dir / "my-agent.md").write_text("No frontmatter here.")
        agents = discover_agents(tmp_path)
        assert "my-agent" in agents

    def test_tools_as_yaml_list(self, tmp_path: Path) -> None:
        agents_dir = tmp_path / ".pi" / "agents"
        agents_dir.mkdir(parents=True)
        (agents_dir / "agent.md").write_text(
            textwrap.dedent("""\
            ---
            tools:
              - read
              - bash
            ---
            Body.
            """)
        )
        agents = discover_agents(tmp_path)
        assert agents["agent"].tools == ["read", "bash"]

    def test_user_level_overrides_project_level(self, tmp_path: Path) -> None:
        # discover_agents looks in <config_dir>/agents/ for user-level
        project_dir = tmp_path / "project"
        user_config = tmp_path / "config"

        project_agents = project_dir / ".pi" / "agents"
        project_agents.mkdir(parents=True)
        (project_agents / "agent.md").write_text("---\ndescription: project\n---\nProject.")

        user_agents = user_config / "agents"
        user_agents.mkdir(parents=True)
        (user_agents / "agent.md").write_text("---\ndescription: user\n---\nUser.")

        agents = discover_agents(project_dir, user_config)
        assert agents["agent"].description == "user"
        assert agents["agent"].trust_level == "user"

    def test_empty_directory(self, tmp_path: Path) -> None:
        agents = discover_agents(tmp_path, tmp_path)
        assert agents == {}


class TestSubagentToolCreation:
    def test_creates_tool_with_no_agents(self, tmp_path: Path) -> None:
        from pi_coding_agent.subagent.tool import create_subagent_tool

        tool = create_subagent_tool(cwd=str(tmp_path))
        assert tool.name == "subagent"
        assert tool.execute is not None
        assert "none discovered" in tool.description

    def test_creates_tool_lists_discovered_agents(self, tmp_path: Path) -> None:
        from pi_coding_agent.subagent.tool import create_subagent_tool

        agents_dir = tmp_path / ".pi" / "agents"
        agents_dir.mkdir(parents=True)
        (agents_dir / "scout.md").write_text("---\nname: scout\n---\nScout.")

        tool = create_subagent_tool(cwd=str(tmp_path))
        assert "scout" in tool.description

    @pytest.mark.asyncio
    async def test_single_mode_unknown_agent_returns_error(self, tmp_path: Path) -> None:
        from pi_coding_agent.subagent.tool import create_subagent_tool

        tool = create_subagent_tool(cwd=str(tmp_path))
        assert tool.execute is not None
        result = await tool.execute("call-1", {"agent": "nonexistent", "task": "do it"}, None, None)
        text = result.content[0].text
        assert "Unknown agent" in text or "nonexistent" in text
