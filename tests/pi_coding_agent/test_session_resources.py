"""Tests for session_manager and resource_loader."""

from __future__ import annotations

from pathlib import Path

from pi_coding_agent.resource_loader import (
    load_append_system_prompt,
    load_project_context_files,
    load_resources,
    load_skills,
    load_system_prompt,
)
from pi_coding_agent.session_manager import SessionEntry, SessionManager


class TestSessionManager:
    def test_create_session(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        info = mgr.create_session(cwd="/tmp", name="test")
        assert info.id != ""
        assert info.name == "test"
        assert info.cwd == "/tmp"
        assert (tmp_path / f"{info.id}.jsonl").exists()

    def test_open_session(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        created = mgr.create_session(name="test")
        opened = mgr.open_session(created.id)
        assert opened is not None
        assert opened.id == created.id
        assert opened.name == "test"

    def test_open_nonexistent(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        assert mgr.open_session("nonexistent") is None

    def test_list_sessions(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        mgr.create_session(name="s1")
        mgr.create_session(name="s2")
        sessions = mgr.list_sessions()
        assert len(sessions) == 2

    def test_delete_session(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        info = mgr.create_session()
        assert mgr.delete_session(info.id) is True
        assert mgr.open_session(info.id) is None
        assert mgr.delete_session("nonexistent") is False

    def test_append_and_get_entries(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        info = mgr.create_session()
        entry1 = SessionEntry(seq=0, parent_seq=None, kind="message", data={"text": "hello"})
        entry2 = SessionEntry(seq=1, parent_seq=0, kind="message", data={"text": "world"})
        mgr.append_entry(info.id, entry1)
        mgr.append_entry(info.id, entry2)
        entries = mgr.get_entries(info.id)
        assert len(entries) == 2
        assert entries[0].data["text"] == "hello"
        assert entries[1].data["text"] == "world"

    def test_continue_recent(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        mgr.create_session(name="s1")
        mgr.create_session(name="s2")
        recent = mgr.continue_recent()
        assert recent is not None
        assert recent.name in ("s1", "s2")

    def test_fork_session(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        original = mgr.create_session(name="original")
        entry = SessionEntry(seq=0, parent_seq=None, kind="message", data={"text": "hello"})
        mgr.append_entry(original.id, entry)
        forked = mgr.fork_session(original.id, name="fork")
        assert forked is not None
        assert forked.id != original.id
        forked_entries = mgr.get_entries(forked.id)
        assert len(forked_entries) == 1
        assert forked_entries[0].data["text"] == "hello"


class TestResourceLoader:
    def test_load_project_context_files(self, tmp_path: Path) -> None:
        agents_file = tmp_path / "AGENTS.md"
        agents_file.write_text("# Project Rules\nBe concise.")
        files = load_project_context_files(tmp_path)
        assert len(files) == 1
        assert "Be concise" in files[0].content

    def test_load_project_context_files_parent_dir(self, tmp_path: Path) -> None:
        parent = tmp_path / "parent"
        child = parent / "child"
        child.mkdir(parents=True)
        agents_file = parent / "AGENTS.md"
        agents_file.write_text("# Parent Rules")
        files = load_project_context_files(child)
        assert len(files) >= 1
        assert any("Parent Rules" in f.content for f in files)

    def test_load_system_prompt_project(self, tmp_path: Path) -> None:
        system_file = tmp_path / "SYSTEM.md"
        system_file.write_text("You are a code reviewer.")
        result = load_system_prompt(tmp_path)
        assert result is not None
        assert "code reviewer" in result

    def test_load_system_prompt_config_dir(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        system_file = config_dir / "SYSTEM.md"
        system_file.write_text("You are a test agent.")
        result = load_system_prompt("/nonexistent", config_dir)
        assert result is not None
        assert "test agent" in result

    def test_load_system_prompt_none(self, tmp_path: Path) -> None:
        assert load_system_prompt(tmp_path) is None

    def test_load_append_system_prompt(self, tmp_path: Path) -> None:
        append_file = tmp_path / "APPEND_SYSTEM.md"
        append_file.write_text("Always use type hints.")
        result = load_append_system_prompt(tmp_path)
        assert result is not None
        assert "type hints" in result

    def test_load_skills(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "python"
        skill_dir.mkdir(parents=True)
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text("---\ndescription: Python coding skill\n---\n# Python Skill\nContent here.")
        skills = load_skills(tmp_path)
        assert len(skills) == 1
        assert skills[0].name == "python"
        assert skills[0].description == "Python coding skill"

    def test_load_skills_empty(self, tmp_path: Path) -> None:
        assert load_skills(tmp_path) == []

    def test_load_resources(self, tmp_path: Path) -> None:
        agents_file = tmp_path / "AGENTS.md"
        agents_file.write_text("# Rules")
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        system_file = config_dir / "SYSTEM.md"
        system_file.write_text("You are helpful.")
        resources = load_resources(tmp_path, config_dir)
        assert len(resources.context_files) >= 1
        assert resources.system_prompt is not None
        assert "helpful" in resources.system_prompt
