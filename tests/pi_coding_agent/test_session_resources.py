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


def _add_user_message(mgr: SessionManager, session_id: str, text: str, *, seq: int = 0) -> None:
    mgr.append_entry(
        session_id,
        SessionEntry(seq=seq, parent_seq=None, kind="message", data={"role": "user", "content": text}),
    )


def _add_assistant_message(mgr: SessionManager, session_id: str, text: str, *, seq: int = 1) -> None:
    mgr.append_entry(
        session_id,
        SessionEntry(
            seq=seq,
            parent_seq=seq - 1,
            kind="message",
            data={"role": "assistant", "content": [{"type": "text", "text": text}]},
        ),
    )


class TestSessionSearch:
    def test_empty_query_returns_nothing(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        mgr.create_session(name="a")
        assert mgr.search_sessions("") == []
        assert mgr.search_sessions("   ") == []

    def test_no_sessions_returns_nothing(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        assert mgr.search_sessions("anything") == []

    def test_matches_by_session_name(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        info = mgr.create_session(name="refactor-auth")
        mgr.create_session(name="unrelated")
        results = mgr.search_sessions("refactor")
        assert [r.info.id for r in results] == [info.id]

    def test_matches_by_cwd(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        info = mgr.create_session(cwd="/home/user/projects/pi-cli", name="x")
        mgr.create_session(cwd="/home/user/projects/other", name="y")
        results = mgr.search_sessions("pi-cli")
        assert [r.info.id for r in results] == [info.id]

    def test_matches_by_message_content(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        info = mgr.create_session(name="chat")
        _add_user_message(mgr, info.id, "how do I refactor the auth middleware?")
        results = mgr.search_sessions("middleware")
        assert [r.info.id for r in results] == [info.id]

    def test_matches_assistant_text_blocks(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        info = mgr.create_session(name="chat")
        _add_assistant_message(mgr, info.id, "Use sqlite-vec for hybrid search.")
        results = mgr.search_sessions("sqlite-vec")
        assert [r.info.id for r in results] == [info.id]

    def test_multiple_words_require_all_to_match(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        both = mgr.create_session(name="chat")
        _add_user_message(mgr, both.id, "refactor the auth middleware please")
        only_one = mgr.create_session(name="chat2")
        _add_user_message(mgr, only_one.id, "refactor the database layer")
        results = mgr.search_sessions("refactor auth")
        assert [r.info.id for r in results] == [both.id]

    def test_case_insensitive(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        info = mgr.create_session(name="Refactor AUTH")
        results = mgr.search_sessions("refactor auth")
        assert [r.info.id for r in results] == [info.id]

    def test_no_match_returns_empty(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        mgr.create_session(name="chat")
        assert mgr.search_sessions("nonexistent-topic-xyz") == []

    def test_ranked_by_match_count_then_recency(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        weak = mgr.create_session(name="weak")
        _add_user_message(mgr, weak.id, "auth")
        strong = mgr.create_session(name="strong")
        _add_user_message(mgr, strong.id, "auth auth auth auth")
        results = mgr.search_sessions("auth")
        assert [r.info.id for r in results] == [strong.id, weak.id]

    def test_result_includes_snippet(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        info = mgr.create_session(name="chat")
        _add_user_message(mgr, info.id, "please refactor the auth middleware")
        results = mgr.search_sessions("middleware")
        assert results[0].snippet
        assert "middleware" in results[0].snippet.lower()

    def test_limit_caps_results(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        for i in range(5):
            mgr.create_session(name=f"auth-session-{i}")
        results = mgr.search_sessions("auth", limit=2)
        assert len(results) == 2


class TestResolveSessionRef:
    def test_exact_id_match(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        info = mgr.create_session(name="a")
        resolved = mgr.resolve_session_ref(info.id)
        assert resolved is not None
        assert resolved.id == info.id

    def test_unambiguous_prefix_match(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        info = mgr.create_session(name="a")
        resolved = mgr.resolve_session_ref(info.id[:4])
        assert resolved is not None
        assert resolved.id == info.id

    def test_ambiguous_prefix_returns_none(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        # Force two ids sharing a prefix by writing session files directly
        # under controlled names rather than relying on random uuids to
        # collide.
        (tmp_path / "abc111111111.jsonl").write_text(
            '{"type": "session_info", "id": "abc111111111", "name": "one", "cwd": "/", '
            '"created_at": 1, "updated_at": 1}\n',
            encoding="utf-8",
        )
        (tmp_path / "abc222222222.jsonl").write_text(
            '{"type": "session_info", "id": "abc222222222", "name": "two", "cwd": "/", '
            '"created_at": 2, "updated_at": 2}\n',
            encoding="utf-8",
        )
        assert mgr.resolve_session_ref("abc") is None

    def test_no_match_returns_none(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        mgr.create_session(name="a")
        assert mgr.resolve_session_ref("doesnotexist") is None


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
