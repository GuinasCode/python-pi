"""Tests for HTML session export."""

from __future__ import annotations

from pathlib import Path

from pi_coding_agent.export_html import render_session_html
from pi_coding_agent.session_manager import SessionEntry, SessionManager


def _entry(seq: int, kind: str, data: dict) -> SessionEntry:
    return SessionEntry(seq=seq, parent_seq=seq - 1 if seq > 0 else None, kind=kind, data=data)


def test_render_session_html_includes_all_roles() -> None:
    entries = [
        _entry(0, "message", {"role": "user", "content": "hello **world**"}),
        _entry(
            1,
            "message",
            {
                "role": "assistant",
                "model": "glm-5.2",
                "content": [
                    {"type": "text", "text": "sure, running a command"},
                    {"type": "toolCall", "id": "t1", "name": "bash", "arguments": {"command": "echo hi"}},
                ],
            },
        ),
        _entry(
            2,
            "message",
            {
                "role": "toolResult",
                "tool_call_id": "t1",
                "tool_name": "bash",
                "is_error": False,
                "content": [{"type": "text", "text": "hi"}],
            },
        ),
    ]

    html = render_session_html(entries, title="My Session")

    assert "<!DOCTYPE html>" in html
    assert "My Session" in html
    assert "world" in html
    assert "sure, running a command" in html
    assert "bash" in html
    assert "hi" in html


def test_render_session_html_skips_non_message_entries() -> None:
    entries = [
        _entry(0, "model_change", {"provider": "nvidia", "model_id": "glm"}),
        _entry(1, "message", {"role": "user", "content": "hi"}),
    ]
    html = render_session_html(entries, title="T")
    assert "hi" in html


def test_export_session_end_to_end(tmp_path: Path) -> None:
    session_mgr = SessionManager(tmp_path / "sessions")
    info = session_mgr.create_session(cwd=str(tmp_path), name="demo")
    session_mgr.append_entry(
        info.id,
        SessionEntry(seq=0, parent_seq=None, kind="message", data={"role": "user", "content": "hello"}),
    )

    entries = session_mgr.get_entries(info.id)
    html = render_session_html(entries, title=info.name or info.id)

    out_file = tmp_path / "out.html"
    out_file.write_text(html, encoding="utf-8")

    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "hello" in content
    assert "demo" in content
