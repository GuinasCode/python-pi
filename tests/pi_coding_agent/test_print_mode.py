"""Tests for print mode."""

from __future__ import annotations

import asyncio

from pi_coding_agent import parse_args
from pi_coding_agent.print_mode import run_print_mode


def test_print_mode_text(capsys: object) -> None:
    args = parse_args(["-p", "hello"])
    result = asyncio.run(run_print_mode(args))
    assert result == 0


def test_print_mode_json(capsys: object) -> None:
    args = parse_args(["--mode", "json", "test prompt"])
    result = asyncio.run(run_print_mode(args))
    assert result == 0


def test_print_mode_no_prompt() -> None:
    args = parse_args(["-p"])
    result = asyncio.run(run_print_mode(args))
    assert result == 1


class TestResolvePromptAndAttachments:
    def test_no_prompt_no_attachments_returns_none(self) -> None:
        from pi_coding_agent.print_mode import _resolve_prompt_and_attachments

        args = parse_args(["-p"])
        assert _resolve_prompt_and_attachments(args) is None

    def test_plain_prompt_with_no_attachments(self) -> None:
        from pi_coding_agent.print_mode import _resolve_prompt_and_attachments

        args = parse_args(["-p", "hello", "world"])
        resolved = _resolve_prompt_and_attachments(args)
        assert resolved is not None
        prompt, images = resolved
        assert prompt == "hello world"
        assert images == []

    def test_image_attachment_becomes_an_image_content_block(self, tmp_path: object) -> None:
        from pathlib import Path

        from pi_coding_agent.print_mode import _resolve_prompt_and_attachments

        image_path = Path(str(tmp_path)) / "photo.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\nfakebytes")

        args = parse_args(["-p", f"@{image_path}", "what", "is", "this?"])
        resolved = _resolve_prompt_and_attachments(args)
        assert resolved is not None
        prompt, images = resolved
        assert prompt == "what is this?"
        assert len(images) == 1
        assert images[0].mime_type == "image/png"

    def test_image_only_no_text_prompt_is_still_valid(self, tmp_path: object) -> None:
        from pathlib import Path

        from pi_coding_agent.print_mode import _resolve_prompt_and_attachments

        image_path = Path(str(tmp_path)) / "photo.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\nfakebytes")

        args = parse_args(["-p", f"@{image_path}"])
        resolved = _resolve_prompt_and_attachments(args)
        assert resolved is not None
        prompt, images = resolved
        assert prompt == ""
        assert len(images) == 1

    def test_text_attachment_is_folded_into_the_prompt(self, tmp_path: object) -> None:
        from pathlib import Path

        from pi_coding_agent.print_mode import _resolve_prompt_and_attachments

        notes_path = Path(str(tmp_path)) / "notes.txt"
        notes_path.write_text("the important bit")

        args = parse_args(["-p", "summarize", f"@{notes_path}"])
        resolved = _resolve_prompt_and_attachments(args)
        assert resolved is not None
        prompt, images = resolved
        assert "summarize" in prompt
        assert "the important bit" in prompt
        assert images == []

    def test_missing_attachment_reports_error_and_returns_none(self, tmp_path: object, capsys: object) -> None:
        from pathlib import Path

        from pi_coding_agent.print_mode import _resolve_prompt_and_attachments

        missing = Path(str(tmp_path)) / "nope.png"
        args = parse_args(["-p", "look at", f"@{missing}"])
        assert _resolve_prompt_and_attachments(args) is None
