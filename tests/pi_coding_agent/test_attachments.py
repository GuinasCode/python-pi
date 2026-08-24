"""Tests for pi_coding_agent.attachments — `@path` resolution shared by
print mode's `--file`/`@path` CLI args and the REPL's inline `@path`
tokens."""

from __future__ import annotations

import base64
from pathlib import Path

from pi_coding_agent.attachments import load_attachment


class TestLoadAttachment:
    def test_missing_file_is_an_error(self, tmp_path: Path) -> None:
        att = load_attachment(str(tmp_path / "nonexistent.png"))
        assert att.error is not None
        assert att.image is None
        assert att.text_block is None

    def test_directory_is_an_error(self, tmp_path: Path) -> None:
        att = load_attachment(str(tmp_path))
        assert att.error is not None
        assert "not a file" in att.error

    def test_image_file_becomes_image_content(self, tmp_path: Path) -> None:
        file_path = tmp_path / "photo.png"
        file_path.write_bytes(b"\x89PNG\r\n\x1a\nfake-but-good-enough-bytes")

        att = load_attachment(str(file_path))
        assert att.error is None
        assert att.text_block is None
        assert att.image is not None
        assert att.image.mime_type == "image/png"
        assert base64.b64decode(att.image.data) == file_path.read_bytes()

    def test_oversized_image_is_an_error(self, tmp_path: Path) -> None:
        import pi_coding_agent.attachments as attachments_module

        file_path = tmp_path / "huge.png"
        file_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        original_max = attachments_module._MAX_IMAGE_BYTES
        attachments_module._MAX_IMAGE_BYTES = 10
        try:
            att = load_attachment(str(file_path))
        finally:
            attachments_module._MAX_IMAGE_BYTES = original_max

        assert att.error is not None
        assert "too large" in att.error
        assert att.image is None

    def test_text_file_becomes_a_labeled_text_block(self, tmp_path: Path) -> None:
        file_path = tmp_path / "notes.txt"
        file_path.write_text("important context")

        att = load_attachment(str(file_path))
        assert att.error is None
        assert att.image is None
        assert att.text_block is not None
        assert "important context" in att.text_block
        assert str(file_path) in att.text_block

    def test_long_text_file_is_truncated(self, tmp_path: Path) -> None:
        import pi_coding_agent.attachments as attachments_module

        file_path = tmp_path / "long.txt"
        file_path.write_text("x" * 1000)

        original_max = attachments_module._MAX_TEXT_CHARS
        attachments_module._MAX_TEXT_CHARS = 100
        try:
            att = load_attachment(str(file_path))
        finally:
            attachments_module._MAX_TEXT_CHARS = original_max

        assert att.text_block is not None
        assert "truncated" in att.text_block
        assert len(att.text_block) < 1000
