"""Load a file referenced via `@path` attachment syntax.

Both entry points into a conversation go through this one function:
`--print`/`--mode` print mode's `@file` CLI args (parsed into
``Args.file_args`` — previously parsed and never consumed at all) and the
REPL's inline ``@file`` tokens typed into a prompt. An image file becomes
an ``ImageContent`` block passed alongside the prompt; anything else is
read as text and appended into the prompt as a labeled block, so
`@path`-ing a non-image file (a log, a diff, a config) still does
something useful rather than silently failing.
"""

from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from pathlib import Path

from pi_ai import ImageContent

# Generous but bounded — an image this large would blow well past most
# providers' request-body limits anyway; better to fail with a clear
# message here than to send it and get an opaque 413 from the API.
_MAX_IMAGE_BYTES = 10 * 1024 * 1024
_MAX_TEXT_CHARS = 50_000


@dataclass
class Attachment:
    """Exactly one of image/text_block/error is set."""

    path: str
    image: ImageContent | None = None
    text_block: str | None = None
    error: str | None = None


def load_attachment(path: str) -> Attachment:
    """Resolve one `@path` reference. Images are detected by mime type
    (not extension-sniffing content), read as bytes and base64-encoded;
    everything else is read as UTF-8 text (replacing undecodable bytes,
    same as the `read` tool) and wrapped in a small header so the model
    can tell where the attached content starts."""
    file_path = Path(path)
    if not file_path.exists():
        return Attachment(path=path, error=f"attachment not found: {path}")
    if not file_path.is_file():
        return Attachment(path=path, error=f"not a file: {path}")

    mime_type, _ = mimetypes.guess_type(str(file_path))
    if mime_type and mime_type.startswith("image/"):
        try:
            data = file_path.read_bytes()
        except OSError as exc:
            return Attachment(path=path, error=f"error reading {path}: {exc}")
        if len(data) > _MAX_IMAGE_BYTES:
            return Attachment(
                path=path, error=f"{path} is too large to attach ({len(data)} bytes, max {_MAX_IMAGE_BYTES})"
            )
        encoded = base64.b64encode(data).decode("ascii")
        return Attachment(path=path, image=ImageContent(data=encoded, mime_type=mime_type))

    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return Attachment(path=path, error=f"error reading {path}: {exc}")
    if len(text) > _MAX_TEXT_CHARS:
        text = text[:_MAX_TEXT_CHARS] + "\n... (truncated)"
    return Attachment(path=path, text_block=f"--- {path} ---\n{text}\n---")


__all__ = ["Attachment", "load_attachment"]
