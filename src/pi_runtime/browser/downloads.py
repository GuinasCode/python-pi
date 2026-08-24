"""Slice B4 — download artifact provenance (spec section 31): every
download becomes a real file on disk plus path/filename/mime/size/
sha256, computed from what was actually written — not asserted from a
response header we didn't verify against."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from playwright.async_api import Download


@dataclass
class DownloadResult:
    ok: bool
    artifact_path: str | None = None
    filename: str | None = None
    mime_type: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    error: str | None = None


async def save_download(download: Download, *, artifacts_dir: Path) -> DownloadResult:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    filename = download.suggested_filename
    target = artifacts_dir / filename
    try:
        await download.save_as(target)
    except Exception as exc:
        return DownloadResult(ok=False, error=str(exc))

    data = target.read_bytes()
    return DownloadResult(
        ok=True,
        artifact_path=str(target),
        filename=filename,
        mime_type=_guess_mime(filename),
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _guess_mime(filename: str) -> str:
    import mimetypes

    guessed, _ = mimetypes.guess_type(filename)
    return guessed or "application/octet-stream"


__all__ = ["DownloadResult", "save_download"]
