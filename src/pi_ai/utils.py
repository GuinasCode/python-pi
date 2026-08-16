"""Utility functions for the pi_ai package, mirroring packages/ai/src/utils/."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import unicodedata
import uuid as _uuid
from collections.abc import Callable, Coroutine
from typing import Any


def uuid() -> str:
    """Generate a v4 UUID string."""
    return str(_uuid.uuid4())


_background_tasks: set[asyncio.Task[Any]] = set()


def spawn_background_task(coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
    """Schedule *coro* on the running loop, keeping a strong reference alive.

    ``asyncio.ensure_future(coro)`` without holding onto the returned task
    lets the task be garbage-collected mid-flight (CPython only keeps a weak
    reference internally), which can silently kill an in-progress stream.
    Retaining it in a module-level set until it completes avoids that.
    """
    task = asyncio.ensure_future(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


def describe_exception(exc: BaseException) -> str:
    """A human-readable, never-empty description of *exc*.

    Some exceptions (e.g. connection-level httpx errors) stringify to "" —
    falling back to the exception's class name keeps error messages
    non-empty so callers don't need a second ``or "Unknown error"`` guard.
    """
    return str(exc) or exc.__class__.__name__


def hash_string(value: str) -> str:
    """SHA-256 hex digest of a string."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sanitize_unicode(text: str) -> str:
    """Sanitize Unicode by normalizing and removing lone surrogates."""
    if not text:
        return text
    return unicodedata.normalize("NFC", text)


def is_whitespace_char(char: str) -> bool:
    """Check if a character is whitespace."""
    return char.isspace() if char else False


PUNCTUATION_REGEX = r"[^\w\s]"

_WORD_BOUNDARY_RE = re.compile(r"[\s\-_./:]")


def get_word_segmenter() -> Callable[[str], list[tuple[str, int, bool]]]:
    """Return a segmenter function that splits text into (segment, index, is_word_like) tuples."""
    return segment_text


_CJK_RANGES = [
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F),
    (0x2B820, 0x2CEAF),
    (0xF900, 0xFAFF),
    (0x2F800, 0x2FA1F),
    (0x3000, 0x303F),
    (0x3040, 0x309F),
    (0x30A0, 0x30FF),
    (0xFF00, 0xFFEF),
]


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return any(low <= code <= high for low, high in _CJK_RANGES)


def segment_text(text: str) -> list[tuple[str, int, bool]]:
    """Segment text into word-like, whitespace, and punctuation runs.

    Returns list of (segment, index, is_word_like) tuples.
    CJK characters are each treated as separate word-like segments.
    """
    segments: list[tuple[str, int, bool]] = []
    i = 0
    while i < len(text):
        char = text[i]
        if _is_cjk(char):
            segments.append((char, i, True))
            i += 1
            continue
        if char.isspace():
            j = i
            while j < len(text) and text[j].isspace() and not _is_cjk(text[j]):
                j += 1
            segments.append((text[i:j], i, False))
            i = j
            continue
        start = i
        j = i
        while j < len(text) and not text[j].isspace() and not _is_cjk(text[j]):
            j += 1
        segments.append((text[start:j], start, True))
        i = j
    return segments


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """Rough token estimate from character count."""
    if not text:
        return 0
    return max(1, round(len(text) / chars_per_token))


def json_parse_partial(text: str) -> Any:
    """Parse potentially incomplete JSON by closing open strings, brackets, and braces."""
    if not text.strip():
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    text = text.strip()
    open_braces = 0
    close_braces = 0
    open_brackets = 0
    close_brackets = 0
    in_string = False
    escape = False
    for char in text:
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            open_braces += 1
        elif char == "}":
            close_braces += 1
        elif char == "[":
            open_brackets += 1
        elif char == "]":
            close_brackets += 1
    if in_string:
        text = text + '"'
    text = text + ("]" * (open_brackets - close_brackets))
    text = text + ("}" * (open_braces - close_braces))
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
