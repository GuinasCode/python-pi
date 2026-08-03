"""Word navigation utilities.

Provides cursor movement by word boundaries, mirroring the TypeScript
``Intl.Segmenter``-based word segmentation. Segmentation splits text into
runs of word-like characters (letters/digits), whitespace, and other
(punctuation/symbols). Each CJK character is treated as a separate
word-like segment.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

# ASCII punctuation characters used to find internal punctuation boundaries
# within word-like segments (mirrors PUNCTUATION_REGEX in utils.ts).
PUNCTUATION_RE = re.compile(r"[(){}\[\]<>.,;:'\"!?+\-=*/\\|&%^$#@~`]")

# Matches a single CJK ideograph (Han, Hiragana, Katakana, Hangul, Bopomofo).
_CJK_RE = re.compile(
    "[\u3000-\u30ff"  # CJK symbols and punctuation / Hiragana / Katakana
    "\u3400-\u4dbf"  # CJK Extension A
    "\u4e00-\u9fff"  # CJK Unified Ideographs
    "\uac00-\ud7af"  # Hangul Syllables
    "\uf900-\ufaff"  # CJK Compatibility Ideographs
    "\uff00-\uffef"  # Halfwidth/Fullwidth forms
    "\U00020000-\U0002a6df"  # CJK Extension B
    "\U0002a700-\U0002b73f"  # CJK Extension C
    "\U0002b740-\U0002b81f"  # CJK Extension D
    "]"
)

# A word-like character: ASCII letter/digit or any Unicode letter/digit.
_WORD_CHAR_RE = re.compile(r"[0-9A-Za-z]|\w", re.UNICODE)


@dataclass(frozen=True)
class Segment:
    """A single word-segmentation unit."""

    segment: str
    is_word_like: bool


class _WordNavigationOptions:
    """Internal options container for word navigation."""

    def __init__(
        self,
        segment: Callable[[str], Iterable[Segment]] | None = None,
        is_atomic_segment: Callable[[str], bool] | None = None,
    ) -> None:
        self.segment = segment
        self.is_atomic_segment = is_atomic_segment


def _is_whitespace(segment: str) -> bool:
    return len(segment) > 0 and segment.strip() == ""


def segment_text(text: str) -> list[Segment]:
    """Segment ``text`` into word-like, whitespace, and other (punctuation) runs.

    Each CJK character becomes its own word-like segment.
    """
    segments: list[Segment] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        # CJK: emit each CJK char as its own word-like segment.
        if _CJK_RE.match(ch):
            segments.append(Segment(segment=ch, is_word_like=True))
            i += 1
            continue
        # Whitespace run
        if ch.isspace():
            j = i
            while j < n and text[j].isspace() and not _CJK_RE.match(text[j]):
                j += 1
            segments.append(Segment(segment=text[i:j], is_word_like=False))
            i = j
            continue
        # Word-like run (letters/digits, not CJK which is handled above)
        if _WORD_CHAR_RE.match(ch):
            j = i
            while j < n:
                c = text[j]
                if _CJK_RE.match(c) or c.isspace() or not _WORD_CHAR_RE.match(c):
                    break
                j += 1
            segments.append(Segment(segment=text[i:j], is_word_like=True))
            i = j
            continue
        # Other (punctuation/symbols) run
        j = i
        while j < n:
            c = text[j]
            if _CJK_RE.match(c) or c.isspace() or _WORD_CHAR_RE.match(c):
                break
            j += 1
        segments.append(Segment(segment=text[i:j], is_word_like=False))
        i = j
    return segments


def _default_segment(text: str) -> list[Segment]:
    return segment_text(text)


def find_word_backward(
    text: str,
    cursor: int,
    options: _WordNavigationOptions | None = None,
) -> int:
    """Find the cursor position after moving one word backward from ``cursor`` in ``text``.

    Skips trailing whitespace, then stops at the next word/punctuation boundary.
    Pure function - does not mutate any state.
    """
    if cursor <= 0:
        return 0

    text_before_cursor = text[:cursor]
    segment_fn = options.segment if options is not None else None
    is_atomic = options.is_atomic_segment if options is not None else None
    segments = list(segment_fn(text_before_cursor)) if segment_fn is not None else _default_segment(text_before_cursor)

    new_cursor = cursor

    # Skip trailing whitespace
    while (
        len(segments) > 0
        and not (is_atomic(segments[-1].segment) if is_atomic is not None else False)
        and _is_whitespace(segments[-1].segment)
    ):
        seg = segments.pop()
        new_cursor -= len(seg.segment)

    if len(segments) == 0:
        return new_cursor

    last = segments[-1]

    if is_atomic is not None and is_atomic(last.segment):
        # Skip one atomic segment.
        new_cursor -= len(last.segment)
    elif last.is_word_like:
        # Skip inside one word-like segment, preserving ASCII punctuation boundaries.
        segment_str = last.segment
        matches = list(PUNCTUATION_RE.finditer(segment_str))
        if len(matches) <= 0:
            new_cursor -= len(segment_str)
        else:
            last_match = matches[-1]
            new_cursor -= len(segment_str) - (last_match.start() + len(last_match.group(0)))
    else:
        # Skip non-word non-whitespace run (punctuation)
        while (
            len(segments) > 0
            and not (is_atomic(segments[-1].segment) if is_atomic is not None else False)
            and not segments[-1].is_word_like
            and not _is_whitespace(segments[-1].segment)
        ):
            seg = segments.pop()
            new_cursor -= len(seg.segment)

    return new_cursor


def find_word_forward(
    text: str,
    cursor: int,
    options: _WordNavigationOptions | None = None,
) -> int:
    """Find the cursor position after moving one word forward from ``cursor`` in ``text``.

    Skips leading whitespace, then stops at the next word/punctuation boundary.
    Pure function - does not mutate any state.
    """
    if cursor >= len(text):
        return len(text)

    text_after_cursor = text[cursor:]
    segment_fn = options.segment if options is not None else None
    is_atomic = options.is_atomic_segment if options is not None else None
    segments = list(segment_fn(text_after_cursor)) if segment_fn is not None else _default_segment(text_after_cursor)

    idx = 0
    new_cursor = cursor

    # Skip leading whitespace
    while idx < len(segments) and not (is_atomic(segments[idx].segment) if is_atomic is not None else False):
        if not _is_whitespace(segments[idx].segment):
            break
        new_cursor += len(segments[idx].segment)
        idx += 1

    if idx >= len(segments):
        return new_cursor

    seg = segments[idx]

    if is_atomic is not None and is_atomic(seg.segment):
        # Skip one atomic segment.
        new_cursor += len(seg.segment)
    elif seg.is_word_like:
        # Skip inside one word-like segment, preserving ASCII punctuation boundaries.
        m = PUNCTUATION_RE.search(seg.segment)
        new_cursor += m.start() if m is not None else len(seg.segment)
    else:
        # Skip non-word non-whitespace run (punctuation)
        while idx < len(segments) and not (is_atomic(segments[idx].segment) if is_atomic is not None else False):
            if segments[idx].is_word_like or _is_whitespace(segments[idx].segment):
                break
            new_cursor += len(segments[idx].segment)
            idx += 1

    return new_cursor
