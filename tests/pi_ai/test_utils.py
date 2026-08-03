"""Tests for pi_ai utils module."""

from __future__ import annotations

from pi_ai.utils import (
    estimate_tokens,
    get_word_segmenter,
    hash_string,
    is_whitespace_char,
    json_parse_partial,
    sanitize_unicode,
    segment_text,
    uuid,
)


def test_uuid_returns_valid_v4_string() -> None:
    result = uuid()
    assert isinstance(result, str)
    assert len(result) == 36
    assert result.count("-") == 4


def test_hash_string_returns_sha256_hex() -> None:
    result = hash_string("hello")
    assert isinstance(result, str)
    assert len(result) == 64
    assert all(c in "0123456789abcdef" for c in result)


def test_sanitize_unicode_normalizes_empty() -> None:
    assert sanitize_unicode("") == ""


def test_sanitize_unicode_normalizes_text() -> None:
    assert sanitize_unicode("hello") == "hello"


def test_is_whitespace_char_detects_space() -> None:
    assert is_whitespace_char(" ") is True
    assert is_whitespace_char("\t") is True
    assert is_whitespace_char("\n") is True
    assert is_whitespace_char("a") is False
    assert is_whitespace_char("") is False


def test_get_word_segmenter_returns_callable() -> None:
    segmenter = get_word_segmenter()
    assert callable(segmenter)


def test_segment_text_basic_words() -> None:
    segments = segment_text("hello world")
    assert len(segments) == 3
    assert segments[0] == ("hello", 0, True)
    assert segments[1] == (" ", 5, False)
    assert segments[2] == ("world", 6, True)


def test_segment_text_punctuation() -> None:
    segments = segment_text("foo.bar")
    assert len(segments) == 1
    assert segments[0] == ("foo.bar", 0, True)


def test_segment_text_cjk() -> None:
    segments = segment_text("你好")
    assert len(segments) == 2
    assert segments[0] == ("你", 0, True)
    assert segments[1] == ("好", 1, True)


def test_segment_text_mixed() -> None:
    segments = segment_text("你好 world")
    assert len(segments) == 4
    assert segments[0] == ("你", 0, True)
    assert segments[1] == ("好", 1, True)
    assert segments[2] == (" ", 2, False)
    assert segments[3] == ("world", 3, True)


def test_segment_text_empty() -> None:
    assert segment_text("") == []


def test_estimate_tokens_basic() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("hello") == 1
    assert estimate_tokens("a" * 100) == 25


def test_json_parse_partial_valid_json() -> None:
    assert json_parse_partial('{"key": "value"}') == {"key": "value"}


def test_json_parse_partial_empty() -> None:
    assert json_parse_partial("") is None
    assert json_parse_partial("   ") is None


def test_json_parse_partial_incomplete_object() -> None:
    result = json_parse_partial('{"key": "val')
    assert result == {"key": "val"}


def test_json_parse_partial_incomplete_array() -> None:
    result = json_parse_partial("[1, 2, 3")
    assert result == [1, 2, 3]


def test_json_parse_partial_completely_invalid() -> None:
    assert json_parse_partial("not json at all") is None
