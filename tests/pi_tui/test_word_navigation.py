from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from pi_tui.word_navigation import Segment, find_word_backward, find_word_forward


class TestFindWordBackward:
    def test_basic_words_hello_world(self) -> None:
        text = "hello world"
        assert find_word_backward(text, 11) == 6
        assert find_word_backward(text, 6) == 0

    def test_dotted_foo_bar(self) -> None:
        text = "foo.bar"
        assert find_word_backward(text, 7) == 4
        assert find_word_backward(text, 4) == 3
        assert find_word_backward(text, 3) == 0

    def test_colon_foo_bar(self) -> None:
        text = "foo:bar"
        assert find_word_backward(text, 7) == 4
        assert find_word_backward(text, 4) == 3
        assert find_word_backward(text, 3) == 0

    def test_path_path_to_file(self) -> None:
        text = "path/to/file"
        assert find_word_backward(text, 12) == 8
        assert find_word_backward(text, 8) == 7
        # "/to" is one word-like segment with "/" as punctuation boundary
        assert find_word_backward(text, 7) == 5
        assert find_word_backward(text, 5) == 4
        assert find_word_backward(text, 4) == 0

    def test_cjk_mixed(self) -> None:
        text = "你好世界 test"
        assert find_word_backward(text, len(text)) == 5
        # Each CJK char is a separate word-like segment (per-char heuristic).
        assert find_word_backward(text, 5) == 3
        assert find_word_backward(text, 3) == 2
        assert find_word_backward(text, 2) == 1
        assert find_word_backward(text, 1) == 0

    def test_whitespace_at_boundaries(self) -> None:
        text = "  hello  "
        assert find_word_backward(text, 9) == 2
        assert find_word_backward(text, 2) == 0

    def test_punctuation_run_foo_bar(self) -> None:
        text = "foo...bar"
        assert find_word_backward(text, 9) == 6
        assert find_word_backward(text, 6) == 3
        assert find_word_backward(text, 3) == 0

    def test_cursor_at_0_returns_0(self) -> None:
        assert find_word_backward("hello", 0) == 0


class TestFindWordForward:
    def test_basic_words_hello_world(self) -> None:
        text = "hello world"
        assert find_word_forward(text, 0) == 5
        assert find_word_forward(text, 5) == 11

    def test_dotted_foo_bar(self) -> None:
        text = "foo.bar"
        assert find_word_forward(text, 0) == 3
        assert find_word_forward(text, 3) == 4
        assert find_word_forward(text, 4) == 7

    def test_colon_foo_bar(self) -> None:
        text = "foo:bar"
        assert find_word_forward(text, 0) == 3
        assert find_word_forward(text, 3) == 4
        assert find_word_forward(text, 4) == 7

    def test_path_path_to_file(self) -> None:
        text = "path/to/file"
        assert find_word_forward(text, 0) == 4
        assert find_word_forward(text, 4) == 5
        assert find_word_forward(text, 5) == 7
        assert find_word_forward(text, 7) == 8
        assert find_word_forward(text, 8) == 12

    def test_cjk_mixed(self) -> None:
        text = "你好世界 test"
        first_end = find_word_forward(text, 0)
        assert first_end > 0
        assert first_end <= 4
        # Walk to end
        pos = 0
        while pos < len(text):
            nxt = find_word_forward(text, pos)
            if nxt == pos:
                break
            pos = nxt
        assert pos == len(text)

    def test_whitespace_at_boundaries(self) -> None:
        text = "  hello  "
        assert find_word_forward(text, 0) == 7
        assert find_word_forward(text, 7) == 9

    def test_punctuation_run_foo_bar(self) -> None:
        text = "foo...bar"
        assert find_word_forward(text, 0) == 3
        assert find_word_forward(text, 3) == 6
        assert find_word_forward(text, 6) == 9

    def test_cursor_at_end_returns_end(self) -> None:
        assert find_word_forward("hello", 5) == 5


@dataclass
class AtomicOpts:
    segment: Callable[[str], Iterable[Segment]]
    is_atomic_segment: Callable[[str], bool]


class TestAtomicSegments:
    marker = "[paste #1 +5 lines]"
    text = f"hello {marker} world"

    def is_atomic(self, s: str) -> bool:
        return s == self.marker

    # The functions slice text before calling segment(), so we map each expected
    # substring to its pre-split segments.
    def segment_map(self) -> dict[str, list[Segment]]:
        text = self.text
        marker = self.marker
        return {
            text: [
                Segment(segment="hello", is_word_like=True),
                Segment(segment=" ", is_word_like=False),
                Segment(segment=marker, is_word_like=True),
                Segment(segment=" ", is_word_like=False),
                Segment(segment="world", is_word_like=True),
            ],
            text[: len(text)]: [
                Segment(segment="hello", is_word_like=True),
                Segment(segment=" ", is_word_like=False),
                Segment(segment=marker, is_word_like=True),
                Segment(segment=" ", is_word_like=False),
                Segment(segment="world", is_word_like=True),
            ],
            # backward from 26: slice(0, 26) = "hello [paste #1 +5 lines] "
            text[:26]: [
                Segment(segment="hello", is_word_like=True),
                Segment(segment=" ", is_word_like=False),
                Segment(segment=marker, is_word_like=True),
                Segment(segment=" ", is_word_like=False),
            ],
            # forward from 6: slice(6) = "[paste #1 +5 lines] world"
            text[6:]: [
                Segment(segment=marker, is_word_like=True),
                Segment(segment=" ", is_word_like=False),
                Segment(segment="world", is_word_like=True),
            ],
        }

    def opts(self) -> AtomicOpts:
        smap = self.segment_map()
        return AtomicOpts(
            segment=lambda inp: smap.get(inp, []),
            is_atomic_segment=self.is_atomic,
        )

    def test_backward_skips_word_then_stops_before_atomic_marker(self) -> None:
        assert find_word_backward(self.text, len(self.text), self.opts()) == 26

    def test_backward_skips_whitespace_then_atomic_marker_as_one_unit(self) -> None:
        assert find_word_backward(self.text, 26, self.opts()) == 6

    def test_forward_skips_atomic_marker_as_one_unit(self) -> None:
        assert find_word_forward(self.text, 6, self.opts()) == 6 + len(self.marker)
