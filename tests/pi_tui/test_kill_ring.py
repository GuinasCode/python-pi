from __future__ import annotations

from pi_tui.kill_ring import KillRing


class TestKillRingPush:
    def test_push_single_entry(self) -> None:
        ring = KillRing()
        ring.push("foo", prepend=False)
        assert ring.length == 1
        assert ring.peek() == "foo"

    def test_push_multiple_entries(self) -> None:
        ring = KillRing()
        ring.push("foo", prepend=False)
        ring.push("bar", prepend=False)
        assert ring.length == 2
        assert ring.peek() == "bar"

    def test_push_empty_string_does_nothing(self) -> None:
        ring = KillRing()
        ring.push("", prepend=False)
        assert ring.length == 0
        assert ring.peek() is None


class TestKillRingAccumulate:
    def test_accumulate_appends_to_last_entry(self) -> None:
        ring = KillRing()
        ring.push("foo", prepend=False)
        ring.push("bar", prepend=False, accumulate=True)
        assert ring.length == 1
        assert ring.peek() == "foobar"

    def test_accumulate_on_empty_ring_creates_new_entry(self) -> None:
        ring = KillRing()
        ring.push("foo", prepend=False, accumulate=True)
        assert ring.length == 1
        assert ring.peek() == "foo"


class TestKillRingPrepend:
    def test_prepend_with_accumulate_prepends_text(self) -> None:
        ring = KillRing()
        ring.push("foo", prepend=False)
        ring.push("bar", prepend=True, accumulate=True)
        assert ring.length == 1
        assert ring.peek() == "barfoo"


class TestKillRingRotate:
    def test_rotate_moves_last_to_front(self) -> None:
        ring = KillRing()
        ring.push("a", prepend=False)
        ring.push("b", prepend=False)
        ring.push("c", prepend=False)
        # ring: [a, b, c], peek = c
        assert ring.peek() == "c"
        ring.rotate()
        # ring: [c, a, b], peek = b
        assert ring.peek() == "b"
        ring.rotate()
        # ring: [b, c, a], peek = a
        assert ring.peek() == "a"
        ring.rotate()
        # ring: [a, b, c], peek = c
        assert ring.peek() == "c"

    def test_rotate_single_entry_no_change(self) -> None:
        ring = KillRing()
        ring.push("only", prepend=False)
        ring.rotate()
        assert ring.length == 1
        assert ring.peek() == "only"

    def test_rotate_empty_ring_no_change(self) -> None:
        ring = KillRing()
        ring.rotate()
        assert ring.length == 0
        assert ring.peek() is None


class TestKillRingPeek:
    def test_peek_returns_most_recent(self) -> None:
        ring = KillRing()
        ring.push("first", prepend=False)
        ring.push("second", prepend=False)
        assert ring.peek() == "second"

    def test_peek_does_not_modify_ring(self) -> None:
        ring = KillRing()
        ring.push("foo", prepend=False)
        ring.peek()
        assert ring.length == 1


class TestKillRingEmpty:
    def test_empty_ring_peek_is_none(self) -> None:
        ring = KillRing()
        assert ring.peek() is None
        assert ring.length == 0
