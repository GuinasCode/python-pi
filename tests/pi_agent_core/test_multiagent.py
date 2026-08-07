"""Tests for pi_agent_core.fork and pi_agent_core.lane."""

from __future__ import annotations

import asyncio

import pytest

from pi_agent_core.fork import derive_child_session_id
from pi_agent_core.lane import Lane, SessionLanes


class TestDeriveChildSessionId:
    def test_deterministic(self) -> None:
        a = derive_child_session_id("parent-1", "call-abc")
        b = derive_child_session_id("parent-1", "call-abc")
        assert a == b

    def test_different_parent_gives_different_id(self) -> None:
        a = derive_child_session_id("parent-1", "call-abc")
        b = derive_child_session_id("parent-2", "call-abc")
        assert a != b

    def test_different_tool_call_gives_different_id(self) -> None:
        a = derive_child_session_id("parent-1", "call-abc")
        b = derive_child_session_id("parent-1", "call-xyz")
        assert a != b

    def test_length(self) -> None:
        result = derive_child_session_id("p", "t")
        assert len(result) == 24

    def test_hex_chars_only(self) -> None:
        result = derive_child_session_id("parent-session", "tool-call-id")
        assert all(c in "0123456789abcdef" for c in result)


class TestLane:
    @pytest.mark.asyncio
    async def test_run_returns_value(self) -> None:
        lane = Lane("main")

        async def _forty_two() -> int:
            return 42

        result = await lane.run(_forty_two())
        assert result == 42

    @pytest.mark.asyncio
    async def test_serialises_concurrent_calls(self) -> None:
        lane = Lane("serial")
        order: list[str] = []

        async def slow(tag: str) -> None:
            order.append(f"start:{tag}")
            await asyncio.sleep(0)
            order.append(f"end:{tag}")

        await asyncio.gather(lane.run(slow("a")), lane.run(slow("b")))
        # One must fully complete before the other starts (serialised per-lane).
        assert order.index("end:a") < order.index("start:b") or order.index("end:b") < order.index("start:a")

    def test_repr(self) -> None:
        assert "main" in repr(Lane("main"))


class TestSessionLanes:
    def test_creates_lane_on_demand(self) -> None:
        lanes = SessionLanes()
        lane = lanes.get("worker")
        assert isinstance(lane, Lane)
        assert lane.name == "worker"

    def test_same_name_returns_same_lane(self) -> None:
        lanes = SessionLanes()
        a = lanes.get("x")
        b = lanes.get("x")
        assert a is b

    def test_different_names_are_independent(self) -> None:
        lanes = SessionLanes()
        assert lanes.get("a") is not lanes.get("b")

    def test_default_name_is_main(self) -> None:
        lanes = SessionLanes()
        assert lanes.get().name == "main"

    def test_names_property(self) -> None:
        lanes = SessionLanes()
        lanes.get("a")
        lanes.get("b")
        assert set(lanes.names) == {"a", "b"}

    @pytest.mark.asyncio
    async def test_parallel_lanes_run_concurrently(self) -> None:
        lanes = SessionLanes()
        results: list[str] = []

        async def task(name: str) -> None:
            results.append(f"start:{name}")
            await asyncio.sleep(0.01)
            results.append(f"end:{name}")

        await asyncio.gather(
            lanes.get("lane-a").run(task("a")),
            lanes.get("lane-b").run(task("b")),
        )
        # Both start before either ends (concurrent across lanes).
        assert results.index("start:a") < results.index("end:b")
        assert results.index("start:b") < results.index("end:a")
