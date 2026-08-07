"""Lane — named parallel execution slot within a session.

A session hosts one or more lanes.  Each lane serialises its own operations
(at most one active coroutine at a time) while different lanes run
concurrently.

The interactive UI uses a single unnamed ``"main"`` lane (the concept is
invisible to the user).  Extensions that receive the full harness can open
additional lanes and run operations on them in parallel.

Example — a subagent tool opens a second lane so it can run alongside the
parent agent's main lane without blocking it::

    lanes = SessionLanes()
    result = await lanes.get("subagent-abc").run(some_coro())
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

T = TypeVar("T")


class Lane:
    """Named execution slot with per-lane serialisation."""

    def __init__(self, name: str = "main") -> None:
        self.name = name
        self._sem: asyncio.Semaphore | None = None

    def _get_sem(self) -> asyncio.Semaphore:
        if self._sem is None:
            self._sem = asyncio.Semaphore(1)
        return self._sem

    async def run(self, coro: Awaitable[T]) -> T:
        """Run *coro* exclusively within this lane (serialised)."""
        async with self._get_sem():
            return await coro

    def __repr__(self) -> str:
        return f"Lane({self.name!r})"


class SessionLanes:
    """Registry of lanes for a session.  Creates lanes on first access."""

    def __init__(self) -> None:
        self._lanes: dict[str, Lane] = {}

    def get(self, name: str = "main") -> Lane:
        """Return the lane with the given name, creating it if absent."""
        if name not in self._lanes:
            self._lanes[name] = Lane(name)
        return self._lanes[name]

    @property
    def names(self) -> list[str]:
        return list(self._lanes)

    def __repr__(self) -> str:
        return f"SessionLanes(lanes={self.names!r})"


__all__ = ["Lane", "SessionLanes"]
