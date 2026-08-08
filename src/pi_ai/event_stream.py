"""Event stream for async iteration of assistant message events.

Mirrors packages/ai/src/utils/event-stream.ts.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, Generic, TypeVar, cast

from pi_ai import AssistantMessage

T = TypeVar("T")
R = TypeVar("R")


class EventStream(Generic[T, R]):
    """Generic async event stream with queue and waiting consumers."""

    def __init__(
        self,
        is_complete: Any,
        extract_result: Any,
    ) -> None:
        self._queue: list[T] = []
        self._waiting: list[asyncio.Future[T | None]] = []
        self._done = False
        self._final_result_future: asyncio.Future[R] | None = None
        self._is_complete = is_complete
        self._extract_result = extract_result

    def _ensure_future(self) -> asyncio.Future[R]:
        if self._final_result_future is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            self._final_result_future = loop.create_future()
        return self._final_result_future

    def push(self, event: T) -> None:
        if self._done:
            return
        if self._is_complete(event):
            self._done = True
            fut = self._ensure_future()
            if not fut.done():
                fut.set_result(self._extract_result(event))

        if self._waiting:
            waiter = self._waiting.pop(0)
            waiter.set_result(event)
        else:
            self._queue.append(event)

    def end(self, result: R | None = None) -> None:
        self._done = True
        if result is not None:
            fut = self._ensure_future()
            if not fut.done():
                fut.set_result(result)
        for waiter in self._waiting:
            if not waiter.done():
                waiter.set_result(None)
        self._waiting.clear()

    def __aiter__(self) -> AsyncIterator[T]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[T]:
        while True:
            if self._queue:
                yield self._queue.pop(0)
            elif self._done:
                return
            else:
                future: asyncio.Future[T | None] = asyncio.get_running_loop().create_future()
                self._waiting.append(future)
                item = await future
                if item is None:
                    return
                yield item

    async def result(self) -> R:
        return await self._ensure_future()


class AssistantMessageEventStream(EventStream[Any, AssistantMessage]):
    """Event stream for assistant message events."""

    def __init__(self) -> None:
        def is_complete(event: Any) -> bool:
            return getattr(event, "type", "") in ("done", "error")

        def extract_result(event: Any) -> AssistantMessage:
            if getattr(event, "type", "") == "done":
                return cast(AssistantMessage, event.message)
            if getattr(event, "type", "") == "error":
                return cast(AssistantMessage, event.error)
            raise ValueError("Unexpected event type for final result")

        super().__init__(is_complete, extract_result)


def create_assistant_message_event_stream() -> AssistantMessageEventStream:
    return AssistantMessageEventStream()
