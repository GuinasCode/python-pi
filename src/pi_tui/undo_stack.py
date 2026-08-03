"""Generic undo stack with clone-on-push semantics.

Stores deep clones of state snapshots. Popped snapshots are returned
directly (no re-cloning) since they are already detached.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Generic, TypeVar

S = TypeVar("S")


@dataclass
class UndoStack(Generic[S]):
    """Generic undo stack that deep-clones state on push."""

    _stack: list[S] = field(default_factory=list)

    def push(self, state: S) -> None:
        """Push a deep clone of the given state onto the stack."""
        self._stack.append(copy.deepcopy(state))

    def pop(self) -> S | None:
        """Pop and return the most recent snapshot, or None if empty."""
        return self._stack.pop() if self._stack else None

    def clear(self) -> None:
        """Remove all snapshots."""
        self._stack.clear()

    @property
    def length(self) -> int:
        return len(self._stack)
