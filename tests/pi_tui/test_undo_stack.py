from __future__ import annotations

from pi_tui.undo_stack import UndoStack


class TestUndoStackPushPop:
    def test_push_then_pop_returns_state(self) -> None:
        stack: UndoStack[dict[str, int]] = UndoStack()
        stack.push({"a": 1})
        result = stack.pop()
        assert result == {"a": 1}
        assert stack.length == 0

    def test_pop_returns_none_when_empty(self) -> None:
        stack: UndoStack[int] = UndoStack()
        assert stack.pop() is None

    def test_push_deep_clones_state(self) -> None:
        stack: UndoStack[dict[str, int]] = UndoStack()
        state = {"a": 1}
        stack.push(state)
        # mutate original after push
        state["a"] = 99
        result = stack.pop()
        assert result == {"a": 1}

    def test_lifo_order(self) -> None:
        stack: UndoStack[int] = UndoStack()
        stack.push(1)
        stack.push(2)
        stack.push(3)
        assert stack.pop() == 3
        assert stack.pop() == 2
        assert stack.pop() == 1
        assert stack.pop() is None


class TestUndoStackClear:
    def test_clear_empties_stack(self) -> None:
        stack: UndoStack[int] = UndoStack()
        stack.push(1)
        stack.push(2)
        stack.clear()
        assert stack.length == 0
        assert stack.pop() is None

    def test_clear_on_empty_stack(self) -> None:
        stack: UndoStack[int] = UndoStack()
        stack.clear()
        assert stack.length == 0


class TestUndoStackLength:
    def test_length_reflects_pushes(self) -> None:
        stack: UndoStack[int] = UndoStack()
        assert stack.length == 0
        stack.push(1)
        assert stack.length == 1
        stack.push(2)
        assert stack.length == 2

    def test_length_reflects_pops(self) -> None:
        stack: UndoStack[int] = UndoStack()
        stack.push(1)
        stack.push(2)
        stack.pop()
        assert stack.length == 1
        stack.pop()
        assert stack.length == 0


class TestUndoStackEmpty:
    def test_empty_stack_pop_is_none(self) -> None:
        stack: UndoStack[int] = UndoStack()
        assert stack.pop() is None
        assert stack.length == 0
