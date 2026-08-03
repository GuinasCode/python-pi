from __future__ import annotations

from pi_tui.native_modifiers import is_native_modifier_pressed


class TestIsNativeModifierPressed:
    def test_shift_returns_false(self) -> None:
        assert is_native_modifier_pressed("shift") is False

    def test_command_returns_false(self) -> None:
        assert is_native_modifier_pressed("command") is False

    def test_control_returns_false(self) -> None:
        assert is_native_modifier_pressed("control") is False

    def test_option_returns_false(self) -> None:
        assert is_native_modifier_pressed("option") is False

    def test_all_modifiers_return_false_on_any_platform(self) -> None:
        for key in ("shift", "command", "control", "option"):
            assert is_native_modifier_pressed(key) is False
