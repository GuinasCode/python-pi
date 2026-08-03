from __future__ import annotations

from pi_tui.terminal_colors import (
    RgbColor,
    is_osc11_background_color_response,
    parse_osc11_background_color,
    parse_terminal_color_scheme_report,
)


class TestParseOsc11BackgroundColor:
    def test_parses_16_bit_osc11_rgb_responses(self) -> None:
        result = parse_osc11_background_color("\x1b]11;rgb:0000/8000/ffff\x07")
        assert result == RgbColor(r=0, g=128, b=255)

    def test_parses_osc11_hex_responses(self) -> None:
        assert parse_osc11_background_color("\x1b]11;#ffffff\x1b\\") == RgbColor(r=255, g=255, b=255)
        assert parse_osc11_background_color("\x1b]11;#000000\x07") == RgbColor(r=0, g=0, b=0)

    def test_rejects_non_strict_osc11_responses(self) -> None:
        assert parse_osc11_background_color("x\x1b]11;#ffffff\x07") is None
        assert parse_osc11_background_color("\x1b]10;#ffffff\x07") is None
        assert parse_osc11_background_color("\x1b]11;#ffffff\x07x") is None


class TestParseTerminalColorSchemeReport:
    def test_parses_color_scheme_reports(self) -> None:
        assert parse_terminal_color_scheme_report("\x1b[?997;1n") == "dark"
        assert parse_terminal_color_scheme_report("\x1b[?997;2n") == "light"
        assert parse_terminal_color_scheme_report("\x1b[?997;3n") is None
        assert parse_terminal_color_scheme_report("\x1b[?996n") is None
        assert parse_terminal_color_scheme_report("x\x1b[?997;1n") is None


class TestIsOsc11BackgroundColorResponse:
    def test_valid_hex_response_bel_terminated(self) -> None:
        assert is_osc11_background_color_response("\x1b]11;#ffffff\x07") is True

    def test_valid_hex_response_st_terminated(self) -> None:
        assert is_osc11_background_color_response("\x1b]11;#ffffff\x1b\\") is True

    def test_valid_rgb_response(self) -> None:
        assert is_osc11_background_color_response("\x1b]11;rgb:00/00/00\x07") is True

    def test_rejects_non_osc11(self) -> None:
        assert is_osc11_background_color_response("x\x1b]11;#ffffff\x07") is False
        assert is_osc11_background_color_response("\x1b]10;#ffffff\x07") is False
        assert is_osc11_background_color_response("\x1b]11;#ffffff\x07x") is False
        assert is_osc11_background_color_response("") is False
