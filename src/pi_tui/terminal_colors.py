"""Terminal color parsing utilities.

Parses OSC 11 background color responses and terminal color scheme reports.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

TerminalColorScheme = Literal["dark", "light"]

_RGB_HEX6_RE = re.compile(r"^[0-9a-f]{6}$", re.IGNORECASE)
_RGB_HEX12_RE = re.compile(r"^[0-9a-f]{12}$", re.IGNORECASE)
_HEX_CHANNEL_RE = re.compile(r"^[0-9a-f]+$", re.IGNORECASE)

# Strict OSC 11 background color response: entire string must match.
# Allows either BEL (\x07) or ST (\x1b\\) terminator.
_OSC11_BACKGROUND_COLOR_RESPONSE_RE = re.compile(r"^\x1b\]11;([^\x07\x1b]*)(?:\x07|\x1b\\)$", re.IGNORECASE)

# Color scheme report: ESC [ ? 997 ; (1|2) n
_COLOR_SCHEME_REPORT_RE = re.compile(r"^\x1b\[\?997;(1|2)n$")


@dataclass(frozen=True)
class RgbColor:
    r: int
    g: int
    b: int


def hex_to_rgb(hex_str: str) -> RgbColor:
    """Convert a 6-digit hex color string (with optional leading '#') to RgbColor."""
    normalized = hex_str[1:] if hex_str.startswith("#") else hex_str
    r = int(normalized[0:2], 16)
    g = int(normalized[2:4], 16)
    b = int(normalized[4:6], 16)
    return RgbColor(r=r, g=g, b=b)


def _parse_osc_hex_channel(channel: str) -> int | None:
    if not _HEX_CHANNEL_RE.match(channel):
        return None
    max_val = 16 ** len(channel) - 1
    if max_val <= 0:
        return None
    result: int | None = round((int(channel, 16) / max_val) * 255)
    return result


def is_osc11_background_color_response(data: str) -> bool:
    """Return True if ``data`` is a strict OSC 11 background color response."""
    return _OSC11_BACKGROUND_COLOR_RESPONSE_RE.match(data) is not None


def parse_osc11_background_color(data: str) -> RgbColor | None:
    """Parse an OSC 11 background color response into an RgbColor, or None if invalid."""
    match = _OSC11_BACKGROUND_COLOR_RESPONSE_RE.match(data)
    if not match:
        return None

    value = match.group(1).strip()

    if value.startswith("#"):
        hex_part = value[1:]
        if _RGB_HEX6_RE.match(hex_part):
            return hex_to_rgb(value)
        if _RGB_HEX12_RE.match(hex_part):
            r = _parse_osc_hex_channel(hex_part[0:4])
            g = _parse_osc_hex_channel(hex_part[4:8])
            b = _parse_osc_hex_channel(hex_part[8:12])
            if r is not None and g is not None and b is not None:
                return RgbColor(r=r, g=g, b=b)
            return None
        return None

    rgb_value = re.sub(r"^rgba?:", "", value, flags=re.IGNORECASE)
    parts = rgb_value.split("/")
    if len(parts) < 3:
        return None
    red, green, blue = parts[0], parts[1], parts[2]
    r = _parse_osc_hex_channel(red)
    g = _parse_osc_hex_channel(green)
    b = _parse_osc_hex_channel(blue)
    if r is not None and g is not None and b is not None:
        return RgbColor(r=r, g=g, b=b)
    return None


def parse_terminal_color_scheme_report(data: str) -> TerminalColorScheme | None:
    """Parse a terminal color scheme report into 'dark' or 'light', or None if invalid."""
    match = _COLOR_SCHEME_REPORT_RE.match(data)
    if not match:
        return None
    return "light" if match.group(1) == "2" else "dark"
