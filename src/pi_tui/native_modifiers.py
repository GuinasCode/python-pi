"""Native modifier key detection.

On non-macOS platforms (and in this Python port, which has no native .node
helper module), ``is_native_modifier_pressed`` always returns False.
"""

from __future__ import annotations

from typing import Literal

ModifierKey = Literal["shift", "command", "control", "option"]


def is_native_modifier_pressed(_key: ModifierKey) -> bool:
    """Return whether a native modifier key is currently pressed.

    In this Python port there is no native helper module, so this always
    returns False on every platform.
    """
    return False
