"""Cyclable permission mode shown in the interactive footer, and the pure
decision logic that gates mutating tool calls (bash/write/edit) based on it.

Only mutating tools are gated — read-only ones (read/grep/ls/recall/...)
always run regardless of mode, matching how Claude Code's permission modes
only ever affect state-changing actions.
"""

from __future__ import annotations

from enum import Enum


class PermissionMode(str, Enum):
    DEFAULT = "default"
    ACCEPT_EDITS = "acceptEdits"
    PLAN = "plan"


_CYCLE_ORDER = (PermissionMode.DEFAULT, PermissionMode.ACCEPT_EDITS, PermissionMode.PLAN)

_LABELS = {
    PermissionMode.DEFAULT: "default mode",
    PermissionMode.ACCEPT_EDITS: "accept edits on",
    PermissionMode.PLAN: "plan mode on",
}


def cycle_permission_mode(current: PermissionMode) -> PermissionMode:
    """Return the next mode in the shift+tab cycle."""
    idx = _CYCLE_ORDER.index(current)
    return _CYCLE_ORDER[(idx + 1) % len(_CYCLE_ORDER)]


def permission_mode_label(mode: PermissionMode) -> str:
    """Human-readable footer label, e.g. 'accept edits on'."""
    return _LABELS[mode]


class PermissionDecision(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


# Tool names that change on-disk or shell state — the only ones any
# permission mode gates. Anything else (read, grep, ls, recall, ...) is
# always allowed.
MUTATING_TOOL_NAMES = frozenset({"bash", "write", "edit"})


def permission_decision(mode: PermissionMode, tool_name: str) -> PermissionDecision:
    """Decide, before executing *tool_name*, whether *mode* allows it
    outright (ALLOW), must prompt the user (ASK), or blocks it (DENY).

    - plan mode: every mutating tool is DENYed (read-only by design).
    - accept edits: every mutating tool (bash included) is ALLOWed
      automatically — the whole point of the mode is to stop asking.
    - default: every mutating tool ASKs.
    """
    if tool_name not in MUTATING_TOOL_NAMES:
        return PermissionDecision.ALLOW
    if mode is PermissionMode.PLAN:
        return PermissionDecision.DENY
    if mode is PermissionMode.ACCEPT_EDITS:
        return PermissionDecision.ALLOW
    return PermissionDecision.ASK
