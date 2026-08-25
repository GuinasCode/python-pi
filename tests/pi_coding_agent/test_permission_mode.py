"""Tests for pi_coding_agent.permission_mode."""

from __future__ import annotations

from pi_coding_agent.permission_mode import (
    PermissionDecision,
    PermissionMode,
    cycle_permission_mode,
    permission_decision,
    permission_mode_label,
)


class TestCyclePermissionMode:
    def test_cycles_default_to_accept_edits_to_plan_and_back(self) -> None:
        mode = PermissionMode.DEFAULT
        mode = cycle_permission_mode(mode)
        assert mode == PermissionMode.ACCEPT_EDITS
        mode = cycle_permission_mode(mode)
        assert mode == PermissionMode.PLAN
        mode = cycle_permission_mode(mode)
        assert mode == PermissionMode.DEFAULT


class TestPermissionModeLabel:
    def test_labels_are_non_empty_for_every_mode(self) -> None:
        for mode in PermissionMode:
            assert permission_mode_label(mode)

    def test_accept_edits_label(self) -> None:
        assert permission_mode_label(PermissionMode.ACCEPT_EDITS) == "accept edits on"


class TestPermissionDecision:
    def test_read_only_tools_always_allowed(self) -> None:
        for mode in PermissionMode:
            for tool in ("read", "grep", "ls", "recall"):
                assert permission_decision(mode, tool) is PermissionDecision.ALLOW

    def test_default_mode_asks_for_mutating_tools(self) -> None:
        for tool in ("bash", "write", "edit"):
            assert permission_decision(PermissionMode.DEFAULT, tool) is PermissionDecision.ASK

    def test_plan_mode_denies_every_mutating_tool(self) -> None:
        for tool in ("bash", "write", "edit"):
            assert permission_decision(PermissionMode.PLAN, tool) is PermissionDecision.DENY

    def test_accept_edits_allows_every_mutating_tool_including_bash(self) -> None:
        for tool in ("bash", "write", "edit"):
            assert permission_decision(PermissionMode.ACCEPT_EDITS, tool) is PermissionDecision.ALLOW
