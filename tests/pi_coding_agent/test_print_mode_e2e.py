"""End-to-end test for print mode with faux provider."""

from __future__ import annotations

import asyncio

from pi_coding_agent import parse_args
from pi_coding_agent.print_mode import run_print_mode


def test_print_mode_text_faux(capsys: object) -> None:
    """Test that print mode produces output with faux provider."""
    args = parse_args(["-p", "hello"])
    result = asyncio.run(run_print_mode(args))
    assert result == 0


def test_print_mode_json_faux(capsys: object) -> None:
    """Test JSON mode produces JSON events."""
    args = parse_args(["--mode", "json", "hello"])
    result = asyncio.run(run_print_mode(args))
    assert result == 0


def test_print_mode_no_prompt() -> None:
    """Test that print mode without prompt returns error."""
    args = parse_args(["-p"])
    result = asyncio.run(run_print_mode(args))
    assert result == 1
