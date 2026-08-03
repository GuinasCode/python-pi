"""Tests for print mode."""

from __future__ import annotations

import asyncio

from pi_coding_agent import Args, parse_args
from pi_coding_agent.print_mode import run_print_mode


def test_print_mode_text(capsys: object) -> None:
    args = parse_args(["-p", "hello"])
    result = asyncio.run(run_print_mode(args))
    assert result == 0


def test_print_mode_json(capsys: object) -> None:
    args = parse_args(["--mode", "json", "test prompt"])
    result = asyncio.run(run_print_mode(args))
    assert result == 0


def test_print_mode_no_prompt() -> None:
    args = parse_args(["-p"])
    result = asyncio.run(run_print_mode(args))
    assert result == 1
