"""Print mode for non-interactive Pi usage.

Mirrors packages/coding-agent/src/modes/print-mode.ts.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from pi_coding_agent import Args


async def run_print_mode(args: Args) -> int:
    """Run Pi in print mode: execute a single prompt and print the result."""
    prompt = " ".join(args.messages) if args.messages else ""
    if not prompt:
        print("Error: No prompt provided", file=sys.stderr)
        return 1

    if args.mode == "json":
        return await _run_json_mode(args, prompt)
    return await _run_text_mode(args, prompt)


async def _run_text_mode(args: Args, prompt: str) -> int:
    """Run in text print mode."""
    print(f"[pi] Processing: {prompt}", file=sys.stderr)
    # In a full implementation, this would call the agent loop
    # For now, output a placeholder
    print("(Python port: agent execution not yet connected)")
    return 0


async def _run_json_mode(args: Args, prompt: str) -> int:
    """Run in JSON event stream mode."""
    events: list[dict[str, Any]] = [
        {"type": "start", "prompt": prompt},
        {"type": "end", "stop_reason": "stop"},
    ]
    for event in events:
        print(json.dumps(event))
    return 0


def run_print_mode_sync(args: Args) -> int:
    """Synchronous wrapper for run_print_mode."""
    return asyncio.run(run_print_mode(args))
