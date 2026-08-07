"""Message helpers for the agent harness.

Mirrors ``packages/agent/src/harness/messages.ts``.

Defines custom agent message types (bash execution, custom, branch summary,
compaction summary) and helpers to convert the full ``AgentMessage`` union into
the ``Message`` union understood by the LLM stream boundary.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from pi_ai import (
    ImageContent,
    Message,
    TextContent,
    UserMessage,
)

__all__ = [
    "BRANCH_SUMMARY_PREFIX",
    "BRANCH_SUMMARY_SUFFIX",
    "COMPACTION_SUMMARY_PREFIX",
    "COMPACTION_SUMMARY_SUFFIX",
    "BashExecutionMessage",
    "BranchSummaryMessage",
    "CompactionSummaryMessage",
    "CustomMessage",
    "bash_execution_to_text",
    "convert_to_llm",
    "create_branch_summary_message",
    "create_compaction_summary_message",
    "create_custom_message",
]

# --- Summary prefixes/suffixes ---

COMPACTION_SUMMARY_PREFIX = (
    "The conversation history before this point was compacted into the following summary:\n\n<summary>\n"
)
COMPACTION_SUMMARY_SUFFIX = "\n</summary>"

BRANCH_SUMMARY_PREFIX = (
    "The following is a summary of a branch that this conversation came back from:\n\n<summary>\n"
)
BRANCH_SUMMARY_SUFFIX = "</summary>"


# --- Custom agent message types ---


@dataclass
class BashExecutionMessage:
    """Mirrors ``BashExecutionMessage`` from the TS messages module."""

    role: str = field(default="bashExecution", init=False)
    command: str = ""
    output: str = ""
    exit_code: int | None = None
    cancelled: bool = False
    truncated: bool = False
    full_output_path: str | None = None
    timestamp: int = 0
    exclude_from_context: bool = False


@dataclass
class CustomMessage:
    """Mirrors ``CustomMessage`` from the TS messages module."""

    role: str = field(default="custom", init=False)
    custom_type: str = ""
    content: str | list[TextContent | ImageContent] = ""
    display: bool = True
    details: Any = None
    timestamp: int = 0


@dataclass
class BranchSummaryMessage:
    """Mirrors ``BranchSummaryMessage`` from the TS messages module."""

    role: str = field(default="branchSummary", init=False)
    summary: str = ""
    from_id: str = ""
    timestamp: int = 0


@dataclass
class CompactionSummaryMessage:
    """Mirrors ``CompactionSummaryMessage`` from the TS messages module."""

    role: str = field(default="compactionSummary", init=False)
    summary: str = ""
    tokens_before: int = 0
    timestamp: int = 0


# Union of all agent message types.
# AgentMessage = Message | BashExecutionMessage | CustomMessage | BranchSummaryMessage | CompactionSummaryMessage


def bash_execution_to_text(msg: BashExecutionMessage) -> str:
    """Render a bash execution message as a human-readable text block."""
    text = f"Ran `{msg.command}`\n"
    if msg.output:
        text += f"```\n{msg.output}\n```"
    else:
        text += "(no output)"
    if msg.cancelled:
        text += "\n\n(command cancelled)"
    elif msg.exit_code is not None and msg.exit_code != 0:
        text += f"\n\nCommand exited with code {msg.exit_code}"
    if msg.truncated and msg.full_output_path:
        text += f"\n\n[Output truncated. Full output: {msg.full_output_path}]"
    return text


def create_branch_summary_message(summary: str, from_id: str, timestamp: str | int) -> BranchSummaryMessage:
    """Create a :class:`BranchSummaryMessage` from a timestamp that may be an ISO string or ms epoch."""
    ts = _to_ms_timestamp(timestamp)
    return BranchSummaryMessage(summary=summary, from_id=from_id, timestamp=ts)


def create_compaction_summary_message(summary: str, tokens_before: int, timestamp: str | int) -> CompactionSummaryMessage:
    """Create a :class:`CompactionSummaryMessage` from a timestamp that may be an ISO string or ms epoch."""
    ts = _to_ms_timestamp(timestamp)
    return CompactionSummaryMessage(summary=summary, tokens_before=tokens_before, timestamp=ts)


def create_custom_message(
    custom_type: str,
    content: str | list[TextContent | ImageContent],
    display: bool,
    details: Any = None,
    timestamp: str | int = 0,
) -> CustomMessage:
    """Create a :class:`CustomMessage` from a timestamp that may be an ISO string or ms epoch."""
    ts = _to_ms_timestamp(timestamp)
    return CustomMessage(custom_type=custom_type, content=content, display=display, details=details, timestamp=ts)


def convert_to_llm(messages: list[Any]) -> list[Message]:
    """Convert agent messages to LLM messages.

    Maps custom agent message roles (``bashExecution``, ``custom``,
    ``branchSummary``, ``compactionSummary``) to user messages, and passes
    ``user``/``assistant``/``toolResult`` messages through unchanged.
    """
    result: list[Message] = []
    for m in messages:
        role = getattr(m, "role", None)
        if role in ("user", "assistant", "toolResult"):
            result.append(m)  # type: ignore[arg-type]
        elif role == "bashExecution":
            if getattr(m, "exclude_from_context", False):
                continue
            result.append(
                UserMessage(content=[TextContent(text=bash_execution_to_text(m))], timestamp=m.timestamp)
            )
        elif role == "custom":
            content = m.content
            if isinstance(content, str):
                content = [TextContent(text=content)]
            result.append(UserMessage(content=content, timestamp=m.timestamp))
        elif role == "branchSummary":
            text = BRANCH_SUMMARY_PREFIX + m.summary + BRANCH_SUMMARY_SUFFIX
            result.append(UserMessage(content=[TextContent(text=text)], timestamp=m.timestamp))
        elif role == "compactionSummary":
            text = COMPACTION_SUMMARY_PREFIX + m.summary + COMPACTION_SUMMARY_SUFFIX
            result.append(UserMessage(content=[TextContent(text=text)], timestamp=m.timestamp))
    return result


def _to_ms_timestamp(value: str | int) -> int:
    """Convert a timestamp that may be an ISO-8601 string or ms epoch to a ms epoch int."""
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        # Parse ISO-8601 string to seconds, then to ms.
        try:
            from datetime import datetime

            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return int(dt.timestamp() * 1000)
        except ValueError:
            # If it's a numeric string, parse as int.
            try:
                return int(value)
            except ValueError:
                return int(time.time() * 1000)
    return int(time.time() * 1000)
