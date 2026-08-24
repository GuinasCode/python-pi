"""Shared OpenAI-compatible chat-completions message conversion.

Every provider in this package speaks some flavor of the OpenAI
chat-completions wire format (openai.py, nvidia_models.py, nvidia_glm.py,
nvidia_moonshot.py, nvidia_minimax.py, openrouter_moonshot.py all POST the
same ``{"role": ..., "content": ...}`` message shape). This one piece —
turning pi_ai content blocks into that shape — used to be duplicated 6
times and had drifted: 2 of the 6 handled ``ImageContent``, the other 4
silently dropped it (``"".join(c.text if hasattr(c, "text") else "" for c
in content)`` turns an image block into an empty string with no error).
Single implementation now; every provider calls this instead.
"""

from __future__ import annotations

from typing import Any

from pi_ai import ImageContent, TextContent, ToolResultMessage


def user_content_to_openai(content: str | list[Any]) -> str | list[dict[str, Any]]:
    """A plain string user message stays a string; otherwise TextContent/
    ImageContent blocks become ``{"type": "text", ...}``/
    ``{"type": "image_url", ...}`` content parts. ImageContent only ever
    carries base64 data (never a bare remote URL — see pi_ai.ImageContent),
    so a data: URI is the only faithful translation."""
    if isinstance(content, str):
        return content
    parts: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, TextContent):
            parts.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageContent):
            parts.append({"type": "image_url", "image_url": {"url": f"data:{block.mime_type};base64,{block.data}"}})
    return parts


def tool_result_to_openai_messages(msg: ToolResultMessage) -> list[dict[str, Any]]:
    """A ToolResultMessage becomes one ``role: tool`` message (text only —
    the OpenAI chat-completions wire format requires tool-role content to
    be a plain string, unlike user-role content, which accepts an array of
    typed parts) plus, only when the result actually contains image(s), a
    synthetic follow-up ``role: user`` message carrying them as
    ``image_url`` parts. That's the standard workaround for getting a
    tool's visual output (e.g. a browser screenshot) in front of a model
    that has no other way to see an image attached to a tool result."""
    text_parts: list[str] = []
    image_parts: list[dict[str, Any]] = []
    for block in msg.content:
        if isinstance(block, TextContent):
            text_parts.append(block.text)
        elif isinstance(block, ImageContent):
            image_parts.append(
                {"type": "image_url", "image_url": {"url": f"data:{block.mime_type};base64,{block.data}"}}
            )

    messages: list[dict[str, Any]] = [
        {"role": "tool", "tool_call_id": msg.tool_call_id, "content": "\n".join(text_parts)}
    ]
    if image_parts:
        messages.append({"role": "user", "content": image_parts})
    return messages


__all__ = ["tool_result_to_openai_messages", "user_content_to_openai"]
