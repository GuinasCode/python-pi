"""Tests for pi_ai types module - validates core dataclasses."""

from __future__ import annotations

from pi_ai import (
    AssistantMessage,
    Context,
    ImageContent,
    Model,
    ModelCost,
    StopReason,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)


def test_text_content_defaults() -> None:
    tc = TextContent(text="hello")
    assert tc.type == "text"
    assert tc.text == "hello"
    assert tc.text_signature is None


def test_thinking_content_defaults() -> None:
    tc = ThinkingContent(thinking="reasoning")
    assert tc.type == "thinking"
    assert tc.thinking == "reasoning"
    assert tc.redacted is None


def test_image_content_defaults() -> None:
    ic = ImageContent(data="abc", mime_type="image/png")
    assert ic.type == "image"
    assert ic.data == "abc"
    assert ic.mime_type == "image/png"


def test_tool_call_defaults() -> None:
    tc = ToolCall(id="call-1", name="read", arguments={"path": "/tmp"})
    assert tc.type == "toolCall"
    assert tc.id == "call-1"
    assert tc.name == "read"
    assert tc.arguments == {"path": "/tmp"}


def test_usage_defaults() -> None:
    usage = Usage()
    assert usage.input == 0
    assert usage.output == 0
    assert usage.cache_read == 0
    assert usage.cache_write == 0
    assert usage.total_tokens == 0
    assert usage.cost.total == 0.0


def test_user_message_defaults() -> None:
    msg = UserMessage(content="hello", timestamp=123)
    assert msg.role == "user"
    assert msg.content == "hello"
    assert msg.timestamp == 123


def test_assistant_message_defaults() -> None:
    msg = AssistantMessage(
        api="openai-completions",
        provider="openai",
        model="gpt-4",
        timestamp=456,
    )
    assert msg.role == "assistant"
    assert msg.api == "openai-completions"
    assert msg.provider == "openai"
    assert msg.model == "gpt-4"
    assert msg.stop_reason == StopReason.PENDING
    assert msg.content == []


def test_tool_result_message_defaults() -> None:
    msg = ToolResultMessage(
        tool_call_id="call-1",
        tool_name="read",
        content=[TextContent(text="result")],
        timestamp=789,
    )
    assert msg.role == "toolResult"
    assert msg.tool_call_id == "call-1"
    assert msg.tool_name == "read"
    assert msg.is_error is False
    assert len(msg.content) == 1


def test_model_defaults() -> None:
    model = Model(
        id="gpt-4",
        name="GPT-4",
        api="openai-completions",
        provider="openai",
        base_url="https://api.openai.com/v1",
        context_window=128000,
        max_tokens=16384,
    )
    assert model.id == "gpt-4"
    assert model.reasoning is False
    assert model.context_window == 128000


def test_model_cost_defaults() -> None:
    cost = ModelCost(input=0.01, output=0.03)
    assert cost.input == 0.01
    assert cost.output == 0.03
    assert cost.tiers is None


def test_tool_defaults() -> None:
    tool = Tool(name="read", description="Read a file", parameters={"type": "object"})
    assert tool.name == "read"
    assert tool.description == "Read a file"
    assert tool.parameters == {"type": "object"}


def test_context_defaults() -> None:
    ctx = Context(system_prompt="You are helpful")
    assert ctx.system_prompt == "You are helpful"
    assert ctx.messages == []
    assert ctx.tools is None


def test_stop_reason_enum() -> None:
    assert StopReason.STOP.value == "stop"
    assert StopReason.ERROR.value == "error"
    assert StopReason.ABORTED.value == "aborted"
    assert StopReason.PENDING.value == "pending"
    assert StopReason.LENGTH.value == "length"
    assert StopReason.TOOL_USE.value == "toolUse"
