"""Tests for pi_ai.providers._openai_compat — the shared OpenAI-compatible
message conversion used by every provider in this package. This is the
one piece that used to be duplicated 6 times and had drifted (4 of 6
providers silently dropped ImageContent); these tests pin the correct
behavior directly, independent of any single provider's HTTP wiring."""

from __future__ import annotations

from pi_ai import ImageContent, TextContent, ToolResultMessage
from pi_ai.providers._openai_compat import tool_result_to_openai_messages, user_content_to_openai


class TestUserContentToOpenai:
    def test_plain_string_stays_a_string(self) -> None:
        assert user_content_to_openai("hello") == "hello"

    def test_text_only_list_becomes_text_parts(self) -> None:
        result = user_content_to_openai([TextContent(text="hi")])
        assert result == [{"type": "text", "text": "hi"}]

    def test_image_becomes_data_uri_image_url_part(self) -> None:
        result = user_content_to_openai([ImageContent(data="Zm9v", mime_type="image/png")])
        assert result == [{"type": "image_url", "image_url": {"url": "data:image/png;base64,Zm9v"}}]

    def test_mixed_text_and_image_preserves_order(self) -> None:
        result = user_content_to_openai(
            [
                TextContent(text="what is this?"),
                ImageContent(data="Zm9v", mime_type="image/jpeg"),
            ]
        )
        assert result == [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,Zm9v"}},
        ]

    def test_empty_list_becomes_empty_list(self) -> None:
        assert user_content_to_openai([]) == []


class TestToolResultToOpenaiMessages:
    def test_text_only_result_is_a_single_tool_message(self) -> None:
        msg = ToolResultMessage(tool_call_id="t1", tool_name="read", content=[TextContent(text="file contents")])
        result = tool_result_to_openai_messages(msg)
        assert result == [{"role": "tool", "tool_call_id": "t1", "content": "file contents"}]

    def test_image_result_adds_a_synthetic_user_message_after_the_tool_message(self) -> None:
        msg = ToolResultMessage(
            tool_call_id="t1", tool_name="screenshot", content=[ImageContent(data="Zm9v", mime_type="image/png")]
        )
        result = tool_result_to_openai_messages(msg)
        assert len(result) == 2
        assert result[0] == {"role": "tool", "tool_call_id": "t1", "content": ""}
        assert result[1] == {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,Zm9v"}}],
        }

    def test_mixed_text_and_image_result(self) -> None:
        msg = ToolResultMessage(
            tool_call_id="t1",
            tool_name="screenshot",
            content=[TextContent(text="captured the page"), ImageContent(data="abc", mime_type="image/png")],
        )
        result = tool_result_to_openai_messages(msg)
        assert result[0] == {"role": "tool", "tool_call_id": "t1", "content": "captured the page"}
        assert result[1]["role"] == "user"
        assert result[1]["content"] == [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]

    def test_empty_result_is_still_a_single_tool_message_with_empty_content(self) -> None:
        msg = ToolResultMessage(tool_call_id="t1", tool_name="noop", content=[])
        result = tool_result_to_openai_messages(msg)
        assert result == [{"role": "tool", "tool_call_id": "t1", "content": ""}]
