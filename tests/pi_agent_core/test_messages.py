"""Tests for pi_agent_core.messages."""

from __future__ import annotations

import time

from pi_agent_core.messages import (
    BRANCH_SUMMARY_PREFIX,
    BRANCH_SUMMARY_SUFFIX,
    COMPACTION_SUMMARY_PREFIX,
    COMPACTION_SUMMARY_SUFFIX,
    BashExecutionMessage,
    BranchSummaryMessage,
    CompactionSummaryMessage,
    CustomMessage,
    bash_execution_to_text,
    convert_to_llm,
    create_branch_summary_message,
    create_compaction_summary_message,
    create_custom_message,
)
from pi_ai import AssistantMessage, TextContent, ToolResultMessage, UserMessage


class TestBashExecutionMessage:
    def test_to_text_with_output(self) -> None:
        msg = BashExecutionMessage(command="ls", output="file.txt", exit_code=0, timestamp=100)
        text = bash_execution_to_text(msg)
        assert "Ran `ls`" in text
        assert "file.txt" in text

    def test_to_text_no_output(self) -> None:
        msg = BashExecutionMessage(command="true", output="", exit_code=0, timestamp=100)
        text = bash_execution_to_text(msg)
        assert "(no output)" in text

    def test_to_text_nonzero_exit(self) -> None:
        msg = BashExecutionMessage(command="false", output="", exit_code=1, timestamp=100)
        text = bash_execution_to_text(msg)
        assert "Command exited with code 1" in text

    def test_to_text_cancelled(self) -> None:
        msg = BashExecutionMessage(command="sleep 10", output="", cancelled=True, timestamp=100)
        text = bash_execution_to_text(msg)
        assert "(command cancelled)" in text

    def test_to_text_truncated(self) -> None:
        msg = BashExecutionMessage(
            command="cat big", output="...", truncated=True, full_output_path="/tmp/out", timestamp=100
        )
        text = bash_execution_to_text(msg)
        assert "[Output truncated. Full output: /tmp/out]" in text


class TestCreateMessages:
    def test_create_branch_summary_message_from_int(self) -> None:
        msg = create_branch_summary_message("summary", "entry1", 12345)
        assert msg.role == "branchSummary"
        assert msg.summary == "summary"
        assert msg.from_id == "entry1"
        assert msg.timestamp == 12345

    def test_create_branch_summary_message_from_iso(self) -> None:
        msg = create_branch_summary_message("summary", "entry1", "2025-01-01T00:00:00Z")
        assert msg.timestamp > 0

    def test_create_compaction_summary_message(self) -> None:
        msg = create_compaction_summary_message("summary", 5000, 12345)
        assert msg.role == "compactionSummary"
        assert msg.summary == "summary"
        assert msg.tokens_before == 5000
        assert msg.timestamp == 12345

    def test_create_custom_message(self) -> None:
        msg = create_custom_message("myType", "hello", True, None, 12345)
        assert msg.role == "custom"
        assert msg.custom_type == "myType"
        assert msg.content == "hello"
        assert msg.display is True
        assert msg.timestamp == 12345


class TestConvertToLlm:
    def test_passthrough_user_message(self) -> None:
        user = UserMessage(content="hello", timestamp=100)
        result = convert_to_llm([user])
        assert len(result) == 1
        assert result[0].role == "user"

    def test_passthrough_assistant_message(self) -> None:
        assistant = AssistantMessage(content=[], timestamp=100)
        result = convert_to_llm([assistant])
        assert len(result) == 1
        assert result[0].role == "assistant"

    def test_passthrough_tool_result_message(self) -> None:
        tr = ToolResultMessage(tool_call_id="t1", tool_name="bash", content=[], timestamp=100)
        result = convert_to_llm([tr])
        assert len(result) == 1
        assert result[0].role == "toolResult"

    def test_bash_execution_converted_to_user(self) -> None:
        msg = BashExecutionMessage(command="ls", output="file", exit_code=0, timestamp=200)
        result = convert_to_llm([msg])
        assert len(result) == 1
        assert result[0].role == "user"
        assert "Ran `ls`" in result[0].content[0].text  # type: ignore[union-attr]

    def test_bash_execution_excluded_from_context(self) -> None:
        msg = BashExecutionMessage(command="ls", output="file", timestamp=200, exclude_from_context=True)
        result = convert_to_llm([msg])
        assert len(result) == 0

    def test_custom_message_string_content(self) -> None:
        msg = CustomMessage(custom_type="note", content="hello", display=True, timestamp=200)
        result = convert_to_llm([msg])
        assert len(result) == 1
        assert result[0].role == "user"
        assert result[0].content[0].text == "hello"  # type: ignore[union-attr]

    def test_custom_message_content_blocks(self) -> None:
        msg = CustomMessage(
            custom_type="note",
            content=[TextContent(text="block")],
            display=True,
            timestamp=200,
        )
        result = convert_to_llm([msg])
        assert len(result) == 1
        assert result[0].content[0].text == "block"  # type: ignore[union-attr]

    def test_branch_summary_converted(self) -> None:
        msg = BranchSummaryMessage(summary="did stuff", from_id="x", timestamp=300)
        result = convert_to_llm([msg])
        assert len(result) == 1
        text = result[0].content[0].text  # type: ignore[union-attr]
        assert BRANCH_SUMMARY_PREFIX in text
        assert "did stuff" in text
        assert text.endswith(BRANCH_SUMMARY_SUFFIX)

    def test_compaction_summary_converted(self) -> None:
        msg = CompactionSummaryMessage(summary="compacted", tokens_before=1000, timestamp=400)
        result = convert_to_llm([msg])
        assert len(result) == 1
        text = result[0].content[0].text  # type: ignore[union-attr]
        assert COMPACTION_SUMMARY_PREFIX in text
        assert "compacted" in text
        assert text.endswith(COMPACTION_SUMMARY_SUFFIX)

    def test_mixed_messages(self) -> None:
        msgs: list[object] = [
            UserMessage(content="hi", timestamp=1),
            AssistantMessage(content=[], timestamp=2),
            CustomMessage(custom_type="n", content="note", display=True, timestamp=3),
        ]
        result = convert_to_llm(msgs)  # type: ignore[arg-type]
        assert len(result) == 3
        assert all(r.role == "user" or r.role == "assistant" for r in result)


class TestTimestamps:
    def test_to_ms_timestamp_from_now(self) -> None:
        ts = int(time.time() * 1000)
        # Test that int passes through
        assert create_branch_summary_message("s", "id", ts).timestamp == ts
