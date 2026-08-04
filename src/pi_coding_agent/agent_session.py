"""Agent session orchestration for Pi coding agent.

Simplified port of packages/coding-agent/src/core/agent-session.ts.
Orchestrates: resolve model → build system prompt → run agent → handle tools.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pi_ai import (
    AssistantMessage,
    Context,
    Message,
    Model,
    StopReason,
    TextContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from pi_ai.models import MutableModels
from pi_coding_agent.resource_loader import LoadedResources, load_resources
from pi_coding_agent.system_prompt import BuildSystemPromptOptions, build_system_prompt
from pi_coding_agent.tools import ToolResult, edit_file, execute_bash, grep_search, list_files, read_file, write_file


def _create_tool_result_message(
    tool_call: ToolCall,
    result: ToolResult,
    timestamp: int,
) -> ToolResultMessage:
    """Create a ToolResultMessage from a ToolResult."""
    content: list[dict[str, str]] = result.content if result.content else [{"type": "text", "text": ""}]
    return ToolResultMessage(
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        content=content,  # type: ignore[arg-type]
        details=result.details,
        is_error=result.is_error,
        timestamp=timestamp,
    )


def _get_builtin_tools() -> list[Tool]:
    """Get the built-in tool definitions."""
    return [
        Tool(
            name="bash",
            description="Execute a bash command and return the output.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 120},
                },
                "required": ["command"],
            },
        ),
        Tool(
            name="read",
            description="Read a file and return its contents with line numbers.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read"},
                    "offset": {"type": "integer", "description": "Line offset to start reading from", "default": 0},
                    "limit": {"type": "integer", "description": "Maximum lines to read", "default": 2000},
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="write",
            description="Write content to a file, creating parent directories if needed.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to write"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        ),
        Tool(
            name="edit",
            description="Edit a file by replacing old_string with new_string.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to edit"},
                    "old_string": {"type": "string", "description": "Text to find"},
                    "new_string": {"type": "string", "description": "Replacement text"},
                    "replace_all": {"type": "boolean", "description": "Replace all occurrences", "default": False},
                },
                "required": ["path", "old_string", "new_string"],
            },
        ),
        Tool(
            name="grep",
            description="Search file contents with a regex pattern.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "path": {"type": "string", "description": "Directory or file to search", "default": "."},
                    "include": {"type": "string", "description": "File glob pattern", "default": "*"},
                    "ignore_case": {"type": "boolean", "description": "Case-insensitive search", "default": False},
                },
                "required": ["pattern"],
            },
        ),
        Tool(
            name="ls",
            description="List files in a directory tree.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path", "default": "."},
                    "max_depth": {"type": "integer", "description": "Max directory depth", "default": 3},
                },
            },
        ),
    ]


def _execute_tool(name: str, args: dict[str, Any]) -> ToolResult:
    """Execute a built-in tool by name."""
    if name == "bash":
        return execute_bash(
            args.get("command", ""),
            timeout=args.get("timeout", 120),
        )
    if name == "read":
        return read_file(
            args.get("path", ""),
            offset=args.get("offset", 0),
            limit=args.get("limit", 2000),
        )
    if name == "write":
        return write_file(args.get("path", ""), args.get("content", ""))
    if name == "edit":
        return edit_file(
            args.get("path", ""),
            args.get("old_string", ""),
            args.get("new_string", ""),
            replace_all=args.get("replace_all", False),
        )
    if name == "grep":
        return grep_search(
            args.get("pattern", ""),
            path=args.get("path", "."),
            include=args.get("include", "*"),
            ignore_case=args.get("ignore_case", False),
        )
    if name == "ls":
        return list_files(
            args.get("path", "."),
            max_depth=args.get("max_depth", 3),
        )
    return ToolResult(
        content=[{"type": "text", "text": f"Unknown tool: {name}"}],
        is_error=True,
    )


@dataclass
class AgentSessionOptions:
    """Options for creating an AgentSession."""

    models: MutableModels
    model: Model
    cwd: str = ""
    system_prompt: str | None = None
    append_system_prompt: str | None = None
    context_files: list[dict[str, str]] | None = None
    skills: list[Any] | None = None
    tools: list[Tool] | None = None
    thinking_level: str = "off"
    max_turns: int = 50


class AgentSession:
    """Orchestrates the agent loop: model → system prompt → LLM → tools → output."""

    def __init__(self, options: AgentSessionOptions) -> None:
        self._models = options.models
        self._model = options.model
        self._cwd = options.cwd
        self._system_prompt = options.system_prompt
        self._append_system_prompt = options.append_system_prompt
        self._context_files = options.context_files or []
        self._skills = options.skills or []
        self._tools = options.tools or _get_builtin_tools()
        self._thinking_level = options.thinking_level
        self._max_turns = options.max_turns
        self._messages: list[Message] = []
        self._event_listeners: list[Callable[[Any], None]] = []
        self._text_buffer = ""

    def on_event(self, listener: Callable[[Any], None]) -> Callable[[]]:
        """Register an event listener."""
        self._event_listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._event_listeners:
                self._event_listeners.remove(listener)

        return unsubscribe

    def _emit(self, event: Any) -> None:
        for listener in list(self._event_listeners):
            listener(event)

    def _build_system_prompt(self) -> str:
        """Build the system prompt from options."""
        if self._system_prompt:
            prompt = self._system_prompt
            if self._append_system_prompt:
                prompt += f"\n\n{self._append_system_prompt}"
            prompt += f"\nCurrent working directory: {self._cwd}"
            return prompt
        return build_system_prompt(BuildSystemPromptOptions(
            cwd=self._cwd,
            append_system_prompt=self._append_system_prompt,
            context_files=self._context_files,
            skills=self._skills,
            selected_tools=[t.name for t in self._tools],
        ))

    async def prompt(self, text: str) -> AssistantMessage:
        """Send a prompt to the model and run the agent loop until completion."""
        system_prompt = self._build_system_prompt()
        user_msg = UserMessage(content=text, timestamp=int(time.time() * 1000))
        self._messages.append(user_msg)

        for _turn in range(self._max_turns):
            self._compact_context_if_needed()
            context = Context(
                system_prompt=system_prompt,
                messages=self._messages,
                tools=self._tools,
            )

            stream = self._models.stream(self._model, context)
            assistant_msg = await self._consume_stream(stream)

            if assistant_msg is None:
                return AssistantMessage(
                    content=[TextContent(text="Error: no response from model")],
                    api=self._model.api,
                    provider=self._model.provider,
                    model=self._model.id,
                    stop_reason=StopReason.ERROR,
                    timestamp=int(time.time() * 1000),
                )

            self._messages.append(assistant_msg)

            if assistant_msg.stop_reason != StopReason.TOOL_USE:
                return assistant_msg

            # Execute tool calls
            for block in assistant_msg.content:
                if block.type != "toolCall":
                    continue
                tool_call = block
                self._emit({"type": "tool_call_start", "name": tool_call.name, "args": tool_call.arguments})
                result = _execute_tool(tool_call.name, tool_call.arguments)
                self._emit({
                    "type": "tool_call_end",
                    "name": tool_call.name,
                    "is_error": result.is_error,
                })
                tool_result = _create_tool_result_message(tool_call, result, int(time.time() * 1000))
                self._messages.append(tool_result)

        return AssistantMessage(
            content=[TextContent(text="Max turns reached")],
            api=self._model.api,
            provider=self._model.provider,
            model=self._model.id,
            stop_reason=StopReason.LENGTH,
            timestamp=int(time.time() * 1000),
        )

    def _compact_context_if_needed(self, max_messages: int = 40) -> None:
        """Basic compaction: keep the first system-relevant turn and most recent messages.

        The original TypeScript implementation uses an LLM summarizer. This
        lightweight Python version prevents unbounded context growth while keeping
        recent conversational state intact.
        """
        if len(self._messages) <= max_messages:
            return
        kept = self._messages[-max_messages:]
        summary_text = f"[Context compacted: {len(self._messages) - len(kept)} older messages omitted]"
        summary = UserMessage(content=summary_text, timestamp=int(time.time() * 1000))
        self._messages = [summary, *kept]

    async def _consume_stream(self, stream: Any) -> AssistantMessage | None:
        """Consume an event stream and return the final AssistantMessage."""
        final_message: AssistantMessage | None = None

        async for event in stream:
            event_type = getattr(event, "type", "")
            self._emit(event)

            if event_type == "text_delta":
                delta = getattr(event, "delta", "")
                if hasattr(self, "_text_buffer"):
                    self._text_buffer += delta
                else:
                    self._text_buffer = delta

            elif event_type == "done":
                final_message = getattr(event, "message", None)
                self._text_buffer = ""

            elif event_type == "error":
                final_message = getattr(event, "error", None)
                self._text_buffer = ""

        return final_message


def create_agent_session(
    *,
    models: MutableModels,
    model: Model,
    cwd: str = "",
    config_dir: str | None = None,
    thinking_level: str = "off",
) -> AgentSession:
    """Create an AgentSession with resources loaded from config_dir and cwd."""
    resources = load_resources(cwd, config_dir) if config_dir else LoadedResources()

    return AgentSession(AgentSessionOptions(
        models=models,
        model=model,
        cwd=cwd,
        system_prompt=resources.system_prompt,
        append_system_prompt=resources.append_system_prompt,
        context_files=[{"path": f.path, "content": f.content} for f in resources.context_files],
        skills=resources.skills,
        thinking_level=thinking_level,
    ))
