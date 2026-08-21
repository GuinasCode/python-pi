"""Agent session orchestration for Pi coding agent.

Simplified port of packages/coding-agent/src/core/agent-session.ts.
Orchestrates: resolve model → build system prompt → run agent → handle tools.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pi_agent_core.types import AgentTool, AgentToolResult
from pi_ai import (
    AssistantMessage,
    Context,
    ImageContent,
    Message,
    Model,
    SimpleStreamOptions,
    StopReason,
    TextContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from pi_ai.models import MutableModels
from pi_coding_agent.extensions import (
    ExtensionFlag,
    ExtensionRunner,
    LoadExtensionsResult,
    RegisteredCommand,
    RegisteredShortcut,
    RegisteredTheme,
)
from pi_coding_agent.extensions.events import (
    AgentEndEvent,
    AgentStartEvent,
    SessionStartEvent,
    ToolCallEvent,
    ToolResultEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from pi_coding_agent.resource_loader import LoadedResources, load_resources
from pi_coding_agent.system_prompt import BuildSystemPromptOptions, build_system_prompt
from pi_coding_agent.tools import ToolResult, edit_file, execute_bash, grep_search, list_files, read_file, write_file
from pi_memory import MemoryStore, create_memory_tools


@dataclass
class AgentSessionStats:
    """Aggregated token usage, tool-call count, and cost for a session so far."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0
    tool_calls: int = 0
    cost_total: float = 0.0


@dataclass
class _ToolCallStartEvent:
    type: str = "tool_call_start"
    name: str = ""
    args: Any = None


@dataclass
class _ToolCallEndEvent:
    type: str = "tool_call_end"
    name: str = ""
    is_error: bool = False
    result_text: str = ""
    details: Any = None


@dataclass
class _MemoryDownloadEvent:
    type: str = "memory_download"
    message: str = ""


@dataclass
class _SubagentProgressEvent:
    type: str = "subagent_progress"
    text: str = ""


def _create_tool_result_message(
    tool_call: ToolCall,
    result: ToolResult,
    timestamp: int,
) -> ToolResultMessage:
    """Create a ToolResultMessage from a ToolResult."""
    raw = result.content if result.content else [{"type": "text", "text": ""}]
    content: list[TextContent | ImageContent] = [
        TextContent(text=b.get("text", "")) for b in raw if b.get("type") == "text"
    ]
    if not content:
        content = [TextContent(text="")]
    return ToolResultMessage(
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        content=content,
        details=result.details,
        is_error=result.is_error,
        timestamp=timestamp,
    )


def get_builtin_tools() -> list[Tool]:
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
    # Can include AgentTool instances alongside plain Tool schemas.
    tools: list[Tool] | None = None
    thinking_level: str = "off"
    temperature: float | None = None
    max_turns: int = 50
    # When True, the subagent tool is automatically added to builtin tools.
    enable_subagents: bool = True
    # Whether a human can be asked a follow-up question and reply in a later
    # turn (True for interactive mode, False for print/subagent runs).
    interactive: bool = True
    # Persistent memory store. When set, remember/recall tools are added and
    # the top matching memories are recalled and injected into the system
    # prompt at the start of every prompt() call.
    memory_store: MemoryStore | None = None
    memory_top_k: int = 3
    # Called before every tool call with (tool_name, arguments); return False
    # to deny it (a denial ToolResult is fed back to the model instead of
    # running the tool). None means every tool call is allowed unconditionally
    # — the caller (e.g. the interactive permission-mode footer) owns the
    # actual policy, this is just the enforcement point.
    permission_gate: Callable[[str, dict[str, Any]], Awaitable[bool]] | None = None
    # Only applies when `tools` is None (i.e. the default builtin toolset).
    # True disables every builtin tool; a list of names excludes just those.
    # An explicitly-passed `tools` list is never filtered by this.
    no_tools: bool | list[str] = False
    # Directory resources (skills/context files/system prompt) were loaded
    # from at construction time — kept only so reload() can re-run that load
    # against current on-disk state. None (the default) means resources were
    # supplied directly via context_files/skills/system_prompt and there is
    # nothing on disk for reload() to re-scan.
    config_dir: str | None = None
    # When set, tools registered by every extension under cwd's/config_dir's
    # .pi/extensions/ are loaded once at construction and re-loaded by
    # reload() — see pi_coding_agent.extensions.ExtensionRunner.
    extension_runner: ExtensionRunner | None = None


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
        self._thinking_level = options.thinking_level
        self._temperature = options.temperature
        self._max_turns = options.max_turns
        self._interactive = options.interactive
        self._messages: list[Message] = []
        self._event_listeners: list[Callable[[Any], None]] = []
        self._text_buffer = ""
        self._memory_store = options.memory_store
        self._memory_top_k = options.memory_top_k
        self._permission_gate = options.permission_gate
        self._config_dir = options.config_dir
        self._extension_runner = options.extension_runner
        self._extension_tool_names: set[str] = set()
        self._session_started = False
        if self._memory_store is not None:
            self._memory_store.embeddings.set_progress_callback(
                lambda message: self._emit(_MemoryDownloadEvent(message=message))
            )

        if options.tools is not None:
            base_tools: list[Tool] = options.tools
        elif options.no_tools is True:
            base_tools = []
        else:
            base_tools = get_builtin_tools()
            if options.no_tools:
                excluded = set(options.no_tools)
                base_tools = [t for t in base_tools if t.name not in excluded]
        if self._memory_store is not None and not any(t.name == "remember" for t in base_tools):
            base_tools = [*base_tools, *create_memory_tools(self._memory_store)]
        if options.enable_subagents and not any(t.name == "subagent" for t in base_tools):
            try:
                from pi_coding_agent.subagent.tool import create_subagent_tool

                def _on_subagent_progress(line: str) -> None:
                    self._emit(_SubagentProgressEvent(text=line))

                base_tools = [
                    *base_tools,
                    create_subagent_tool(
                        cwd=self._cwd,
                        config_dir=None,
                        on_progress=_on_subagent_progress,
                    ),
                ]
            except Exception:
                pass

        if self._extension_runner is not None:
            self._extension_runner.load()
            extension_tools = self._extension_runner.get_tools()
            base_tools = [*base_tools, *extension_tools]
            self._extension_tool_names = {t.name for t in extension_tools}

        self._tools = base_tools

    def on_event(self, listener: Callable[[Any], None]) -> Callable[[], None]:
        """Register an event listener."""
        self._event_listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._event_listeners:
                self._event_listeners.remove(listener)

        return unsubscribe

    def _emit(self, event: Any) -> None:
        for listener in list(self._event_listeners):
            listener(event)

    def get_active_tool_names(self) -> list[str]:
        """Names of tools currently registered on this session."""
        return [t.name for t in self._tools]

    def get_extensions(self) -> LoadExtensionsResult:
        """Result of the most recent extension load — empty when no
        extension_runner was configured for this session."""
        if self._extension_runner is None:
            return LoadExtensionsResult()
        return self._extension_runner.get_extensions()

    def get_extension_paths(self) -> list[str]:
        """Source paths of every successfully loaded extension."""
        if self._extension_runner is None:
            return []
        return self._extension_runner.get_extension_paths()

    def get_extension_commands(self) -> list[RegisteredCommand]:
        """Every slash command registered by a loaded extension."""
        if self._extension_runner is None:
            return []
        return self._extension_runner.get_commands()

    def get_extension_shortcuts(self) -> list[RegisteredShortcut]:
        """Every keybinding registered by a loaded extension."""
        if self._extension_runner is None:
            return []
        return self._extension_runner.get_shortcuts()

    def get_extension_themes(self) -> list[RegisteredTheme]:
        """Every color theme registered by a loaded extension."""
        if self._extension_runner is None:
            return []
        return self._extension_runner.get_themes()

    def get_extension_flags(self) -> dict[str, ExtensionFlag]:
        """Every CLI flag declared by a loaded extension."""
        if self._extension_runner is None:
            return {}
        return self._extension_runner.get_flags()

    def set_extension_flag_value(self, name: str, value: bool | str) -> None:
        """Set a flag's value, visible to every extension's pi.get_flag(name)."""
        if self._extension_runner is not None:
            self._extension_runner.set_flag_value(name, value)

    def get_last_assistant_text(self) -> str:
        """Concatenated text of the most recent assistant message, or ''."""
        for message in reversed(self._messages):
            if isinstance(message, AssistantMessage):
                return "".join(block.text for block in message.content if isinstance(block, TextContent))
        return ""

    def get_session_stats(self) -> AgentSessionStats:
        """Aggregate token usage, tool-call count, and cost across the session so far."""
        stats = AgentSessionStats()
        for message in self._messages:
            if isinstance(message, AssistantMessage):
                usage = message.usage
                stats.input_tokens += usage.input
                stats.output_tokens += usage.output
                stats.cache_read_tokens += usage.cache_read
                stats.cache_write_tokens += usage.cache_write
                stats.total_tokens += usage.total_tokens
                stats.cost_total += usage.cost.total
            elif isinstance(message, ToolResultMessage):
                stats.tool_calls += 1
        return stats

    def get_system_prompt(self) -> str:
        """The system prompt that would be sent on the next prompt() call (no memories)."""
        return self._build_system_prompt()

    def set_system_prompt_override(self, text: str) -> None:
        """Replace the system prompt outright for every subsequent prompt() call.

        Used by callers that need to transform the default prompt (e.g. an
        eval harness comparing prompt variants) without re-deriving it from
        context_files/skills each time.
        """
        self._system_prompt = text

    async def reload(self) -> None:
        """Re-run resource loading (system prompt/context files/skills) from
        config_dir, and re-load extensions, against current on-disk state.

        Each half is independently a no-op when its prerequisite wasn't
        supplied (config_dir for resources, extension_runner for
        extensions) — resources/tools that came in directly via
        AgentSessionOptions have nothing on disk to re-scan. Useful between
        prompt steps that create or modify project resources the session
        should pick up mid-run (e.g. an eval step that writes a new skill
        or extension file and expects the following prompt to see it).
        """
        if self._config_dir is not None:
            resources = load_resources(self._cwd, self._config_dir)
            self._system_prompt = resources.system_prompt
            self._append_system_prompt = resources.append_system_prompt
            self._context_files = [{"path": f.path, "content": f.content} for f in resources.context_files]
            self._skills = resources.skills

        if self._extension_runner is not None:
            self._tools = [t for t in self._tools if t.name not in self._extension_tool_names]
            self._extension_runner.load()
            extension_tools = self._extension_runner.get_tools()
            self._tools = [*self._tools, *extension_tools]
            self._extension_tool_names = {t.name for t in extension_tools}

    def _build_system_prompt(self, memories: list[str] | None = None) -> str:
        """Build the system prompt from options."""
        if self._system_prompt:
            prompt = self._system_prompt
            if self._append_system_prompt:
                prompt += f"\n\n{self._append_system_prompt}"
            prompt += f"\nCurrent working directory: {self._cwd}"
            return prompt
        return build_system_prompt(
            BuildSystemPromptOptions(
                cwd=self._cwd,
                append_system_prompt=self._append_system_prompt,
                context_files=self._context_files,
                skills=self._skills,
                selected_tools=[t.name for t in self._tools],
                interactive=self._interactive,
                memories=memories,
                memory_enabled=self._memory_store is not None,
            )
        )

    async def _recall_memories(self, text: str) -> list[str]:
        """Fetch the top matching memories for this turn, never raising."""
        store = self._memory_store
        if store is None:
            return []
        loop = asyncio.get_running_loop()
        try:
            records = await loop.run_in_executor(
                None,
                lambda: store.search(text, top_k=self._memory_top_k, project_cwd=self._cwd),
            )
        except Exception:
            return []
        return [f"[{r.type.value}] {r.title}: {r.content}" for r in records]

    async def _emit_ext(self, event_name: str, event: Any) -> None:
        """Fire a notification-only extension lifecycle event, if an
        extension_runner is configured. No-op otherwise."""
        if self._extension_runner is not None:
            await self._extension_runner.emit(event_name, event)

    async def prompt(self, text: str) -> AssistantMessage:
        """Send a prompt to the model and run the agent loop until completion."""
        if self._extension_runner is not None and not self._session_started:
            # Fired lazily on first use rather than in __init__, which is
            # synchronous while extension handlers are async.
            await self._extension_runner.emit("session_start", SessionStartEvent())
            self._session_started = True

        memories = await self._recall_memories(text)
        system_prompt = self._build_system_prompt(memories)
        user_msg = UserMessage(content=text, timestamp=int(time.time() * 1000))
        self._messages.append(user_msg)

        await self._emit_ext("agent_start", AgentStartEvent())
        try:
            return await self._run_turns(system_prompt)
        finally:
            await self._emit_ext("agent_end", AgentEndEvent())

    async def _run_turns(self, system_prompt: str) -> AssistantMessage:
        for turn_index in range(self._max_turns):
            await self._emit_ext("turn_start", TurnStartEvent(turn=turn_index))
            try:
                turn_result = await self._run_one_turn(system_prompt)
                if turn_result is not None:
                    return turn_result
            finally:
                await self._emit_ext("turn_end", TurnEndEvent(turn=turn_index))

        return AssistantMessage(
            content=[TextContent(text="Max turns reached")],
            api=self._model.api,
            provider=self._model.provider,
            model=self._model.id,
            stop_reason=StopReason.LENGTH,
            timestamp=int(time.time() * 1000),
        )

    async def _run_one_turn(self, system_prompt: str) -> AssistantMessage | None:
        """Run one model round-trip (+ any resulting tool calls). Returns the
        final AssistantMessage if the conversation should stop here, or None
        to continue looping (another turn follows, e.g. after tool use)."""
        self._compact_context_if_needed()
        context = Context(
            system_prompt=system_prompt,
            messages=self._messages,
            tools=self._tools,
        )

        stream_options = SimpleStreamOptions(temperature=self._temperature) if self._temperature is not None else None
        stream = self._models.stream(self._model, context, stream_options)
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

        # Execute tool calls — support both AgentTool (async) and builtin (sync).
        for block in assistant_msg.content:
            if not isinstance(block, ToolCall):
                continue
            await self._run_tool_call(block)

        return None

    async def _run_tool_call(self, tool_call: ToolCall) -> None:
        self._emit(_ToolCallStartEvent(name=tool_call.name, args=tool_call.arguments))

        if self._extension_runner is not None:
            call_event = ToolCallEvent(
                tool_call_id=tool_call.id, tool_name=tool_call.name, arguments=tool_call.arguments
            )
            call_result = await self._extension_runner.emit_tool_call(call_event)
            if call_result is not None and call_result.block:
                blocked_text = call_result.reason or "Blocked by an extension."
                self._emit(
                    _ToolCallEndEvent(name=tool_call.name, is_error=True, result_text=blocked_text, details=None)
                )
                self._messages.append(
                    ToolResultMessage(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        content=[TextContent(text=blocked_text)],
                        details=None,
                        is_error=True,
                        timestamp=int(time.time() * 1000),
                    )
                )
                return

        if self._permission_gate is not None and not await self._permission_gate(tool_call.name, tool_call.arguments):
            denial_text = "Permission denied: the user (or the current permission mode) did not approve this action."
            self._emit(_ToolCallEndEvent(name=tool_call.name, is_error=True, result_text=denial_text, details=None))
            self._messages.append(
                ToolResultMessage(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    content=[TextContent(text=denial_text)],
                    details=None,
                    is_error=True,
                    timestamp=int(time.time() * 1000),
                )
            )
            return

        result, is_error, details = await self._execute_tool_call(tool_call)
        tool_content: list[TextContent | ImageContent] = [
            TextContent(text=b.get("text", "")) for b in result if b.get("type") == "text"
        ]

        if self._extension_runner is not None:
            result_event = ToolResultEvent(
                tool_call_id=tool_call.id, tool_name=tool_call.name, content=tool_content, is_error=is_error
            )
            override = await self._extension_runner.emit_tool_result(result_event)
            if override is not None:
                if override.content is not None:
                    tool_content = override.content
                if override.is_error is not None:
                    is_error = override.is_error

        result_text = "".join(b.text for b in tool_content if isinstance(b, TextContent))
        self._emit(_ToolCallEndEvent(name=tool_call.name, is_error=is_error, result_text=result_text, details=details))
        self._messages.append(
            ToolResultMessage(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                content=tool_content,
                details=details,
                is_error=is_error,
                timestamp=int(time.time() * 1000),
            )
        )

    async def _execute_tool_call(
        self,
        tool_call: ToolCall,
    ) -> tuple[list[dict[str, str]], bool, Any]:
        """Execute a tool call, returning (content_dicts, is_error, details).

        If the tool is an :class:`AgentTool` with an async ``execute``
        callback, that is called directly.  Otherwise the call is dispatched
        to the synchronous built-in executor.
        """
        for tool in self._tools:
            if tool.name != tool_call.name:
                continue
            if isinstance(tool, AgentTool) and tool.execute is not None:
                try:
                    agent_result: AgentToolResult = await tool.execute(tool_call.id, tool_call.arguments, None, None)
                    content = [{"type": "text", "text": b.text} for b in agent_result.content if hasattr(b, "text")]
                    return content or [{"type": "text", "text": ""}], False, None
                except Exception as exc:
                    return [{"type": "text", "text": str(exc)}], True, None

        # Built-in tools: run off the event loop since execute_bash/read_file/etc.
        # call blocking subprocess/filesystem APIs and would otherwise stall
        # every other coroutine (streaming, input handling) for the duration.
        loop = asyncio.get_running_loop()
        sync_result = await loop.run_in_executor(None, _execute_tool, tool_call.name, tool_call.arguments)
        return sync_result.content, sync_result.is_error, sync_result.details

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

    return AgentSession(
        AgentSessionOptions(
            models=models,
            model=model,
            cwd=cwd,
            system_prompt=resources.system_prompt,
            append_system_prompt=resources.append_system_prompt,
            context_files=[{"path": f.path, "content": f.content} for f in resources.context_files],
            skills=resources.skills,
            thinking_level=thinking_level,
        )
    )
