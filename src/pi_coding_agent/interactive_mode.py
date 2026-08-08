"""Interactive mode for Pi coding agent.

Provides a REPL (Read-Eval-Print Loop) that maintains conversation context,
executes tools, persists sessions, and loads project resources.

This is the main interactive experience when running `pi` without `-p`.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markup import escape

from pi_ai import (
    AssistantMessage,
    Message,
    Model,
    StopReason,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from pi_ai.models import MutableModels, Provider
from pi_coding_agent import Args
from pi_coding_agent.agent_session import AgentSession, AgentSessionOptions
from pi_coding_agent.config import ensure_config_dir, ensure_session_dir, get_config_dir, get_session_dir
from pi_coding_agent.diff_render import render_diff
from pi_coding_agent.markdown_render import LeftMarkdown as Markdown
from pi_coding_agent.resource_loader import load_resources
from pi_coding_agent.session_manager import SessionEntry, SessionManager
from pi_coding_agent.styles import DIM_STYLE, PASTEL_BLUE, PASTEL_GREEN, PASTEL_RED, PASTEL_YELLOW

_console = Console(highlight=False, soft_wrap=True)
_err_console = Console(highlight=False, soft_wrap=True, stderr=True)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt_args(args: Any) -> str:
    """Format tool arguments into a concise single-line representation."""
    if not args:
        return ""
    if isinstance(args, dict):
        parts: list[str] = []
        for k, v in args.items():
            if v is None:
                continue
            if isinstance(v, str):
                # Truncate long strings
                display = v if len(v) <= 60 else v[:57] + "..."
                parts.append(f'{k}="{escape(display)}"')
            else:
                parts.append(f"{k}={escape(str(v))}")
        return ", ".join(parts)
    return escape(str(args))


def _fmt_result_preview(text: str, max_lines: int = 8) -> str:
    """Return a dim-styled preview of tool output."""
    if not text or not text.strip():
        return ""
    lines = text.rstrip().splitlines()
    shown = lines[:max_lines]
    preview_lines = [f"  [dim]{escape(line)}[/dim]" for line in shown]
    if len(lines) > max_lines:
        preview_lines.append(f"  [dim]... ({len(lines) - max_lines} more lines)[/dim]")
    return "\n".join(preview_lines)


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------


def _setup_models(args: Args) -> tuple[MutableModels, Any]:
    """Set up models with available providers.

    Priority order:
    1. NVAPI_KEY  -> NVIDIA GLM 5.2
    2. OPENAI_API_KEY -> OpenAI
    3. Faux provider (fallback for testing)
    """
    models = MutableModels()

    nvapi_key = args.api_key or os.environ.get("NVAPI_KEY")
    if nvapi_key:
        try:
            from pi_ai.providers.nvidia_glm import nvidia_glm_provider

            model, provider_models, _meta = nvidia_glm_provider(
                Model(id="test"),
                api_key=nvapi_key,
            )
            return provider_models, model
        except Exception as e:
            _err_console.print(f"[yellow]warning:[/yellow] NVIDIA provider failed: {e}")

    openai_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if openai_key:
        try:
            from pi_ai.providers.openai import openai_provider

            model, stream_fn = openai_provider(api_key=openai_key)
            provider: Provider[Any] = Provider(
                id="openai",
                name="OpenAI",
                models=[model],
                stream_fn=stream_fn,
            )
            models.set_provider(provider)
            return models, model
        except Exception:
            pass

    from pi_ai.providers.faux import faux_assistant_message, faux_provider

    def faux_reply(context: Any, _options: Any, _state: dict[str, int], _model: Any) -> AssistantMessage:
        last_user = ""
        for message in reversed(context.messages):
            if message.role == "user":
                last_user = message.content if isinstance(message.content, str) else str(message.content)
                break
        return faux_assistant_message(
            "(faux provider: set NVAPI_KEY or OPENAI_API_KEY for real LLM responses)\n"
            f"I received your message: {last_user}"
        )

    handle = faux_provider()
    handle.set_responses([faux_reply for _ in range(500)])
    models.set_provider(handle.provider)
    return models, handle.get_model()


def _setup_models_with_settings(args: Args) -> tuple[MutableModels, Any]:
    """Set up models using settings_manager and model_resolver if available."""
    try:
        from pi_coding_agent.settings_manager import SettingsManager

        settings_mgr = SettingsManager.create(cwd=str(Path.cwd()), agent_dir=get_config_dir())
        provider_name = settings_mgr.get_default_provider()
        model_id = settings_mgr.get_default_model()
    except Exception:
        provider_name = None
        model_id = None

    if args.provider:
        provider_name = args.provider
    if args.model:
        model_id = args.model

    models, default_model = _setup_models(args)

    if model_id and provider_name:
        found = models.get_model(provider_name, model_id)
        if found:
            return models, found

    return models, default_model


# ---------------------------------------------------------------------------
# Interactive session
# ---------------------------------------------------------------------------


class InteractiveSession:
    """Interactive REPL session that maintains context and persists to disk."""

    def __init__(
        self,
        *,
        models: MutableModels,
        model: Any,
        cwd: str,
        config_dir: Path | None = None,
        session_manager: SessionManager | None = None,
        session_id: str | None = None,
    ) -> None:
        self._models = models
        self._model = model
        self._cwd = cwd
        self._config_dir = config_dir or get_config_dir()

        resources = load_resources(cwd, self._config_dir)

        self._agent_session = AgentSession(
            AgentSessionOptions(
                models=models,
                model=model,
                cwd=cwd,
                system_prompt=resources.system_prompt,
                append_system_prompt=resources.append_system_prompt,
                context_files=[{"path": f.path, "content": f.content} for f in resources.context_files],
                skills=resources.skills,
            )
        )

        self._session_manager = session_manager
        self._session_id = session_id
        self._message_count = 0
        self._restore_persisted_messages()

        self._running = False
        self._event_handlers: list[Callable[[str], None]] = []
        self._turn_has_text = False
        self._thinking_open = False
        # Buffer for the current text block — flushed as Markdown on text_end.
        self._text_block_buf = ""
        # Human-readable current status shown below the input box.
        self._status = "ready"

    def _restore_persisted_messages(self) -> None:
        if not self._session_manager or not self._session_id:
            return
        entries = self._session_manager.get_entries(self._session_id)
        for entry in entries:
            if entry.kind != "message":
                continue
            data = entry.data
            role = data.get("role", "user")
            if role == "user":
                msg: Message = UserMessage(content=data.get("content", ""), timestamp=data.get("timestamp", 0))
            elif role == "assistant":
                content_blocks: list[TextContent | ThinkingContent | ToolCall] = []
                for b in data.get("content", []):
                    if b.get("type") == "text":
                        content_blocks.append(TextContent(text=b.get("text", "")))
                msg = AssistantMessage(
                    content=content_blocks,
                    model=data.get("model", ""),
                    stop_reason=StopReason[data.get("stop_reason", "stop").upper()],
                    timestamp=data.get("timestamp", 0),
                )
            elif role == "toolResult":
                continue
            else:
                continue
            self._agent_session._messages.append(msg)
            self._message_count += 1

    def _persist_message(self, message: Message) -> None:
        if not self._session_manager or not self._session_id:
            return

        data: dict[str, Any] = {"role": message.role}

        if isinstance(message, UserMessage):
            content = message.content
            if isinstance(content, str):
                data["content"] = content
            elif isinstance(content, list):
                data["content"] = [
                    {"type": b.type, "text": b.text} if isinstance(b, TextContent) else str(b) for b in content
                ]
        elif isinstance(message, AssistantMessage):

            def _block_to_dict(b: TextContent | ThinkingContent | ToolCall) -> dict[str, Any]:
                if isinstance(b, TextContent):
                    return {"type": b.type, "text": b.text}
                if isinstance(b, ThinkingContent):
                    return {"type": b.type, "thinking": b.thinking}
                return {"type": b.type, "id": b.id, "name": b.name, "arguments": b.arguments}

            data["content"] = [_block_to_dict(b) for b in message.content]
            data["model"] = message.model
            data["stop_reason"] = (
                message.stop_reason.value if hasattr(message.stop_reason, "value") else str(message.stop_reason)
            )
        elif isinstance(message, ToolResultMessage):
            data["tool_call_id"] = message.tool_call_id
            data["tool_name"] = message.tool_name
            data["is_error"] = message.is_error
            data["content"] = [
                {"type": b.type, "text": b.text} if isinstance(b, TextContent) else {"type": b.type}
                for b in message.content
            ]
            if message.details is not None:
                data["details"] = message.details

        entry = SessionEntry(
            seq=self._message_count,
            parent_seq=self._message_count - 1 if self._message_count > 0 else None,
            kind="message",
            data=data,
            timestamp=int(time.time() * 1000),
        )
        self._session_manager.append_entry(self._session_id, entry)
        self._message_count += 1

    # ------------------------------------------------------------------
    # Rich event display
    # ------------------------------------------------------------------

    def _flush_text_block(self) -> None:
        """Render the accumulated text block as left-aligned Markdown and reset."""
        buf = self._text_block_buf
        self._text_block_buf = ""
        if not buf:
            return
        if buf.strip():
            _console.print(Markdown(buf))
        else:
            _console.print(buf)

    def _handle_event(self, event: Any) -> None:
        """Display agent events with rich formatting."""
        t = getattr(event, "type", "")

        # ── thinking ──────────────────────────────────────────────────
        if t == "thinking_start":
            self._thinking_open = True
            _console.print(f"[{DIM_STYLE} italic]thinking[/{DIM_STYLE} italic]")

        elif t == "thinking_delta":
            delta = getattr(event, "delta", "")
            _console.print(f"[{DIM_STYLE} italic]{escape(delta)}[/{DIM_STYLE} italic]", end="")

        elif t == "thinking_end":
            self._thinking_open = False
            _console.print()

        # ── text streaming: buffer and render as Markdown on text_end ─
        elif t == "text_start":
            self._turn_has_text = True
            self._text_block_buf = ""

        elif t == "text_delta":
            self._text_block_buf += getattr(event, "delta", "")

        elif t == "text_end":
            self._flush_text_block()

        # ── model announcing a tool call ──────────────────────────────
        elif t == "toolcall_end":
            tool_call = getattr(event, "tool_call", None)
            if tool_call:
                args_str = _fmt_args(getattr(tool_call, "arguments", {}))
                _console.print(f"\n[bold {PASTEL_BLUE}]> {escape(tool_call.name)}[/bold {PASTEL_BLUE}]({args_str})")

        # ── tool execution (AgentSession) ─────────────────────────────
        elif t == "tool_call_start":
            name = getattr(event, "name", "")
            args = getattr(event, "args", {}) or {}
            args_str = _fmt_args(args)
            self._status = f"running: {name}"
            _console.print(f"[{PASTEL_YELLOW}]~[/{PASTEL_YELLOW}] [bold]{escape(name)}[/bold]({args_str})")

        elif t == "tool_call_end":
            name = getattr(event, "name", "")
            is_error = getattr(event, "is_error", False)
            result_text = getattr(event, "result_text", "")
            details = getattr(event, "details", None)
            self._status = "thinking..."
            icon_style = PASTEL_RED if is_error else PASTEL_GREEN
            icon = "!" if is_error else "+"
            _console.print(f"[{icon_style}]{icon}[/{icon_style}] [bold]{escape(name)}[/bold]")
            diff = (details or {}).get("diff") if isinstance(details, dict) else None
            if diff:
                _console.print(render_diff(diff))
            else:
                preview = _fmt_result_preview(result_text)
                if preview:
                    _console.print(preview)

        # ── done: flush any remaining text (stream ended without text_end)
        elif t == "done":
            self._flush_text_block()
            self._status = "ready"

        # ── error ─────────────────────────────────────────────────────
        elif t == "error":
            self._flush_text_block()
            self._status = "ready"
            err = getattr(event, "error", None)
            if isinstance(err, str):
                msg = err or "Unknown error"
            else:
                msg = getattr(err, "error_message", None) or "Unknown error"
            _console.print(f"[{PASTEL_RED}]error:[/{PASTEL_RED}] {escape(msg)}")

    async def run_turn(self, user_input: str) -> AssistantMessage | None:
        """Run a single conversation turn."""
        user_msg = UserMessage(content=user_input, timestamp=int(time.time() * 1000))
        self._persist_message(user_msg)

        self._turn_has_text = False
        self._thinking_open = False
        unsub = self._agent_session.on_event(self._handle_event)
        try:
            result = await self._agent_session.prompt(user_input)
        except Exception as exc:
            _console.print(f"\n[{PASTEL_RED}]error:[/{PASTEL_RED}] {escape(str(exc))}")
            return None
        finally:
            unsub()

        if result:
            self._persist_message(result)

        return result

    # ------------------------------------------------------------------
    # Input area layout
    # ------------------------------------------------------------------

    def _state_line(self) -> str:
        """One-line status shown below the input box."""
        provider = getattr(self._model, "provider", "?")
        model_id = getattr(self._model, "id", "?")
        session_info = f"session:{self._session_id[:8]}" if self._session_id else "no session"
        dot = "[green]●[/green]" if self._status == "ready" else "[yellow]●[/yellow]"
        return f"{dot} [dim]{self._status}  ·  {provider}/{model_id}  ·  {session_info}[/dim]"

    async def _prompt_input(self) -> str | None:
        """Draw the bordered input area, return stripped text or None on exit."""
        w = _console.width

        # Tips — left-aligned, one line above the top border
        tips = "/help  /clear  /model  /session  /exit"
        _console.print(f"[{DIM_STYLE}]{tips}[/{DIM_STYLE}]")

        # Top rule — plain horizontal line spanning the terminal width
        sys.stdout.write("─" * w + "\n")
        sys.stdout.flush()

        # Input line — the terminal wraps long input across multiple lines
        # on its own, so the rule above/below grows with the input naturally.
        try:
            raw = await asyncio.get_running_loop().run_in_executor(None, lambda: input("\033[1m>\033[0m "))
        except EOFError:
            sys.stdout.write("\n" + "─" * w + "\n")
            sys.stdout.flush()
            return None
        except KeyboardInterrupt:
            sys.stdout.write("\n" + "─" * w + "\n")
            sys.stdout.flush()
            _console.print("[dim]interrupted[/dim]")
            return ""

        # Bottom rule
        sys.stdout.write("─" * w + "\n")
        sys.stdout.flush()

        # State line immediately below bottom border
        _console.print(self._state_line())

        return raw.strip()

    def _print_user_message(self, text: str) -> None:
        """Display the user's message with a highlighted background."""
        w = _console.width
        # Pad to full terminal width so background fills the line
        padded = f"  {text}  "
        _console.print(f"[bold on grey23]{escape(padded):<{w}}[/bold on grey23]")
        _console.print()

    # ------------------------------------------------------------------
    # REPL loop
    # ------------------------------------------------------------------

    async def repl_loop(self) -> None:
        """Main REPL loop."""
        self._running = True

        provider_name = getattr(self._model, "provider", "faux")
        model_name = getattr(self._model, "name", "unknown")
        model_id = getattr(self._model, "id", "?")

        # One-line banner — left-aligned
        _console.print(
            f"[bold]pi[/bold] v0.83.0  [dim]·[/dim]"
            f"  [{PASTEL_BLUE}]{provider_name}[/{PASTEL_BLUE}] / [{PASTEL_BLUE}]{model_name}[/{PASTEL_BLUE}]"
            f"  [dim]({model_id})[/dim]"
            f"  [dim]{self._cwd}[/dim]"
        )
        _console.print()

        while self._running:
            user_input = await self._prompt_input()

            if user_input is None:  # EOF
                break
            if not user_input:  # empty or interrupted
                continue

            if user_input.startswith("/"):
                if not self._handle_command(user_input):
                    break
                continue

            _console.print()
            self._print_user_message(user_input)
            self._status = "thinking..."
            result = await self.run_turn(user_input)
            self._status = "ready"
            _console.print()

            if result and result.stop_reason == StopReason.ERROR:
                error_msg = result.error_message or "Unknown error"
                _console.print(f"[{PASTEL_RED}]error:[/{PASTEL_RED}] {escape(error_msg)}")

    def _handle_command(self, command: str) -> bool:
        """Handle slash commands. Returns True to continue, False to exit."""
        cmd = command.lower().strip()

        if cmd in ("/exit", "/quit", "/q"):
            return False

        if cmd == "/help":
            _console.print(
                "\n[bold]Commands[/bold]\n"
                f"  [{PASTEL_BLUE}]/help[/{PASTEL_BLUE}]     Show this help\n"
                f"  [{PASTEL_BLUE}]/exit[/{PASTEL_BLUE}]     Exit Pi\n"
                f"  [{PASTEL_BLUE}]/model[/{PASTEL_BLUE}]    Show current model\n"
                f"  [{PASTEL_BLUE}]/clear[/{PASTEL_BLUE}]    Clear conversation history\n"
                f"  [{PASTEL_BLUE}]/tools[/{PASTEL_BLUE}]    List available tools\n"
                f"  [{PASTEL_BLUE}]/session[/{PASTEL_BLUE}]  Show session info\n"
            )
            return True

        if cmd == "/model":
            _console.print(
                f"Provider: [{PASTEL_BLUE}]{getattr(self._model, 'provider', '?')}[/{PASTEL_BLUE}]\n"
                f"Model:    [{PASTEL_BLUE}]{getattr(self._model, 'id', '?')}[/{PASTEL_BLUE}]\n"
                f"Context:  {getattr(self._model, 'context_window', '?')} tokens"
            )
            return True

        if cmd == "/clear":
            self._agent_session._messages = []
            _console.print("[dim]conversation cleared[/dim]")
            return True

        if cmd == "/tools":
            for tool in self._agent_session._tools:
                _console.print(f"  [{PASTEL_BLUE}]{tool.name}[/{PASTEL_BLUE}]: {tool.description}")
            return True

        if cmd == "/session":
            if self._session_id:
                _console.print(f"Session ID: [dim]{self._session_id}[/dim]\nMessages:   {self._message_count}")
            else:
                _console.print("[dim]no active session[/dim]")
            return True

        _console.print(f"[yellow]Unknown command:[/yellow] {command}")
        return True


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


async def run_interactive_mode(args: Args) -> int:
    """Run Pi in interactive mode."""
    cwd = str(Path.cwd())
    config_dir = get_config_dir()
    ensure_config_dir()
    session_dir = get_session_dir()
    ensure_session_dir()

    models, model = _setup_models_with_settings(args)
    if model is None:
        _console.print(f"[{PASTEL_RED}]error:[/{PASTEL_RED}] No model available")
        return 1

    session_mgr = SessionManager(session_dir)

    session_id = None
    if args.continue_session or args.resume:
        recent = session_mgr.continue_recent()
        if recent:
            session_id = recent.id
            _console.print(f"[dim]continuing session: {recent.id}[/dim]")
        else:
            _console.print("[dim]no previous session found, starting new[/dim]")

    if not session_id and not args.no_session:
        session_name = args.name or Path(cwd).name
        info = session_mgr.create_session(cwd=cwd, name=session_name)
        session_id = info.id

    session = InteractiveSession(
        models=models,
        model=model,
        cwd=cwd,
        config_dir=config_dir,
        session_manager=session_mgr if session_id else None,
        session_id=session_id,
    )

    try:
        await session.repl_loop()
    except KeyboardInterrupt:
        _console.print("\n[dim]exiting[/dim]")

    return 0


def run_interactive_sync(args: Args) -> int:
    """Synchronous wrapper for run_interactive_mode."""
    return asyncio.run(run_interactive_mode(args))
