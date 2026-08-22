"""Interactive mode for Pi coding agent.

Provides a REPL (Read-Eval-Print Loop) that maintains conversation context,
executes tools, persists sessions, and loads project resources.

This is the main interactive experience when running `pi` without `-p`.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import sys
import time
from collections.abc import Awaitable, Callable
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
from pi_coding_agent.extension_ui import ExtensionUIContext, NoopExtensionUIContext
from pi_coding_agent.extensions import ExtensionRunner
from pi_coding_agent.git_info import get_git_repo_line
from pi_coding_agent.markdown_render import LeftMarkdown as Markdown
from pi_coding_agent.output_sink import ConsoleOutputSink, OutputSink
from pi_coding_agent.permission_mode import (
    PermissionDecision,
    PermissionMode,
    cycle_permission_mode,
    permission_decision,
    permission_mode_label,
)
from pi_coding_agent.resource_loader import load_resources
from pi_coding_agent.session_manager import SessionEntry, SessionManager
from pi_coding_agent.styles import DIM_STYLE, PASTEL_BLUE, PASTEL_GREEN, PASTEL_RED, PASTEL_YELLOW
from pi_memory import MemoryStore
from pi_tui.raw_input import read_line_with_cycle

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
        output: OutputSink | None = None,
    ) -> None:
        self._models = models
        self._model = model
        self._cwd = cwd
        self._config_dir = config_dir or get_config_dir()
        # Where run_turn()/_handle_command() render to. Defaults to the
        # classic REPL's Console — the Textual app (--ui-mode fullscreen)
        # passes its own sink so the exact same event/command handling
        # renders into a transcript widget instead of straight to stdout.
        self._output: OutputSink = output or ConsoleOutputSink(_console)
        # How _permission_gate's "ask" decision actually asks. Defaults to
        # the REPL's blocking y/N input() prompt; the Textual app (Phase
        # T2) swaps this for an async modal dialog instead, since a
        # blocking input() call would freeze its event loop.
        self._confirm_tool_fn: Callable[[str, dict[str, Any]], Awaitable[bool]] = self._confirm_tool
        # Interactive-prompt surface passed to extension handlers via
        # ExtensionContext.ui (Phase H). Defaults to a no-op — the REPL has
        # no way to show a select/input dialog without hand-building one,
        # which isn't worth doing for a surface the Textual app already
        # covers; the Textual app (PiApp.on_mount) swaps this for a
        # dialog-backed implementation.
        self._ui_context: ExtensionUIContext = NoopExtensionUIContext()

        resources = load_resources(cwd, self._config_dir)

        memory_store: MemoryStore | None = None
        memory_top_k = 3
        try:
            from pi_coding_agent.settings_manager import SettingsManager

            settings_mgr = SettingsManager.create(cwd=cwd, agent_dir=self._config_dir)
            if settings_mgr.get_memory_enabled():
                memory_store = MemoryStore(settings_mgr.get_memory_db_path())
                memory_top_k = settings_mgr.get_memory_top_k()
        except Exception:
            memory_store = None

        # Cyclable permission mode (shift+tab), shown in the footer and
        # enforced via _permission_gate below — set before constructing the
        # AgentSession since the gate closure reads it on every tool call.
        self._permission_mode: PermissionMode = PermissionMode.DEFAULT

        self._agent_session = AgentSession(
            AgentSessionOptions(
                models=models,
                model=model,
                cwd=cwd,
                system_prompt=resources.system_prompt,
                append_system_prompt=resources.append_system_prompt,
                context_files=[{"path": f.path, "content": f.content} for f in resources.context_files],
                skills=resources.skills,
                memory_store=memory_store,
                memory_top_k=memory_top_k,
                permission_gate=self._permission_gate,
                extension_runner=ExtensionRunner(cwd, self._config_dir, models=models),
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
        # Phase T6: called whenever _handle_event has just processed an
        # event (i.e. self._status may have changed) — PiApp uses this to
        # refresh its footer live during a turn (thinking/running tool/
        # ready), instead of only at submission start/end. None in the
        # classic REPL, which prints its status line directly rather than
        # keeping a persistent footer to refresh.
        self._on_status_change: Callable[[], None] | None = None

    def _cycle_permission_mode(self) -> None:
        self._permission_mode = cycle_permission_mode(self._permission_mode)

    async def _permission_gate(self, tool_name: str, args: dict[str, Any]) -> bool:
        """AgentSessionOptions.permission_gate: allow/ask/deny a tool call
        based on the current permission mode, before it runs."""
        decision = permission_decision(self._permission_mode, tool_name)
        if decision is PermissionDecision.ALLOW:
            return True
        if decision is PermissionDecision.DENY:
            self._output.print(
                f"[{PASTEL_RED}]blocked[/{PASTEL_RED}] [bold]{escape(tool_name)}[/bold] — plan mode is "
                "read-only [dim](shift+tab to change mode)[/dim]"
            )
            return False
        return await self._confirm_tool_fn(tool_name, args)

    async def _confirm_tool(self, tool_name: str, args: dict[str, Any]) -> bool:
        """Ask the user to approve a single mutating tool call (y/N)."""
        args_str = _fmt_args(args)
        _console.print(
            f"[{PASTEL_YELLOW}]?[/{PASTEL_YELLOW}] Allow [bold]{escape(tool_name)}[/bold]({args_str})? "
            "[dim](y/N)[/dim]",
            end=" ",
        )
        loop = asyncio.get_running_loop()
        try:
            answer = await loop.run_in_executor(None, input)
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        return answer.strip().lower() in ("y", "yes")

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

    def _render_entry(self, tool_name: str, phase: str, event: dict[str, Any]) -> bool:
        """Phase G: if an extension registered an entry renderer for
        `tool_name`, render its "start"/"end" transcript line through it
        instead of the default. Returns True if it handled rendering
        (whether or not it actually printed anything — a renderer that
        returns None still falls through to the caller's default, but one
        that returns "" is a deliberate no-op the caller must not
        double-render), False if no renderer is registered for this tool."""
        renderer = self._agent_session.get_extension_entry_renderer(tool_name)
        if renderer is None:
            return False
        from pi_coding_agent.extensions.events import ExtensionContext

        rendered = renderer(phase, event, ExtensionContext(cwd=self._cwd, ui=self._ui_context))
        if rendered is None:
            return False
        if isinstance(rendered, str):
            if rendered:
                self._output.print(rendered)
        else:
            self._output.print_renderable(rendered)
        return True

    def _flush_text_block(self) -> None:
        """Render the accumulated text block as left-aligned Markdown and reset."""
        buf = self._text_block_buf
        self._text_block_buf = ""
        if not buf:
            return
        if buf.strip():
            from pi_coding_agent.extensions.events import ExtensionContext

            ctx = ExtensionContext(cwd=self._cwd, ui=self._ui_context)
            for transformer in self._agent_session.get_extension_markdown_transformers():
                buf = transformer(buf, ctx)
            renderer = self._agent_session.get_extension_message_renderer("assistant")
            if renderer is not None:
                rendered = renderer(buf, ctx)
                if rendered is not None:
                    if isinstance(rendered, str):
                        self._output.print(rendered)
                    else:
                        self._output.print_renderable(rendered)
                    return
            self._output.print_renderable(Markdown(buf))
        else:
            self._output.print(buf)

    def _handle_event(self, event: Any) -> None:
        """Display agent events with rich formatting."""
        t = getattr(event, "type", "")

        # ── thinking ──────────────────────────────────────────────────
        if t == "thinking_start":
            self._thinking_open = True
            self._output.print(f"[{DIM_STYLE} italic]thinking[/{DIM_STYLE} italic]")

        elif t == "thinking_delta":
            delta = getattr(event, "delta", "")
            self._output.print(f"[{DIM_STYLE} italic]{escape(delta)}[/{DIM_STYLE} italic]", end="")

        elif t == "thinking_end":
            self._thinking_open = False
            self._output.print()

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
                self._output.print(f"\n[bold {PASTEL_BLUE}]> {escape(tool_call.name)}[/bold {PASTEL_BLUE}]({args_str})")

        # ── tool execution (AgentSession) ─────────────────────────────
        elif t == "tool_call_start":
            name = getattr(event, "name", "")
            args = getattr(event, "args", {}) or {}
            args_str = _fmt_args(args)
            self._status = f"running: {name}"
            if not self._render_entry(name, "start", {"args": args}):
                self._output.print(f"[{PASTEL_YELLOW}]~[/{PASTEL_YELLOW}] [bold]{escape(name)}[/bold]({args_str})")

        elif t == "tool_call_end":
            name = getattr(event, "name", "")
            is_error = getattr(event, "is_error", False)
            result_text = getattr(event, "result_text", "")
            details = getattr(event, "details", None)
            self._status = "thinking..."
            if not self._render_entry(name, "end", {"result_text": result_text, "is_error": is_error}):
                icon_style = PASTEL_RED if is_error else PASTEL_GREEN
                icon = "!" if is_error else "+"
                self._output.print(f"[{icon_style}]{icon}[/{icon_style}] [bold]{escape(name)}[/bold]")
                if self._ui_context.get_tools_expanded():
                    diff = (details or {}).get("diff") if isinstance(details, dict) else None
                    if diff:
                        self._output.print_renderable(render_diff(diff))
                    else:
                        preview = _fmt_result_preview(result_text)
                        if preview:
                            self._output.print(preview)

        # ── one-time local memory embedding model download ──────────────
        elif t == "memory_download":
            message = getattr(event, "message", "")
            self._output.print(f"[{DIM_STYLE}]{escape(message)}[/{DIM_STYLE}]")

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
            self._output.print(f"[{PASTEL_RED}]error:[/{PASTEL_RED}] {escape(msg)}")

        if self._on_status_change is not None:
            self._on_status_change()

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
            self._output.print(f"\n[{PASTEL_RED}]error:[/{PASTEL_RED}] {escape(str(exc))}")
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

    def _mode_line(self) -> str:
        """Permission-mode line shown below the state line (shift+tab to cycle)."""
        label = permission_mode_label(self._permission_mode)
        color = PASTEL_YELLOW if self._permission_mode is not PermissionMode.DEFAULT else DIM_STYLE
        return f"[{color}]{escape(label)}[/{color}] [{DIM_STYLE}](shift+tab to cycle)[/{DIM_STYLE}]"

    def _repo_line(self) -> str | None:
        """'(repo-name:branch)' line shown below the permission-mode line, if cwd is a git repo."""
        line = get_git_repo_line(self._cwd)
        if not line:
            return None
        return f"[{DIM_STYLE}]{escape(line)}[/{DIM_STYLE}]"

    async def _prompt_input(self) -> str | None:
        """Draw the bordered input area with a live footer, return stripped text or None on exit.

        The whole block (prompt + typed text + bottom rule + state/mode
        [/repo] lines) is fully redrawn, anchored at a cursor position
        saved once before typing starts, on every keystroke — not just
        when Shift+Tab is pressed. That's a deliberate trade: input can
        wrap across multiple terminal rows once it's long enough (the
        terminal does this on its own), and the footer sits immediately
        below it — incrementally patching the screen in place while
        tracking exactly how many rows the wrapped input currently
        occupies (and where within them the cursor now sits, since
        Left/Right/Home/End can put it anywhere in the buffer, not just
        the end) is fragile row-arithmetic that used to corrupt the
        display (new characters landing on top of already-drawn footer
        text once input grew past one row, since the footer's position
        was computed once up front assuming it never would). A full
        repaint from a fixed anchor, positioning everything as an
        explicit (row, column) offset *from that anchor* rather than
        relative to wherever the cursor happens to currently be, sidesteps
        needing that arithmetic at all — and a human's typing rate makes
        the repaint itself unnoticeable.
        """
        w = _console.width

        # Tips — left-aligned, one line above the top border
        tips = "/help  /clear  /model  /session  /exit"
        _console.print(f"[{DIM_STYLE}]{tips}[/{DIM_STYLE}]")

        # Top rule — plain horizontal line spanning the terminal width.
        # \r\n (not bare \n): this runs before _raw_mode is entered inside
        # read_line_with_cycle, but _render below also writes during raw
        # mode, where ONLCR is off — using \r\n unconditionally everywhere
        # in this method keeps its behavior identical in both cases rather
        # than depending on which mode happens to be active when it runs.
        sys.stdout.write("─" * w + "\r\n")
        sys.stdout.flush()

        repo_line = self._repo_line()
        footer_rows = 3 + (1 if repo_line else 0)  # bottom rule + state + mode [+ repo]

        prompt = "\033[1m>\033[0m "
        prompt_visible_len = 2  # "> " — the bold/reset codes above are zero-width

        def _row_col(pos: int) -> tuple[int, int]:
            """0-indexed row and 1-indexed column, both relative to the
            anchor, of the position right after `pos` characters (prompt
            included) have been laid out at terminal width `w`."""
            return pos // w, (pos % w) + 1

        # Anchor: the still-blank row input is about to start on. Every
        # write below is positioned as an explicit offset from here,
        # restoring to it first — never relative to wherever the cursor
        # currently is, which is what let the old scheme's assumptions
        # drift out of sync with reality once input wrapped.
        sys.stdout.write("\x1b7")  # DECSC
        sys.stdout.flush()

        # Remembers the last render so land_below_footer (called after
        # read_line_with_cycle returns/raises, when _edit_loop's buffer is
        # no longer reachable) can still compute where the footer ended up.
        last_text = ""

        def _render(text: str, cursor: int, selection: tuple[int, int] | None) -> None:
            nonlocal last_text
            last_text = text

            sys.stdout.write("\x1b8\x1b[0J")  # anchor, wipe any longer previous render
            if selection is None:
                sys.stdout.write(prompt + text)
            else:
                # SGR 7/27 (reverse video on/off) around the selected span
                # — zero-width control codes, don't affect the row/column
                # math below, which only ever counts visible characters.
                lo, hi = selection
                sys.stdout.write(prompt + text[:lo] + "\x1b[7m" + text[lo:hi] + "\x1b[27m" + text[hi:])

            # Row the footer's top (bottom rule) line starts on: right
            # after the last character, *unless* that character exactly
            # filled its row (column 1 of the row after it — an empty,
            # not-yet-used row), in which case the footer starts there
            # instead of one further down.
            end_row, end_col = _row_col(prompt_visible_len + len(text))
            footer_row = end_row if end_col == 1 else end_row + 1

            sys.stdout.write("\x1b8")
            if footer_row:
                sys.stdout.write(f"\x1b[{footer_row}B")
            sys.stdout.write("\r")
            footer_lines = [self._state_line(), self._mode_line()]
            if repo_line:
                footer_lines.append(repo_line)
            with _console.capture() as capture:
                _console.print(("─" * w) + "\n" + "\n".join(footer_lines), end="")
            sys.stdout.write(capture.get().replace("\n", "\r\n"))

            cursor_row, cursor_col = _row_col(prompt_visible_len + cursor)
            sys.stdout.write("\x1b8")
            if cursor_row:
                sys.stdout.write(f"\x1b[{cursor_row}B")
            sys.stdout.write(f"\x1b[{cursor_col}G")
            sys.stdout.flush()

        def on_cycle() -> None:
            self._cycle_permission_mode()

        def land_below_footer() -> None:
            end_row, end_col = _row_col(prompt_visible_len + len(last_text))
            footer_row = end_row if end_col == 1 else end_row + 1
            landing_row = footer_row + footer_rows
            sys.stdout.write("\x1b8")
            if landing_row:
                sys.stdout.write(f"\x1b[{landing_row}B")
            # A real \r\n for the last step (not another CSI-B) so the
            # terminal scrolls if the footer was sitting at the bottom of
            # the visible viewport — cursor-only movement would just clamp
            # there instead of producing a fresh line to print into.
            sys.stdout.write("\r\n")
            sys.stdout.flush()

        try:
            raw = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: read_line_with_cycle(prompt, on_render=_render, on_cycle=on_cycle),
            )
        except EOFError:
            land_below_footer()
            return None
        except KeyboardInterrupt:
            land_below_footer()
            _console.print("[dim]interrupted[/dim]")
            return ""

        land_below_footer()
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
                if not await self._handle_command(user_input):
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

    async def _handle_extension_command(self, command: str) -> bool:
        """Dispatch `command` to a matching extension-registered slash
        command, if any. Returns True if one handled it (whether or not it
        printed anything), False if no extension owns this command name."""
        from pi_coding_agent.extensions.events import ExtensionContext

        stripped = command[1:].strip()
        if not stripped:
            return False
        parts = stripped.split(None, 1)
        name = parts[0]
        args_text = parts[1] if len(parts) > 1 else ""

        for registered in self._agent_session.get_extension_commands():
            if registered.name != name:
                continue
            result = registered.handler(args_text, ExtensionContext(cwd=self._cwd, ui=self._ui_context))
            if inspect.isawaitable(result):
                result = await result
            if isinstance(result, str) and result:
                self._output.print(result)
            return True
        return False

    async def _handle_extension_shortcut(self, key: str) -> bool:
        """Dispatch `key` (a Textual key-name, e.g. "ctrl+g") to the first
        matching extension-registered shortcut, if any. Returns True if one
        handled it (whether or not it printed anything), False if no
        extension registered this key — the T3 keybinding dispatcher in
        tui_app.py calls this from PiApp.on_key; the classic REPL has no
        keybinding dispatcher to call it from."""
        from pi_coding_agent.extensions.events import ExtensionContext

        for registered in self._agent_session.get_extension_shortcuts():
            if registered.key != key:
                continue
            result = registered.handler(ExtensionContext(cwd=self._cwd, ui=self._ui_context))
            if inspect.isawaitable(result):
                result = await result
            if isinstance(result, str) and result:
                self._output.print(result)
            return True
        return False

    def _resolve_model(self, query: str) -> Model | None:
        """Find a model by "<id>" (searched across every configured
        provider) or "<provider>/<id>" (unambiguous, for when two
        providers happen to share a model id) — the argument form
        `/model <query>` accepts."""
        if "/" in query:
            provider_id, _, model_id = query.partition("/")
            provider = self._models.get_provider(provider_id)
            if provider is None:
                return None
            for model in provider.get_models():
                if model.id == model_id:
                    return model
            return None
        for provider in self._models.get_providers():
            for model in provider.get_models():
                if model.id == query:
                    return model
        return None

    async def _handle_command(self, command: str) -> bool:
        """Handle slash commands. Returns True to continue, False to exit."""
        cmd = command.lower().strip()

        if cmd in ("/exit", "/quit", "/q"):
            return False

        if cmd == "/help":
            self._output.print(
                "\n[bold]Commands[/bold]\n"
                f"  [{PASTEL_BLUE}]/help[/{PASTEL_BLUE}]     Show this help\n"
                f"  [{PASTEL_BLUE}]/exit[/{PASTEL_BLUE}]     Exit Pi\n"
                f"  [{PASTEL_BLUE}]/model[/{PASTEL_BLUE}]    List providers/models; /model <id> to switch\n"
                f"  [{PASTEL_BLUE}]/clear[/{PASTEL_BLUE}]    Clear conversation history\n"
                f"  [{PASTEL_BLUE}]/tools[/{PASTEL_BLUE}]    List available tools\n"
                f"  [{PASTEL_BLUE}]/session[/{PASTEL_BLUE}]  Show session info\n"
                f"  [{PASTEL_BLUE}]/extensions[/{PASTEL_BLUE}]  List loaded extensions and load errors\n"
            )
            return True

        if cmd == "/model" or cmd.startswith("/model "):
            # Args from the original (non-lowercased) command — model IDs
            # can be case-sensitive, unlike the command name itself.
            args_text = command[len("/model") :].strip()
            if args_text:
                target = self._resolve_model(args_text)
                if target is None:
                    self._output.print(f"[{PASTEL_RED}]no such model:[/{PASTEL_RED}] {escape(args_text)}")
                else:
                    self._model = target
                    self._agent_session.set_model(target)
                    self._output.print(
                        f"[{PASTEL_GREEN}]switched to[/{PASTEL_GREEN}] "
                        f"[{PASTEL_BLUE}]{escape(target.provider)}/{escape(target.id)}[/{PASTEL_BLUE}]"
                    )
                return True

            current_provider = getattr(self._model, "provider", None)
            current_id = getattr(self._model, "id", None)
            lines = [
                f"Current: [{PASTEL_BLUE}]{current_provider}[/{PASTEL_BLUE}]/"
                f"[{PASTEL_BLUE}]{current_id}[/{PASTEL_BLUE}] "
                f"[dim]({getattr(self._model, 'context_window', '?')} tokens)[/dim]",
                "",
            ]
            providers = self._models.get_providers()
            if not providers:
                lines.append("[dim]no providers configured[/dim]")
            for provider in providers:
                lines.append(f"[bold]{escape(provider.name)}[/bold] [dim]({escape(provider.id)})[/dim]")
                models = provider.get_models()
                if not models:
                    lines.append("  [dim](no models)[/dim]")
                for model in models:
                    is_current = provider.id == current_provider and model.id == current_id
                    marker = f" [{PASTEL_GREEN}]*[/{PASTEL_GREEN}]" if is_current else ""
                    lines.append(f"  [{PASTEL_BLUE}]{escape(model.id)}[/{PASTEL_BLUE}]{marker}")
            lines.append("")
            lines.append("[dim]/model <id> or /model <provider>/<id> to switch[/dim]")
            self._output.print("\n".join(lines))
            return True

        if cmd == "/clear":
            self._agent_session._messages = []
            self._output.print("[dim]conversation cleared[/dim]")
            return True

        if cmd == "/tools":
            for tool in self._agent_session._tools:
                self._output.print(f"  [{PASTEL_BLUE}]{tool.name}[/{PASTEL_BLUE}]: {tool.description}")
            return True

        if cmd == "/session":
            if self._session_id:
                self._output.print(f"Session ID: [dim]{self._session_id}[/dim]\nMessages:   {self._message_count}")
            else:
                self._output.print("[dim]no active session[/dim]")
            return True

        if cmd == "/extensions":
            extensions = self._agent_session.get_extensions()
            if not extensions.extensions and not extensions.errors:
                self._output.print("[dim]no extensions loaded[/dim]")
                return True
            for ext in extensions.extensions:
                tools = ", ".join(ext.tool_names) or "(no tools)"
                self._output.print(f"  [{PASTEL_GREEN}]{ext.path}[/{PASTEL_GREEN}]: {tools}")
            for err in extensions.errors:
                self._output.print(f"  [{PASTEL_RED}]{err.path}[/{PASTEL_RED}]: {escape(err.error)}")
            return True

        if await self._handle_extension_command(command):
            return True

        self._output.print(f"[yellow]Unknown command:[/yellow] {command}")
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

    if _resolve_ui_mode(args, cwd, config_dir) == "fullscreen":
        from pi_coding_agent.tui_app import PiApp

        await PiApp(session).run_async()
        return 0

    try:
        await session.repl_loop()
    except KeyboardInterrupt:
        _console.print("\n[dim]exiting[/dim]")

    return 0


def _resolve_ui_mode(args: Args, cwd: str, config_dir: Path) -> str:
    """--ui-mode/--alt wins; otherwise the persisted setting; "regular" if
    neither is available (never let a settings-load failure block startup)."""
    if args.ui_mode:
        return args.ui_mode
    try:
        from pi_coding_agent.settings_manager import SettingsManager

        settings_mgr = SettingsManager.create(cwd=cwd, agent_dir=config_dir)
        return settings_mgr.get_ui_mode()
    except Exception:
        return "regular"


def run_interactive_sync(args: Args) -> int:
    """Synchronous wrapper for run_interactive_mode."""
    return asyncio.run(run_interactive_mode(args))
