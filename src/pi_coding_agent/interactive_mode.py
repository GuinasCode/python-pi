"""Interactive mode for Pi coding agent.

Provides a REPL (Read-Eval-Print Loop) that maintains conversation context,
executes tools, persists sessions, and loads project resources.

This is the main interactive experience when running `pi` without `-p`.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import re
import sys
import threading
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.text import Text

from pi_ai import (
    AssistantMessage,
    ImageContent,
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
from pi_coding_agent.attachments import load_attachment
from pi_coding_agent.config import ensure_config_dir, ensure_session_dir, get_config_dir, get_session_dir
from pi_coding_agent.diff_render import render_diff
from pi_coding_agent.extension_ui import ExtensionUIContext, NoopExtensionUIContext
from pi_coding_agent.extensions import ExtensionRunner
from pi_coding_agent.git_info import get_git_repo_line
from pi_coding_agent.markdown_render import LeftMarkdown as Markdown
from pi_coding_agent.output_sink import ConsoleOutputSink, FooterAwareOutputSink, OutputSink
from pi_coding_agent.permission_mode import (
    PermissionDecision,
    PermissionMode,
    cycle_permission_mode,
    permission_decision,
    permission_mode_label,
)
from pi_coding_agent.resource_loader import load_resources
from pi_coding_agent.session_manager import SessionEntry, SessionInfo, SessionManager
from pi_coding_agent.styles import DIM_STYLE, PASTEL_BLUE, PASTEL_GREEN, PASTEL_RED, PASTEL_YELLOW, PI_THEME
from pi_memory import MemoryStore, MemoryType
from pi_tui.raw_input import read_line_with_cycle

_console = Console(highlight=False, soft_wrap=True, theme=PI_THEME)
_err_console = Console(highlight=False, soft_wrap=True, stderr=True, theme=PI_THEME)

# `@path` inline attachment tokens in a typed prompt — see run_turn().
_ATTACHMENT_RE = re.compile(r"@(\S+)")


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


def _derive_soul_title(text: str) -> str:
    """Short label for a Soul entry entered as free text via /soul add/edit
    or the onboarding flow — first few words, not a separate user input."""
    words = text.strip().split()
    title = " ".join(words[:8])
    if len(words) > 8:
        title += "…"
    return title or "Principle"


def _suggest_soul_merge(old_text: str, new_text: str) -> str:
    """Heuristic (non-LLM, deterministic) merge suggestion for two
    overlapping Soul principles, offered as an editable starting point —
    never applied without the user confirming or rewriting it. If one
    text already contains the other, the longer one already says
    everything; otherwise concatenate them as two clauses."""
    old_norm, new_norm = old_text.strip(), new_text.strip()
    if old_norm.lower() in new_norm.lower():
        return new_norm
    if new_norm.lower() in old_norm.lower():
        return old_norm
    return f"{old_norm} Além disso: {new_norm}"


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


def _visible_row_count(markup: str, width: int) -> int:
    """How many terminal rows one logical `markup` line occupies once
    auto-wrapped at `width` — used by InteractiveSession._render_live_box
    so its row-count math (used both for drawing and for erasing) has one
    single source of truth for what "one row" means, rather than
    independently-maintained copies of this math drifting apart."""
    visible_len = len(Text.from_markup(markup).plain)
    return max(1, -(-max(visible_len, 1) // width))


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------


def _setup_models(args: Args) -> tuple[MutableModels, Any]:
    """Set up models with available providers.

    Priority order:
    1. NVAPI_KEY  -> NVIDIA (default: MiniMax M3 — also registers
       Nemotron 3 Super/Ultra, gpt-oss 120B/20B, and nvidia/auto, a
       fallback chain across those four; all selectable via /model)
    2. OPENAI_API_KEY -> OpenAI
    3. Faux provider (fallback for testing)
    """
    models = MutableModels()

    nvapi_key = args.api_key or os.environ.get("NVAPI_KEY")
    if nvapi_key:
        try:
            from pi_ai.providers.nvidia_models import nvidia_models_provider

            model, provider_models, _meta = nvidia_models_provider(
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
        # The live input box (prompt + typed text + rule + state/mode/repo
        # footer) — kept on screen continuously for the whole REPL
        # session, not just while actively being typed into. See
        # _render_live_box/_erase_live_box/_read_live_line and
        # repl_loop/_run_turn_accepting_mid_turn_input below for the full
        # picture. `_input_render_lock` is a real threading.Lock, not an
        # asyncio one: keystroke redraws happen on a background thread
        # (read_line_with_cycle blocks there — see _read_live_line), while
        # turn output redraws happen on the main event-loop thread; both
        # sides must serialize on the exact same primitive or their
        # terminal writes can interleave mid-escape-sequence.
        self._input_render_lock = threading.Lock()
        self._live_text = ""
        self._live_cursor = 0
        self._live_selection: tuple[int, int] | None = None
        self._live_cursor_row = 0
        self._live_footer_rows = 0
        self._live_shown = False
        self._live_prompt = "\033[1m>\033[0m "
        self._live_prompt_visible_len = 2  # "> " — the bold/reset codes above are zero-width
        # A _read_live_line() task left pending when a turn finished before
        # the user pressed Enter — reused as the next prompt's read instead
        # of starting a second, competing raw-mode stdin reader.
        self._pending_live_input_task: asyncio.Task[str | None] | None = None

        # Where run_turn()/_handle_command() render to. Defaults to the
        # classic REPL's Console — the Textual app (--ui-mode fullscreen)
        # passes its own sink so the exact same event/command handling
        # renders into a transcript widget instead of straight to stdout,
        # and already keeps its input/footer visible natively via
        # Textual's own docked-widget layout, so it gets the plain console
        # sink, not the live-box-aware wrapper (which exists to fake that
        # for a raw terminal).
        self._output: OutputSink = (
            output
            if output is not None
            else FooterAwareOutputSink(
                ConsoleOutputSink(_console),
                before=self._box_before_output,
                after=self._box_after_output,
            )
        )
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

        # Submitted lines (not slash commands — see repl_loop), oldest
        # first, that Up/Down in the prompt editor walk through. Not
        # persisted across sessions — a fresh REPL process starts empty.
        self._prompt_history: list[str] = []

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
        # ready), instead of only at submission start/end. The classic
        # REPL wires the same idea to its own mid-turn footer (see
        # _refresh_status_footer) so "running: bash" etc. appear the
        # instant the status changes, not only the next time something
        # happens to print; still None when an external sink was
        # supplied (fullscreen/tests own their own status refresh).
        self._on_status_change: Callable[[], None] | None = None if output is not None else self._refresh_status_footer

    def _cycle_permission_mode(self) -> None:
        self._permission_mode = cycle_permission_mode(self._permission_mode)

    async def _permission_gate(self, tool_name: str, args: dict[str, Any]) -> bool:
        """AgentSessionOptions.permission_gate: allow/ask/deny a tool call
        based on the current permission mode, before it runs.

        A `remember(type="soul")` call is a special case: writing a
        permanent, cross-session principle deserves a real human
        confirmation regardless of the current PermissionMode (even
        acceptEdits, which is only about auto-approving file edits) — a
        second self-issued tool call from the model is not evidence a human
        actually saw and approved it, so this always routes to the same
        blocking y/N prompt used for mutating tools, unconditionally.
        """
        if tool_name == "remember" and args.get("type") == "soul":
            return await self._confirm_tool_fn(tool_name, args)
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
        """Ask the user to approve a single mutating tool call (y/N).

        Prints straight to `_console`, bypassing self._output/
        FooterAwareOutputSink — so the live box is erased first and
        redrawn after by hand here, same as the wrapped sink does for
        every other print, just done directly since this call site needs
        blocking input(), not just a print.

        Known limitation: this can run concurrently with a mid-turn
        /steer read (_run_turn_accepting_mid_turn_input's own background
        stdin reader) if a tool confirmation happens to fire while the
        user is actively typing — both are blocking reads on the same
        stdin from different threads, and there's no coordination
        between them beyond the terminal-write lock this method already
        takes. In that narrow window a keystroke could be delivered to
        the "wrong" reader. Not corruption of terminal *state* (the lock
        still prevents interleaved writes), just a rare mis-routed key —
        a real gap, not one this pass closes; documented rather than
        silently ignored (spec: "não prometa isolamento que não
        existe")."""
        args_str = _fmt_args(args)
        self._box_before_output()
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
        finally:
            self._box_after_output()
        return answer.strip().lower() in ("y", "yes")

    async def _soul_yes_no(self, prompt_text: str) -> bool:
        self._box_before_output()
        _console.print(f"[{PASTEL_YELLOW}]?[/{PASTEL_YELLOW}] {prompt_text} [dim](y/N)[/dim]", end=" ")
        loop = asyncio.get_running_loop()
        try:
            answer = await loop.run_in_executor(None, input)
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        finally:
            self._box_after_output()
        return answer.strip().lower() in ("y", "yes")

    async def _soul_read_line(self, prompt_text: str) -> str:
        self._box_before_output()
        _console.print(f"{prompt_text} ", end="")
        loop = asyncio.get_running_loop()
        try:
            answer = await loop.run_in_executor(None, input)
        except (EOFError, KeyboardInterrupt):
            return ""
        finally:
            self._box_after_output()
        return answer.strip()

    async def _soul_choice(self, prompt_text: str, options: dict[str, str], *, default: str) -> str:
        """Read a numbered choice (blank input picks `default`); re-prompts
        on an unrecognized answer instead of silently guessing."""
        menu = "  ".join(f"[{key}] {label}" for key, label in options.items())
        while True:
            raw = await self._soul_read_line(f"{prompt_text}\n  {menu}")
            if not raw:
                return default
            if raw in options:
                return raw
            self._output.print(f"[{PASTEL_RED}]escolha inválida:[/{PASTEL_RED}] {escape(raw)}")

    async def _soul_resolve_overlap(
        self, store: MemoryStore, candidate_title: str, candidate_content: str, *, exclude_id: int | None = None
    ) -> bool:
        """Deterministic overlap check (MemoryStore.find_overlapping_soul)
        run before a NEW Soul principle is written. Surfaces the closest
        existing match — reasoning is just showing both texts side by side,
        not an LLM judgment call, so the trigger stays deterministic — and
        lets the user decide. Returns True if the caller should proceed to
        write candidate_title/candidate_content as a new record; False if
        this function already resolved it (replaced/merged into the
        existing record, or the user cancelled)."""
        matches = store.find_overlapping_soul(candidate_content, exclude_id=exclude_id)
        if not matches:
            return True
        record, ratio = matches[0]
        self._output.print(
            f"[{PASTEL_YELLOW}]possível sobreposição[/{PASTEL_YELLOW}] ({ratio:.0%} de similaridade textual) "
            f"com o princípio existente #{record.id}:"
        )
        self._output.print(f"  existente: {escape(record.content)}")
        self._output.print(f"  novo:      {escape(candidate_content)}")
        choice = await self._soul_choice(
            "o que fazer?",
            {"1": "manter os dois", "2": "substituir o existente", "3": "mesclar", "4": "cancelar"},
            default="1",
        )
        if choice == "2":
            store.update(record.id, title=candidate_title, content=candidate_content)
            self._output.print(f"[{PASTEL_GREEN}]substituído[/{PASTEL_GREEN}] soul #{record.id}")
            return False
        if choice == "3":
            merged = await self._soul_prompt_merge(record.content, candidate_content)
            if merged is None:
                self._output.print("[dim]cancelado[/dim]")
                return False
            store.update(record.id, title=_derive_soul_title(merged), content=merged)
            self._output.print(f"[{PASTEL_GREEN}]mesclado[/{PASTEL_GREEN}] em soul #{record.id}")
            return False
        if choice == "4":
            self._output.print("[dim]cancelado[/dim]")
            return False
        return True  # "1" — keep both, proceed to write the new record

    async def _soul_prompt_merge(self, old_text: str, new_text: str) -> str | None:
        """Offer the heuristic merge suggestion, editable; None means the
        user backed out entirely."""
        suggestion = _suggest_soul_merge(old_text, new_text)
        self._output.print(f'  sugestão de mesclagem: "{suggestion}"')
        if await self._soul_yes_no("usar esta sugestão?"):
            return suggestion
        typed = await self._soul_read_line("digite o texto final mesclado (vazio para cancelar)")
        return typed.strip() or None

    async def _handle_soul_command(self, args_text: str) -> None:
        """/soul [show|add|edit|remove|clear|audit] — direct, explicit
        management of the user's own MemoryStore, type=soul. Explicit
        invocation of this command IS the confirmation (unlike the model
        calling `remember(type="soul")`, which is routed through
        _permission_gate's real y/N prompt instead). `add` and the
        onboarding flow run a deterministic overlap check
        (_soul_resolve_overlap) before writing; `audit` runs the same
        check proactively across every already-saved principle."""
        store = self._agent_session._memory_store
        if store is None:
            self._output.print("[dim]memory is disabled — /soul is unavailable[/dim]")
            return

        parts = args_text.split(maxsplit=1)
        sub = parts[0].lower() if parts else "show"
        rest = parts[1] if len(parts) > 1 else ""

        if sub in ("show", ""):
            records = store.list_by_type(MemoryType.SOUL)
            if not records:
                self._output.print("[dim]no Soul principles yet.[/dim]")
                if await self._soul_yes_no("Want to set some up now?"):
                    await self._soul_onboarding(store)
                else:
                    self._output.print(f"[dim]use [{PASTEL_BLUE}]/soul add <principle>[/{PASTEL_BLUE}] any time[/dim]")
                return
            for r in records:
                self._output.print(f"  [{PASTEL_BLUE}]#{r.id}[/{PASTEL_BLUE}] {escape(r.title)}: {escape(r.content)}")
            return

        if sub == "add":
            if not rest:
                self._output.print(f"[{PASTEL_RED}]usage:[/{PASTEL_RED}] /soul add <principle text>")
                return
            title = _derive_soul_title(rest)
            if not await self._soul_resolve_overlap(store, title, rest):
                return
            self._output.print(f'About to save as a permanent principle: "{escape(rest)}"')
            if not await self._soul_yes_no("Confirm?"):
                self._output.print("[dim]cancelled[/dim]")
                return
            record = store.write(type=MemoryType.SOUL, title=title, content=rest, source="manual")
            self._output.print(f"[{PASTEL_GREEN}]saved[/{PASTEL_GREEN}] soul #{record.id}: {escape(record.title)}")
            return

        if sub == "edit":
            id_part, _, text = rest.partition(" ")
            if not id_part.isdigit() or not text.strip():
                self._output.print(f"[{PASTEL_RED}]usage:[/{PASTEL_RED}] /soul edit <id> <new text>")
                return
            new_text = text.strip()
            updated = store.update(int(id_part), title=_derive_soul_title(new_text), content=new_text)
            if updated is None:
                self._output.print(f"[{PASTEL_RED}]no such soul entry:[/{PASTEL_RED}] #{id_part}")
                return
            self._output.print(f"[{PASTEL_GREEN}]updated[/{PASTEL_GREEN}] soul #{updated.id}: {escape(updated.title)}")
            return

        if sub == "remove":
            if not rest.isdigit():
                self._output.print(f"[{PASTEL_RED}]usage:[/{PASTEL_RED}] /soul remove <id>")
                return
            if store.delete(int(rest)):
                self._output.print(f"[{PASTEL_GREEN}]removed[/{PASTEL_GREEN}] soul #{rest}")
            else:
                self._output.print(f"[{PASTEL_RED}]no such soul entry:[/{PASTEL_RED}] #{rest}")
            return

        if sub == "clear":
            records = store.list_by_type(MemoryType.SOUL)
            if not records:
                self._output.print("[dim]nothing to clear.[/dim]")
                return
            if not await self._soul_yes_no(f"Remove all {len(records)} soul principle(s)? This cannot be undone."):
                self._output.print("[dim]cancelled[/dim]")
                return
            for r in records:
                store.delete(r.id)
            self._output.print(f"[{PASTEL_GREEN}]cleared[/{PASTEL_GREEN}] {len(records)} soul principle(s)")
            return

        if sub == "audit":
            await self._soul_audit(store)
            return

        self._output.print(f"[{PASTEL_RED}]unknown /soul subcommand:[/{PASTEL_RED}] {escape(sub)}")

    async def _soul_audit(self, store: MemoryStore) -> None:
        """/soul audit — proactive, deterministic pairwise scan over every
        already-saved Soul principle, for overlaps that crept in without
        anyone noticing at write time (e.g. saved in different sessions).
        Reuses the same MemoryStore.find_overlapping_soul detector as the
        reactive check in _soul_resolve_overlap; no separate search system."""
        records = store.list_by_type(MemoryType.SOUL)
        if len(records) < 2:
            self._output.print("[dim]menos de 2 princípios — nada para auditar.[/dim]")
            return

        shown_pairs: set[frozenset[int]] = set()
        deleted: set[int] = set()
        found_any = False
        for record in records:
            if record.id in deleted:
                continue
            matches = [
                (other, ratio)
                for other, ratio in store.find_overlapping_soul(record.content, exclude_id=record.id)
                if other.id not in deleted and frozenset((record.id, other.id)) not in shown_pairs
            ]
            if not matches:
                continue
            other, ratio = matches[0]
            shown_pairs.add(frozenset((record.id, other.id)))
            found_any = True
            self._output.print(
                f"[{PASTEL_YELLOW}]sobreposição[/{PASTEL_YELLOW}] ({ratio:.0%} de similaridade textual) "
                f"entre #{record.id} e #{other.id}:"
            )
            self._output.print(f"  #{record.id}: {escape(record.content)}")
            self._output.print(f"  #{other.id}: {escape(other.content)}")
            choice = await self._soul_choice(
                "o que fazer?",
                {"1": "manter os dois", "2": f"manter só #{record.id}", "3": "mesclar", "4": "pular"},
                default="4",
            )
            if choice == "2":
                store.delete(other.id)
                deleted.add(other.id)
                self._output.print(f"[{PASTEL_GREEN}]removido[/{PASTEL_GREEN}] soul #{other.id}")
            elif choice == "3":
                merged = await self._soul_prompt_merge(record.content, other.content)
                if merged is None:
                    self._output.print("[dim]pulado[/dim]")
                    continue
                store.update(record.id, title=_derive_soul_title(merged), content=merged)
                store.delete(other.id)
                deleted.add(other.id)
                self._output.print(
                    f"[{PASTEL_GREEN}]mesclado[/{PASTEL_GREEN}] em soul #{record.id}, #{other.id} removido"
                )
        if not found_any:
            self._output.print("[dim]nenhuma sobreposição encontrada.[/dim]")
        else:
            self._output.print("[dim]auditoria concluída.[/dim]")

    async def _soul_onboarding(self, store: MemoryStore) -> None:
        """Iterative, one-question-at-a-time construction flow — never a
        30-question form. Each answer is confirmed individually before
        being written; a blank answer skips that question."""
        questions = [
            "Como você quer que eu me comporte?",
            "Quais princípios devo seguir sempre?",
            "Como devo lidar com ambiguidades?",
            "Quais princípios de engenharia são importantes para você?",
        ]
        saved = 0
        for question in questions:
            answer = await self._soul_read_line(f"{question} [dim](enter para pular)[/dim]")
            if not answer:
                continue
            if not await self._soul_yes_no(f'Salvar como princípio permanente: "{answer}"?'):
                continue
            title = _derive_soul_title(answer)
            if not await self._soul_resolve_overlap(store, title, answer):
                continue
            record = store.write(type=MemoryType.SOUL, title=title, content=answer, source="soul_onboarding")
            saved += 1
            self._output.print(f"[{PASTEL_GREEN}]saved[/{PASTEL_GREEN}] soul #{record.id}")
        self._output.print(f"[dim]done — {saved} principle(s) saved. Use /soul any time to review.[/dim]")

    async def _handle_session_command(self, args_text: str) -> None:
        """/session [show|list|resume] — show is the default (unchanged
        behavior). `list [query]` lists recent sessions, or searches every
        stored session (SessionManager.search_sessions) when a query is
        given. `resume <id-or-prefix>` switches the REPL's active session
        at runtime: the current session's file is left untouched on disk
        (still fully replayable later), the in-memory transcript is
        replaced with the target session's via the same
        _restore_persisted_messages() path startup uses, and all further
        turns persist into the target session from then on."""
        parts = args_text.split(maxsplit=1)
        sub = parts[0].lower() if parts else "show"
        rest = parts[1].strip() if len(parts) > 1 else ""

        if sub == "show":
            if self._session_id:
                self._output.print(f"Session ID: [dim]{self._session_id}[/dim]\nMessages:   {self._message_count}")
            else:
                self._output.print("[dim]no active session[/dim]")
            return

        if self._session_manager is None:
            self._output.print("[dim]session management is disabled (--no-session)[/dim]")
            return

        if sub == "list":
            if rest:
                results = self._session_manager.search_sessions(rest)
                if not results:
                    self._output.print(f"[dim]no sessions match {escape(rest)!r}[/dim]")
                    return
                for r in results:
                    self._print_session_row(r.info, snippet=r.snippet)
            else:
                sessions = self._session_manager.list_sessions()
                if not sessions:
                    self._output.print("[dim]no sessions yet.[/dim]")
                    return
                for info in sessions[:20]:
                    self._print_session_row(info)
            return

        if sub == "resume":
            if not rest:
                self._output.print(f"[{PASTEL_RED}]usage:[/{PASTEL_RED}] /session resume <id-or-prefix>")
                return
            target = self._session_manager.resolve_session_ref(rest)
            if target is None:
                candidates = [s for s in self._session_manager.list_sessions() if s.id.startswith(rest)]
                if len(candidates) > 1:
                    ids = ", ".join(c.id for c in candidates[:5])
                    self._output.print(f"[{PASTEL_RED}]ambiguous id prefix[/{PASTEL_RED}] {escape(rest)}: {ids}")
                else:
                    self._output.print(f"[{PASTEL_RED}]no such session:[/{PASTEL_RED}] {escape(rest)}")
                return
            if target.id == self._session_id:
                self._output.print("[dim]already the active session[/dim]")
                return
            self._session_id = target.id
            self._agent_session._messages = []
            self._message_count = 0
            self._restore_persisted_messages()
            self._output.print(
                f"[{PASTEL_GREEN}]resumed[/{PASTEL_GREEN}] session [dim]{target.id}[/dim] "
                f"({escape(target.name or target.cwd)}) — {self._message_count} message(s)"
            )
            return

        self._output.print(f"[{PASTEL_RED}]unknown /session subcommand:[/{PASTEL_RED}] {escape(sub)}")

    def _print_session_row(self, info: SessionInfo, *, snippet: str | None = None) -> None:
        when = datetime.fromtimestamp(info.updated_at / 1000).strftime("%Y-%m-%d %H:%M")
        label = escape(info.name or "(unnamed)")
        self._output.print(
            f"  [{PASTEL_BLUE}]{info.id}[/{PASTEL_BLUE}]  {label}  [dim]{when} · "
            f"{info.message_count} msg · {escape(info.cwd)}[/dim]"
        )
        if snippet:
            self._output.print(f"      [dim]{escape(snippet)}[/dim]")

    async def _handle_agents_command(self, args_text: str) -> None:
        """/agents [list|stop|steer] — live view/control of subagent child
        processes spawned via the `subagent` tool. `list` (default) shows
        every live child plus recently finished ones. `stop <id-or-prefix>`
        asks a live child to stop (RPC, honored at its next turn boundary;
        force-killed if it doesn't within a few seconds — see
        SubagentHandle.stop). `steer <id-or-prefix> <text>` queues extra
        guidance for a live child, delivered the same way (next turn
        boundary, never mid-generation — see AgentSession.queue_steer_message).
        This is direct, explicit control — unlike the subagent tool itself,
        which the model calls autonomously, these subcommands only run when
        you type them."""
        from pi_coding_agent.subagent.registry import SubagentRegistry

        registry: SubagentRegistry | None = self._agent_session.get_subagent_registry()
        if registry is None:
            self._output.print("[dim]subagents are disabled for this session[/dim]")
            return

        parts = args_text.split(maxsplit=2)
        sub = parts[0].lower() if parts else "list"

        if sub == "list":
            handles = registry.list_all()
            if not handles:
                self._output.print("[dim]no subagents have run yet.[/dim]")
                return
            for h in handles:
                elapsed = time.time() - h.started_at
                task_preview = h.task if len(h.task) <= 70 else h.task[:69] + "…"
                status_color = PASTEL_GREEN if h.status == "running" else DIM_STYLE
                self._output.print(
                    f"  [{PASTEL_BLUE}]{h.id}[/{PASTEL_BLUE}]  {escape(h.agent_name)}  "
                    f"[{status_color}]{h.status}[/{status_color}]  [dim]{elapsed:.0f}s · {escape(task_preview)}[/dim]"
                )
            return

        if sub == "stop":
            ref = parts[1] if len(parts) > 1 else ""
            if not ref:
                self._output.print(f"[{PASTEL_RED}]usage:[/{PASTEL_RED}] /agents stop <id-or-prefix>")
                return
            handle = registry.get(ref)
            if handle is None:
                self._output.print(f"[{PASTEL_RED}]no such subagent:[/{PASTEL_RED}] {escape(ref)}")
                return
            if handle.status != "running":
                self._output.print(f"[dim]#{handle.id} already finished ({handle.status})[/dim]")
                return
            result = await handle.stop()
            self._output.print(f"[{PASTEL_GREEN}]stopped[/{PASTEL_GREEN}] #{handle.id} ({result.status})")
            return

        if sub == "steer":
            if len(parts) < 3 or not parts[1]:
                self._output.print(f"[{PASTEL_RED}]usage:[/{PASTEL_RED}] /agents steer <id-or-prefix> <text>")
                return
            ref, text = parts[1], parts[2]
            handle = registry.get(ref)
            if handle is None:
                self._output.print(f"[{PASTEL_RED}]no such subagent:[/{PASTEL_RED}] {escape(ref)}")
                return
            ok = await handle.steer(text)
            if ok:
                self._output.print(f"[{PASTEL_GREEN}]steered[/{PASTEL_GREEN}] #{handle.id}")
            else:
                self._output.print(f"[dim]#{handle.id} is no longer running — nothing to steer[/dim]")
            return

        self._output.print(f"[{PASTEL_RED}]unknown /agents subcommand:[/{PASTEL_RED}] {escape(sub)}")

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
                raw_content = data.get("content", "")
                if isinstance(raw_content, list):
                    user_blocks: list[TextContent | ImageContent] = []
                    for b in raw_content:
                        if b.get("type") == "text":
                            user_blocks.append(TextContent(text=b.get("text", "")))
                        elif b.get("type") == "image":
                            user_blocks.append(ImageContent(data=b.get("data", ""), mime_type=b.get("mime_type", "")))
                    msg: Message = UserMessage(content=user_blocks, timestamp=data.get("timestamp", 0))
                else:
                    msg = UserMessage(content=raw_content, timestamp=data.get("timestamp", 0))
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

                def _user_block_to_dict(b: TextContent | ImageContent) -> dict[str, Any]:
                    if isinstance(b, TextContent):
                        return {"type": b.type, "text": b.text}
                    return {"type": b.type, "data": b.data, "mime_type": b.mime_type}

                data["content"] = [_user_block_to_dict(b) for b in content]
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

            def _tool_block_to_dict(b: TextContent | ImageContent) -> dict[str, Any]:
                if isinstance(b, TextContent):
                    return {"type": b.type, "text": b.text}
                if isinstance(b, ImageContent):
                    return {"type": b.type, "data": b.data, "mime_type": b.mime_type}
                return {"type": getattr(b, "type", "unknown")}

            data["content"] = [_tool_block_to_dict(b) for b in message.content]
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

        elif t == "subagent_progress":
            text = getattr(event, "text", "")
            if text:
                self._output.print(f"[{DIM_STYLE}]  ⤷ {escape(text)}[/{DIM_STYLE}]")

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

        # A full box redraw is real terminal I/O (erase + repaint every
        # visible row) — firing it on *every* event, including
        # thinking_delta/text_delta (one per streamed token, potentially
        # dozens a second), repainted the input box dozens of times a
        # second and was the actual source of the flicker this guards
        # against now. Those two event types never touch self._status
        # (see above — nothing they do assigns to it) and carry their own
        # incremental echo via self._output.print(..., end=""), so
        # skipping the footer refresh for just them loses no information
        # the footer would have shown; every other event type (including
        # "done"/"error", which must still refresh even when the status
        # text happens to end up unchanged — e.g. a turn with no tool
        # calls goes ready -> ready) keeps refreshing every time, same as
        # before.
        if self._on_status_change is not None and t not in ("thinking_delta", "text_delta"):
            self._on_status_change()

    async def run_turn(self, user_input: str) -> AssistantMessage | None:
        """Run a single conversation turn.

        `@path` tokens in `user_input` are resolved as attachments (same
        mechanism as print mode's `@file` CLI args — see
        pi_coding_agent.attachments): an image becomes a content block
        sent alongside the text; anything else is read as text and folded
        into the prompt actually sent to the model. Only real, existing
        paths are treated as attachments — `@handle`-looking text that
        doesn't resolve to a file is left alone, so this can't misfire on
        an unrelated `@` in the message."""
        prompt_text = user_input
        images: list[ImageContent] = []
        for match in _ATTACHMENT_RE.finditer(user_input):
            # Strip trailing punctuation a sentence would naturally have
            # right after the path ("@screenshot.png?", "@log.txt.") —
            # without this, \S+ swallows it into the path and the
            # existence check below always fails.
            candidate = match.group(1).rstrip(".,!?;:'\")]}")
            if not candidate or not Path(candidate).exists():
                continue
            attachment = load_attachment(candidate)
            if attachment.error:
                self._output.print(f"[{PASTEL_RED}]attachment error:[/{PASTEL_RED}] {escape(attachment.error)}")
                continue
            if attachment.image:
                images.append(attachment.image)
            elif attachment.text_block:
                prompt_text = f"{prompt_text}\n\n{attachment.text_block}"

        persisted_content: str | list[TextContent | ImageContent] = (
            [TextContent(text=user_input), *images] if images else user_input
        )
        self._persist_message(UserMessage(content=persisted_content, timestamp=int(time.time() * 1000)))

        self._turn_has_text = False
        self._thinking_open = False
        unsub = self._agent_session.on_event(self._handle_event)
        try:
            result = await self._agent_session.prompt(prompt_text, images=images or None)
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

    def _footer_lines(self) -> list[str]:
        lines = [self._state_line(), self._mode_line()]
        repo_line = self._repo_line()
        if repo_line:
            lines.append(repo_line)
        return lines

    def _row_col_after(self, n: int) -> tuple[int, int]:
        """0-indexed row and 1-indexed column of the position right after
        the n-th character has been laid out at the console's current
        width (n counts from the input row's own start, prompt included)
        — "deferred wrap" convention: filling a row exactly stays put on
        that row's last column rather than jumping to a new, still-empty
        one, matching how a plain character write actually leaves the
        terminal's cursor (real wrapping only happens once a *further*
        character forces it)."""
        w = _console.width
        row = (n - 1) // w
        col = min(((n - 1) % w) + 2, w)
        return row, col

    def _render_live_box(self, text: str, cursor: int, selection: tuple[int, int] | None) -> None:
        """Full redraw of the live box (prompt + typed text + rule +
        state/mode[/repo] footer) at the current cursor position. Caller
        must hold `_input_render_lock`.

        The whole block is fully repainted on every call, not patched in
        place — input can wrap across multiple terminal rows once it's
        long enough (the terminal does this on its own), and the footer
        sits immediately below it, so incrementally patching the screen
        while tracking exactly how many rows the wrapped input currently
        occupies (and where within them the cursor now sits, since
        Left/Right/Home/End can put it anywhere in the buffer, not just
        the end) is fragile row-arithmetic. A full repaint sidesteps
        needing that arithmetic — a human's typing rate, or a turn's own
        pace of printing, makes the repaint itself unnoticeable.

        Positioning is done with *relative* cursor moves only (CSI
        A/B — up/down N rows — never DECSC/DECRC save/restore): every
        move is computed from `_live_cursor_row`, the 0-indexed row (from
        the box's own first row) the cursor was left on by the previous
        render, tracked here rather than asked of the terminal. DECSC/
        DECRC was tried first and had to be abandoned — it saves a
        position relative to the *current screen*, and doesn't reliably
        keep tracking that position across a scroll event (e.g. one
        caused by a long assistant reply, or even by the box's own lines
        pushing the viewport) that happens between the save and a later
        restore; when that desync happens, DECRC lands one or more rows
        off from where the box actually is, and each redraw's restore
        being off by a different, drifting amount is exactly what caused
        the box to visibly duplicate itself down the screen. Plain
        relative moves have no "remembered absolute position" to desync
        in the first place — they always act on wherever the cursor
        genuinely is right now."""
        self._live_text, self._live_cursor, self._live_selection = text, cursor, selection

        # Synchronized-output (DEC private mode 2026): tells a supporting
        # terminal to hold the erase+redraw below off-screen until the
        # matching "end" arrives, instead of painting each intermediate
        # write as its own frame. Ignored harmlessly by terminals that
        # don't recognize it (it's a private-mode toggle with no visible
        # side effect on its own), so it's safe to always send. This is
        # the actual fix for the box flickering on every keystroke/status
        # change: erase and redraw used to be two *separately flushed*
        # writes (_erase_live_box flushed on its own, then this method's
        # own writes flushed again after), so the terminal had a real
        # chance to paint the box in its erased, blank state as its own
        # frame before the redraw landed — visible as a flash on every
        # single render. Batching both into one flush already mostly
        # fixed that; wrapping them in sync markers additionally protects
        # against terminals (some ConPTY/Windows Terminal versions
        # included) that can still repaint mid-write on a large-enough
        # single write.
        sys.stdout.write("\x1b[?2026h")
        self._erase_live_box(flush=False)  # no-op if nothing's shown yet

        if selection is None:
            sys.stdout.write(self._live_prompt + text)
        else:
            # SGR 7/27 (reverse video on/off) around the selected span —
            # zero-width control codes, don't affect the row/column math
            # below, which only ever counts visible characters.
            lo, hi = selection
            sys.stdout.write(self._live_prompt + text[:lo] + "\x1b[7m" + text[lo:hi] + "\x1b[27m" + text[hi:])

        # The footer always starts on the row right after the input's own
        # last row, via an explicit \r\n — not by relying on the
        # terminal's autowrap eventually carrying it there, which
        # inherits the same deferred-wrap ambiguity _row_col_after
        # documents (whether a just-filled row already counts as "used"
        # is exactly what's unclear about that pending state; an explicit
        # \r\n sidesteps needing an answer).
        w = _console.width
        input_last_row, _ = self._row_col_after(self._live_prompt_visible_len + len(text))
        footer_lines = self._footer_lines()
        with _console.capture() as capture:
            _console.print(("─" * w) + "\n" + "\n".join(footer_lines), end="")
        sys.stdout.write("\r\n" + capture.get().replace("\n", "\r\n"))

        # 1 row for the rule (always exactly `w` "─" chars) + however many
        # rows each footer line actually wraps to — not a fixed count
        # (state_line, in particular, can wrap on its own once a longer
        # model id is active on a narrower terminal; a fixed count used
        # to under-count that and cause a duplication bug identical in
        # shape to the one this same arithmetic once had here).
        footer_rows = 1 + sum(_visible_row_count(line, w) for line in footer_lines)
        footer_last_row = input_last_row + footer_rows
        cursor_row, cursor_col = self._row_col_after(self._live_prompt_visible_len + cursor)
        rows_up = footer_last_row - cursor_row
        if rows_up:
            sys.stdout.write(f"\x1b[{rows_up}A")
        sys.stdout.write(f"\x1b[{cursor_col}G")
        sys.stdout.write("\x1b[?2026l")
        sys.stdout.flush()

        self._live_cursor_row = cursor_row
        self._live_footer_rows = footer_rows
        self._live_shown = True

    def _erase_live_box(self, *, flush: bool = True) -> None:
        """Caller must hold `_input_render_lock`.

        Moves up `_live_cursor_row` — not `_live_footer_rows` — to land
        back on the box's own *first* row (the prompt/input line), the
        same quantity `_render_live_box`'s own internal erase-before-
        redraw branch uses (it's "rows from the input's own first row to
        wherever the cursor currently sits", tracked across renders,
        exactly like the original _prompt_input's `_render` closure this
        replaced). `_live_footer_rows` counts only the rule+footer-lines
        rows — using it here instead landed one or more rows *below* the
        input's own first row whenever that row wasn't also the last
        rendered row (e.g. right after the box's very first draw, cursor
        and input row coincide at row 0 either way, but as soon as real
        content is printed before/after it they don't), erasing only
        part of the box and leaving stale content — the actual shape of
        a real bug this comment replaces, not a hypothetical one.

        A single `\\x1b[0J` from that row clears everything below it too
        (footer included) — no need to separately account for the
        footer's own row count at all once the cursor is back at the
        input row.

        `flush=False` (used by `_render_live_box`, which immediately
        follows this with its own writes and a single flush at the very
        end) queues the erase without handing it to the terminal yet —
        flushing here too would let the terminal paint the box's blank,
        erased state as its own visible frame before the redraw lands
        right after, which is exactly what caused the box to flash on
        every keystroke/status change. Every other caller erases as a
        standalone operation (something else is about to be printed
        through the normal Console, which does its own flushing) and
        keeps the default `flush=True`."""
        if not self._live_shown:
            return
        if self._live_cursor_row:
            sys.stdout.write(f"\x1b[{self._live_cursor_row}A")
        sys.stdout.write("\r\x1b[0J")
        if flush:
            sys.stdout.flush()
        self._live_shown = False

    def _box_before_output(self) -> None:
        """FooterAwareOutputSink's `before` hook — erases the live box
        (if currently shown) so the next thing printed lands where it
        would have anyway, above where the box reappears."""
        with self._input_render_lock:
            self._erase_live_box()

    def _box_after_output(self) -> None:
        """FooterAwareOutputSink's `after` hook — redraws the box
        (prompt + whatever's currently typed + footer) immediately below
        whatever was just printed, using its own last-known text/cursor/
        selection — so it stays pinned to the bottom of the growing
        transcript instead of left behind above new output, and a
        message queued mid-turn doesn't wipe out whatever the user was
        still typing."""
        with self._input_render_lock:
            self._render_live_box(self._live_text, self._live_cursor, self._live_selection)

    def _refresh_status_footer(self) -> None:
        """Wired as self._on_status_change — redraws the box in place
        whenever self._status changes (e.g. "thinking..." -> "running:
        bash" -> "thinking..." -> "ready"), so the state line is live
        even between prints, not just refreshed as a side effect of the
        next thing that happens to print."""
        with self._input_render_lock:
            self._render_live_box(self._live_text, self._live_cursor, self._live_selection)

    async def _read_live_line(self) -> str | None:
        """Read one line through the live box, without reprinting the
        tips/rule header (see repl_loop, which prints that once at
        startup — the box itself persists for the rest of the session).
        Used both for the top-level prompt and for mid-turn /steer //
        /stop reads (_run_turn_accepting_mid_turn_input) — the box stays
        on screen and editable throughout either way, unlike the old
        per-turn footer that only existed while nothing else was
        printing.

        Blocks on a background thread (`read_line_with_cycle` reads raw
        keys synchronously) — every render it triggers goes through
        `_input_render_lock`, the same lock `_box_before_output`/
        `_box_after_output` use from the main event-loop thread for
        output printed while this is still pending, so a keystroke
        redraw and a turn's output redraw can never interleave mid
        escape-sequence.
        """

        def on_render(text: str, cursor: int, selection: tuple[int, int] | None) -> None:
            with self._input_render_lock:
                self._render_live_box(text, cursor, selection)

        def on_cycle() -> None:
            self._cycle_permission_mode()

        try:
            raw = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: read_line_with_cycle(
                    self._live_prompt, on_render=on_render, on_cycle=on_cycle, history=self._prompt_history
                ),
            )
        except EOFError:
            return None
        except KeyboardInterrupt:
            self._output.print("[dim]interrupted[/dim]")
            return ""

        # The box still visually shows the just-submitted text — cleared
        # here so the *next* erase/redraw cycle (triggered by whatever
        # prints next: the highlighted echo of this very line, a turn's
        # output, or the next _read_live_line call) picks up an empty,
        # ready-for-more-input box instead of redrawing stale text.
        with self._input_render_lock:
            self._live_text, self._live_cursor, self._live_selection = "", 0, None
        return raw.strip()

    def _print_user_message(self, text: str) -> None:
        """Display the user's message with a highlighted background."""
        w = _console.width
        # Pad to full terminal width so background fills the line
        padded = f"  {text}  "
        self._output.print(f"[bold on grey23]{escape(padded):<{w}}[/bold on grey23]")
        self._output.print()

    def _handle_mid_turn_input(self, text: str) -> None:
        """A line submitted while run_turn() is still in flight — never
        starts a second, concurrent turn (AgentSession.prompt() isn't
        reentrant). "/stop" requests the current turn abort at the next
        tool-call boundary; anything else (with or without an explicit
        "/steer" prefix — the box doesn't need the prefix to know a turn
        is already running) is queued via queue_steer_message and
        delivered as an ordinary follow-up UserMessage at that same
        boundary, exactly like a normal next prompt would be, just
        without waiting for this one to finish first."""
        if text == "/stop":
            self._agent_session.request_stop()
            self._output.print("[dim]stop requested — finishing at the next turn boundary[/dim]")
            return
        steer_text = text[len("/steer") :].strip() if text.startswith("/steer") else text
        if not steer_text:
            self._output.print(f"[{PASTEL_RED}]usage:[/{PASTEL_RED}] /steer <text>")
            return
        self._agent_session.queue_steer_message(steer_text)
        self._output.print(f"[dim]queued for the next turn boundary:[/dim] {escape(steer_text)}")

    async def _run_turn_accepting_mid_turn_input(
        self, turn_task: asyncio.Task[AssistantMessage | None]
    ) -> AssistantMessage | None:
        """Runs alongside turn_task, concurrently reading more input
        lines through the same live box — anything submitted while the
        turn is still going is routed to _handle_mid_turn_input (steer/
        stop) rather than starting a second, competing turn. Leaves
        `self._pending_live_input_task` set to whatever read was still
        in flight when the turn finished, so repl_loop's next prompt
        reuses that same reader instead of starting a second, competing
        one on the same stdin."""
        input_task: asyncio.Task[str | None] | None = asyncio.create_task(self._read_live_line())
        try:
            while True:
                wait_set: set[asyncio.Task[Any]] = {turn_task}
                if input_task is not None:
                    wait_set.add(input_task)
                done, _pending = await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)
                if turn_task in done:
                    self._pending_live_input_task = input_task if input_task not in done else None
                    return turn_task.result()
                assert input_task is not None
                submitted = input_task.result()
                if submitted is None:
                    # EOF mid-turn: wind the session down, but don't spawn
                    # another reader — stdin is gone, it would just EOF
                    # again immediately. Keep waiting on turn_task alone
                    # until it actually finishes.
                    self._running = False
                    self._agent_session.request_stop()
                    input_task = None
                    continue
                if submitted:
                    self._handle_mid_turn_input(submitted)
                input_task = asyncio.create_task(self._read_live_line())
        except BaseException:
            turn_task.cancel()
            raise

    # ------------------------------------------------------------------
    # REPL loop
    # ------------------------------------------------------------------

    async def repl_loop(self) -> None:
        """Main REPL loop.

        The live box (prompt + typed text + rule + state/mode/repo
        footer) is drawn once here and then persists for the *entire*
        session — never torn down between turns the way the old
        per-cycle input area was. During a turn it stays visible and
        editable: `_run_turn_accepting_mid_turn_input` reads further
        submitted lines concurrently and routes them to /steer or /stop
        (`_handle_mid_turn_input`) instead of starting a second,
        competing turn — matching how Claude Code's own CLI keeps taking
        input while it works, rather than going dark until a turn ends."""
        self._running = True

        provider_name = getattr(self._model, "provider", "faux")
        model_name = getattr(self._model, "name", "unknown")
        model_id = getattr(self._model, "id", "?")

        # One-line banner — left-aligned. Printed directly (not through
        # self._output): the live box isn't drawn yet, nothing to erase.
        _console.print(
            f"[bold]pi[/bold] v0.83.0  [dim]·[/dim]"
            f"  [{PASTEL_BLUE}]{provider_name}[/{PASTEL_BLUE}] / [{PASTEL_BLUE}]{model_name}[/{PASTEL_BLUE}]"
            f"  [dim]({model_id})[/dim]"
            f"  [dim]{self._cwd}[/dim]"
        )
        _console.print()

        # Tips + top rule, printed once here rather than before every
        # prompt cycle the way the old per-turn box used to — the live
        # box now persists for the whole session instead of disappearing
        # and reappearing, so reprinting the hint every cycle would just
        # be noise scrolling past.
        tips = "/help  /clear  /model  /session  /exit"
        _console.print(f"[{DIM_STYLE}]{tips}[/{DIM_STYLE}]")
        sys.stdout.write("─" * _console.width + "\r\n")
        sys.stdout.flush()
        with self._input_render_lock:
            self._render_live_box("", 0, None)

        try:
            while self._running:
                if self._pending_live_input_task is not None:
                    input_task = self._pending_live_input_task
                    self._pending_live_input_task = None
                else:
                    input_task = asyncio.create_task(self._read_live_line())
                user_input = await input_task

                if user_input is None:  # EOF
                    break
                if not user_input:  # empty or interrupted
                    continue

                self._prompt_history.append(user_input)

                if user_input.startswith("/"):
                    if not await self._handle_command(user_input):
                        break
                    continue

                self._output.print()
                self._print_user_message(user_input)
                self._status = "thinking..."
                turn_task = asyncio.create_task(self.run_turn(user_input))
                result = await self._run_turn_accepting_mid_turn_input(turn_task)
                self._status = "ready"
                self._output.print()
                # No error print here: every provider failure already
                # reaches _handle_event as an "error" event (see
                # AgentSession._consume_stream, which both emits that
                # event *and* returns it as this turn's `result`) — a
                # second print here keyed off `result.stop_reason ==
                # StopReason.ERROR` printed the exact same message a
                # second time, back to back, for every single error.
        finally:
            self._finalize_live_box()

    def _finalize_live_box(self) -> None:
        """Move the cursor past the live box before the REPL loop exits,
        so whatever runs next (the shell's own prompt) lands below it
        instead of overlapping it."""
        with self._input_render_lock:
            if not self._live_shown:
                return
            input_last_row, _ = self._row_col_after(self._live_prompt_visible_len + len(self._live_text))
            footer_last_row = input_last_row + self._live_footer_rows
            rows_down = footer_last_row - self._live_cursor_row
            if rows_down:
                sys.stdout.write(f"\x1b[{rows_down}B")
            sys.stdout.write("\r\n")
            sys.stdout.flush()
            self._live_shown = False

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

    async def _select_model_interactive_repl(self, choices: list[Model], initial: int) -> Model | None:
        """Arrow-key model picker for the classic REPL's bare ``/model``.

        Same relative-cursor-movement redraw technique as
        ``_render_live_box`` (see its docstring for why: DECSC/DECRC was
        tried first and abandoned there after it caused the box-duplication
        bug this session already fixed once) — but simpler, since a menu's row
        count is fixed for its whole lifetime (no wrapping text to
        account for), so there's no row arithmetic to get wrong beyond
        "move up exactly as many rows as were drawn last time".
        """
        from pi_tui.raw_input import select_from_list

        rows = len(choices)

        def _line(model: Model, highlighted: bool) -> str:
            label = escape(f"{model.provider}/{model.id}")
            return f"[reverse {PASTEL_BLUE}]> {label}[/]" if highlighted else f"  {label}"

        drawn = False

        def _render(index: int) -> None:
            nonlocal drawn
            if drawn and rows > 1:
                sys.stdout.write(f"\x1b[{rows - 1}A")
            sys.stdout.write("\r\x1b[0J")
            with _console.capture() as capture:
                _console.print("\n".join(_line(m, i == index) for i, m in enumerate(choices)), end="")
            sys.stdout.write(capture.get().replace("\n", "\r\n"))
            sys.stdout.flush()
            drawn = True

        # The live box must stay hidden for this picker's whole lifetime
        # (not just erased-then-redrawn around one print) — _render draws
        # its own rows directly, with no erase/redraw handshake with the
        # box, so the two would corrupt each other's row-tracking if both
        # were live at once.
        self._box_before_output()
        _console.print(f"[{DIM_STYLE}]↑/↓ move · enter select · esc cancel[/{DIM_STYLE}]")
        loop = asyncio.get_running_loop()
        index = await loop.run_in_executor(None, lambda: select_from_list(rows, on_render=_render, initial=initial))
        sys.stdout.write("\r\n")
        sys.stdout.flush()
        self._box_after_output()
        return choices[index] if index is not None else None

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
                f"  [{PASTEL_BLUE}]/model[/{PASTEL_BLUE}]    Interactive model picker; /model <id> to switch directly\n"
                f"  [{PASTEL_BLUE}]/clear[/{PASTEL_BLUE}]    Clear conversation history\n"
                f"  [{PASTEL_BLUE}]/tools[/{PASTEL_BLUE}]    List available tools\n"
                f"  [{PASTEL_BLUE}]/session[/{PASTEL_BLUE}]  Show session info "
                "(list [query] / resume <id>)\n"
                f"  [{PASTEL_BLUE}]/extensions[/{PASTEL_BLUE}]  List loaded extensions and load errors\n"
                f"  [{PASTEL_BLUE}]/soul[/{PASTEL_BLUE}]     Show/manage permanent principles "
                "(add/edit/remove/clear/audit)\n"
                f"  [{PASTEL_BLUE}]/agents[/{PASTEL_BLUE}]   List/stop/steer running subagents "
                "(list / stop <id> / steer <id> <text>)\n"
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
            providers = self._models.get_providers()
            choices = [model for provider in providers for model in provider.get_models()]

            if not choices:
                self._output.print("[dim]no providers configured[/dim]")
                return True

            # Nothing to pick between — showing a picker with one option
            # would just be an extra keypress for no reason (and, in the
            # Textual app, a modal dialog nobody asked for).
            if len(choices) == 1:
                only = choices[0]
                only_label = f"{escape(only.provider)}/{escape(only.id)}"
                self._output.print(f"[dim]only model available:[/dim] [{PASTEL_BLUE}]{only_label}[/{PASTEL_BLUE}]")
                return True

            # Piped/non-interactive stdin can't navigate a menu at all —
            # same plain listing this command always showed, unchanged.
            if isinstance(self._ui_context, NoopExtensionUIContext) and not sys.stdin.isatty():
                lines = [
                    f"Current: [{PASTEL_BLUE}]{current_provider}[/{PASTEL_BLUE}]/"
                    f"[{PASTEL_BLUE}]{current_id}[/{PASTEL_BLUE}] "
                    f"[dim]({getattr(self._model, 'context_window', '?')} tokens)[/dim]",
                    "",
                ]
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

            initial = next(
                (i for i, m in enumerate(choices) if m.provider == current_provider and m.id == current_id),
                0,
            )

            selected: Model | None
            if isinstance(self._ui_context, NoopExtensionUIContext):
                selected = await self._select_model_interactive_repl(choices, initial)
            else:
                labels = [f"{m.provider}/{m.id}" for m in choices]
                picked = await self._ui_context.select("Select a model", labels)
                selected = next((m for m, label in zip(choices, labels, strict=True) if label == picked), None)

            if selected is None:
                self._output.print("[dim]cancelled[/dim]")
                return True

            self._model = selected
            self._agent_session.set_model(selected)
            self._output.print(
                f"[{PASTEL_GREEN}]switched to[/{PASTEL_GREEN}] "
                f"[{PASTEL_BLUE}]{escape(selected.provider)}/{escape(selected.id)}[/{PASTEL_BLUE}]"
            )
            return True

        if cmd == "/clear":
            self._agent_session._messages = []
            self._output.print("[dim]conversation cleared[/dim]")
            return True

        if cmd == "/tools":
            for tool in self._agent_session._tools:
                self._output.print(f"  [{PASTEL_BLUE}]{tool.name}[/{PASTEL_BLUE}]: {tool.description}")
            return True

        if cmd == "/session" or cmd.startswith("/session "):
            args_text = command[len("/session") :].strip()
            await self._handle_session_command(args_text)
            return True

        if cmd == "/agents" or cmd.startswith("/agents "):
            args_text = command[len("/agents") :].strip()
            await self._handle_agents_command(args_text)
            return True

        if cmd == "/soul" or cmd.startswith("/soul "):
            args_text = command[len("/soul") :].strip()
            await self._handle_soul_command(args_text)
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
    if args.session:
        target = session_mgr.resolve_session_ref(args.session)
        if target:
            session_id = target.id
            _console.print(f"[dim]resuming session: {target.id} ({target.name or target.cwd})[/dim]")
        else:
            _console.print(f"[{PASTEL_RED}]error:[/{PASTEL_RED}] no session matches --session {args.session!r}")
            return 1
    elif args.continue_session or args.resume:
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
