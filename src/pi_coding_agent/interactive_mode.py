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
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.text import Text

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
from pi_coding_agent.session_manager import SessionEntry, SessionInfo, SessionManager
from pi_coding_agent.styles import DIM_STYLE, PASTEL_BLUE, PASTEL_GREEN, PASTEL_RED, PASTEL_YELLOW, PI_THEME
from pi_memory import MemoryStore, MemoryType
from pi_tui.raw_input import read_line_with_cycle

_console = Console(highlight=False, soft_wrap=True, theme=PI_THEME)
_err_console = Console(highlight=False, soft_wrap=True, stderr=True, theme=PI_THEME)


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
        # ready), instead of only at submission start/end. None in the
        # classic REPL, which prints its status line directly rather than
        # keeping a persistent footer to refresh.
        self._on_status_change: Callable[[], None] | None = None

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

    async def _soul_yes_no(self, prompt_text: str) -> bool:
        _console.print(f"[{PASTEL_YELLOW}]?[/{PASTEL_YELLOW}] {prompt_text} [dim](y/N)[/dim]", end=" ")
        loop = asyncio.get_running_loop()
        try:
            answer = await loop.run_in_executor(None, input)
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        return answer.strip().lower() in ("y", "yes")

    async def _soul_read_line(self, prompt_text: str) -> str:
        _console.print(f"{prompt_text} ", end="")
        loop = asyncio.get_running_loop()
        try:
            answer = await loop.run_in_executor(None, input)
        except (EOFError, KeyboardInterrupt):
            return ""
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
        [/repo] lines) is fully redrawn on every keystroke, not just when
        Shift+Tab is pressed — input can wrap across multiple terminal
        rows once it's long enough (the terminal does this on its own),
        and the footer sits immediately below it, so incrementally
        patching the screen in place while tracking exactly how many rows
        the wrapped input currently occupies (and where within them the
        cursor now sits, since Left/Right/Home/End can put it anywhere in
        the buffer, not just the end) is fragile row-arithmetic. A full
        repaint sidesteps needing that arithmetic — a human's typing rate
        makes the repaint itself unnoticeable.

        Positioning is done with *relative* cursor moves only (CSI
        A/B — up/down N rows — never DECSC/DECRC save/restore): every
        move is computed from ``last_cursor_row``, the 0-indexed row (from
        the input's own first row) the cursor was left on by the
        previous render, which this method tracks itself rather than
        asking the terminal to remember a position for it. DECSC/DECRC
        was tried first and had to be abandoned — it saves a position
        relative to the *current screen*, and doesn't reliably keep
        tracking that position across a scroll event (e.g. one caused by
        a long assistant reply, or even by the footer's own lines pushing
        the viewport) that happens between the save and a later restore;
        when that desync happens, DECRC lands one or more rows off from
        where the input row actually is, and each keystroke's restore
        being off by a different, drifting amount is exactly what caused
        the footer to visibly duplicate itself down the screen. Plain
        relative moves have no "remembered absolute position" to desync
        in the first place — they always act on wherever the cursor
        genuinely is right now.
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

        prompt = "\033[1m>\033[0m "
        prompt_visible_len = 2  # "> " — the bold/reset codes above are zero-width

        def _line_rows(markup: str) -> int:
            """How many terminal rows printing one *logical* line of
            `markup` (rich markup, followed by a newline) actually
            occupies once the terminal auto-wraps it at width `w` — 1
            unless its visible (tag-stripped) length exceeds `w`, in
            which case every extra full width's worth of overflow forces
            another. state_line in particular (status + provider/model +
            session id all on one line) routinely runs long enough to
            wrap on its own once a longer model id is active, and
            footer_rows treating every footer line as exactly one row
            regardless used to under-count whenever that happened — the
            following render's move-up then landed short of the input's
            actual first row, leaving it only partially cleared and
            visibly duplicated below the stale remainder.
            """
            visible_len = len(Text.from_markup(markup).plain)
            return max(1, -(-max(visible_len, 1) // w))

        def _row_col_after(n: int) -> tuple[int, int]:
            """0-indexed row and 1-indexed column of the position right
            after the n-th character has been laid out at terminal width
            `w` (n counts from the input row's own start, prompt
            included) — "deferred wrap" convention: filling a row exactly
            stays put on that row's last column rather than jumping to a
            new, still-empty one, matching how a plain character write
            actually leaves the terminal's cursor (real wrapping only
            happens once a *further* character forces it)."""
            row = (n - 1) // w
            col = min(((n - 1) % w) + 2, w)
            return row, col

        # 0-indexed row (from the input's own first row) the cursor is
        # currently sitting on — 0 before the first render, since nothing
        # has been drawn yet and the cursor is right where the input row
        # starts. Every render moves up exactly this many rows before
        # redrawing, then updates it to reflect where it left off.
        last_cursor_row = 0
        last_text = ""
        last_footer_rows = 0

        def _render(text: str, cursor: int, selection: tuple[int, int] | None) -> None:
            nonlocal last_text, last_cursor_row, last_footer_rows
            last_text = text

            if last_cursor_row:
                sys.stdout.write(f"\x1b[{last_cursor_row}A")
            sys.stdout.write("\r\x1b[0J")  # column 0, wipe any longer previous render

            if selection is None:
                sys.stdout.write(prompt + text)
            else:
                # SGR 7/27 (reverse video on/off) around the selected span
                # — zero-width control codes, don't affect the row/column
                # math below, which only ever counts visible characters.
                lo, hi = selection
                sys.stdout.write(prompt + text[:lo] + "\x1b[7m" + text[lo:hi] + "\x1b[27m" + text[hi:])

            # The footer always starts on the row right after the input's
            # own last row, via an explicit \r\n — not by relying on the
            # terminal's autowrap eventually carrying it there, which
            # inherits the same deferred-wrap ambiguity _row_col_after
            # documents (whether a just-filled row already counts as
            # "used" is exactly what's unclear about that pending state;
            # an explicit \r\n sidesteps needing an answer).
            input_last_row, _ = _row_col_after(prompt_visible_len + len(text))
            footer_lines = [self._state_line(), self._mode_line()]
            if repo_line:
                footer_lines.append(repo_line)
            with _console.capture() as capture:
                _console.print(("─" * w) + "\n" + "\n".join(footer_lines), end="")
            sys.stdout.write("\r\n" + capture.get().replace("\n", "\r\n"))

            # 1 row for the rule (always exactly `w` "─" chars) + however
            # many rows each footer line actually wraps to — not a fixed
            # count, see _line_rows.
            footer_rows = 1 + sum(_line_rows(line) for line in footer_lines)
            footer_last_row = input_last_row + footer_rows
            cursor_row, cursor_col = _row_col_after(prompt_visible_len + cursor)
            rows_up = footer_last_row - cursor_row
            if rows_up:
                sys.stdout.write(f"\x1b[{rows_up}A")
            sys.stdout.write(f"\x1b[{cursor_col}G")
            sys.stdout.flush()

            last_cursor_row = cursor_row
            last_footer_rows = footer_rows

        def on_cycle() -> None:
            self._cycle_permission_mode()

        def land_below_footer() -> None:
            input_last_row, _ = _row_col_after(prompt_visible_len + len(last_text))
            footer_last_row = input_last_row + last_footer_rows
            rows_down = footer_last_row - last_cursor_row
            if rows_down:
                sys.stdout.write(f"\x1b[{rows_down}B")
            # A real \r\n for the last step (not another CSI-B) so the
            # terminal scrolls if the footer was sitting at the bottom of
            # the visible viewport — cursor-only movement would just clamp
            # there instead of producing a fresh line to print into.
            sys.stdout.write("\r\n")
            sys.stdout.flush()

        try:
            raw = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: read_line_with_cycle(
                    prompt, on_render=_render, on_cycle=on_cycle, history=self._prompt_history
                ),
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

            self._prompt_history.append(user_input)

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

    async def _select_model_interactive_repl(self, choices: list[Model], initial: int) -> Model | None:
        """Arrow-key model picker for the classic REPL's bare ``/model``.

        Same relative-cursor-movement redraw technique as ``_prompt_input``
        (see its docstring for why: DECSC/DECRC was tried first and
        abandoned there after it caused the footer-duplication bug this
        session already fixed once) — but simpler, since a menu's row
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

        self._output.print(f"[{DIM_STYLE}]↑/↓ move · enter select · esc cancel[/{DIM_STYLE}]")
        loop = asyncio.get_running_loop()
        index = await loop.run_in_executor(None, lambda: select_from_list(rows, on_render=_render, initial=initial))
        sys.stdout.write("\r\n")
        sys.stdout.flush()
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
