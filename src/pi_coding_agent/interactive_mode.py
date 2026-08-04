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

from pi_ai import AssistantMessage, Message, StopReason, UserMessage
from pi_ai.models import MutableModels, Provider
from pi_coding_agent import Args
from pi_coding_agent.agent_session import AgentSession, AgentSessionOptions
from pi_coding_agent.config import ensure_config_dir, ensure_session_dir, get_config_dir, get_session_dir
from pi_coding_agent.resource_loader import load_resources
from pi_coding_agent.session_manager import SessionEntry, SessionManager


def _setup_models(args: Args) -> tuple[MutableModels, Any]:
    """Set up models with available providers."""
    models = MutableModels()

    # Try OpenAI if API key is available
    openai_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if openai_key:
        try:
            from pi_ai.providers.openai import openai_provider

            model, stream_fn = openai_provider(api_key=openai_key)
            provider = Provider(
                id="openai",
                name="OpenAI",
                models=[model],
                stream_fn=stream_fn,  # type: ignore[arg-type]
            )
            models.set_provider(provider)
            return models, model
        except Exception:
            pass

    # Fall back to faux provider. Queue many context-aware responses so the
    # interactive REPL can run multiple turns without a real API key.
    from pi_ai.providers.faux import faux_assistant_message, faux_provider

    def faux_reply(context: Any, _options: Any, _state: dict[str, int], _model: Any) -> AssistantMessage:
        last_user = ""
        for message in reversed(context.messages):
            if message.role == "user":
                last_user = message.content if isinstance(message.content, str) else str(message.content)
                break
        return faux_assistant_message(
            "(faux provider: set OPENAI_API_KEY for real LLM responses)\n"
            f"I received your message and kept it in context: {last_user}"
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
        settings = settings_mgr.get_settings()
        provider_name = settings.get("provider")
        model_id = settings.get("model")
    except Exception:
        provider_name = None
        model_id = None

    # Override with CLI args
    if args.provider:
        provider_name = args.provider
    if args.model:
        model_id = args.model

    models, default_model = _setup_models(args)

    # Try to find a specific model if specified
    if model_id and provider_name:
        found = models.get_model(provider_name, model_id)
        if found:
            return models, found

    return models, default_model


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

        # Load resources (AGENTS.md, SYSTEM.md, skills)
        resources = load_resources(cwd, self._config_dir)

        # Create agent session
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

        # Session persistence
        self._session_manager = session_manager
        self._session_id = session_id
        self._message_count = 0

        # Restore persisted messages if this is a resumed session.
        self._restore_persisted_messages()

        # State
        self._running = False
        self._event_handlers: list[Callable[[str], None]] = []

    def _restore_persisted_messages(self) -> None:
        """Load messages from the session JSONL file into the agent session."""
        if not self._session_manager or not self._session_id:
            return
        entries = self._session_manager.get_entries(self._session_id)
        for entry in entries:
            if entry.kind != "message":
                continue
            data = entry.data
            role = data.get("role", "user")
            if role == "user":
                msg = UserMessage(content=data.get("content", ""), timestamp=data.get("timestamp", 0))
            elif role == "assistant":
                from pi_ai import TextContent

                content_blocks = []
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
                # Skip tool results in the restored context for simplicity
                continue
            else:
                continue
            self._agent_session._messages.append(msg)
            self._message_count += 1

    def _on_tool_call_start(self, name: str, args: dict) -> None:
        """Print tool call start."""
        args_str = " ".join(f"{k}={v!r}" for k, v in args.items() if v is not None)
        print(f"\n[tool] {name}({args_str})", file=sys.stderr)

    def _on_tool_call_end(self, name: str, is_error: bool) -> None:
        """Print tool call end."""
        status = "error" if is_error else "done"
        print(f"[tool] {name} {status}", file=sys.stderr)

    def _persist_message(self, message: Message) -> None:
        """Persist a message to the session JSONL file."""
        if not self._session_manager or not self._session_id:
            return

        kind = "message"
        data: dict[str, Any] = {"role": message.role}

        if message.role == "user":
            content = message.content
            if isinstance(content, str):
                data["content"] = content
            elif isinstance(content, list):
                data["content"] = [{"type": b.type, "text": b.text} if hasattr(b, "type") else str(b) for b in content]
        elif message.role == "assistant":
            data["content"] = [
                {"type": b.type, "text": b.text}
                if b.type == "text"
                else {"type": b.type, "thinking": b.thinking}
                if b.type == "thinking"
                else {"type": b.type, "id": b.id, "name": b.name, "arguments": b.arguments}
                for b in message.content
            ]
            data["model"] = message.model
            data["stop_reason"] = (
                message.stop_reason.value if hasattr(message.stop_reason, "value") else str(message.stop_reason)
            )
        elif message.role == "toolResult":
            data["tool_call_id"] = message.tool_call_id
            data["tool_name"] = message.tool_name
            data["is_error"] = message.is_error

        entry = SessionEntry(
            seq=self._message_count,
            parent_seq=self._message_count - 1 if self._message_count > 0 else None,
            kind=kind,
            data=data,
            timestamp=int(time.time() * 1000),
        )
        self._session_manager.append_entry(self._session_id, entry)
        self._message_count += 1

    async def run_turn(self, user_input: str) -> AssistantMessage | None:
        """Run a single conversation turn."""
        # Persist user message
        user_msg = UserMessage(content=user_input, timestamp=int(time.time() * 1000))
        self._persist_message(user_msg)

        # Register event handlers for tool calls
        self._agent_session.on_event(self._handle_event)

        # Run the agent
        try:
            result = await self._agent_session.prompt(user_input)
        except Exception as exc:
            print(f"\n[error] {exc}", file=sys.stderr)
            return None

        # Persist assistant message
        if result:
            self._persist_message(result)

        return result

    def _handle_event(self, event: Any) -> None:
        """Handle agent events for display."""
        event_type = getattr(event, "type", "")
        if event_type == "text_delta":
            delta = getattr(event, "delta", "")
            print(delta, end="", flush=True)
        elif event_type == "tool_call_start":
            name = getattr(event, "name", "")
            args = getattr(event, "args", {})
            self._on_tool_call_start(name, args)
        elif event_type == "tool_call_end":
            name = getattr(event, "name", "")
            is_error = getattr(event, "is_error", False)
            self._on_tool_call_end(name, is_error)
        elif event_type == "text_end":
            print()  # newline after streaming text

    async def repl_loop(self) -> None:
        """Main REPL loop."""
        self._running = True

        # Print banner
        print(f"pi v0.83.0 (Python) — cwd: {self._cwd}")
        provider_name = getattr(self._model, "provider", "faux")
        model_name = getattr(self._model, "name", "unknown")
        print(f"Provider: {provider_name} | Model: {model_name}")
        print("Type your message. Use /help for commands, /exit to quit.\n")

        while self._running:
            try:
                # Read input
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input("\033[1m> \033[0m").strip()
                )
            except EOFError:
                break
            except KeyboardInterrupt:
                print("\n[interrupted]")
                continue

            if not user_input:
                continue

            # Handle slash commands
            if user_input.startswith("/"):
                if self._handle_command(user_input):
                    continue
                else:
                    break
                continue

            # Run agent turn
            print()  # blank line before response
            result = await self.run_turn(user_input)
            print()  # blank line after response

            if result and result.stop_reason == StopReason.ERROR:
                error_msg = result.error_message or "Unknown error"
                print(f"[error] {error_msg}", file=sys.stderr)

    def _handle_command(self, command: str) -> bool:
        """Handle slash commands. Returns True to continue, False to exit."""
        cmd = command.lower().strip()

        if cmd in ("/exit", "/quit", "/q"):
            return False

        if cmd == "/help":
            print("Commands:")
            print("  /help     Show this help")
            print("  /exit     Exit Pi")
            print("  /quit     Exit Pi")
            print("  /model    Show current model")
            print("  /clear    Clear conversation history")
            print("  /tools    List available tools")
            print("  /session  Show session info")
            return True

        if cmd == "/model":
            print(f"Provider: {getattr(self._model, 'provider', '?')}")
            print(f"Model: {getattr(self._model, 'id', '?')}")
            print(f"Context window: {getattr(self._model, 'context_window', '?')}")
            return True

        if cmd == "/clear":
            self._agent_session._messages = []
            print("[conversation cleared]")
            return True

        if cmd == "/tools":
            for tool in self._agent_session._tools:
                print(f"  {tool.name}: {tool.description}")
            return True

        if cmd == "/session":
            if self._session_id:
                print(f"Session ID: {self._session_id}")
                print(f"Messages: {self._message_count}")
            else:
                print("[no session]")
            return True

        print(f"Unknown command: {command}")
        return True


async def run_interactive_mode(args: Args) -> int:
    """Run Pi in interactive mode."""
    cwd = str(Path.cwd())
    config_dir = get_config_dir()
    ensure_config_dir()
    session_dir = get_session_dir()
    ensure_session_dir()

    # Set up models
    models, model = _setup_models_with_settings(args)
    if model is None:
        print("Error: No model available", file=sys.stderr)
        return 1

    # Set up session
    session_mgr = SessionManager(session_dir)

    session_id = None
    if args.continue_session or args.resume:
        # Continue most recent session
        recent = session_mgr.continue_recent()
        if recent:
            session_id = recent.id
            print(f"[continuing session: {recent.id}]")
        else:
            print("[no previous session found, starting new]")

    if not session_id and not args.no_session:
        session_name = args.name or Path(cwd).name
        info = session_mgr.create_session(cwd=cwd, name=session_name)
        session_id = info.id

    # Create and run interactive session
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
        print("\n[exiting]")

    return 0


def run_interactive_sync(args: Args) -> int:
    """Synchronous wrapper for run_interactive_mode."""
    return asyncio.run(run_interactive_mode(args))
