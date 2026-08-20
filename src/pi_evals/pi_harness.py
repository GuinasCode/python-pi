"""Adapter binding pi_coding_agent.AgentSession to pytest-evals.

Mirrors ``packages/evals/src/pi-harness.ts``: given a prompt (or a sequence
of prompt/reload steps), spins up an isolated ``AgentSession`` in a
throwaway cwd/agent_dir, runs it, normalizes the resulting transcript into
JSON-safe events, and aggregates usage stats — all without pytest-evals
needing to know anything about Pi.

Unlike ``vitest-evals``, ``pytest-evals`` has no ``createHarness``/``Harness``
abstraction of its own — ``@pytest.mark.eval`` tests just populate
``eval_bag`` directly. So this module supplies its own minimal harness
result shape; eval tests call ``await harness.run(...)`` and store whatever
they need from the result on ``eval_bag``.
"""

from __future__ import annotations

import inspect
import json
import shutil
import tempfile
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypedDict

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
from pi_ai.models import MutableModels
from pi_coding_agent import Args
from pi_coding_agent.agent_session import AgentSession, AgentSessionOptions
from pi_coding_agent.extensions import ExtensionRunner

__all__ = [
    "PiCodingAgentHarness",
    "PiCodingAgentHarnessOptions",
    "PiEvalInput",
    "PiHarnessResult",
    "PiHarnessUsage",
    "create_pi_coding_agent_harness",
    "resolve_model_selection",
]


class PromptStep(TypedDict):
    type: Literal["prompt"]
    content: str


class ReloadStep(TypedDict):
    type: Literal["reload"]


PiEvalInput = str | Sequence[PromptStep | ReloadStep]


@dataclass
class PiHarnessUsage:
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    tool_calls: int = 0
    cost_total: float = 0.0


@dataclass
class PiHarnessResult:
    output: Any
    events: list[dict[str, Any]]
    usage: PiHarnessUsage
    session_snapshot: str
    total_ms: float


@dataclass
class PiCodingAgentHarnessOptions:
    """Options for :func:`create_pi_coding_agent_harness`.

    ``model`` overrides the runner's default (``PI_PROVIDER``/``PI_MODEL``)
    for this harness specifically — used by comparative eval sets that pin
    each candidate to a fixed model regardless of the runner invocation.
    """

    name: str = "pi-coding-agent"
    model: tuple[str, str] | None = None
    no_tools: bool | list[str] = False
    transform_system_prompt: Callable[[str], str] | None = None
    output: Callable[[str, AgentSession], Any | Awaitable[Any]] | None = None
    # When True, an ExtensionRunner is wired in for this run's isolated
    # cwd/agent_dir — needed for evals that create/use a .pi/extensions/
    # extension mid-run (e.g. extensions.eval.ts). Off by default since most
    # harnesses don't touch extensions and loading is pure overhead for them.
    enable_extensions: bool = False


def resolve_model_selection(
    explicit_model: tuple[str, str] | None,
    environment: dict[str, str] | None = None,
) -> tuple[str, str]:
    """Resolve (provider, model_id) from an explicit override or PI_PROVIDER/PI_MODEL."""
    import os

    env = environment if environment is not None else dict(os.environ)
    provider = (explicit_model[0] if explicit_model else env.get("PI_PROVIDER", "")).strip()
    model_id = (explicit_model[1] if explicit_model else env.get("PI_MODEL", "")).strip()
    if not provider or not model_id:
        raise ValueError("Select a harness model explicitly or set both PI_PROVIDER and PI_MODEL as defaults.")
    return provider, model_id


def _resolve_models(provider: str, model_id: str) -> tuple[MutableModels, Model]:
    """Auto-discover configured providers (same priority as the CLI) and
    resolve the requested provider/model_id, raising if it isn't available.
    """
    from pi_coding_agent.interactive_mode import _setup_models_with_settings

    models, resolved = _setup_models_with_settings(Args(provider=provider, model=model_id))
    if resolved is None or resolved.provider != provider or resolved.id != model_id:
        raise ValueError(f"Eval model not found: {provider}/{model_id}")
    return models, resolved


def _serialize_content_block(block: Any) -> dict[str, Any]:
    if isinstance(block, TextContent):
        return {"type": "text", "text": block.text}
    if isinstance(block, ThinkingContent):
        return {"type": "thinking", "thinking": block.thinking}
    if isinstance(block, ToolCall):
        return {"type": "toolCall", "id": block.id, "name": block.name, "arguments": block.arguments}
    if isinstance(block, ImageContent):
        return {"type": "image"}
    return {"type": getattr(block, "type", "unknown")}


def _serialize_message(message: Message) -> dict[str, Any]:
    """JSON-safe dict for one pi_ai message, for the session snapshot artifact."""
    if isinstance(message, UserMessage):
        content: Any = message.content
        if not isinstance(content, str):
            content = [_serialize_content_block(b) for b in content]
        return {"role": "user", "content": content, "timestamp": message.timestamp}
    if isinstance(message, AssistantMessage):
        return {
            "role": "assistant",
            "content": [_serialize_content_block(b) for b in message.content],
            "model": message.model,
            "stopReason": message.stop_reason.value,
            "timestamp": message.timestamp,
        }
    if isinstance(message, ToolResultMessage):
        return {
            "role": "toolResult",
            "toolCallId": message.tool_call_id,
            "toolName": message.tool_name,
            "content": [_serialize_content_block(b) for b in message.content],
            "isError": message.is_error,
            "timestamp": message.timestamp,
        }
    raise TypeError(f"Unknown message type: {type(message)!r}")


def _serialize_session_snapshot(messages: Sequence[Message]) -> str:
    """JSONL text (one message per line) for the ``.eval/sessions/*.jsonl`` artifact."""
    return "\n".join(json.dumps(_serialize_message(m)) for m in messages)


def _content_text(blocks: Sequence[Any]) -> str:
    return "".join(b.text for b in blocks if isinstance(b, TextContent))


def _to_transcript_events(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Normalize the message list into JSON-safe transcript events, mirroring
    ``toTranscriptEvents`` in the original TS harness."""
    events: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, UserMessage):
            content = message.content if isinstance(message.content, str) else _content_text(message.content)
            events.append({"type": "message", "role": "user", "content": content})
        elif isinstance(message, AssistantMessage):
            text = _content_text(message.content)
            if text:
                events.append({"type": "message", "role": "assistant", "content": text})
            for block in message.content:
                if isinstance(block, ToolCall):
                    events.append(
                        {"type": "tool_call", "id": block.id, "name": block.name, "arguments": block.arguments}
                    )
        elif isinstance(message, ToolResultMessage):
            text = _content_text(message.content)
            event: dict[str, Any] = {
                "type": "tool_result",
                "tool_call_id": message.tool_call_id,
                "name": message.tool_name,
                "content": text,
            }
            if message.is_error:
                event["error"] = {"message": text or "Tool failed"}
            events.append(event)
    return events


async def _prompt_and_get_text(session: AgentSession, content: str) -> str:
    result = await session.prompt(content)
    if result.stop_reason != StopReason.STOP:
        raise RuntimeError(
            result.error_message or f"Agent run ended with unexpected stop reason: {result.stop_reason}."
        )
    output = session.get_last_assistant_text()
    if not output:
        raise RuntimeError("Agent run produced no assistant text.")
    return output


async def _run_pi_coding_agent(pi_input: PiEvalInput, options: PiCodingAgentHarnessOptions) -> PiHarnessResult:
    started = time.perf_counter()
    provider, model_id = resolve_model_selection(options.model)
    models, model = _resolve_models(provider, model_id)

    root = Path(tempfile.mkdtemp(prefix="pi-eval-"))
    cwd = root / "workspace"
    agent_dir = root / "agent"
    cwd.mkdir()
    agent_dir.mkdir()

    try:
        extension_runner = ExtensionRunner(cwd, agent_dir) if options.enable_extensions else None
        session = AgentSession(
            AgentSessionOptions(
                models=models,
                model=model,
                cwd=str(cwd),
                config_dir=str(agent_dir),
                no_tools=options.no_tools,
                enable_subagents=False,
                interactive=False,
                extension_runner=extension_runner,
            )
        )
        # Note: unlike the TS harness, AgentSession doesn't consume a
        # SettingsManager at all yet (settings only affect the CLI/interactive
        # layer above it), so there's nothing to wire an in-memory one into —
        # isolation here comes entirely from the throwaway cwd/agent_dir.

        if options.transform_system_prompt is not None:
            transformed = options.transform_system_prompt(session.get_system_prompt())
            if not transformed.strip():
                raise ValueError("Transformed eval system prompt must not be empty.")
            session.set_system_prompt_override(transformed)

        steps: Sequence[PromptStep | ReloadStep] = (
            [{"type": "prompt", "content": pi_input}] if isinstance(pi_input, str) else pi_input
        )
        response: str | None = None
        for step in steps:
            if step["type"] == "prompt":
                response = await _prompt_and_get_text(session, step["content"])
            elif step["type"] == "reload":
                await session.reload()
            else:
                raise ValueError(f"Unknown eval step type: {step['type']!r}")

        if response is None:
            raise ValueError("Pi eval input must include at least one prompt step.")

        output: Any = options.output(response, session) if options.output is not None else response
        if inspect.isawaitable(output):
            output = await output

        stats = session.get_session_stats()
        usage = PiHarnessUsage(
            provider=model.provider,
            model=model.id,
            input_tokens=stats.input_tokens,
            output_tokens=stats.output_tokens,
            total_tokens=stats.total_tokens,
            cache_read_tokens=stats.cache_read_tokens,
            cache_write_tokens=stats.cache_write_tokens,
            tool_calls=stats.tool_calls,
            cost_total=stats.cost_total,
        )
        return PiHarnessResult(
            output=output,
            events=_to_transcript_events(session._messages),
            usage=usage,
            session_snapshot=_serialize_session_snapshot(session._messages),
            total_ms=(time.perf_counter() - started) * 1000,
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


@dataclass
class PiCodingAgentHarness:
    """Callable-ish harness: ``await harness.run(input)`` runs one isolated
    AgentSession turn (or prompt/reload sequence) and returns a
    :class:`PiHarnessResult`."""

    options: PiCodingAgentHarnessOptions = field(default_factory=PiCodingAgentHarnessOptions)

    async def run(self, pi_input: PiEvalInput) -> PiHarnessResult:
        return await _run_pi_coding_agent(pi_input, self.options)


def create_pi_coding_agent_harness(
    *,
    name: str = "pi-coding-agent",
    model: tuple[str, str] | None = None,
    no_tools: bool | list[str] = False,
    transform_system_prompt: Callable[[str], str] | None = None,
    output: Callable[[str, AgentSession], Any | Awaitable[Any]] | None = None,
) -> PiCodingAgentHarness:
    """Create a :class:`PiCodingAgentHarness` — mirrors ``createPiCodingAgentHarness`` in pi-harness.ts."""
    return PiCodingAgentHarness(
        PiCodingAgentHarnessOptions(
            name=name,
            model=model,
            no_tools=no_tools,
            transform_system_prompt=transform_system_prompt,
            output=output,
        )
    )
