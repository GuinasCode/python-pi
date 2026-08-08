"""Print mode for non-interactive Pi usage.

Mirrors packages/coding-agent/src/modes/print-mode.ts.
Runs a prompt through the full agent loop (:class:`AgentSession`) — same
tool-calling machinery as interactive mode — so a single ``pi --print``
invocation can actually read/grep/ls/bash, not just produce a bare LLM
completion. This matters most for the ``subagent`` tool, which spawns a
child ``pi --print`` process per subagent: without a real tool loop here,
a subagent asked to "read every file in X" has no way to do so and
fabricates a plausible-sounding answer instead.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

from pi_ai import Model, StopReason
from pi_ai.models import MutableModels
from pi_coding_agent import Args
from pi_coding_agent.agent_session import AgentSession, AgentSessionOptions, get_builtin_tools


async def run_print_mode(args: Args) -> int:
    """Run Pi in print mode: execute a single prompt and print the result."""
    prompt = " ".join(args.messages) if args.messages else ""
    if not prompt:
        print("Error: No prompt provided", file=sys.stderr)
        return 1

    if args.mode == "json":
        return await _run_json_mode(args, prompt)
    return await _run_text_mode(args, prompt)


def _setup_models(args: Args) -> tuple[MutableModels, Any]:
    """Set up the models collection with available providers."""
    from pi_ai.models import Provider
    from pi_ai.providers.faux import faux_assistant_message, faux_provider

    models = MutableModels()

    # Try NVIDIA GLM 5.2 first if NVAPI_KEY is available
    nvapi_key = args.api_key or os.environ.get("NVAPI_KEY")
    if nvapi_key:
        try:
            from pi_ai.providers.nvidia_glm import nvidia_glm_provider

            model, provider_models, _meta = nvidia_glm_provider(
                Model(id="test"),
                api_key=nvapi_key,
            )
            return provider_models, model
        except Exception:
            pass

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

    # Fall back to faux provider
    handle = faux_provider()
    handle.set_responses([faux_assistant_message("(faux provider: set OPENAI_API_KEY for real LLM responses)")])
    models.set_provider(handle.provider)
    return models, handle.get_model()


def _resolve_tools(args: Args) -> tuple[list[Any] | None, bool]:
    """Resolve the tool list and whether subagent auto-wiring should apply.

    Returns ``(tools, enable_subagents)``. When the caller passed an
    explicit ``--tools``/``--exclude-tools``/``--no-tools`` restriction
    (as ``run_subagent`` does, from an agent definition's ``tools:``
    frontmatter), that restriction is authoritative — the subagent tool
    is not silently re-added on top of it.
    """
    if args.no_tools or args.no_builtin_tools:
        return [], False

    restricted = bool(args.tools) or bool(args.exclude_tools)
    tools = get_builtin_tools()
    if args.tools:
        allowed = set(args.tools)
        tools = [t for t in tools if t.name in allowed]
    if args.exclude_tools:
        excluded = set(args.exclude_tools)
        tools = [t for t in tools if t.name not in excluded]

    return tools, not restricted


def _build_session(models: MutableModels, model: Any, args: Args) -> AgentSession:
    tools, enable_subagents = _resolve_tools(args)
    append_system_prompt = "\n\n".join(args.append_system_prompt) if args.append_system_prompt else None
    return AgentSession(
        AgentSessionOptions(
            models=models,
            model=model,
            cwd=os.getcwd(),
            system_prompt=args.system_prompt,
            append_system_prompt=append_system_prompt,
            tools=tools,
            enable_subagents=enable_subagents,
            temperature=args.temperature,
            interactive=False,
        )
    )


async def _run_text_mode(args: Args, prompt: str) -> int:
    """Run in text print mode: run the agent loop and print the result."""
    print(f"[pi] {prompt}", file=sys.stderr)

    models, model = _setup_models(args)
    if model is None:
        print("Error: No model available", file=sys.stderr)
        return 1

    session = _build_session(models, model, args)

    printed_any = False

    def _on_event(event: Any) -> None:
        nonlocal printed_any
        event_type = getattr(event, "type", "")
        if event_type == "text_delta":
            delta = getattr(event, "delta", "")
            print(delta, end="", flush=True)
            printed_any = True
        elif event_type == "tool_call_start":
            print(f"\n[pi] tool: {getattr(event, 'name', '')}", file=sys.stderr)
        elif event_type == "tool_call_end" and getattr(event, "is_error", False):
            print(f"[pi] tool error: {getattr(event, 'result_text', '')}", file=sys.stderr)

    session.on_event(_on_event)
    result = await session.prompt(prompt)

    if printed_any:
        print()

    if result.stop_reason == StopReason.ERROR:
        print(f"\nError: {result.error_message or 'Unknown error'}", file=sys.stderr)
        return 1
    return 0


async def _run_json_mode(args: Args, prompt: str) -> int:
    """Run in JSON event stream mode."""
    models, model = _setup_models(args)
    if model is None:
        print(json.dumps({"type": "error", "message": "No model available"}))
        return 1

    session = _build_session(models, model, args)

    def _on_event(event: Any) -> None:
        event_type = getattr(event, "type", "")
        if event_type == "start":
            print(json.dumps({"type": "start"}))
        elif event_type == "text_delta":
            print(json.dumps({"type": "text_delta", "delta": getattr(event, "delta", "")}))
        elif event_type == "text_end":
            print(json.dumps({"type": "text_end", "content": getattr(event, "content", "")}))
        elif event_type == "tool_call_start":
            print(json.dumps({"type": "tool_call_start", "name": getattr(event, "name", "")}))
        elif event_type == "tool_call_end":
            print(
                json.dumps(
                    {
                        "type": "tool_call_end",
                        "name": getattr(event, "name", ""),
                        "is_error": getattr(event, "is_error", False),
                    }
                )
            )

    session.on_event(_on_event)
    result = await session.prompt(prompt)

    if result.stop_reason == StopReason.ERROR:
        print(json.dumps({"type": "error", "message": result.error_message or "Unknown error"}))
        return 1

    print(json.dumps({"type": "done", "stop_reason": result.stop_reason.value}))
    return 0


def run_print_mode_sync(args: Args) -> int:
    """Synchronous wrapper for run_print_mode."""
    return asyncio.run(run_print_mode(args))
