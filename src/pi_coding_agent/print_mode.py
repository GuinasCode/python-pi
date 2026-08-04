"""Print mode for non-interactive Pi usage.

Mirrors packages/coding-agent/src/modes/print-mode.ts.
Connects to the agent loop with faux or real LLM providers.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any

from pi_ai import Context, Model, UserMessage
from pi_ai.models import MutableModels
from pi_coding_agent import Args


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


async def _run_text_mode(args: Args, prompt: str) -> int:
    """Run in text print mode: send prompt to LLM and print response."""
    print(f"[pi] {prompt}", file=sys.stderr)

    models, model = _setup_models(args)
    if model is None:
        print("Error: No model available", file=sys.stderr)
        return 1

    context = Context(
        system_prompt=args.system_prompt or "You are a helpful coding assistant.",
        messages=[UserMessage(content=prompt, timestamp=int(time.time() * 1000))],
    )

    stream = models.stream(model, context)

    output_text = ""
    async for event in stream:
        event_type = getattr(event, "type", "")
        if event_type == "text_delta":
            delta = getattr(event, "delta", "")
            print(delta, end="", flush=True)
            output_text += delta
        elif event_type == "done":
            if output_text:
                print()  # newline after streaming output
        elif event_type == "error":
            msg = getattr(event, "error", None)
            if msg and hasattr(msg, "error_message") and msg.error_message:
                print(f"\nError: {msg.error_message}", file=sys.stderr)
            return 1

    return 0


async def _run_json_mode(args: Args, prompt: str) -> int:
    """Run in JSON event stream mode."""
    models, model = _setup_models(args)
    if model is None:
        print(json.dumps({"type": "error", "message": "No model available"}))
        return 1

    context = Context(
        system_prompt=args.system_prompt or "You are a helpful coding assistant.",
        messages=[UserMessage(content=prompt, timestamp=int(time.time() * 1000))],
    )

    stream = models.stream(model, context)

    async for event in stream:
        event_type = getattr(event, "type", "")
        if event_type == "start":
            print(json.dumps({"type": "start"}))
        elif event_type == "text_delta":
            print(json.dumps({"type": "text_delta", "delta": getattr(event, "delta", "")}))
        elif event_type == "text_end":
            print(json.dumps({"type": "text_end", "content": getattr(event, "content", "")}))
        elif event_type == "done":
            msg = getattr(event, "message", None)
            print(
                json.dumps(
                    {
                        "type": "done",
                        "stop_reason": getattr(msg, "stop_reason", "stop"),
                    }
                )
            )
            return 0
        elif event_type == "error":
            msg = getattr(event, "error", None)
            print(
                json.dumps(
                    {
                        "type": "error",
                        "message": getattr(msg, "error_message", "Unknown error"),
                    }
                )
            )
            return 1

    return 0


def run_print_mode_sync(args: Args) -> int:
    """Synchronous wrapper for run_print_mode."""
    return asyncio.run(run_print_mode(args))
