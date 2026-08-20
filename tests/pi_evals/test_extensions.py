"""Extension-authoring eval: port of packages/evals/src/extensions.eval.ts.

Runs a real model through creating a Pi extension, reloading, and using
its tool — comparing the default system prompt (Guidelines + Pi docs)
against a stripped variant, to measure whether that guidance actually
helps the model succeed. Like test_smoke.py, this needs PI_PROVIDER/
PI_MODEL and only runs under `--run-eval` (see test_smoke.py's docstring
for why a plain `pytest` run never triggers it).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from pi_ai import TextContent, ToolResultMessage
from pi_coding_agent.agent_session import AgentSession
from pi_evals import PiHarnessResult, create_judge, create_pi_coding_agent_harness, harness_table
from pi_evals.harness_table import compute_pass_rate_lift, passed_from_score


def _exclude_guidelines_and_documentation(default_prompt: str) -> str:
    marker = "\nGuidelines:\n"
    idx = default_prompt.find(marker)
    if idx == -1:
        raise ValueError("Default Pi system prompt has no Guidelines section.")
    return default_prompt[:idx]


def _prepare_default_prompt_override(default_prompt: str) -> str:
    marker = "\nCurrent working directory: "
    idx = default_prompt.rfind(marker)
    if idx == -1:
        raise ValueError("Default Pi system prompt has no working-directory section.")
    return default_prompt[:idx]


def _hello_tool_succeeded(session: AgentSession) -> bool:
    for message in session._messages:
        if not (isinstance(message, ToolResultMessage) and message.tool_name == "hello" and not message.is_error):
            continue
        text = "".join(b.text for b in message.content if isinstance(b, TextContent))
        if text.strip() == "Hello, Bob!":
            return True
    return False


def _extension_authoring_output(response: str, session: AgentSession) -> dict[str, Any]:
    extensions_result = session.get_extensions()
    # The prompt doesn't mandate a filename, but (matching the original
    # eval) we only check the conventional one — a model that names it
    # something else fails this check, same known quirk as upstream.
    extension_path = Path(session._cwd) / ".pi" / "extensions" / "hello.py"
    extension_source = extension_path.read_text(encoding="utf-8") if extension_path.is_file() else None
    return {
        "response": response,
        "system_prompt_has_guidelines": "\nGuidelines:\n" in session.get_system_prompt(),
        "system_prompt_has_pi_docs": "Pi documentation (read only" in session.get_system_prompt(),
        "extension_errors": extensions_result.errors,
        "loaded_extensions": [{"path": e.path, "tools": e.tool_names} for e in extensions_result.extensions],
        "extension_source": extension_source,
        "hello_tool_succeeded": _hello_tool_succeeded(session),
    }


def _create_extension_authoring_harness(name: str, transform_system_prompt: Any = None) -> Any:
    return create_pi_coding_agent_harness(
        name=name,
        transform_system_prompt=transform_system_prompt,
        enable_extensions=True,
        output=_extension_authoring_output,
    )


def _extension_authoring_score(output: dict[str, Any], **_: Any) -> float:
    failures: list[str] = []
    source = output["extension_source"]
    if source is None:
        failures.append("generated extension source is unavailable")
    else:
        if "AgentTool" not in source or "pi_agent_core" not in source:
            failures.append("extension does not appear to use pi_agent_core's AgentTool")
        if "@mariozechner/" in source or "@sinclair/typebox" in source or "@earendil-works/" in source:
            failures.append("extension imports a legacy TS-era package name")
    if output["extension_errors"]:
        failures.append("extension loader reported errors")
    if not any("hello" in ext["tools"] for ext in output["loaded_extensions"]):
        failures.append('no loaded extension registered the "hello" tool')
    if not output["hello_tool_succeeded"]:
        failures.append('no successful hello tool call returned "Hello, Bob!"')
    if output["response"].strip() != "Hello, Bob!":
        failures.append('final response was not exactly "Hello, Bob!"')
    return 0.0 if failures else 1.0


_extension_authoring_judge = create_judge("ExtensionAuthoringJudge", _extension_authoring_score, threshold=None)

_rows = harness_table(
    baseline=_create_extension_authoring_harness("system-prompt-without-docs", _exclude_guidelines_and_documentation),
    candidate=_create_extension_authoring_harness("default-system-prompt", _prepare_default_prompt_override),
)


@pytest.mark.eval(name="pi_extension_authoring")
@pytest.mark.parametrize("row", _rows, ids=[f"{r.name}-{r.repetition}" for r in _rows])
def test_creates_reloads_and_uses_a_hello_extension(row: Any, eval_bag: Any) -> None:
    result: PiHarnessResult = asyncio.run(
        row.harness.run(
            [
                {
                    "type": "prompt",
                    "content": (
                        "Create a Pi extension with a hello tool that takes a name and returns a greeting. "
                        "For example, passing Bob should return `Hello, Bob!`."
                    ),
                },
                {"type": "reload"},
                {
                    "type": "prompt",
                    "content": (
                        "Use the hello tool to greet Bob. Respond with exactly the tool's greeting and nothing else."
                    ),
                },
            ]
        )
    )
    output = result.output
    eval_bag.harness_name = row.name
    eval_bag.output = output
    eval_bag.usage = result.usage

    score = asyncio.run(_extension_authoring_judge(output=output))
    eval_bag.score = score.score
    eval_bag.passed = passed_from_score(score.score)

    expects_full_prompt = row.name == "candidate"
    assert output["system_prompt_has_guidelines"] == expects_full_prompt
    assert output["system_prompt_has_pi_docs"] == expects_full_prompt


@pytest.mark.eval_analysis(name="pi_extension_authoring")
def test_pi_extension_authoring_analysis(eval_results: Any) -> None:
    assert len(eval_results) > 0
    lifts = compute_pass_rate_lift(
        eval_results,
        harness_name=lambda r: r.result.harness_name,
        passed=lambda r: r.result.passed,
        baseline_name="baseline",
    )
    for lift in lifts:
        print(
            f"{lift.candidate}: baseline={lift.baseline_pass_rate:.0%} "
            f"candidate={lift.candidate_pass_rate:.0%} lift={lift.lift_pp:+.1f}pp"
        )
    # judgeThreshold: null in the original — this is an observation, not a
    # hard gate. A comparative set where the candidate never loses is
    # probably measuring nothing.
