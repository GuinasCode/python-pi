"""Tests for system prompt construction."""

from __future__ import annotations

from pi_coding_agent.system_prompt import BuildSystemPromptOptions, build_system_prompt


def test_default_prompt() -> None:
    opts = BuildSystemPromptOptions(cwd="/home/user/project")
    prompt = build_system_prompt(opts)
    assert "coding assistant" in prompt
    assert "Available tools:" in prompt
    assert "Guidelines:" in prompt
    assert "/home/user/project" in prompt


def test_custom_prompt() -> None:
    opts = BuildSystemPromptOptions(
        custom_prompt="You are a code reviewer.",
        cwd="/tmp",
    )
    prompt = build_system_prompt(opts)
    assert "You are a code reviewer." in prompt
    assert "/tmp" in prompt


def test_default_prompt_has_pi_documentation_section() -> None:
    opts = BuildSystemPromptOptions(cwd="/home/user/project")
    prompt = build_system_prompt(opts)
    assert "\nGuidelines:\n" in prompt
    assert "Pi documentation (read only" in prompt
    assert "extensions.md" in prompt
    assert "hello.py" in prompt


def test_custom_prompt_has_no_pi_documentation_section() -> None:
    opts = BuildSystemPromptOptions(custom_prompt="You are a code reviewer.", cwd="/tmp")
    prompt = build_system_prompt(opts)
    assert "Pi documentation" not in prompt


def test_append_system_prompt() -> None:
    opts = BuildSystemPromptOptions(
        cwd="/tmp",
        append_system_prompt="Always use type hints.",
    )
    prompt = build_system_prompt(opts)
    assert "Always use type hints." in prompt


def test_context_files() -> None:
    opts = BuildSystemPromptOptions(
        cwd="/tmp",
        context_files=[
            {"path": "AGENTS.md", "content": "Follow PEP 8."},
        ],
    )
    prompt = build_system_prompt(opts)
    assert "AGENTS.md" in prompt
    assert "Follow PEP 8." in prompt
    assert "project_context" in prompt


def test_selected_tools() -> None:
    opts = BuildSystemPromptOptions(
        cwd="/tmp",
        selected_tools=["read", "bash"],
        tool_snippets={"read": "Read file contents", "bash": "Execute command"},
    )
    prompt = build_system_prompt(opts)
    assert "read: Read file contents" in prompt
    assert "bash: Execute command" in prompt


def test_prompt_guidelines() -> None:
    opts = BuildSystemPromptOptions(
        cwd="/tmp",
        prompt_guidelines=["Never use global variables"],
    )
    prompt = build_system_prompt(opts)
    assert "Never use global variables" in prompt


def test_windows_path_normalized() -> None:
    opts = BuildSystemPromptOptions(cwd="C:\\Users\\test\\project")
    prompt = build_system_prompt(opts)
    assert "C:/Users/test/project" in prompt


def test_no_tools() -> None:
    opts = BuildSystemPromptOptions(
        cwd="/tmp",
        selected_tools=[],
    )
    prompt = build_system_prompt(opts)
    assert "(none)" in prompt


def test_contract_adherence_guideline_always_present() -> None:
    prompt = build_system_prompt(BuildSystemPromptOptions(cwd="/tmp"))
    assert "literal contract" in prompt


def test_interactive_prompts_to_ask_and_wait() -> None:
    prompt = build_system_prompt(BuildSystemPromptOptions(cwd="/tmp", interactive=True))
    assert "ask the user which path" in prompt
    assert "stop and wait for" in prompt


def test_non_interactive_flags_instead_of_asking() -> None:
    prompt = build_system_prompt(BuildSystemPromptOptions(cwd="/tmp", interactive=False))
    assert "no interactive session" in prompt
    assert "ask the user which path" not in prompt


def test_custom_prompt_skips_contract_guideline() -> None:
    """Subagent personas (custom_prompt) keep their own scope — no injected guideline."""
    prompt = build_system_prompt(BuildSystemPromptOptions(custom_prompt="You are a scout.", cwd="/tmp"))
    assert "literal contract" not in prompt


def test_referenced_docs_and_example_paths_actually_exist() -> None:
    """The paths the system prompt tells the model to read must be real —
    a broken path here means the model gets sent to read nothing."""
    from pi_coding_agent.config import get_docs_path, get_examples_path

    assert (get_docs_path() / "extensions.md").is_file()
    assert (get_examples_path() / "extensions" / "hello.py").is_file()
