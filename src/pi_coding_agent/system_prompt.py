"""System prompt construction and project context loading.

Mirrors packages/coding-agent/src/core/system-prompt.ts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pi_coding_agent.config import get_docs_path, get_examples_path


@dataclass
class BuildSystemPromptOptions:
    """Options for building the system prompt."""

    custom_prompt: str | None = None
    selected_tools: list[str] | None = None
    tool_snippets: dict[str, str] | None = None
    prompt_guidelines: list[str] | None = None
    append_system_prompt: str | None = None
    cwd: str = ""
    context_files: list[dict[str, str]] | None = None
    skills: list[Any] | None = None
    # Whether the user can be asked a follow-up question and reply in a
    # later turn (true for interactive mode, false for print/subagent runs).
    interactive: bool = True
    # Snippets of relevant persisted memories for the current turn, already
    # formatted as "[type] title: content". Changes every turn, unlike
    # context_files/skills which are fixed for the session.
    memories: list[str] | None = None
    # Whether the memory subsystem is active for this session — controls
    # whether the <memory_policy> guideline (instructing the model to call
    # `remember` proactively) is included.
    memory_enabled: bool = False
    # Every persisted Soul principle (type=soul), already formatted as
    # "title: content". Unlike `memories`, this is a guaranteed direct read
    # by type (see AgentSession._load_soul), not a similarity-search result
    # — so it's included whenever any Soul entries exist, independent of
    # whether they happen to match the current turn's text.
    soul: list[str] | None = None


def build_system_prompt(options: BuildSystemPromptOptions) -> str:
    """Build the system prompt with tools, guidelines, and context."""
    prompt_cwd = options.cwd.replace("\\", "/")
    append_section = f"\n\n{options.append_system_prompt}" if options.append_system_prompt else ""
    context_files = options.context_files or []
    skills = options.skills or []

    if options.custom_prompt:
        prompt = options.custom_prompt
        if append_section:
            prompt += append_section
        prompt += _format_soul_for_prompt(options.soul)
        prompt += _format_memories_for_prompt(options.memories)
        prompt += _memory_policy_block(options.memory_enabled)
        if context_files:
            prompt += "\n\n<project_context>\n\n"
            prompt += "Project-specific instructions and guidelines:\n\n"
            for cf in context_files:
                prompt += f'<project_instructions path="{cf["path"]}">\n{cf["content"]}\n</project_instructions>\n\n'
            prompt += "</project_context>\n"

        has_read = not options.selected_tools or "read" in options.selected_tools
        if has_read and skills:
            prompt += _format_skills_for_prompt(skills)

        prompt += f"\nCurrent working directory: {prompt_cwd}"
        return prompt

    tools = options.selected_tools or ["read", "bash", "edit", "write"]
    tool_snippets = options.tool_snippets or {}
    visible_tools = [name for name in tools if name in tool_snippets]
    tools_list = "\n".join(f"- {name}: {tool_snippets[name]}" for name in visible_tools) if visible_tools else "(none)"

    guidelines_set: set[str] = set()
    guidelines_list: list[str] = []

    def add_guideline(guideline: str) -> None:
        if guideline not in guidelines_set:
            guidelines_set.add(guideline)
            guidelines_list.append(guideline)

    has_bash = "bash" in tools
    has_grep = "grep" in tools
    has_find = "find" in tools
    has_ls = "ls" in tools
    has_read = "read" in tools

    if has_bash and not has_grep and not has_find and not has_ls:
        add_guideline("Use bash for file operations like ls, rg, find")

    has_webfetch = "webfetch" in tools
    has_browser = "browser" in tools
    if has_webfetch and has_browser:
        add_guideline(
            "You have real web access — use it rather than declining or asking the user to fetch "
            "something for you. Try `webfetch` first (fast, no JavaScript); if the result looks "
            "like an empty app shell despite a 200 status, the page is client-side rendered — use "
            "`browser` instead, which runs a real headless browser and returns the rendered page's "
            "text. If `browser` reports Playwright isn't installed, say so plainly rather than "
            "silently giving up or fabricating results."
        )
    elif has_webfetch:
        add_guideline(
            "You have real web access via `webfetch` — use it rather than declining or asking the "
            "user to fetch something for you. It doesn't execute JavaScript, so client-side "
            "rendered pages may come back as an empty app shell; say so plainly if that happens "
            "rather than fabricating results."
        )

    for guideline in options.prompt_guidelines or []:
        normalized = guideline.strip()
        if normalized:
            add_guideline(normalized)

    add_guideline("Be concise in your responses")
    add_guideline("Show file paths clearly when working with files")
    add_guideline(
        "Follow the literal contract of the user's request: the exact deliverable "
        'format, scope, and constraints they specified (e.g. "only propose a diff, '
        "don't touch the code\" means literally emit a diff, not prose or an actual "
        "edit; \"just answer, don't implement\" means don't call write/edit tools)."
    )
    if options.interactive:
        add_guideline(
            "If your own reasoning concludes the requested contract is not the best "
            "approach, do not silently substitute your own approach. State your "
            "reasoning and ask the user which path they want, then stop and wait for "
            "their reply before proceeding — don't decide on their behalf."
        )
    else:
        add_guideline(
            "If your own reasoning concludes the requested contract is not the best "
            "approach, do not silently substitute your own approach either. There is "
            "no interactive session to ask a follow-up in, so: follow the literal "
            "contract as given, and clearly flag the concern and your suggested "
            "alternative in the response instead of unilaterally deviating."
        )

    guidelines = "\n".join(f"- {g}" for g in guidelines_list)
    docs_path = get_docs_path()
    examples_path = get_examples_path()

    prompt = (
        "You are an expert coding assistant operating inside pi, a coding agent harness."
        " You help users by reading files, executing commands, editing code,"
        " and writing new files.\n\n"
        f"Available tools:\n{tools_list}\n\n"
        "In addition to the tools above, you may have access to other custom tools"
        " depending on the project.\n\n"
        f"Guidelines:\n{guidelines}\n\n"
        "Pi documentation (read only when the user asks about pi itself, its SDK, or extensions):\n"
        f"- Extension authoring guide: {docs_path / 'extensions.md'}\n"
        f"- Example extension: {examples_path / 'extensions' / 'hello.py'}\n"
        "- When asked to write a Pi extension, read the extension authoring guide and the example "
        "extension first, then follow their conventions exactly (entry point name, AgentTool shape, "
        "the .pi/extensions/ directory convention) rather than guessing the API."
    )

    if append_section:
        prompt += append_section

    prompt += _format_soul_for_prompt(options.soul)
    prompt += _format_memories_for_prompt(options.memories)
    prompt += _memory_policy_block(options.memory_enabled)

    if context_files:
        prompt += "\n\n<project_context>\n\n"
        prompt += "Project-specific instructions and guidelines:\n\n"
        for cf in context_files:
            prompt += f'<project_instructions path="{cf["path"]}">\n{cf["content"]}\n</project_instructions>\n\n'
        prompt += "</project_context>\n"

    if has_read and skills:
        prompt += _format_skills_for_prompt(skills)

    prompt += f"\nCurrent working directory: {prompt_cwd}"
    return prompt


_MEMORY_POLICY = (
    "\n\n<memory_policy>\n"
    "You have access to a persistent memory store via the `remember` and `recall` tools. "
    "Whenever you notice a technical/architectural decision or a stated style preference from "
    "the user, call `remember` (type=decision or type=style) proactively, without waiting for "
    "confirmation. Relevant memories from past sessions are already provided above in "
    "<memories> when applicable.\n"
    "If `remember` reports a POSSIBLE_DUPLICATE, do not silently pick an outcome: proactively "
    "tell the user which existing memory it resembles, what the two have in common, and the "
    "subtle difference you noticed, then propose a merged version. Let the user accept your "
    "merge, dictate their own merged wording, or decline and keep both as separate entries — "
    "then call `remember` again once with the outcome they chose.\n"
    "</memory_policy>\n"
)


_SOUL_HEADER = (
    "\n\n<soul>\n"
    "These are permanent, cross-session principles — your default behavior, "
    "listed below. They take precedence over anything in <memories> (which is "
    "episodic/contextual, not permanent).\n"
    "They do NOT override a specific, valid instruction the user gives you this "
    "turn, nor any rule from your system instructions — those come first. If the "
    "user's request for this turn conflicts with a principle below, that is "
    "normally just a one-turn deviation: follow the turn's instruction for this "
    "response only, and do not change or re-confirm the principle. Only treat it "
    "as a change to the principle itself when the user's language signals "
    "permanence (e.g. \"from now on\", \"always\", \"I want this to be a "
    "permanent rule\") — in that case, propose the update via `remember` "
    "(type=soul), which will ask the user to confirm before anything is saved.\n"
)


def _format_soul_for_prompt(soul: list[str] | None) -> str:
    if not soul:
        return ""
    lines = "\n".join(f"- {s}" for s in soul)
    return f"{_SOUL_HEADER}{lines}\n</soul>\n"


def _format_memories_for_prompt(memories: list[str] | None) -> str:
    if not memories:
        return ""
    lines = "\n".join(f"- {m}" for m in memories)
    return f"\n\n<memories>\n{lines}\n</memories>\n"


def _memory_policy_block(enabled: bool) -> str:
    return _MEMORY_POLICY if enabled else ""


def _format_skills_for_prompt(skills: list[Any]) -> str:
    """Format skills for inclusion in the system prompt."""
    if not skills:
        return ""
    sections = ["\n\n<skills>\n"]
    for skill in skills:
        name = getattr(skill, "name", str(skill))
        description = getattr(skill, "description", "")
        sections.append(f"- {name}: {description}")
    sections.append("</skills>")
    return "\n".join(sections)
