"""System prompt construction and project context loading.

Mirrors packages/coding-agent/src/core/system-prompt.ts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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

    prompt = (
        "You are an expert coding assistant operating inside pi, a coding agent harness."
        " You help users by reading files, executing commands, editing code,"
        " and writing new files.\n\n"
        f"Available tools:\n{tools_list}\n\n"
        "In addition to the tools above, you may have access to other custom tools"
        " depending on the project.\n\n"
        f"Guidelines:\n{guidelines}"
    )

    if append_section:
        prompt += append_section

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
