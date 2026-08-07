"""Load agent definitions from Markdown files with YAML frontmatter.

Agent files live at:
  ~/.pi/agents/<name>.md       user-level — trusted by default
  .pi/agents/<name>.md         project-level — requires confirmation

Format::

    ---
    name: scout
    description: Fast codebase recon
    tools: read, grep, find, ls, bash
    model: nvidia/glm-5.2
    ---
    System prompt goes here...

The ``name`` field defaults to the filename stem when absent.
The ``tools`` field accepts a comma-separated string or YAML list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AgentDef:
    """A single agent definition loaded from a Markdown file."""

    name: str
    description: str = ""
    tools: list[str] = field(default_factory=list)
    model: str | None = None
    system_prompt: str = ""
    source_path: str = ""
    trust_level: str = "user"  # "user" | "project"


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split YAML frontmatter from body.  Returns (meta, body)."""
    import yaml

    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return {}, text

    end: int | None = None
    for i, line in enumerate(lines[1:], 1):
        if line.strip() == "---":
            end = i
            break

    if end is None:
        return {}, text

    fm_text = "".join(lines[1:end])
    body = "".join(lines[end + 1:])

    try:
        meta = yaml.safe_load(fm_text) or {}
    except Exception:
        meta = {}

    return meta, body.strip()


def _load_from_file(path: Path, trust_level: str) -> AgentDef | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None

    meta, body = _parse_frontmatter(text)

    name = meta.get("name", path.stem)
    tools_raw = meta.get("tools", "")
    if isinstance(tools_raw, str):
        tools = [t.strip() for t in tools_raw.split(",") if t.strip()]
    elif isinstance(tools_raw, list):
        tools = [str(t).strip() for t in tools_raw]
    else:
        tools = []

    return AgentDef(
        name=str(name),
        description=str(meta.get("description", "")),
        tools=tools,
        model=meta.get("model"),
        system_prompt=body,
        source_path=str(path),
        trust_level=trust_level,
    )


def discover_agents(
    cwd: str | Path,
    config_dir: str | Path | None = None,
) -> dict[str, AgentDef]:
    """Discover agent definitions from project and user directories.

    Project-level agents (in ``.pi/agents/``) require confirmation before
    running in non-trusted contexts.  User-level agents (in
    ``~/.pi/agents/``) are trusted by default.
    """
    agents: dict[str, AgentDef] = {}

    # Project-level (discovered first so user-level can override)
    project_dir = Path(cwd) / ".pi" / "agents"
    if project_dir.is_dir():
        for md in sorted(project_dir.glob("*.md")):
            agent = _load_from_file(md, "project")
            if agent:
                agents[agent.name] = agent

    # User-level (takes precedence over project-level)
    if config_dir is not None:
        user_dir = Path(config_dir) / "agents"
        if user_dir.is_dir():
            for md in sorted(user_dir.glob("*.md")):
                agent = _load_from_file(md, "user")
                if agent:
                    agents[agent.name] = agent

    return agents


__all__ = ["AgentDef", "discover_agents"]
