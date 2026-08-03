"""Resource loader for Pi coding agent.

Simplified port of packages/coding-agent/src/core/resource-loader.ts.
Loads AGENTS.md, SYSTEM.md, skills from ~/.pi/ and project directories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ContextFile:
    """A loaded context file."""

    path: str
    content: str


@dataclass
class Skill:
    """A loaded skill."""

    name: str
    description: str = ""
    path: str = ""
    content: str = ""


@dataclass
class LoadedResources:
    """All loaded project resources."""

    context_files: list[ContextFile] = field(default_factory=list)
    system_prompt: str | None = None
    append_system_prompt: str | None = None
    skills: list[Skill] = field(default_factory=list)


def load_project_context_files(cwd: str | Path, max_depth: int = 10) -> list[ContextFile]:
    """Walk cwd and ancestors looking for AGENTS.md and CLAUDE.md files."""
    cwd_path = Path(cwd).resolve()
    context_files: list[ContextFile] = []

    context_names = ["AGENTS.md", "CLAUDE.md", ".cursorrules"]
    current = cwd_path
    depth = 0
    visited: set[Path] = set()

    while current.is_dir() and depth < max_depth and current not in visited:
        visited.add(current)
        for name in context_names:
            file_path = current / name
            if file_path.is_file():
                try:
                    content = file_path.read_text(encoding="utf-8")
                    context_files.append(ContextFile(path=str(file_path), content=content))
                except Exception:
                    pass
        parent = current.parent
        if parent == current:
            break
        current = parent
        depth += 1

    return context_files


def load_system_prompt(cwd: str | Path, config_dir: str | Path | None = None) -> str | None:
    """Find and load SYSTEM.md from config dir or project."""
    # Check ~/.pi/SYSTEM.md
    if config_dir is not None:
        system_file = Path(config_dir) / "SYSTEM.md"
        if system_file.is_file():
            try:
                return system_file.read_text(encoding="utf-8")
            except Exception:
                pass

    # Check project SYSTEM.md
    project_system = Path(cwd) / "SYSTEM.md"
    if project_system.is_file():
        try:
            return project_system.read_text(encoding="utf-8")
        except Exception:
            pass

    return None


def load_append_system_prompt(cwd: str | Path, config_dir: str | Path | None = None) -> str | None:
    """Find and load APPEND_SYSTEM.md."""
    if config_dir is not None:
        append_file = Path(config_dir) / "APPEND_SYSTEM.md"
        if append_file.is_file():
            try:
                return append_file.read_text(encoding="utf-8")
            except Exception:
                pass

    project_append = Path(cwd) / "APPEND_SYSTEM.md"
    if project_append.is_file():
        try:
            return project_append.read_text(encoding="utf-8")
        except Exception:
            pass

    return None


def load_skills(config_dir: str | Path) -> list[Skill]:
    """Discover and load skills from config_dir/skills/."""
    skills_dir = Path(config_dir) / "skills"
    if not skills_dir.is_dir():
        return []

    skills: list[Skill] = []
    for item in sorted(skills_dir.iterdir()):
        if not item.is_dir():
            continue
        skill_file = item / "SKILL.md"
        if not skill_file.is_file():
            continue
        try:
            content = skill_file.read_text(encoding="utf-8")
            name = item.name
            description = _extract_description(content)
            skills.append(
                Skill(
                    name=name,
                    description=description,
                    path=str(skill_file),
                    content=content,
                )
            )
        except Exception:
            pass

    return skills


def _extract_description(content: str) -> str:
    """Extract description from SKILL.md frontmatter or first paragraph."""
    lines = content.strip().splitlines()
    in_frontmatter = False
    for line in lines:
        stripped = line.strip()
        if stripped == "---":
            in_frontmatter = not in_frontmatter
            continue
        if in_frontmatter and stripped.startswith("description:"):
            return stripped[len("description:") :].strip().strip('"').strip("'")
    # Fall back to first non-empty, non-heading line
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and not stripped.startswith("---"):
            return stripped[:200]
    return ""


def load_resources(
    cwd: str | Path,
    config_dir: str | Path | None = None,
) -> LoadedResources:
    """Load all project resources: context files, system prompt, skills."""
    context_files = load_project_context_files(cwd)
    system_prompt = load_system_prompt(cwd, config_dir)
    append_system_prompt = load_append_system_prompt(cwd, config_dir)
    skills: list[Skill] = []
    if config_dir is not None:
        skills = load_skills(config_dir)

    return LoadedResources(
        context_files=context_files,
        system_prompt=system_prompt,
        append_system_prompt=append_system_prompt,
        skills=skills,
    )
