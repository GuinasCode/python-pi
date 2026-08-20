"""ExtensionRunner: loads and holds every extension for one session.

Phase B slice of packages/coding-agent/src/core/extensions/runner.ts —
tool registration and error reporting only. The event bus, command/
shortcut/flag registries, and provider registration that the full
``ExtensionRunner`` also owns are added by later phases directly onto
this class.
"""

from __future__ import annotations

from pathlib import Path

from pi_agent_core.types import AgentTool
from pi_coding_agent.extensions.loader import discover_extension_paths, load_extensions
from pi_coding_agent.extensions.types import LoadExtensionsResult

__all__ = ["ExtensionRunner"]


class ExtensionRunner:
    """Discovers, loads, and holds every extension for a session's cwd."""

    def __init__(
        self,
        cwd: str | Path,
        agent_dir: str | Path | None = None,
        configured_paths: list[str] | None = None,
    ) -> None:
        self._cwd = cwd
        self._agent_dir = agent_dir
        self._configured_paths = configured_paths or []
        self._result = LoadExtensionsResult()

    def load(self) -> LoadExtensionsResult:
        """(Re)discover and (re)load every extension, replacing any
        previous result. Safe to call again (e.g. from a session reload)
        after project files on disk have changed."""
        paths = discover_extension_paths(self._cwd, self._agent_dir, self._configured_paths)
        self._result = load_extensions(paths)
        return self._result

    def get_extensions(self) -> LoadExtensionsResult:
        """The result of the most recent load() — empty (no extensions,
        no errors) if load() hasn't been called yet."""
        return self._result

    def get_tools(self) -> list[AgentTool]:
        """Every tool registered by every successfully loaded extension."""
        return [tool for ext in self._result.extensions for tool in ext.tools]

    def get_extension_paths(self) -> list[str]:
        """Source paths of every successfully loaded extension."""
        return [ext.path for ext in self._result.extensions]
