"""Tests for pi_coding_agent.extensions.loader."""

from __future__ import annotations

from pathlib import Path

from pi_agent_core.types import AgentTool
from pi_coding_agent.extensions.loader import (
    discover_extension_paths,
    load_extension_from_path,
    load_extensions,
)
from pi_coding_agent.extensions.types import ExtensionAPI


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


_HELLO_EXTENSION = """
from pi_agent_core.types import AgentTool

def extension(pi):
    pi.register_tool(AgentTool(name="hello", description="Greets someone", parameters={}))
"""

_PI_EXTENSION_VAR = """
from pi_agent_core.types import AgentTool

def _factory(pi):
    pi.register_tool(AgentTool(name="via_var", description="d", parameters={}))

PI_EXTENSION = _factory
"""

_BROKEN_EXTENSION = "raise RuntimeError('boom during import')\n"

_NO_ENTRY_POINT = "x = 1\n"

_ASYNC_FACTORY = """
async def extension(pi):
    pass
"""


class TestDiscoverExtensionPaths:
    def test_finds_loose_py_files_in_project_dir(self, tmp_path: Path) -> None:
        cwd = tmp_path / "project"
        agent_dir = tmp_path / "agent"
        _write(cwd / ".pi" / "extensions" / "hello.py", _HELLO_EXTENSION)

        paths = discover_extension_paths(cwd, agent_dir)
        assert [p.name for p in paths] == ["hello.py"]

    def test_ignores_underscore_prefixed_files(self, tmp_path: Path) -> None:
        cwd = tmp_path / "project"
        agent_dir = tmp_path / "agent"
        _write(cwd / ".pi" / "extensions" / "_helper.py", "x = 1\n")

        assert discover_extension_paths(cwd, agent_dir) == []

    def test_project_and_global_both_discovered(self, tmp_path: Path) -> None:
        cwd = tmp_path / "project"
        agent_dir = tmp_path / "agent"
        _write(cwd / ".pi" / "extensions" / "local.py", _HELLO_EXTENSION)
        _write(agent_dir / "extensions" / "global_ext.py", _HELLO_EXTENSION)

        paths = discover_extension_paths(cwd, agent_dir)
        assert {p.name for p in paths} == {"local.py", "global_ext.py"}

    def test_package_dir_resolved_via_manifest(self, tmp_path: Path) -> None:
        cwd = tmp_path / "project"
        agent_dir = tmp_path / "agent"
        pkg = cwd / ".pi" / "extensions" / "mypkg"
        _write(pkg / "manifest_target.py", _HELLO_EXTENSION)
        _write(pkg / "pi_manifest.json", '{"extensions": ["manifest_target.py"]}')

        paths = discover_extension_paths(cwd, agent_dir)
        assert [p.name for p in paths] == ["manifest_target.py"]

    def test_package_dir_resolved_via_init_py_fallback(self, tmp_path: Path) -> None:
        cwd = tmp_path / "project"
        agent_dir = tmp_path / "agent"
        pkg = cwd / ".pi" / "extensions" / "mypkg"
        _write(pkg / "__init__.py", _HELLO_EXTENSION)

        paths = discover_extension_paths(cwd, agent_dir)
        assert [p.name for p in paths] == ["__init__.py"]

    def test_configured_path_file_is_included(self, tmp_path: Path) -> None:
        cwd = tmp_path / "project"
        agent_dir = tmp_path / "agent"
        extra = _write(tmp_path / "extra_ext.py", _HELLO_EXTENSION)

        paths = discover_extension_paths(cwd, agent_dir, configured_paths=[str(extra)])
        assert paths == [extra]

    def test_duplicate_paths_are_deduplicated(self, tmp_path: Path) -> None:
        cwd = tmp_path / "project"
        agent_dir = tmp_path / "agent"
        local = _write(cwd / ".pi" / "extensions" / "hello.py", _HELLO_EXTENSION)

        paths = discover_extension_paths(cwd, agent_dir, configured_paths=[str(local)])
        assert paths == [local]

    def test_no_extensions_dirs_returns_empty(self, tmp_path: Path) -> None:
        cwd = tmp_path / "project"
        agent_dir = tmp_path / "agent"
        assert discover_extension_paths(cwd, agent_dir) == []


class TestLoadExtensionFromPath:
    def test_loads_and_registers_tool(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "hello.py", _HELLO_EXTENSION)
        api = ExtensionAPI()
        error = load_extension_from_path(path, api)
        assert error is None
        assert len(api.tools) == 1
        assert isinstance(api.tools[0], AgentTool)
        assert api.tools[0].name == "hello"

    def test_pi_extension_variable_entry_point(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "var_ext.py", _PI_EXTENSION_VAR)
        api = ExtensionAPI()
        error = load_extension_from_path(path, api)
        assert error is None
        assert api.tools[0].name == "via_var"

    def test_broken_extension_returns_error_not_raise(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "broken.py", _BROKEN_EXTENSION)
        api = ExtensionAPI()
        error = load_extension_from_path(path, api)
        assert error is not None
        assert "boom during import" in error.error
        assert api.tools == []

    def test_missing_entry_point_returns_error(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "no_entry.py", _NO_ENTRY_POINT)
        api = ExtensionAPI()
        error = load_extension_from_path(path, api)
        assert error is not None
        assert "no extension entry point" in error.error

    def test_async_factory_returns_error(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "async_ext.py", _ASYNC_FACTORY)
        api = ExtensionAPI()
        error = load_extension_from_path(path, api)
        assert error is not None
        assert "async extension factories are not supported" in error.error

    def test_missing_file_returns_error(self, tmp_path: Path) -> None:
        error = load_extension_from_path(tmp_path / "nope.py", ExtensionAPI())
        assert error is not None
        assert "not found" in error.error


class TestLoadExtensions:
    def test_aggregates_successes_and_errors(self, tmp_path: Path) -> None:
        good = _write(tmp_path / "good.py", _HELLO_EXTENSION)
        bad = _write(tmp_path / "bad.py", _BROKEN_EXTENSION)

        result = load_extensions([good, bad])
        assert len(result.extensions) == 1
        assert result.extensions[0].path == str(good)
        assert result.extensions[0].tool_names == ["hello"]
        assert len(result.errors) == 1
        assert result.errors[0].path == str(bad)

    def test_empty_path_list_yields_empty_result(self) -> None:
        result = load_extensions([])
        assert result.extensions == []
        assert result.errors == []
