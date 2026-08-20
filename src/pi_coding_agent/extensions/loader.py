"""Extension discovery and dynamic loading.

Python port of packages/coding-agent/src/core/extensions/loader.ts. The
original uses ``jiti`` to transpile/load ``.ts`` files at runtime and
injects "virtual modules" so extensions can import bundled Pi packages
without them being separately installed. Neither concern applies here:
Python extensions are plain ``.py`` files loaded via ``importlib``, and
they import ``pi_ai``/``pi_agent_core``/``pi_coding_agent`` exactly like
any other code in this venv — no transpilation or virtual-module
machinery needed. That's a real scope reduction versus the original, not
a corner cut.

Discovery mirrors ``discoverAndLoadExtensions``'s three tiers: project
(``<cwd>/.pi/extensions/``), global (``<agent_dir>/extensions/``), then
explicitly configured paths. A directory entry can be a single ``.py``
file, or a package directory resolved via ``pi_manifest.json`` (a
``{"extensions": [...]}`` list of relative paths) or an ``extension.py``/
``__init__.py`` entry point, mirroring the original's ``package.json``
"pi" field / ``index.ts`` fallback.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from pi_coding_agent.config import CONFIG_DIR_NAME, get_config_dir
from pi_coding_agent.extensions.types import (
    ExtensionAPI,
    ExtensionError,
    LoadedExtension,
    LoadExtensionsResult,
)

__all__ = [
    "discover_extension_paths",
    "load_extension_from_path",
    "load_extensions",
]

_MANIFEST_NAME = "pi_manifest.json"
_ENTRY_POINT_NAMES = ("extension.py", "__init__.py")


def _is_extension_file(name: str) -> bool:
    return name.endswith(".py") and not name.startswith("_")


def _resolve_extension_entries(directory: Path) -> list[Path] | None:
    """Resolve a subdirectory into its extension entry file(s), or None if
    it isn't recognizable as an extension package (caller then falls back
    to discovering loose .py files inside it directly)."""
    manifest_path = directory / _MANIFEST_NAME
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = None
        entries = manifest.get("extensions") if isinstance(manifest, dict) else None
        if entries:
            resolved = [directory / str(e) for e in entries if (directory / str(e)).is_file()]
            if resolved:
                return resolved

    for name in _ENTRY_POINT_NAMES:
        candidate = directory / name
        if candidate.is_file():
            return [candidate]
    return None


def _discover_in_dir(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    discovered: list[Path] = []
    try:
        entries = sorted(directory.iterdir())
    except OSError:
        return []
    for entry in entries:
        if entry.is_file() and _is_extension_file(entry.name):
            discovered.append(entry)
        elif entry.is_dir():
            resolved = _resolve_extension_entries(entry)
            if resolved:
                discovered.extend(resolved)
    return discovered


def discover_extension_paths(
    cwd: str | Path,
    agent_dir: str | Path | None = None,
    configured_paths: list[str] | None = None,
) -> list[Path]:
    """Find every extension entry file, layered project -> global -> explicit.

    Later duplicates (by resolved absolute path) are dropped, keeping the
    first occurrence's position — project-local extensions win priority
    over a same-path global/explicit entry.
    """
    resolved_cwd = Path(cwd).resolve()
    resolved_agent_dir = Path(agent_dir).resolve() if agent_dir else get_config_dir()

    seen: set[Path] = set()
    ordered: list[Path] = []

    def _add(paths: list[Path]) -> None:
        for p in paths:
            resolved = p.resolve()
            if resolved not in seen:
                seen.add(resolved)
                ordered.append(p)

    _add(_discover_in_dir(resolved_cwd / CONFIG_DIR_NAME / "extensions"))
    _add(_discover_in_dir(resolved_agent_dir / "extensions"))

    for raw_path in configured_paths or []:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = resolved_cwd / candidate
        if candidate.is_dir():
            resolved_entries = _resolve_extension_entries(candidate)
            _add(resolved_entries if resolved_entries is not None else _discover_in_dir(candidate))
        else:
            _add([candidate])

    return ordered


def _resolve_factory(module: Any) -> Any:
    entry = getattr(module, "PI_EXTENSION", None)
    if callable(entry):
        return entry
    entry = getattr(module, "extension", None)
    if callable(entry):
        return entry
    return None


def load_extension_from_path(path: Path, api: ExtensionAPI) -> ExtensionError | None:
    """Import one extension file and call its entry point with `api`.

    Returns an :class:`ExtensionError` instead of raising — a broken
    extension must never take down the whole session. The imported module
    is removed from ``sys.modules`` afterward; each extension gets a fresh,
    isolated module object rather than sharing a name with any other.
    """
    if not path.is_file():
        return ExtensionError(path=str(path), error="extension file not found")

    module_name = f"pi_extension_{uuid.uuid4().hex}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            return ExtensionError(path=str(path), error="could not create a module spec for this file")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        factory = _resolve_factory(module)
        if factory is None:
            return ExtensionError(
                path=str(path),
                error="no extension entry point found (expected a callable `extension(pi)` or `PI_EXTENSION`)",
            )

        result = factory(api)
        if inspect.isawaitable(result):
            close = getattr(result, "close", None)
            if callable(close):
                close()  # avoid a "coroutine was never awaited" warning — we're deliberately not running it
            return ExtensionError(
                path=str(path),
                error="async extension factories are not supported — do async setup lazily inside a tool's execute()",
            )
        return None
    except Exception as exc:
        return ExtensionError(path=str(path), error=str(exc))
    finally:
        sys.modules.pop(module_name, None)


def load_extensions(paths: list[Path]) -> LoadExtensionsResult:
    """Load every path, aggregating registered tools and any errors."""
    result = LoadExtensionsResult()
    for path in paths:
        api = ExtensionAPI()
        error = load_extension_from_path(path, api)
        if error is not None:
            result.errors.append(error)
            continue
        result.extensions.append(LoadedExtension(path=str(path), tools=api.tools, handlers=api.handlers))
    return result
