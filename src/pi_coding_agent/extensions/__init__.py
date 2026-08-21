"""Extension system: lifecycle events and custom tools loaded from
``.pi/extensions/``. Python port of
``packages/coding-agent/src/core/extensions/``.
"""

from __future__ import annotations

from pi_coding_agent.extensions.loader import (
    discover_extension_paths,
    load_extension_from_path,
    load_extensions,
)
from pi_coding_agent.extensions.runner import ExtensionRunner
from pi_coding_agent.extensions.types import (
    EntryRenderer,
    ExtensionAPI,
    ExtensionError,
    ExtensionFactory,
    ExtensionFlag,
    LoadedExtension,
    LoadExtensionsResult,
    MarkdownTransformer,
    MessageRenderer,
    RegisteredCommand,
    RegisteredShortcut,
    RegisteredTheme,
)

__all__ = [
    "EntryRenderer",
    "ExtensionAPI",
    "ExtensionError",
    "ExtensionFactory",
    "ExtensionFlag",
    "ExtensionRunner",
    "LoadExtensionsResult",
    "LoadedExtension",
    "MarkdownTransformer",
    "MessageRenderer",
    "RegisteredCommand",
    "RegisteredShortcut",
    "RegisteredTheme",
    "discover_extension_paths",
    "load_extension_from_path",
    "load_extensions",
]
