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
from pi_coding_agent.extensions.types import (
    ExtensionAPI,
    ExtensionError,
    ExtensionFactory,
    LoadedExtension,
    LoadExtensionsResult,
)

__all__ = [
    "ExtensionAPI",
    "ExtensionError",
    "ExtensionFactory",
    "LoadExtensionsResult",
    "LoadedExtension",
    "discover_extension_paths",
    "load_extension_from_path",
    "load_extensions",
]
