"""Default stream function registry, mirroring ``packages/agent/src/stream-fn.ts``."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import StreamFn

_default_stream_fn: "StreamFn | None" = None


def set_default_stream_fn(stream_fn: "StreamFn | None") -> None:
    """Configure the fallback used by Agent and low-level loops when callers omit stream_fn.

    Hosts that provide a default model runtime can install its stream function here
    without making pi_agent_core depend on a provider catalog or compatibility layer.
    """
    global _default_stream_fn
    _default_stream_fn = stream_fn


def get_default_stream_fn() -> "StreamFn":
    """Return the configured default stream function.

    Raises:
        RuntimeError: if no default stream function has been configured.
    """
    if _default_stream_fn is None:
        raise RuntimeError(
            "No default stream function configured. Pass stream_fn explicitly or call set_default_stream_fn()."
        )
    return _default_stream_fn


__all__ = ["get_default_stream_fn", "set_default_stream_fn"]
