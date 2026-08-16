"""AgentTool factory for manual memory access (``remember``/``recall``).

Mirrors the pattern used by ``pi_coding_agent.subagent.tool.create_subagent_tool``:
a stateful capability is wrapped as an :class:`AgentTool` with an async
``execute`` callback, rather than a stateless entry in the built-in
``tools.py`` dispatcher (which has no access to a ``MemoryStore`` instance).
"""

from __future__ import annotations

import asyncio
from typing import Any

from pi_agent_core.types import AgentTool, AgentToolResult
from pi_ai import TextContent
from pi_memory.store import MemoryStore, MemoryType

_MEMORY_TYPES = [t.value for t in MemoryType]


def _ok(text: str) -> AgentToolResult:
    return AgentToolResult(content=[TextContent(text=text)])


def create_memory_tools(store: MemoryStore) -> list[AgentTool]:
    """Return ``[remember, recall]`` AgentTools bound to *store*."""

    async def _remember(
        _tool_call_id: str,
        args: dict[str, Any],
        _context: Any,
        _on_update: Any,
    ) -> AgentToolResult:
        raw_type = args.get("type", "")
        try:
            memory_type = MemoryType(raw_type)
        except ValueError:
            return _ok(f"Invalid type {raw_type!r}. Must be one of: {', '.join(_MEMORY_TYPES)}")

        title = args.get("title", "")
        content = args.get("content", "")
        if not title or not content:
            return _ok("Both 'title' and 'content' are required.")

        loop = asyncio.get_running_loop()
        record = await loop.run_in_executor(
            None,
            lambda: store.write(type=memory_type, title=title, content=content, source="manual"),
        )
        return _ok(f"Remembered [{record.type.value}] {record.title}")

    async def _recall(
        _tool_call_id: str,
        args: dict[str, Any],
        _context: Any,
        _on_update: Any,
    ) -> AgentToolResult:
        query = args.get("query", "")
        if not query:
            return _ok("'query' is required.")
        top_k = int(args.get("top_k", 5))

        loop = asyncio.get_running_loop()
        records = await loop.run_in_executor(None, lambda: store.search(query, top_k=top_k))
        if not records:
            return _ok("No matching memories found.")
        lines = [f"[{r.type.value}] {r.title}: {r.content}" for r in records]
        return _ok("\n".join(lines))

    remember_tool = AgentTool(
        name="remember",
        description=(
            "Persist a fact, decision, or preference so it can be recalled in future sessions. "
            "Use type=decision for technical/architectural decisions and type=style for the "
            "user's stated stylistic preferences — these should be captured proactively, without "
            "waiting for the user to ask."
        ),
        parameters={
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": _MEMORY_TYPES, "description": "Category of memory"},
                "title": {"type": "string", "description": "Short title summarizing the memory"},
                "content": {"type": "string", "description": "The fact or preference to remember"},
            },
            "required": ["type", "title", "content"],
        },
        label="Remember",
        execute=_remember,
    )

    recall_tool = AgentTool(
        name="recall",
        description="Search persisted memories by free-text query and return the best matches.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for"},
                "top_k": {"type": "integer", "description": "Max results to return", "default": 5},
            },
            "required": ["query"],
        },
        label="Recall",
        execute=_recall,
    )

    return [remember_tool, recall_tool]
