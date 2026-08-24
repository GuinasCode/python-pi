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
from pi_memory.store import MemoryStore, MemoryType, SecretDetectedError

_MEMORY_TYPES = [t.value for t in MemoryType]


def _ok(text: str) -> AgentToolResult:
    return AgentToolResult(content=[TextContent(text=text)])


def create_memory_tools(store: MemoryStore, *, interactive: bool = True) -> list[AgentTool]:
    """Return ``[remember, recall]`` AgentTools bound to *store*.

    *interactive* mirrors ``AgentSessionOptions.interactive``: when False
    (print mode, subagent runs — no human present to confirm anything),
    writes of type=soul are refused outright. There is no permission gate
    to route a confirmation through in that context, so a permanent
    principle must never be written unattended.
    """

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

        if memory_type is MemoryType.SOUL and not interactive:
            return _ok(
                "Refused: type=soul requires an interactive session where the user can be asked "
                "to confirm — not available in this (non-interactive) context."
            )

        merge_id = args.get("merge_id")
        force = bool(args.get("force", False))
        loop = asyncio.get_running_loop()

        if merge_id is not None:
            try:
                record = await loop.run_in_executor(
                    None, lambda: store.update(int(merge_id), title=title, content=content)
                )
            except SecretDetectedError as exc:
                return _ok(str(exc))
            if record is None:
                return _ok(f"No memory with id {merge_id} found to merge into.")
            return _ok(f"Merged into [{record.type.value}] #{record.id} {record.title}")

        if not force:
            candidate = await loop.run_in_executor(None, lambda: store.find_similar(title, content))
            if candidate is not None:
                existing, distance = candidate
                return _ok(
                    "POSSIBLE_DUPLICATE — do not save yet.\n"
                    f"An existing memory looks similar (distance {distance:.3f}, lower = more similar):\n"
                    f"  existing #{existing.id} [{existing.type.value}] {existing.title}: {existing.content}\n"
                    f"  new (unsaved) [{memory_type.value}] {title}: {content}\n\n"
                    "Tell the user what the two have in common and what's subtly different between them, "
                    "propose how a merged version could read, and ask how they'd like to proceed:\n"
                    f"  - merge: call remember again with merge_id={existing.id} and the agreed merged "
                    "title/content (overwrites the existing entry)\n"
                    "  - keep both as separate entries: call remember again with force=true\n"
                    "  - save the new entry as edited by the user: call remember again with force=true "
                    "and the corrected title/content"
                )

        try:
            record = await loop.run_in_executor(
                None,
                lambda: store.write(type=memory_type, title=title, content=content, source="manual"),
            )
        except SecretDetectedError as exc:
            return _ok(str(exc))
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
            "waiting for the user to ask.\n"
            "type=soul is different from every other type: it is a PERMANENT principle that "
            "applies to every future session, not an episodic fact. Only use type=soul when the "
            "user explicitly asks you to remember something as a lasting principle/rule (e.g. "
            "'remember this permanently', 'from now on always...') — never proactively, and never "
            "just because something sounds important. In an interactive session this call will be "
            "routed to the user for a real yes/no confirmation before anything is written; if the "
            "user declines, do not retry with the same content.\n"
            "By default this checks for a likely-duplicate existing memory first: if one is found, "
            "nothing is saved and the result asks you to summarize the overlap/difference to the "
            "user and get their call before calling remember again. Pass force=true to save anyway "
            "(new separate entry, or an edited version), or merge_id=<id> to overwrite an existing "
            "memory with merged title/content instead of creating a new one."
        ),
        parameters={
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": _MEMORY_TYPES,
                    "description": (
                        "Category of memory. 'soul' is a permanent, cross-session principle — "
                        "only use it on an explicit, permanence-signaling user request; see the "
                        "tool description."
                    ),
                },
                "title": {"type": "string", "description": "Short title summarizing the memory"},
                "content": {"type": "string", "description": "The fact or preference to remember"},
                "merge_id": {
                    "type": "integer",
                    "description": (
                        "If set, overwrite the existing memory with this id using title/content "
                        "instead of creating a new entry — used after the user agrees to merge a "
                        "reported duplicate."
                    ),
                },
                "force": {
                    "type": "boolean",
                    "description": "Skip the duplicate check and always save as a new entry.",
                    "default": False,
                },
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
