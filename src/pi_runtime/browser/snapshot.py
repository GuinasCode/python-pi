"""Slice B2 — accessibility-tree snapshots and the ElementRef model.

Spec section 26: never return raw HTML. `page.locator("body").aria_snapshot()`
(Playwright's own accessibility-tree serialization) gives a real,
bounded, human-readable tree — e.g.:

    - heading "Fixture Home" [level=1]
    - textbox
    - combobox:
      - option "Red" [selected]
      - option "Blue"
    - button "Submit"

Each line here becomes one `SnapshotNode` with a stable ref (`e1`,
`e2`, ...). Spec section 27's ref model: instead of a selector the
model must invent, a ref resolves via the same role+accessible-name
strategy Playwright itself recommends for stable locators
(`page.get_by_role(role, name=...)`), disambiguated by occurrence index
when multiple elements share both role and name. A ref map is stored on
the session and replaced wholesale by the next snapshot — resolving a
ref against a stale map (or one that no longer matches any element,
because the DOM changed) raises `StaleRefError` explicitly rather than
silently clicking whatever the selector happens to match now (the
spec's invariant: "não clicar silenciosamente em outro elemento").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from playwright._impl._api_structures import AriaRole
    from playwright.async_api import Locator, Page

_LINE_RE = re.compile(
    r"^(?P<indent>\s*)-\s+(?P<role>[a-zA-Z][\w-]*)"
    r'(?:\s+"(?P<name>(?:[^"\\]|\\.)*)")?'
    r"\s*:?\s*(?:\[(?P<attrs>[^\]]*)\])?\s*$"
)

_MAX_NODES = 500
_MAX_TEXT_CHARS = 20_000


@dataclass
class SnapshotNode:
    ref: str
    role: str
    name: str | None
    attrs: str | None
    depth: int


@dataclass
class RefTarget:
    """What a ref actually resolves to: role+name (Playwright's
    recommended stable locator strategy) plus an occurrence index to
    disambiguate elements that share both."""

    role: str
    name: str | None
    nth: int
    page_id: str


@dataclass
class PageSnapshot:
    url: str
    title: str
    nodes: list[SnapshotNode] = field(default_factory=list)
    text: str = ""
    truncated: bool = False


class StaleRefError(Exception):
    """A ref that isn't in the session's current ref map (superseded by
    a later snapshot, or never existed) or that no longer resolves to
    any live element. Never silently falls back to clicking something
    else — spec section 27's core invariant."""


def _parse_aria_snapshot(raw: str) -> list[SnapshotNode]:
    nodes: list[SnapshotNode] = []
    counter = 0
    for line in raw.splitlines():
        if not line.strip():
            continue
        match = _LINE_RE.match(line)
        if match is None:
            continue
        counter += 1
        indent = match.group("indent") or ""
        nodes.append(
            SnapshotNode(
                ref=f"e{counter}",
                role=match.group("role"),
                name=match.group("name"),
                attrs=match.group("attrs"),
                depth=len(indent) // 2,
            )
        )
        if counter >= _MAX_NODES:
            break
    return nodes


def _render_text(nodes: list[SnapshotNode]) -> tuple[str, bool]:
    lines = []
    for node in nodes:
        indent = "  " * node.depth
        label = f"{indent}[{node.role} @{node.ref}]"
        if node.name:
            label += f" {node.name}"
        if node.attrs:
            label += f" ({node.attrs})"
        lines.append(label)
    text = "\n".join(lines)
    truncated = len(text) > _MAX_TEXT_CHARS
    if truncated:
        text = text[:_MAX_TEXT_CHARS] + "\n...(truncated)..."
    return text, truncated


async def capture_snapshot(page: Page, *, page_id: str) -> tuple[PageSnapshot, dict[str, RefTarget]]:
    """Returns the bounded snapshot plus the ref->target map the caller
    (BrowserSession) should store, replacing whatever map it had
    before — every snapshot invalidates every ref from the previous
    one, on purpose."""
    raw = await page.locator("body").aria_snapshot()
    nodes = _parse_aria_snapshot(raw)
    text, truncated = _render_text(nodes)

    ref_map: dict[str, RefTarget] = {}
    seen_role_name_count: dict[tuple[str, str | None], int] = {}
    for node in nodes:
        key = (node.role, node.name)
        nth = seen_role_name_count.get(key, 0)
        seen_role_name_count[key] = nth + 1
        ref_map[node.ref] = RefTarget(role=node.role, name=node.name, nth=nth, page_id=page_id)

    snapshot = PageSnapshot(url=page.url, title=await page.title(), nodes=nodes, text=text, truncated=truncated)
    return snapshot, ref_map


def resolve_locator(page: Page, target: RefTarget) -> Locator:
    # target.role comes from parsing Playwright's own aria_snapshot() output,
    # so at runtime it is always one of AriaRole's literal values — mypy
    # can't see that since the value only exists after a live snapshot.
    role = cast("AriaRole", target.role)
    locator = page.get_by_role(role, name=target.name, exact=True) if target.name else page.get_by_role(role)
    return locator.nth(target.nth)


__all__ = [
    "PageSnapshot",
    "RefTarget",
    "SnapshotNode",
    "StaleRefError",
    "capture_snapshot",
    "resolve_locator",
]
