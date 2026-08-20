# Writing a Pi extension

An extension is a plain Python file that registers custom tools (and, in
future versions of this SDK, hooks into agent lifecycle events). Pi loads
extensions from three places, in this order:

1. `<project>/.pi/extensions/` — project-local, checked into the repo
2. `~/.pi/extensions/` — global, available in every project
3. Any path explicitly configured for the session

Each extension is either a single `.py` file, or a directory containing
either a `pi_manifest.json` (`{"extensions": ["relative/path.py", ...]}`)
or an `extension.py`/`__init__.py` entry point.

## Minimal example

```python
# .pi/extensions/hello.py
from pi_agent_core.types import AgentTool, AgentToolResult
from pi_ai import TextContent


async def _hello(_tool_call_id, args, _context, _on_update):
    name = args.get("name", "")
    return AgentToolResult(content=[TextContent(text=f"Hello, {name}!")])


def extension(pi):
    pi.register_tool(
        AgentTool(
            name="hello",
            description="Greets someone by name.",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            label="Hello",
            execute=_hello,
        )
    )
```

The entry point is a top-level callable named `extension` (or a callable
assigned to `PI_EXTENSION`), taking one argument — the `pi` API object —
and registering things on it. Nothing is returned.

## The `pi` object

`pi.register_tool(tool: AgentTool)` is the only registration call in this
version of the SDK. `AgentTool` (from `pi_agent_core.types`) is the same
type Pi's own built-in tools use: `name`, `description`, a JSON Schema
`parameters` dict, a `label`, and an async `execute(tool_call_id, args,
context, on_update) -> AgentToolResult` callback.

## Constraints

- **The extension's own entry point must be synchronous** (`def
  extension(pi)`, not `async def`). If you need to do async setup (e.g.
  fetch a remote config), do it lazily inside your tool's `execute()` —
  that callback is already async.
- A broken extension (import error, missing entry point, an exception
  raised during registration) is recorded as a load error and skipped —
  it never crashes the session. Ask the user to check their extension
  file if a tool you expected to exist isn't showing up.
- Extensions are re-scanned whenever the session reloads (e.g. after you
  create or edit one mid-conversation), so a newly written extension's
  tools become available without restarting.

## Examples

See `examples/extensions/hello.py` (resolved from Examples in this
documentation section) for a complete, runnable version of the extension
above.
