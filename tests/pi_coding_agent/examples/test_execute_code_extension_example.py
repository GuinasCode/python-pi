"""Slice A6 — the official example extension exposing execute_code
through the real extension mechanism (pi.register_tool), matching the
already-established pattern of pi_coding_agent/examples/extensions/hello.py.
"""

from __future__ import annotations

import asyncio

from pi_coding_agent.examples.extensions.execute_code import extension
from pi_coding_agent.extensions.types import ExtensionAPI


class TestExecuteCodeExtensionRegistration:
    def test_registers_a_tool_named_execute_code(self) -> None:
        pi = ExtensionAPI()
        extension(pi)
        assert "execute_code" in [t.name for t in pi.tools]

    def test_the_tool_requires_code_and_declares_mode_enum(self) -> None:
        pi = ExtensionAPI()
        extension(pi)
        tool = next(t for t in pi.tools if t.name == "execute_code")
        assert tool.parameters["required"] == ["code"]
        assert tool.parameters["properties"]["mode"]["enum"] == ["strict", "project"]


class TestExecuteCodeExtensionRunsRealCode:
    def test_calling_the_registered_tool_actually_runs_python(self) -> None:
        pi = ExtensionAPI()
        extension(pi)
        tool = next(t for t in pi.tools if t.name == "execute_code")

        result = asyncio.run(tool.execute("call-1", {"code": "print(1 + 1)"}, None, lambda _update: None))
        assert len(result.content) == 1
        text = result.content[0].text
        assert "status: success" in text
        assert "2" in text

    def test_rpc_allowlist_is_reachable_from_the_registered_tool(self, tmp_path: object) -> None:
        from pathlib import Path

        target = Path(str(tmp_path)) / "note.txt"
        target.write_text("hello from disk")

        pi = ExtensionAPI()
        extension(pi)
        tool = next(t for t in pi.tools if t.name == "execute_code")

        code = f"""
from pi_tools import read_file
print(read_file({str(target)!r}).strip().split("|")[-1].strip())
"""
        result = asyncio.run(tool.execute("call-2", {"code": code}, None, lambda _update: None))
        text = result.content[0].text
        assert "hello from disk" in text
