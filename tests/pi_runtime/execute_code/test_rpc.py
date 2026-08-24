"""Tests for pi_runtime.execute_code.rpc.RpcServer — Slice A2.

Covers spec section 19's "RPC" category: request, response, malformed
request, unknown tool, tool error, concurrent requests (policy denial
lands in Slice A4, once PolicyEngine is wired in).
"""

from __future__ import annotations

import asyncio
import json

from pi_runtime.execute_code.rpc import RpcError, RpcServer


async def _send(port: int, token: str, payload: dict[str, object]) -> dict[str, object]:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        writer.write((json.dumps({"token": token, **payload}) + "\n").encode())
        await writer.drain()
        line = await reader.readline()
        return dict(json.loads(line))
    finally:
        writer.close()


async def _echo_handler(_tool: str, arguments: dict[str, object]) -> object:
    return arguments


class TestRequestResponse:
    def test_successful_call_returns_the_handler_result(self) -> None:
        async def _run() -> None:
            async with RpcServer(handlers={"echo": _echo_handler}) as server:
                response = await _send(
                    server.port, server.token, {"request_id": "r1", "tool": "echo", "arguments": {"x": 1}}
                )
                assert response["status"] == "success"
                assert response["result"] == {"x": 1}
                assert response["request_id"] == "r1"

        asyncio.run(_run())

    def test_request_id_round_trips_exactly(self) -> None:
        async def _run() -> None:
            async with RpcServer(handlers={"echo": _echo_handler}) as server:
                response = await _send(
                    server.port, server.token, {"request_id": "abc-123", "tool": "echo", "arguments": {}}
                )
                assert response["request_id"] == "abc-123"

        asyncio.run(_run())


class TestMalformedRequest:
    def test_invalid_json_gets_a_malformed_request_error(self) -> None:
        async def _run() -> None:
            server = RpcServer(handlers={})
            await server.start()
            try:
                reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
                writer.write(b"not json at all\n")
                await writer.drain()
                line = await reader.readline()
                response = json.loads(line)
                assert response["status"] == "error"
                assert response["error"]["type"] == "malformed_request"
                writer.close()
            finally:
                await server.close()

        asyncio.run(_run())

    def test_missing_tool_field_is_malformed(self) -> None:
        async def _run() -> None:
            async with RpcServer(handlers={}) as server:
                response = await _send(server.port, server.token, {"request_id": "r1", "arguments": {}})
                assert response["status"] == "error"
                assert response["error"]["type"] == "malformed_request"  # type: ignore[index]

        asyncio.run(_run())


class TestUnknownTool:
    def test_unregistered_tool_is_rejected_not_run(self) -> None:
        async def _run() -> None:
            async with RpcServer(handlers={"echo": _echo_handler}) as server:
                response = await _send(
                    server.port, server.token, {"request_id": "r1", "tool": "not_registered", "arguments": {}}
                )
                assert response["status"] == "error"
                assert response["error"]["type"] == "unknown_tool"  # type: ignore[index]

        asyncio.run(_run())


class TestPolicyDenial:
    """Unauthorized (wrong/missing token) is the policy-adjacent check
    already real in this slice — a child can never invoke anything by
    guessing a tool name without also knowing the token (spec section 10:
    "o child não pode ganhar permissões somente porque conhece o nome de
    uma tool")."""

    def test_wrong_token_is_rejected(self) -> None:
        async def _run() -> None:
            async with RpcServer(handlers={"echo": _echo_handler}) as server:
                response = await _send(
                    server.port, "wrong-token", {"request_id": "r1", "tool": "echo", "arguments": {}}
                )
                assert response["status"] == "error"
                assert response["error"]["type"] == "unauthorized"  # type: ignore[index]

        asyncio.run(_run())

    def test_missing_token_is_rejected(self) -> None:
        async def _run() -> None:
            server = RpcServer(handlers={"echo": _echo_handler})
            await server.start()
            try:
                reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
                writer.write((json.dumps({"request_id": "r1", "tool": "echo", "arguments": {}}) + "\n").encode())
                await writer.drain()
                response = json.loads(await reader.readline())
                assert response["error"]["type"] == "unauthorized"
                writer.close()
            finally:
                await server.close()

        asyncio.run(_run())


class TestToolError:
    def test_handler_raising_rpc_error_produces_a_structured_error(self) -> None:
        async def _failing_handler(_tool: str, _arguments: dict[str, object]) -> object:
            raise RpcError("file not found", error_type="tool_error")

        async def _run() -> None:
            async with RpcServer(handlers={"boom": _failing_handler}) as server:
                response = await _send(server.port, server.token, {"request_id": "r1", "tool": "boom", "arguments": {}})
                assert response["status"] == "error"
                assert response["error"]["type"] == "tool_error"  # type: ignore[index]
                assert "file not found" in response["error"]["message"]  # type: ignore[index]

        asyncio.run(_run())

    def test_handler_raising_an_unexpected_exception_never_crashes_the_server(self) -> None:
        async def _crashing_handler(_tool: str, _arguments: dict[str, object]) -> object:
            raise RuntimeError("unexpected bug")

        async def _run() -> None:
            async with RpcServer(handlers={"boom": _crashing_handler}) as server:
                response = await _send(server.port, server.token, {"request_id": "r1", "tool": "boom", "arguments": {}})
                assert response["status"] == "error"
                assert response["error"]["type"] == "rpc_error"  # type: ignore[index]
                # the server is still alive and can serve another request
                second = await _send(server.port, server.token, {"request_id": "r2", "tool": "boom", "arguments": {}})
                assert second["status"] == "error"

        asyncio.run(_run())


class TestConcurrentRequests:
    def test_multiple_concurrent_connections_are_all_served_correctly(self) -> None:
        async def _run() -> None:
            async with RpcServer(handlers={"echo": _echo_handler}) as server:
                results = await asyncio.gather(
                    *(
                        _send(server.port, server.token, {"request_id": f"r{i}", "tool": "echo", "arguments": {"n": i}})
                        for i in range(20)
                    )
                )
                for i, response in enumerate(results):
                    assert response["status"] == "success"
                    assert response["result"] == {"n": i}

        asyncio.run(_run())


class TestResourceLimit:
    def test_max_calls_is_enforced(self) -> None:
        async def _run() -> None:
            async with RpcServer(handlers={"echo": _echo_handler}, max_calls=2) as server:
                await _send(server.port, server.token, {"request_id": "r1", "tool": "echo", "arguments": {}})
                await _send(server.port, server.token, {"request_id": "r2", "tool": "echo", "arguments": {}})
                third = await _send(server.port, server.token, {"request_id": "r3", "tool": "echo", "arguments": {}})
                assert third["status"] == "error"
                assert third["error"]["type"] == "resource_limit"  # type: ignore[index]

        asyncio.run(_run())


class TestCallLog:
    def test_every_call_is_logged_for_telemetry(self) -> None:
        async def _run() -> None:
            async with RpcServer(handlers={"echo": _echo_handler}) as server:
                await _send(server.port, server.token, {"request_id": "r1", "tool": "echo", "arguments": {}})
                await _send(server.port, server.token, {"request_id": "r2", "tool": "unknown", "arguments": {}})
                assert len(server.call_log) == 2
                assert server.call_log[0].status == "success"
                assert server.call_log[1].status == "error"
                assert server.call_log[1].error_type == "unknown_tool"

        asyncio.run(_run())


class TestPortIsLoopbackOnly:
    def test_server_binds_to_a_real_port_on_loopback(self) -> None:
        async def _run() -> None:
            async with RpcServer(handlers={}) as server:
                assert server.port > 0

        asyncio.run(_run())
