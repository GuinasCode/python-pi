from __future__ import annotations

import pytest

from pi_protocol import (
    PROTOCOL_VERSION,
    ClientMessageDecoder,
    FrameDecoder,
    ProtocolValidationError,
    ServerMessageDecoder,
    decode_cbor,
    encode_cbor,
    encode_client_message,
    encode_frame,
    encode_server_message,
    is_supported_protocol_version,
    parse_client_message,
    parse_server_message,
)

EMPTY_SERVER_SNAPSHOT = {
    "serverId": "server-1",
    "protocolVersion": PROTOCOL_VERSION,
    "revision": 0,
    "sessions": [],
    "models": [],
}

CLIENT_HELLO = {"type": "hello", "version": PROTOCOL_VERSION, "token": "secret"}
SERVER_HELLO = {
    "type": "hello",
    "version": PROTOCOL_VERSION,
    "connectionId": "connection-1",
    "snapshot": EMPTY_SERVER_SNAPSHOT,
}


def item_message(item: object, type_: str = "item_finished") -> dict[str, object]:
    return {
        "type": "event",
        "event": {
            "type": "session_progress",
            "sessionId": "session-1",
            "progress": {"type": type_, "item": item},
        },
    }


def test_protocol_version() -> None:
    assert PROTOCOL_VERSION == 2
    assert is_supported_protocol_version(2)
    assert not is_supported_protocol_version(1)
    assert not is_supported_protocol_version(2.5)


@pytest.mark.parametrize("version", [0, 1, PROTOCOL_VERSION])
def test_accepts_integer_client_hello_version_for_negotiation(version: int) -> None:
    message = {**CLIENT_HELLO, "version": version}
    assert parse_client_message(message) == message


@pytest.mark.parametrize(
    "message",
    [
        {"type": "hello", "version": "2", "token": "secret"},
        {"type": "hello", "version": 2.5, "token": "secret"},
        {"type": "hello", "version": 2, "token": ""},
        {"type": "hello", "version": 2, "token": "secret", "extra": True},
    ],
)
def test_rejects_invalid_strict_handshake(message: object) -> None:
    with pytest.raises(ProtocolValidationError):
        parse_client_message(message)


def test_does_not_parse_json_strings_as_wire_messages() -> None:
    with pytest.raises(ProtocolValidationError):
        parse_client_message('{"type":"hello"}')
    with pytest.raises(ProtocolValidationError):
        parse_server_message('{"type":"hello"}')


def test_rejects_image_input_while_mvp_remains_text_only() -> None:
    with pytest.raises(ProtocolValidationError):
        parse_client_message(
            {
                "type": "request",
                "id": "request-1",
                "request": {
                    "command": "prompt",
                    "sessionId": "session-1",
                    "text": "inspect",
                    "images": [{"type": "image", "data": "abc", "mimeType": "image/png"}],
                },
            }
        )


def test_parses_server_handshake_snapshot() -> None:
    assert parse_server_message(SERVER_HELLO) == SERVER_HELLO


@pytest.mark.parametrize(
    "wire",
    [
        {"type": "hello", "version": 1, "connectionId": "connection-1", "snapshot": EMPTY_SERVER_SNAPSHOT},
        {"type": "response", "id": "request-1", "ok": True, "result": {"command": "unknown"}},
        {"type": "event", "event": {"type": "session_removed", "sessionId": 42}},
    ],
)
def test_rejects_invalid_server_messages(wire: object) -> None:
    with pytest.raises(ProtocolValidationError):
        parse_server_message(wire)


def test_validates_nested_json_tool_details() -> None:
    message = item_message(
        {
            "id": "tool-1",
            "role": "tool",
            "toolCallId": "call-1",
            "toolName": "read",
            "input": {"path": "/tmp/file"},
            "content": [{"type": "text", "text": "done"}],
            "details": {"lines": [1, 2, 3], "cached": False},
            "status": "complete",
            "isError": False,
            "timestamp": 1,
        }
    )
    assert parse_server_message(message) == message


@pytest.mark.parametrize(
    "state",
    [
        {"status": "streaming"},
        {"status": "complete", "stopReason": "stop"},
        {"status": "error", "stopReason": "error"},
        {"status": "error", "stopReason": "error", "errorMessage": "failed"},
        {"status": "aborted", "stopReason": "aborted"},
    ],
)
def test_accepts_consistent_assistant_item(state: dict[str, object]) -> None:
    message = item_message(
        {
            "id": "assistant-1",
            "role": "assistant",
            "content": [{"type": "text", "text": "hello"}],
            "model": {"provider": "test", "id": "model"},
            "timestamp": 1,
            **state,
        },
        "item_updated" if state["status"] == "streaming" else "item_finished",
    )
    assert parse_server_message(message) == message


@pytest.mark.parametrize(
    "state",
    [
        {"status": "streaming", "stopReason": "stop"},
        {"status": "complete"},
        {"status": "complete", "stopReason": "error"},
        {"status": "error", "stopReason": "error", "errorMessage": ""},
        {"status": "aborted", "stopReason": "stop"},
    ],
)
def test_rejects_inconsistent_assistant_item(state: dict[str, object]) -> None:
    with pytest.raises(ProtocolValidationError):
        parse_server_message(
            item_message(
                {
                    "id": "assistant-1",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hello"}],
                    "model": {"provider": "test", "id": "model"},
                    "timestamp": 1,
                    **state,
                }
            )
        )


@pytest.mark.parametrize(
    "state",
    [
        {"status": "running", "isError": False},
        {"status": "complete", "isError": False},
        {"status": "error", "isError": True},
    ],
)
def test_accepts_consistent_tool_item(state: dict[str, object]) -> None:
    message = item_message(
        {
            "id": "tool-1",
            "role": "tool",
            "toolCallId": "call-1",
            "toolName": "read",
            "input": {},
            "content": [],
            "timestamp": 1,
            **state,
        },
        "item_updated" if state["status"] == "running" else "item_finished",
    )
    assert parse_server_message(message) == message


def test_rejects_nonterminal_items_reported_as_finished() -> None:
    assistant = {
        "id": "assistant-1",
        "role": "assistant",
        "content": [],
        "model": {"provider": "test", "id": "model"},
        "status": "streaming",
        "timestamp": 1,
    }
    tool = {
        "id": "tool-1",
        "role": "tool",
        "toolCallId": "call-1",
        "toolName": "read",
        "input": {},
        "content": [],
        "status": "running",
        "isError": False,
        "timestamp": 1,
    }
    with pytest.raises(ProtocolValidationError):
        parse_server_message(item_message(assistant))
    with pytest.raises(ProtocolValidationError):
        parse_server_message(item_message(tool))


@pytest.mark.parametrize(
    "state",
    [
        {"status": "running", "isError": True},
        {"status": "complete", "isError": True},
        {"status": "error", "isError": False},
    ],
)
def test_rejects_inconsistent_tool_item(state: dict[str, object]) -> None:
    with pytest.raises(ProtocolValidationError):
        parse_server_message(
            item_message(
                {
                    "id": "tool-1",
                    "role": "tool",
                    "toolCallId": "call-1",
                    "toolName": "read",
                    "input": {},
                    "content": [],
                    "timestamp": 1,
                    **state,
                }
            )
        )


def test_validation_errors_do_not_retain_rejected_payloads() -> None:
    with pytest.raises(ProtocolValidationError) as exc:
        parse_client_message({"type": "hello", "version": "2", "token": "x" * 2_000_000})
    assert not hasattr(exc.value, "value")
    assert len(str(exc.value)) < 1000


def test_encodes_complete_client_and_server_frames() -> None:
    client_frames = FrameDecoder().push(encode_client_message(CLIENT_HELLO))
    assert len(client_frames) == 1
    assert parse_client_message(decode_cbor(client_frames[0])) == CLIENT_HELLO

    server_frames = FrameDecoder().push(encode_server_message(SERVER_HELLO))
    assert len(server_frames) == 1
    assert parse_server_message(decode_cbor(server_frames[0])) == SERVER_HELLO


def test_enforces_outbound_frame_limit_before_returning_encoded_bytes() -> None:
    with pytest.raises(ProtocolValidationError):
        encode_client_message(CLIENT_HELLO, max_frame_length=8)
    with pytest.raises(ProtocolValidationError):
        encode_server_message(SERVER_HELLO, max_frame_length=8)


def test_validates_messages_before_encoding() -> None:
    with pytest.raises(ProtocolValidationError):
        encode_client_message({"type": "hello", "version": 2, "token": ""})


def test_incrementally_decodes_fragmented_and_coalesced_client_messages() -> None:
    request = {"type": "request", "id": "request-1", "request": {"command": "list"}}
    first = encode_client_message(CLIENT_HELLO)
    second = encode_client_message(request)
    wire = first + second
    for split in range(len(wire) + 1):
        decoder = ClientMessageDecoder()
        messages = [*decoder.push(wire[:split]), *decoder.push(wire[split:])]
        decoder.end()
        assert messages == [CLIENT_HELLO, request]


def test_incrementally_decodes_server_messages() -> None:
    error_message = {"type": "hello_error", "error": {"code": "auth", "message": "Invalid token"}}
    decoder = ServerMessageDecoder()
    assert decoder.push(encode_server_message(error_message)) == [error_message]
    decoder.end()


@pytest.mark.parametrize(
    "wire",
    [
        encode_frame(b""),
        encode_frame(b"\xff"),
        encode_frame(encode_cbor({"type": "hello", "version": 2, "token": "", "extra": True})),
    ],
)
def test_rejects_invalid_framed_client_input(wire: bytes) -> None:
    decoder = ClientMessageDecoder()
    with pytest.raises(ProtocolValidationError):
        decoder.push(wire)
    with pytest.raises(ProtocolValidationError, match="failed"):
        decoder.push(encode_client_message(CLIENT_HELLO))


def test_rejects_cbor_byte_strings_nested_in_json_valued_fields() -> None:
    wire = encode_frame(
        encode_cbor(
            {
                "type": "response",
                "id": "request-1",
                "ok": False,
                "error": {
                    "code": "invalid_request",
                    "message": "invalid",
                    "details": {"nested": b"\x01\x02\x03"},
                },
            }
        )
    )
    with pytest.raises(ProtocolValidationError):
        ServerMessageDecoder().push(wire)


def test_rejects_truncated_and_oversized_framing_through_validated_decoder() -> None:
    truncated = ServerMessageDecoder()
    assert truncated.push(b"\x00\x00\x00\x02\x01") == []
    with pytest.raises(ProtocolValidationError):
        truncated.end()

    oversized = ClientMessageDecoder(max_frame_length=3)
    with pytest.raises(ProtocolValidationError):
        oversized.push(b"\x00\x00\x00\x04")
