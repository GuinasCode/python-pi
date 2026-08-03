from __future__ import annotations

import math
import struct
from collections.abc import Mapping
from typing import TypeAlias

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]

UINT32_BASE = 0x1_0000_0000
MAX_UINT32 = 0xFFFF_FFFF
MAX_SAFE_INTEGER = 2**53 - 1
MAX_CONFIGURED_DEPTH = 512

DEFAULT_MAX_CBOR_BYTE_LENGTH = 16 * 1024 * 1024
DEFAULT_MAX_CBOR_CONTAINER_LENGTH = 1_000_000
DEFAULT_MAX_CBOR_DEPTH = 64
DEFAULT_MAX_FRAME_LENGTH = 16 * 1024 * 1024
PROTOCOL_VERSION = 2


class CborError(Exception):
    """Raised when a value is outside Pi's strict CBOR subset."""


class FrameError(Exception):
    """Raised when a length-prefixed frame is malformed."""


class ProtocolValidationError(Exception):
    """Raised when a decoded message does not match the wire protocol."""


def _resolve_limit(name: str, value: int | float, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > maximum:
        raise ValueError(f"{name} must be an integer between 0 and {maximum}")
    return value


def _resolve_cbor_options(
    max_byte_length: int = DEFAULT_MAX_CBOR_BYTE_LENGTH,
    max_container_length: int = DEFAULT_MAX_CBOR_CONTAINER_LENGTH,
    max_depth: int = DEFAULT_MAX_CBOR_DEPTH,
) -> tuple[int, int, int]:
    return (
        _resolve_limit("maxByteLength", max_byte_length, MAX_UINT32),
        _resolve_limit("maxContainerLength", max_container_length, MAX_UINT32),
        _resolve_limit("maxDepth", max_depth, MAX_CONFIGURED_DEPTH),
    )


class _CborWriter:
    def __init__(self, max_byte_length: int) -> None:
        self._buffer = bytearray()
        self._max_byte_length = max_byte_length

    def write(self, data: bytes) -> None:
        if len(self._buffer) + len(data) > self._max_byte_length:
            raise CborError(f"CBOR byte length exceeds configured limit of {self._max_byte_length}")
        self._buffer.extend(data)

    def finish(self) -> bytes:
        return bytes(self._buffer)


def _write_argument(writer: _CborWriter, major_type: int, value: int) -> None:
    prefix = major_type << 5
    if value < 24:
        writer.write(bytes([prefix | value]))
    elif value <= 0xFF:
        writer.write(bytes([prefix | 24, value]))
    elif value <= 0xFFFF:
        writer.write(bytes([prefix | 25]) + value.to_bytes(2, "big"))
    elif value <= MAX_UINT32:
        writer.write(bytes([prefix | 26]) + value.to_bytes(4, "big"))
    else:
        writer.write(bytes([prefix | 27]) + value.to_bytes(8, "big"))


def _encode_text(writer: _CborWriter, value: str, max_byte_length: int) -> None:
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise CborError("CBOR text strings must contain valid Unicode scalar values") from exc
    if len(encoded) > max_byte_length:
        raise CborError(f"CBOR text string length exceeds configured limit of {max_byte_length}")
    _write_argument(writer, 3, len(encoded))
    writer.write(encoded)


def _encode_value(
    writer: _CborWriter,
    value: object,
    max_byte_length: int,
    max_container_length: int,
    max_depth: int,
    depth: int,
    ancestors: set[int],
) -> None:
    if depth > max_depth:
        raise CborError(f"CBOR nesting depth exceeds configured limit of {max_depth}")
    if value is None:
        writer.write(b"\xf6")
        return
    if isinstance(value, bool):
        writer.write(b"\xf5" if value else b"\xf4")
        return
    if isinstance(value, int) and not isinstance(value, bool):
        if abs(value) > MAX_SAFE_INTEGER:
            raise CborError("CBOR integers must be safe JavaScript integers")
        if value >= 0:
            _write_argument(writer, 0, value)
        else:
            _write_argument(writer, 1, -1 - value)
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CborError("CBOR numbers must be finite")
        if value.is_integer() and value != 0.0:
            integer_value = int(value)
            if abs(integer_value) > MAX_SAFE_INTEGER:
                raise CborError("CBOR integers must be safe JavaScript integers")
        writer.write(b"\xfb" + struct.pack(">d", value))
        return
    if isinstance(value, str):
        _encode_text(writer, value, max_byte_length)
        return
    if isinstance(value, bytes | bytearray | memoryview):
        raw = bytes(value)
        if len(raw) > max_byte_length:
            raise CborError(f"CBOR byte string length exceeds configured limit of {max_byte_length}")
        _write_argument(writer, 2, len(raw))
        writer.write(raw)
        return
    if isinstance(value, list | tuple):
        identity = id(value)
        if identity in ancestors:
            raise CborError("CBOR values must not contain cycles")
        if len(value) > max_container_length:
            raise CborError(f"CBOR array length exceeds configured limit of {max_container_length}")
        ancestors.add(identity)
        try:
            _write_argument(writer, 4, len(value))
            for item in value:
                _encode_value(writer, item, max_byte_length, max_container_length, max_depth, depth + 1, ancestors)
        finally:
            ancestors.remove(identity)
        return
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in ancestors:
            raise CborError("CBOR values must not contain cycles")
        entries = list(value.items())
        if len(entries) > max_container_length:
            raise CborError(f"CBOR map length exceeds configured limit of {max_container_length}")
        ancestors.add(identity)
        try:
            _write_argument(writer, 5, len(entries))
            for key, item in entries:
                if not isinstance(key, str):
                    raise CborError("CBOR map keys must be strings")
                _encode_text(writer, key, max_byte_length)
                _encode_value(writer, item, max_byte_length, max_container_length, max_depth, depth + 1, ancestors)
        finally:
            ancestors.remove(identity)
        return
    raise CborError(f"Unsupported CBOR value type: {type(value).__name__}")


def encode_cbor(
    value: object,
    *,
    max_byte_length: int = DEFAULT_MAX_CBOR_BYTE_LENGTH,
    max_container_length: int = DEFAULT_MAX_CBOR_CONTAINER_LENGTH,
    max_depth: int = DEFAULT_MAX_CBOR_DEPTH,
) -> bytes:
    """Encode one value using Pi's strict, definite-length RFC 8949 subset."""
    resolved_max_byte_length, resolved_max_container_length, resolved_max_depth = _resolve_cbor_options(
        max_byte_length, max_container_length, max_depth
    )
    writer = _CborWriter(resolved_max_byte_length)
    _encode_value(writer, value, resolved_max_byte_length, resolved_max_container_length, resolved_max_depth, 0, set())
    return writer.finish()


class _CborReader:
    def __init__(self, data: bytes, max_byte_length: int, max_container_length: int, max_depth: int) -> None:
        self._data = data
        self._offset = 0
        self._max_byte_length = max_byte_length
        self._max_container_length = max_container_length
        self._max_depth = max_depth

    def decode(self) -> object:
        value = self._read_item(0)
        if self._offset != len(self._data):
            raise CborError("CBOR payload contains trailing data")
        return value

    def _read_item(self, depth: int) -> object:
        if depth > self._max_depth:
            raise CborError(f"CBOR nesting depth exceeds configured limit of {self._max_depth}")
        initial = self._read_byte()
        major_type = initial >> 5
        additional = initial & 0x1F
        if major_type == 0:
            return self._read_argument(additional)
        if major_type == 1:
            value = -1 - self._read_argument(additional)
            if abs(value) > MAX_SAFE_INTEGER:
                raise CborError("Decoded CBOR integer is outside the safe range")
            return value
        if major_type == 2:
            length = self._read_length(additional, "byte string", self._max_byte_length)
            return self._read_bytes(length)
        if major_type == 3:
            length = self._read_length(additional, "text string", self._max_byte_length)
            raw = self._read_bytes(length)
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise CborError("CBOR text string contains invalid UTF-8") from exc
        if major_type == 4:
            length = self._read_length(additional, "array", self._max_container_length)
            return [self._read_item(depth + 1) for _ in range(length)]
        if major_type == 5:
            length = self._read_length(additional, "map", self._max_container_length)
            result: dict[str, object] = {}
            for _ in range(length):
                key = self._read_item(depth + 1)
                if not isinstance(key, str):
                    raise CborError("CBOR map keys must be strings")
                if key in result:
                    raise CborError("CBOR map contains a duplicate key")
                result[key] = self._read_item(depth + 1)
            return result
        if major_type == 6:
            raise CborError("CBOR tags are not supported")
        if major_type == 7:
            return self._read_simple(additional)
        raise CborError("Malformed CBOR major type")

    def _read_simple(self, additional: int) -> object:
        if additional == 20:
            return False
        if additional == 21:
            return True
        if additional == 22:
            return None
        if additional == 27:
            value = struct.unpack(">d", self._read_bytes(8))[0]
            if not math.isfinite(value):
                raise CborError("Decoded CBOR number must be finite")
            if value.is_integer() and value != 0.0 and abs(int(value)) > MAX_SAFE_INTEGER:
                raise CborError("Decoded CBOR integer is outside the safe range")
            return value
        if additional == 31:
            raise CborError("CBOR break marker is not supported")
        raise CborError("Unsupported CBOR simple value or floating-point width")

    def _read_length(self, additional: int, kind: str, limit: int) -> int:
        if additional == 31:
            raise CborError(f"Indefinite-length CBOR {kind}s are not supported")
        length = self._read_argument(additional)
        if length > limit:
            raise CborError(f"CBOR {kind} length exceeds configured limit of {limit}")
        return length

    def _read_argument(self, additional: int) -> int:
        if additional < 24:
            return additional
        if additional == 24:
            return self._read_byte()
        if additional == 25:
            return int.from_bytes(self._read_bytes(2), "big")
        if additional == 26:
            return int.from_bytes(self._read_bytes(4), "big")
        if additional == 27:
            value = int.from_bytes(self._read_bytes(8), "big")
            if value > MAX_SAFE_INTEGER:
                raise CborError("Decoded CBOR integer or length is outside the safe range")
            return value
        if additional == 31:
            raise CborError("Indefinite-length CBOR items are not supported")
        raise CborError("Malformed CBOR additional information")

    def _read_byte(self) -> int:
        if self._offset >= len(self._data):
            raise CborError("Truncated CBOR payload")
        value = self._data[self._offset]
        self._offset += 1
        return value

    def _read_bytes(self, length: int) -> bytes:
        if length > len(self._data) - self._offset:
            raise CborError("Truncated CBOR payload")
        value = self._data[self._offset : self._offset + length]
        self._offset += length
        return value


def decode_cbor(
    data: bytes | bytearray | memoryview,
    *,
    max_byte_length: int = DEFAULT_MAX_CBOR_BYTE_LENGTH,
    max_container_length: int = DEFAULT_MAX_CBOR_CONTAINER_LENGTH,
    max_depth: int = DEFAULT_MAX_CBOR_DEPTH,
) -> object:
    """Decode exactly one item from Pi's strict CBOR subset."""
    if not isinstance(data, bytes | bytearray | memoryview):
        raise TypeError("CBOR input must be bytes-like")
    raw = bytes(data)
    resolved_max_byte_length, resolved_max_container_length, resolved_max_depth = _resolve_cbor_options(
        max_byte_length, max_container_length, max_depth
    )
    if len(raw) > resolved_max_byte_length:
        raise CborError(f"CBOR byte length exceeds configured limit of {resolved_max_byte_length}")
    return _CborReader(raw, resolved_max_byte_length, resolved_max_container_length, resolved_max_depth).decode()


def _resolve_max_frame_length(max_frame_length: int = DEFAULT_MAX_FRAME_LENGTH) -> int:
    return _resolve_limit("maxFrameLength", max_frame_length, MAX_UINT32)


def encode_frame(payload: bytes | bytearray | memoryview) -> bytes:
    if not isinstance(payload, bytes | bytearray | memoryview):
        raise TypeError("Frame payload must be bytes-like")
    raw = bytes(payload)
    if len(raw) > MAX_UINT32:
        raise ValueError("Frame payload exceeds the unsigned 32-bit length limit")
    return len(raw).to_bytes(4, "big") + raw


def assert_complete_frame(
    frame: bytes | bytearray | memoryview, *, max_frame_length: int = DEFAULT_MAX_FRAME_LENGTH
) -> None:
    if not isinstance(frame, bytes | bytearray | memoryview):
        raise TypeError("Frame must be bytes-like")
    raw = bytes(frame)
    if len(raw) < 4:
        raise FrameError("Frame does not contain a complete length prefix")
    length = int.from_bytes(raw[:4], "big")
    limit = _resolve_max_frame_length(max_frame_length)
    if length > limit:
        raise FrameError(f"Frame length {length} exceeds configured limit of {limit}")
    if len(raw) != 4 + length:
        raise FrameError("Frame must contain exactly one complete payload")


class FrameDecoder:
    def __init__(self, *, max_frame_length: int = DEFAULT_MAX_FRAME_LENGTH) -> None:
        self._max_frame_length = _resolve_max_frame_length(max_frame_length)
        self._buffer = bytearray()
        self._state = "open"

    def push(self, chunk: bytes | bytearray | memoryview) -> list[bytes]:
        if self._state == "ended":
            raise FrameError("Frame decoder has ended")
        if self._state == "failed":
            raise FrameError("Frame decoder has failed")
        if not isinstance(chunk, bytes | bytearray | memoryview):
            raise TypeError("Frame chunk must be bytes-like")
        self._buffer.extend(bytes(chunk))
        frames: list[bytes] = []
        try:
            while len(self._buffer) >= 4:
                length = int.from_bytes(self._buffer[:4], "big")
                if length > self._max_frame_length:
                    self._fail(f"Frame length {length} exceeds configured limit of {self._max_frame_length}")
                if len(self._buffer) < 4 + length:
                    break
                frames.append(bytes(self._buffer[4 : 4 + length]))
                del self._buffer[: 4 + length]
            return frames
        except Exception:
            if self._state != "failed":
                self._fail("Frame decoder has failed")
            raise

    def end(self) -> None:
        if self._state == "ended":
            raise FrameError("Frame decoder has ended")
        if self._state == "failed":
            raise FrameError("Frame decoder has failed")
        if self._buffer:
            self._fail("Truncated frame at end of stream")
        self._state = "ended"

    def _fail(self, message: str) -> None:
        self._state = "failed"
        self._buffer.clear()
        raise FrameError(message)


def _is_id(value: object) -> bool:
    return isinstance(value, str) and len(value) > 0


def _is_non_negative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_non_negative_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _is_json_value(value: object, ancestors: set[int] | None = None) -> bool:
    if ancestors is None:
        ancestors = set()
    if value is None or isinstance(value, bool | str):
        return True
    if isinstance(value, int | float) and not isinstance(value, bool):
        return math.isfinite(value)
    if isinstance(value, bytes | bytearray | memoryview):
        return False
    if isinstance(value, list):
        identity = id(value)
        if identity in ancestors:
            return False
        ancestors.add(identity)
        try:
            return all(_is_json_value(item, ancestors) for item in value)
        finally:
            ancestors.remove(identity)
    if isinstance(value, dict):
        identity = id(value)
        if identity in ancestors:
            return False
        ancestors.add(identity)
        try:
            return all(isinstance(key, str) and _is_json_value(item, ancestors) for key, item in value.items())
        finally:
            ancestors.remove(identity)
    return False


def _strict_keys(value: Mapping[str, object], allowed: set[str]) -> bool:
    return set(value) <= allowed


def _model_ref(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"provider", "id"}
        and _is_id(value["provider"])
        and _is_id(value["id"])
    )


def _thinking_level(value: object) -> bool:
    return value in {"off", "minimal", "low", "medium", "high", "xhigh", "max"}


def _session_phase(value: object) -> bool:
    return value in {"idle", "turn", "compaction", "branch_summary", "retry"}


def _content(value: object, allowed: set[str]) -> bool:
    if not isinstance(value, dict) or "type" not in value or not isinstance(value["type"], str):
        return False
    type_ = value["type"]
    if type_ not in allowed:
        return False
    if type_ == "text":
        return set(value) == {"type", "text"} and isinstance(value.get("text"), str)
    if type_ == "thinking":
        return (
            _strict_keys(value, {"type", "thinking", "redacted"})
            and isinstance(value.get("thinking"), str)
            and ("redacted" not in value or isinstance(value.get("redacted"), bool))
        )
    if type_ == "image":
        return (
            set(value) == {"type", "data", "mimeType"}
            and isinstance(value.get("data"), str)
            and _is_id(value.get("mimeType"))
        )
    if type_ == "toolCall":
        return (
            set(value) == {"type", "toolCallId", "toolName", "input"}
            and _is_id(value.get("toolCallId"))
            and _is_id(value.get("toolName"))
            and _is_json_value(value.get("input"))
        )
    return False


def _usage(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    if not _strict_keys(value, {"input", "output", "cacheRead", "cacheWrite", "reasoning", "totalTokens", "cost"}):
        return False
    required_ints = ["input", "output", "cacheRead", "cacheWrite", "totalTokens"]
    if not all(_is_non_negative_int(value.get(key)) for key in required_ints):
        return False
    if "reasoning" in value and not _is_non_negative_int(value.get("reasoning")):
        return False
    cost = value.get("cost")
    return (
        isinstance(cost, dict)
        and set(cost) == {"input", "output", "cacheRead", "cacheWrite", "total"}
        and all(_is_non_negative_number(cost.get(key)) for key in cost)
    )


def _transcript_item(value: object) -> bool:
    if not isinstance(value, dict) or not _is_id(value.get("id")) or not _is_non_negative_int(value.get("timestamp")):
        return False
    role = value.get("role")
    if role == "user":
        return (
            set(value) == {"id", "role", "content", "timestamp"}
            and isinstance(value.get("content"), list)
            and all(_content(item, {"text", "image"}) for item in value["content"])
        )
    if role == "assistant":
        if not _strict_keys(
            value,
            {
                "id",
                "role",
                "content",
                "model",
                "responseModel",
                "usage",
                "timestamp",
                "status",
                "stopReason",
                "errorMessage",
            },
        ):
            return False
        if not isinstance(value.get("content"), list) or not all(
            _content(item, {"text", "thinking", "toolCall"}) for item in value["content"]
        ):
            return False
        if not _model_ref(value.get("model")):
            return False
        if "responseModel" in value and not _is_id(value.get("responseModel")):
            return False
        if "usage" in value and not _usage(value.get("usage")):
            return False
        status = value.get("status")
        stop_reason = value.get("stopReason")
        if status == "streaming":
            return "stopReason" not in value and "errorMessage" not in value
        if status == "complete":
            return stop_reason in {"stop", "length", "toolUse"} and "errorMessage" not in value
        if status == "error":
            return stop_reason == "error" and ("errorMessage" not in value or _is_id(value.get("errorMessage")))
        if status == "aborted":
            return stop_reason == "aborted" and (
                "errorMessage" not in value or isinstance(value.get("errorMessage"), str)
            )
        return False
    if role == "tool":
        if not _strict_keys(
            value,
            {
                "id",
                "role",
                "toolCallId",
                "toolName",
                "input",
                "content",
                "details",
                "usage",
                "timestamp",
                "status",
                "isError",
            },
        ):
            return False
        if (
            not _is_id(value.get("toolCallId"))
            or not _is_id(value.get("toolName"))
            or not _is_json_value(value.get("input"))
        ):
            return False
        if not isinstance(value.get("content"), list) or not all(
            _content(item, {"text", "image"}) for item in value["content"]
        ):
            return False
        if "details" in value and not _is_json_value(value.get("details")):
            return False
        if "usage" in value and not _usage(value.get("usage")):
            return False
        return (value.get("status"), value.get("isError")) in {("running", False), ("complete", False), ("error", True)}
    return False


def _transcript_progress(value: object) -> bool:
    if not isinstance(value, dict) or not isinstance(value.get("type"), str):
        return False
    type_ = value["type"]
    if type_ == "item_started":
        return set(value) == {"type", "item"} and _transcript_item(value.get("item"))
    if type_ == "assistant_delta":
        return (
            set(value) == {"type", "messageId", "contentIndex", "kind", "delta"}
            and _is_id(value.get("messageId"))
            and _is_non_negative_int(value.get("contentIndex"))
            and value.get("kind") in {"text", "thinking", "toolCall"}
            and isinstance(value.get("delta"), str)
        )
    if type_ == "item_updated":
        return (
            set(value) == {"type", "item"}
            and _transcript_item(value.get("item"))
            and isinstance(value.get("item"), dict)
            and value["item"].get("role") in {"assistant", "tool"}
        )
    if type_ == "item_finished":
        item = value.get("item")
        return (
            set(value) == {"type", "item"}
            and _transcript_item(item)
            and isinstance(item, dict)
            and (
                (item.get("role") == "assistant" and item.get("status") in {"complete", "error", "aborted"})
                or (item.get("role") == "tool" and item.get("status") in {"complete", "error"})
            )
        )
    return False


def _session_summary(value: object) -> bool:
    return (
        isinstance(value, dict)
        and _strict_keys(
            value,
            {"id", "name", "cwd", "createdAt", "updatedAt", "phase", "model", "thinkingLevel", "attached", "locked"},
        )
        and _is_id(value.get("id"))
        and ("name" not in value or isinstance(value.get("name"), str))
        and _is_id(value.get("cwd"))
        and _is_non_negative_int(value.get("createdAt"))
        and _is_non_negative_int(value.get("updatedAt"))
        and _session_phase(value.get("phase"))
        and _model_ref(value.get("model"))
        and _thinking_level(value.get("thinkingLevel"))
        and isinstance(value.get("attached"), bool)
        and isinstance(value.get("locked"), bool)
    )


def _session_snapshot(value: object) -> bool:
    return (
        _session_summary(value)
        and isinstance(value, dict)
        and set(value) >= {"revision", "transcript", "queuedSteer", "queuedSteerCount"}
        and _is_non_negative_int(value.get("revision"))
        and isinstance(value.get("transcript"), list)
        and all(_transcript_item(item) for item in value["transcript"])
        and isinstance(value.get("queuedSteer"), list)
        and all(
            _transcript_item(item) and isinstance(item, dict) and item.get("role") == "user"
            for item in value["queuedSteer"]
        )
        and _is_non_negative_int(value.get("queuedSteerCount"))
    )


def _model_metadata(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "provider",
        "id",
        "name",
        "api",
        "reasoning",
        "input",
        "contextWindow",
        "maxTokens",
        "cost",
        "supportedThinkingLevels",
        "authenticated",
    }:
        return False
    cost = value.get("cost")
    return (
        _is_id(value.get("provider"))
        and _is_id(value.get("id"))
        and _is_id(value.get("name"))
        and _is_id(value.get("api"))
        and isinstance(value.get("reasoning"), bool)
        and isinstance(value.get("input"), list)
        and all(item in {"text", "image"} for item in value["input"])
        and _is_non_negative_int(value.get("contextWindow"))
        and value.get("contextWindow") != 0
        and _is_non_negative_int(value.get("maxTokens"))
        and value.get("maxTokens") != 0
        and isinstance(cost, dict)
        and set(cost) == {"input", "output", "cacheRead", "cacheWrite"}
        and all(_is_non_negative_number(cost.get(key)) for key in cost)
        and isinstance(value.get("supportedThinkingLevels"), list)
        and len(value["supportedThinkingLevels"]) > 0
        and all(_thinking_level(item) for item in value["supportedThinkingLevels"])
        and isinstance(value.get("authenticated"), bool)
    )


def _server_snapshot(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"serverId", "protocolVersion", "revision", "sessions", "models"}
        and _is_id(value.get("serverId"))
        and value.get("protocolVersion") == PROTOCOL_VERSION
        and _is_non_negative_int(value.get("revision"))
        and isinstance(value.get("sessions"), list)
        and all(_session_summary(item) for item in value["sessions"])
        and isinstance(value.get("models"), list)
        and all(_model_metadata(item) for item in value["models"])
    )


def _protocol_error(value: object) -> bool:
    return (
        isinstance(value, dict)
        and _strict_keys(value, {"code", "message", "details"})
        and value.get("code") in {"auth", "version", "busy", "session_locked", "not_found", "invalid_request"}
        and isinstance(value.get("message"), str)
        and ("details" not in value or _is_json_value(value.get("details")))
    )


def _command(value: object) -> bool:
    if not isinstance(value, dict) or not isinstance(value.get("command"), str):
        return False
    command = value["command"]
    if command == "list":
        return set(value) == {"command"}
    if command == "create":
        return (
            _strict_keys(value, {"command", "cwd", "name", "model", "thinkingLevel"})
            and ("cwd" not in value or _is_id(value.get("cwd")))
            and ("name" not in value or isinstance(value.get("name"), str))
            and ("model" not in value or _model_ref(value.get("model")))
            and ("thinkingLevel" not in value or _thinking_level(value.get("thinkingLevel")))
        )
    if command in {"attach", "detach", "abort"}:
        return set(value) == {"command", "sessionId"} and _is_id(value.get("sessionId"))
    if command in {"prompt", "steer"}:
        return (
            set(value) == {"command", "sessionId", "text"}
            and _is_id(value.get("sessionId"))
            and isinstance(value.get("text"), str)
        )
    if command == "set_model":
        return (
            set(value) == {"command", "sessionId", "model"}
            and _is_id(value.get("sessionId"))
            and _model_ref(value.get("model"))
        )
    if command == "set_thinking":
        return (
            set(value) == {"command", "sessionId", "thinkingLevel"}
            and _is_id(value.get("sessionId"))
            and _thinking_level(value.get("thinkingLevel"))
        )
    return False


def _command_result(value: object) -> bool:
    if not isinstance(value, dict) or not isinstance(value.get("command"), str):
        return False
    command = value["command"]
    if command == "list":
        return (
            set(value) == {"command", "sessions"}
            and isinstance(value.get("sessions"), list)
            and all(_session_summary(item) for item in value["sessions"])
        )
    if command == "detach":
        return set(value) == {"command", "sessionId"} and _is_id(value.get("sessionId"))
    if command in {"create", "attach", "prompt", "steer", "abort", "set_model", "set_thinking"}:
        return set(value) == {"command", "session"} and _session_snapshot(value.get("session"))
    return False


def _client_message(value: object) -> bool:
    if not isinstance(value, dict) or not isinstance(value.get("type"), str):
        return False
    if value["type"] == "hello":
        return (
            set(value) == {"type", "version", "token"}
            and _is_non_negative_int(value.get("version"))
            and _is_id(value.get("token"))
        )
    if value["type"] == "request":
        return set(value) == {"type", "id", "request"} and _is_id(value.get("id")) and _command(value.get("request"))
    return False


def _server_event(value: object) -> bool:
    if not isinstance(value, dict) or not isinstance(value.get("type"), str):
        return False
    if value["type"] == "server_snapshot":
        return set(value) == {"type", "snapshot"} and _server_snapshot(value.get("snapshot"))
    if value["type"] == "session_snapshot":
        return set(value) == {"type", "snapshot"} and _session_snapshot(value.get("snapshot"))
    if value["type"] == "session_progress":
        return (
            set(value) == {"type", "sessionId", "progress"}
            and _is_id(value.get("sessionId"))
            and _transcript_progress(value.get("progress"))
        )
    if value["type"] == "session_removed":
        return set(value) == {"type", "sessionId"} and _is_id(value.get("sessionId"))
    return False


def _server_message(value: object) -> bool:
    if not isinstance(value, dict) or not isinstance(value.get("type"), str):
        return False
    if value["type"] == "hello":
        return (
            set(value) == {"type", "version", "connectionId", "snapshot"}
            and value.get("version") == PROTOCOL_VERSION
            and _is_id(value.get("connectionId"))
            and _server_snapshot(value.get("snapshot"))
        )
    if value["type"] == "hello_error":
        return set(value) == {"type", "error"} and _protocol_error(value.get("error"))
    if value["type"] == "response":
        if (
            not _strict_keys(value, {"type", "id", "ok", "result", "error"})
            or not _is_id(value.get("id"))
            or not isinstance(value.get("ok"), bool)
        ):
            return False
        return (
            (_command_result(value.get("result")) and "error" not in value)
            if value["ok"]
            else (_protocol_error(value.get("error")) and "result" not in value)
        )
    if value["type"] == "event":
        return set(value) == {"type", "event"} and _server_event(value.get("event"))
    return False


def parse_client_message(value: object) -> dict[str, object]:
    if not _is_json_value(value) or not _client_message(value):
        raise ProtocolValidationError("Invalid client protocol message")
    if not isinstance(value, dict):
        raise ProtocolValidationError("Invalid client protocol message")
    return value


def parse_server_message(value: object) -> dict[str, object]:
    if not _is_json_value(value) or not _server_message(value):
        raise ProtocolValidationError("Invalid server protocol message")
    if not isinstance(value, dict):
        raise ProtocolValidationError("Invalid server protocol message")
    return value


def _bounded_error_message(error: Exception) -> str:
    message = str(error) or "Unknown codec error"
    return message if len(message) <= 500 else f"{message[:497]}..."


def _encode_protocol_message(value: object, parser: object, kind: str, max_frame_length: int) -> bytes:
    parse = parser  # keep narrow helper simple
    try:
        validated = parse(value)  # type: ignore[operator]
        frame = encode_frame(encode_cbor(validated, max_byte_length=max_frame_length))
        assert_complete_frame(frame, max_frame_length=max_frame_length)
        return frame
    except ProtocolValidationError:
        raise
    except Exception as exc:
        raise ProtocolValidationError(
            f"Unable to encode {kind} protocol message: {_bounded_error_message(exc)}"
        ) from exc


def encode_client_message(message: object, *, max_frame_length: int = DEFAULT_MAX_FRAME_LENGTH) -> bytes:
    return _encode_protocol_message(message, parse_client_message, "client", max_frame_length)


def encode_server_message(message: object, *, max_frame_length: int = DEFAULT_MAX_FRAME_LENGTH) -> bytes:
    return _encode_protocol_message(message, parse_server_message, "server", max_frame_length)


class _ValidatedMessageDecoder:
    def __init__(self, kind: str, parser: object, *, max_frame_length: int = DEFAULT_MAX_FRAME_LENGTH) -> None:
        self._failed = False
        self._frames = FrameDecoder(max_frame_length=max_frame_length)
        self._kind = kind
        self._max_frame_length = max_frame_length
        self._parser = parser

    def push(self, chunk: bytes | bytearray | memoryview) -> list[dict[str, object]]:
        if self._failed:
            raise ProtocolValidationError(f"{self._kind} message decoder has failed")
        parse = self._parser
        try:
            messages: list[dict[str, object]] = []
            for frame in self._frames.push(chunk):
                messages.append(parse(decode_cbor(frame, max_byte_length=self._max_frame_length)))  # type: ignore[operator]
            return messages
        except Exception as exc:
            self._failed = True
            if isinstance(exc, ProtocolValidationError):
                raise
            raise ProtocolValidationError(
                f"Invalid {self._kind} protocol frame: {_bounded_error_message(exc)}"
            ) from exc

    def end(self) -> None:
        if self._failed:
            raise ProtocolValidationError(f"{self._kind} message decoder has failed")
        try:
            self._frames.end()
        except Exception as exc:
            self._failed = True
            raise ProtocolValidationError(
                f"Invalid {self._kind} protocol framing: {_bounded_error_message(exc)}"
            ) from exc


class ClientMessageDecoder:
    def __init__(self, *, max_frame_length: int = DEFAULT_MAX_FRAME_LENGTH) -> None:
        self._decoder = _ValidatedMessageDecoder("client", parse_client_message, max_frame_length=max_frame_length)

    def push(self, chunk: bytes | bytearray | memoryview) -> list[dict[str, object]]:
        return self._decoder.push(chunk)

    def end(self) -> None:
        self._decoder.end()


class ServerMessageDecoder:
    def __init__(self, *, max_frame_length: int = DEFAULT_MAX_FRAME_LENGTH) -> None:
        self._decoder = _ValidatedMessageDecoder("server", parse_server_message, max_frame_length=max_frame_length)

    def push(self, chunk: bytes | bytearray | memoryview) -> list[dict[str, object]]:
        return self._decoder.push(chunk)

    def end(self) -> None:
        self._decoder.end()


def create_client_message_decoder(*, max_frame_length: int = DEFAULT_MAX_FRAME_LENGTH) -> ClientMessageDecoder:
    return ClientMessageDecoder(max_frame_length=max_frame_length)


def create_server_message_decoder(*, max_frame_length: int = DEFAULT_MAX_FRAME_LENGTH) -> ServerMessageDecoder:
    return ServerMessageDecoder(max_frame_length=max_frame_length)


def is_supported_protocol_version(version: object) -> bool:
    return isinstance(version, int) and not isinstance(version, bool) and version == PROTOCOL_VERSION
