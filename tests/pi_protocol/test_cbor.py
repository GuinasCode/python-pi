from __future__ import annotations

import math

import pytest

from pi_protocol import (
    DEFAULT_MAX_CBOR_BYTE_LENGTH,
    DEFAULT_MAX_CBOR_CONTAINER_LENGTH,
    DEFAULT_MAX_CBOR_DEPTH,
    CborError,
    decode_cbor,
    encode_cbor,
)


def from_hex(hex_value: str) -> bytes:
    return bytes.fromhex(hex_value)


def to_hex(data: bytes) -> str:
    return data.hex()


KNOWN_VECTORS = [
    (None, "f6"),
    (False, "f4"),
    (True, "f5"),
    (0, "00"),
    (1, "01"),
    (10, "0a"),
    (23, "17"),
    (24, "1818"),
    (25, "1819"),
    (100, "1864"),
    (1000, "1903e8"),
    (1_000_000, "1a000f4240"),
    (1_000_000_000_000, "1b000000e8d4a51000"),
    (2**53 - 1, "1b001fffffffffffff"),
    (-1, "20"),
    (-10, "29"),
    (-24, "37"),
    (-25, "3818"),
    (-100, "3863"),
    (-1000, "3903e7"),
    (-1_000_000, "3a000f423f"),
    (-(2**53) + 1, "3b001ffffffffffffe"),
    (1.1, "fb3ff199999999999a"),
    (-0.0, "fb8000000000000000"),
    (b"\x01\x02\x03\x04", "4401020304"),
    ("", "60"),
    ("IETF", "6449455446"),
    ("ü", "62c3bc"),
    ("水", "63e6b0b4"),
    ("𐅑", "64f0908591"),
    ([], "80"),
    ([1, 2, 3], "83010203"),
    ([1, [2, 3], [4, 5]], "8301820203820405"),
    ({"a": 1, "b": [2, 3]}, "a26161016162820203"),
]


@pytest.mark.parametrize(("value", "wire"), KNOWN_VECTORS)
def test_cbor_encodes_and_decodes_rfc_8949_vectors(value: object, wire: str) -> None:
    encoded = encode_cbor(value)
    assert to_hex(encoded) == wire
    decoded = decode_cbor(from_hex(wire))
    if isinstance(value, float) and value == 0 and math.copysign(1, value) < 0:
        assert isinstance(decoded, float)
        assert math.copysign(1, decoded) < 0
    else:
        assert decoded == value


def test_cbor_preserves_bom_and_proto_as_data() -> None:
    assert decode_cbor(from_hex("63efbbbf")) == "\ufeff"
    decoded = decode_cbor(encode_cbor({"__proto__": "safe"}))
    assert decoded == {"__proto__": "safe"}


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, 2**53, -(2**53), object(), {"x": object()}])
def test_cbor_rejects_unsupported_encoder_values(value: object) -> None:
    with pytest.raises(CborError):
        encode_cbor(value)


def test_cbor_rejects_cycles_and_excessive_depth() -> None:
    cyclic: list[object] = []
    cyclic.append(cyclic)
    with pytest.raises(CborError, match="cycles"):
        encode_cbor(cyclic)

    too_deep: object = None
    for _ in range(DEFAULT_MAX_CBOR_DEPTH + 1):
        too_deep = [too_deep]
    with pytest.raises(CborError, match="depth"):
        encode_cbor(too_deep)


@pytest.mark.parametrize(
    "wire",
    [
        "",
        "18",
        "1c",
        "5f",
        "7f",
        "9f",
        "bf",
        "c000",
        "f7",
        "e0",
        "ff",
        "f93c00",
        "fa3f800000",
        "fb7ff0000000000000",
        "fb7ff8000000000000",
        "fb3ff00000",
        "44010203",
        "636162",
        "8201",
        "a16161",
        "0000",
        "a10102",
        "a2616101616102",
        "61ff",
        "62c080",
        "63eda080",
        "1b0020000000000000",
        "3b001fffffffffffff",
        "fb4340000000000000",
    ],
)
def test_cbor_rejects_invalid_decoder_input(wire: str) -> None:
    with pytest.raises(CborError):
        decode_cbor(from_hex(wire))


def test_cbor_enforces_limits() -> None:
    too_deep = bytes([0x81] * (DEFAULT_MAX_CBOR_DEPTH + 1) + [0xF6])
    with pytest.raises(CborError, match="depth"):
        decode_cbor(too_deep)

    oversized_bytes = from_hex(f"5a{DEFAULT_MAX_CBOR_BYTE_LENGTH + 1:08x}")
    oversized_text = from_hex(f"7a{DEFAULT_MAX_CBOR_BYTE_LENGTH + 1:08x}")
    oversized_array = from_hex(f"9a{DEFAULT_MAX_CBOR_CONTAINER_LENGTH + 1:08x}")
    oversized_map = from_hex(f"ba{DEFAULT_MAX_CBOR_CONTAINER_LENGTH + 1:08x}")
    for wire in [oversized_bytes, oversized_text, oversized_array, oversized_map]:
        with pytest.raises(CborError, match="limit"):
            decode_cbor(wire)

    with pytest.raises(CborError, match="limit"):
        decode_cbor(from_hex("83010203"), max_container_length=2)
    with pytest.raises(CborError, match="limit"):
        decode_cbor(from_hex("626162"), max_byte_length=2)
    with pytest.raises(CborError, match="limit"):
        encode_cbor([1, 2, 3], max_container_length=2)
    with pytest.raises(CborError, match="limit"):
        encode_cbor("ab", max_byte_length=2)
