from __future__ import annotations

import pytest

from pi_protocol import DEFAULT_MAX_FRAME_LENGTH, FrameDecoder, FrameError, assert_complete_frame, encode_frame


def concatenate(*chunks: bytes) -> bytes:
    return b"".join(chunks)


def test_frame_prefixes_payloads_with_four_byte_big_endian_length() -> None:
    assert encode_frame(b"\xaa\xbb\xcc") == b"\x00\x00\x00\x03\xaa\xbb\xcc"
    assert encode_frame(b"") == b"\x00\x00\x00\x00"


def test_frame_validates_one_complete_bounded_frame() -> None:
    assert_complete_frame(b"\x00\x00\x00\x02\x01\x02", max_frame_length=2)
    with pytest.raises(FrameError, match="complete"):
        assert_complete_frame(b"\x00\x00\x00\x02\x01")
    with pytest.raises(FrameError, match="exactly"):
        assert_complete_frame(b"\x00\x00\x00\x01\x01\x02")
    with pytest.raises(FrameError, match="limit"):
        assert_complete_frame(b"\x00\x00\x00\x03\x01\x02\x03", max_frame_length=2)


def test_frame_decodes_fragmented_coalesced_and_empty_frames_in_order() -> None:
    wire = concatenate(encode_frame(b"\x01\x02\x03"), encode_frame(b""), encode_frame(b"\x04"))
    decoder = FrameDecoder()
    frames: list[bytes] = []
    for byte in wire:
        frames.extend(decoder.push(bytes([byte])))
    decoder.end()
    assert frames == [b"\x01\x02\x03", b"", b"\x04"]

    coalesced = FrameDecoder()
    assert coalesced.push(wire) == frames
    coalesced.end()


def test_frame_assembles_payloads_spanning_internal_blocks() -> None:
    payload = bytes(index % 251 for index in range(70_000))
    wire = encode_frame(payload)
    decoder = FrameDecoder()
    frames = [*decoder.push(wire[:101]), *decoder.push(wire[101:65_541]), *decoder.push(wire[65_541:])]
    decoder.end()
    assert frames == [payload]


def test_frame_handles_every_split_point_across_a_frame() -> None:
    wire = encode_frame(b"\x0a\x14\x1e\x28")
    for split in range(len(wire) + 1):
        decoder = FrameDecoder()
        frames = [*decoder.push(wire[:split]), *decoder.push(wire[split:])]
        decoder.end()
        assert frames == [b"\x0a\x14\x1e\x28"]


def test_frame_copies_payload_bytes_instead_of_aliasing_input_chunks() -> None:
    chunk = bytearray(encode_frame(b"\x01\x02\x03"))
    decoder = FrameDecoder()
    frames = decoder.push(chunk)
    chunk[:] = b"\x09" * len(chunk)
    assert frames == [b"\x01\x02\x03"]


def test_frame_accepts_empty_chunks_and_clean_empty_stream() -> None:
    decoder = FrameDecoder()
    assert decoder.push(b"") == []
    decoder.end()


@pytest.mark.parametrize("wire", [b"\x00\x00\x00", b"\x00\x00\x00\x02\x01"])
def test_frame_rejects_truncated_stream_at_end(wire: bytes) -> None:
    decoder = FrameDecoder()
    assert decoder.push(wire) == []
    with pytest.raises(FrameError):
        decoder.end()


def test_frame_rejects_oversized_declared_length_as_soon_as_header_is_complete() -> None:
    decoder = FrameDecoder(max_frame_length=3)
    with pytest.raises(FrameError, match="limit"):
        decoder.push(b"\x00\x00\x00\x04")
    with pytest.raises(FrameError, match="failed"):
        decoder.push(b"\x01")


def test_frame_accepts_frame_exactly_at_configured_maximum() -> None:
    decoder = FrameDecoder(max_frame_length=3)
    assert decoder.push(encode_frame(b"\x01\x02\x03")) == [b"\x01\x02\x03"]
    decoder.end()


def test_frame_cannot_be_pushed_after_end() -> None:
    decoder = FrameDecoder()
    decoder.end()
    with pytest.raises(FrameError, match="ended"):
        decoder.push(b"")
    with pytest.raises(FrameError, match="ended"):
        decoder.end()


@pytest.mark.parametrize("max_frame_length", [-1, 1.5, DEFAULT_MAX_FRAME_LENGTH * 1000])
def test_frame_rejects_invalid_maximum_frame_length(max_frame_length: int | float) -> None:
    with pytest.raises(ValueError):
        FrameDecoder(max_frame_length=max_frame_length)
