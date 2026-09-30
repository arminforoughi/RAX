"""Dynamixel protocol 2.0: the CRC, the byte stuffing, and reading replies off a noisy line.

These are the parts that fail SILENTLY. A wrong CRC or a missed stuffed byte does not
raise -- it produces a packet the servo interprets as some other number, and the first
symptom is an arm moving somewhere nobody asked for. So the packets are checked against
values computed by hand from the specification rather than against the implementation's
own output, which would only prove it is consistent with itself.

The stuffing rule is the subtle one. ``FF FF FD`` is the header a device scans for, so
any payload containing it would resynchronise the receiver mid-packet; protocol 2.0
inserts an extra ``FD`` after that sequence and the receiver removes it. A goal position
of 0x00FDFFFF is a perfectly ordinary number that contains it.
"""

from __future__ import annotations

import pytest

from rax.robots.dynamixel import (
    ADDR,
    DynamixelBus,
    DynamixelError,
    _stuff,
    _unstuff,
    build_packet,
    crc16,
    parse_statuses,
)


class TestCRC:
    def test_matches_the_documented_reference_packet(self):
        # The broadcast PING worked through in the protocol 2.0 documentation, quoted
        # byte for byte. Checking against the published bytes is the point: comparing
        # this implementation's CRC with its own output would only prove it agrees with
        # itself, which is exactly what a wrong polynomial also does.
        assert build_packet(0xFE, 0x01).hex(" ") == "ff ff fd 00 fe 03 00 01 31 42"
        assert crc16(bytes.fromhex("fffffd00fe030001")) == 0x4231

    def test_crc_trails_every_packet(self):
        pkt = build_packet(3, 0x02, b"\x84\x00\x04\x00")
        assert crc16(pkt[:-2]) == (pkt[-2] | (pkt[-1] << 8))

    def test_crc_covers_the_whole_body(self):
        a = build_packet(1, 0x01)
        b = build_packet(2, 0x01)
        assert a[-2:] != b[-2:], "changing the id must change the CRC"

    def test_empty_input(self):
        assert crc16(b"") == 0


class TestStuffing:
    def test_leaves_ordinary_payloads_alone(self):
        for p in (b"", b"\x00", b"\x01\x02\x03", b"\xff", b"\xff\xff", b"\xfd\xfd"):
            assert _stuff(p) == p
            assert _unstuff(_stuff(p)) == p

    def test_inserts_after_the_header_sequence(self):
        assert _stuff(b"\xff\xff\xfd") == b"\xff\xff\xfd\xfd"

    def test_round_trips_a_payload_containing_the_header(self):
        # 0x00FDFFFF little-endian is FF FF FD 00 -- an ordinary goal position.
        raw = (0x00FDFFFF).to_bytes(4, "little")
        assert raw == b"\xff\xff\xfd\x00"
        assert _unstuff(_stuff(raw)) == raw
        assert len(_stuff(raw)) == 5

    def test_a_stuffed_payload_cannot_contain_a_bare_header(self):
        raw = b"\x01\xff\xff\xfd\x00\x02"
        assert b"\xff\xff\xfd\x00" not in _stuff(raw)

    def test_length_field_counts_the_stuffed_payload(self):
        pkt = build_packet(1, 0x03, b"\xff\xff\xfd\x00")
        length = pkt[5] | (pkt[6] << 8)
        assert length == len(pkt) - 7
        assert crc16(pkt[:-2]) == (pkt[-2] | (pkt[-1] << 8))


def _status(dxl_id, params=b"", error=0):
    """Build a well-formed status packet the way a servo would."""
    p = _stuff(params)
    length = len(p) + 4
    body = (b"\xff\xff\xfd\x00" + bytes([dxl_id, length & 0xFF, (length >> 8) & 0xFF,
                                         0x55, error]) + p)
    c = crc16(body)
    return body + bytes([c & 0xFF, (c >> 8) & 0xFF])


class TestParsing:
    def test_reads_one_reply(self):
        got = parse_statuses(_status(3, b"\x34\x12"))
        assert len(got) == 1
        assert got[0].dxl_id == 3
        assert got[0].params == b"\x34\x12"
        assert got[0].error == 0

    def test_reads_several_replies_in_order(self):
        buf = _status(2) + _status(5) + _status(7)
        assert [s.dxl_id for s in parse_statuses(buf)] == [2, 5, 7]

    def test_ignores_leading_rubbish(self):
        # A half-duplex line carries the tail of the previous reply.
        got = parse_statuses(b"\x00\xaa\x55" + _status(4, b"\x01"))
        assert [s.dxl_id for s in got] == [4]

    def test_ignores_a_truncated_trailing_packet(self):
        buf = _status(2, b"\x01\x02") + _status(9, b"\x03\x04")[:6]
        assert [s.dxl_id for s in parse_statuses(buf)] == [2]

    def test_rejects_a_corrupted_packet(self):
        bad = bytearray(_status(6, b"\x01\x02"))
        bad[-1] ^= 0xFF                      # break the CRC
        assert parse_statuses(bytes(bad)) == []

    def test_carries_the_error_byte(self):
        got = parse_statuses(_status(3, b"", error=0x04))
        assert got[0].error == 0x04

    def test_unstuffs_reply_parameters(self):
        raw = b"\xff\xff\xfd\x00"
        got = parse_statuses(_status(3, raw))
        assert got[0].params == raw


class TestBusSafety:
    def test_every_address_is_a_pair(self):
        for name, v in ADDR.items():
            assert len(v) == 2, name
            assert v[1] in (1, 2, 4), f"{name} has an odd length {v[1]}"

    def test_refuses_to_talk_before_opening(self):
        bus = DynamixelBus("COM-does-not-exist")
        assert not bus.is_open
        with pytest.raises(DynamixelError, match="not open"):
            bus.read(1, "present_position")

    def test_closing_an_unopened_bus_is_harmless(self):
        DynamixelBus("COM-does-not-exist").close()
