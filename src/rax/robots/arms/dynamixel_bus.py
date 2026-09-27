"""Dynamixel protocol 2.0 over a serial port, with no SDK behind it.

WHY THIS EXISTS. The X250's servos speak protocol 2.0; the SO-101's speak Feetech, and
``scservo_sdk`` is the only servo SDK installed on the rig. Rather than add a dependency
to talk to six motors, the protocol is small enough to write down: a four-byte header, a
length, an instruction, a CRC, and a byte-stuffing rule. What follows is that, and
nothing else.

It is deliberately NOT a robot. It reads and writes registers on numbered ids. Which id
is the elbow, what a tick means in degrees, and whether a pose is safe are all questions
for :mod:`rax.robots.arms.x250`, which is where the answers can be tested without a
serial port attached.

THE ONE SUBTLETY IS BYTE STUFFING. The header ``FF FF FD`` is what a device scans for,
so any payload that happens to contain it would resynchronise the receiver mid-packet.
Protocol 2.0 handles that by inserting an extra ``FD`` after any ``FF FF FD`` in the
payload, and the receiver removes it. Skip the stuffing and the bus works perfectly until
a goal position lands on the wrong value and the arm lunges -- so it is done here, in one
place, for every packet.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

__all__ = [
    "DynamixelBus",
    "DynamixelError",
    "ADDR",
    "crc16",
]

#: Control-table addresses for the X-series (XM430 / XL430), as (address, length).
ADDR = {
    "torque_enable": (64, 1),
    "led": (65, 1),
    "goal_position": (116, 4),
    "moving_speed": (112, 4),
    "present_position": (132, 4),
    "present_velocity": (128, 4),
    "present_load": (126, 2),
    "hardware_error": (70, 1),
    "model_number": (0, 2),
}

_INSTR_PING = 0x01
_INSTR_READ = 0x02
_INSTR_WRITE = 0x03
_INSTR_STATUS = 0x55
_BROADCAST = 0xFE
_HEADER = b"\xff\xff\xfd\x00"


class DynamixelError(RuntimeError):
    """A servo did not answer, or answered with an error flag set."""


def crc16(data: bytes) -> int:
    """CRC-16/IBM-3740, the one protocol 2.0 specifies.

    Written out rather than table-driven: six motors at a few hundred packets a second
    is nowhere near the speed where a 256-entry table earns its place, and a loop is
    easier to check against the specification.
    """
    crc = 0
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x8005) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def _stuff(payload: bytes) -> bytes:
    """Insert the extra 0xFD that stops a payload imitating the header."""
    out = bytearray()
    run = 0
    for b in payload:
        out.append(b)
        if run == 0 and b == 0xFF:
            run = 1
        elif run == 1 and b == 0xFF:
            run = 2
        elif run == 2 and b == 0xFD:
            out.append(0xFD)
            run = 0
        elif run == 2 and b == 0xFF:
            run = 2
        else:
            run = 1 if b == 0xFF else 0
    return bytes(out)


def _unstuff(payload: bytes) -> bytes:
    out = bytearray()
    i = 0
    while i < len(payload):
        out.append(payload[i])
        if (payload[i:i + 3] == b"\xff\xff\xfd" and i + 3 < len(payload)
                and payload[i + 3] == 0xFD):
            out.extend(payload[i + 1:i + 3])
            i += 4
            continue
        i += 1
    return bytes(out)


def build_packet(dxl_id: int, instruction: int, params: bytes = b"") -> bytes:
    """One protocol-2.0 instruction packet, stuffed and CRC'd."""
    p = _stuff(params)
    length = len(p) + 3                     # instruction + params + 2 CRC
    body = _HEADER + bytes([dxl_id & 0xFF, length & 0xFF, (length >> 8) & 0xFF,
                            instruction & 0xFF]) + p
    c = crc16(body)
    return body + bytes([c & 0xFF, (c >> 8) & 0xFF])


@dataclass
class Status:
    """One reply: which id sent it, its error byte, and the parameters."""

    dxl_id: int
    error: int
    params: bytes


def parse_statuses(buf: bytes) -> list[Status]:
    """Every complete status packet in a buffer, in order.

    Tolerates leading rubbish and partial trailing packets, because on a shared half
    duplex line both happen: the reply to the previous instruction can still be arriving,
    and a read can return mid-packet.
    """
    out: list[Status] = []
    i = 0
    while True:
        i = buf.find(_HEADER, i)
        if i < 0 or i + 9 > len(buf):
            break
        length = buf[i + 5] | (buf[i + 6] << 8)
        end = i + 7 + length
        if length < 4 or end > len(buf):
            break
        if buf[i + 7] == _INSTR_STATUS:
            body = buf[i:end - 2]
            got = buf[end - 2] | (buf[end - 1] << 8)
            if crc16(body) == got:
                out.append(Status(buf[i + 4], buf[i + 8],
                                  _unstuff(bytes(buf[i + 9:end - 2]))))
        i = end if end > i else i + 4
    return out


class DynamixelBus:
    """A serial line with Dynamixel servos on it.

    Opened lazily and closed explicitly, so constructing one touches no hardware and the
    class stays testable. Every call is synchronous: write, wait, read. At 1 Mbaud a
    round trip is well under a millisecond, and a control loop that reads six joints is
    nowhere near the rate where pipelining would matter.
    """

    def __init__(self, port: str, baudrate: int = 1_000_000, timeout: float = 0.2):
        self.port, self.baudrate, self.timeout = port, int(baudrate), float(timeout)
        self._sp = None

    # ---- lifecycle -------------------------------------------------------------
    def open(self) -> "DynamixelBus":
        import serial                                   # optional at import time

        if self._sp is None:
            self._sp = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
        return self

    def close(self) -> None:
        if self._sp is not None:
            try:
                self._sp.close()
            finally:
                self._sp = None

    @property
    def is_open(self) -> bool:
        return self._sp is not None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    # ---- transport -------------------------------------------------------------
    def _txrx(self, pkt: bytes, wait: float, expect: int) -> list[Status]:
        if self._sp is None:
            raise DynamixelError("bus is not open")
        self._sp.reset_input_buffer()
        self._sp.write(pkt)
        self._sp.flush()
        # Collect until the expected number of replies arrive or the window closes.
        # Fixed sleeps either waste milliseconds on every call or truncate a slow reply;
        # polling costs neither.
        deadline = time.time() + wait
        buf = b""
        while time.time() < deadline:
            chunk = self._sp.read(self._sp.in_waiting or 1)
            if chunk:
                buf += chunk
                if len(parse_statuses(buf)) >= expect:
                    break
            else:
                time.sleep(0.001)
        return parse_statuses(buf)

    # ---- instructions ----------------------------------------------------------
    def ping(self, dxl_id: int = _BROADCAST, wait: float = 0.25) -> dict[int, int]:
        """``{id: model_number}`` for whoever answers. Reads nothing else and moves nothing."""
        n = 253 if dxl_id == _BROADCAST else 1
        got = self._txrx(build_packet(dxl_id, _INSTR_PING), wait, n)
        return {s.dxl_id: (s.params[0] | (s.params[1] << 8)) if len(s.params) >= 2 else 0
                for s in got}

    def read(self, dxl_id: int, item: str, wait: float = 0.06) -> int:
        """One control-table item from one servo, as an unsigned integer."""
        addr, length = ADDR[item]
        got = self._txrx(build_packet(dxl_id, _INSTR_READ,
                                      bytes([addr & 0xFF, addr >> 8, length, 0])),
                         wait, 1)
        for s in got:
            if s.dxl_id == dxl_id and len(s.params) >= length:
                if s.error:
                    raise DynamixelError(
                        f"servo {dxl_id} reports hardware error {s.error:#04x} "
                        f"while reading {item}")
                return int.from_bytes(s.params[:length], "little")
        raise DynamixelError(f"servo {dxl_id} did not answer a read of {item}")

    def write(self, dxl_id: int, item: str, value: int, wait: float = 0.06) -> None:
        """Write one control-table item. THIS CAN MOVE THE ARM."""
        addr, length = ADDR[item]
        payload = bytes([addr & 0xFF, addr >> 8]) + \
            int(value).to_bytes(length, "little", signed=False)
        got = self._txrx(build_packet(dxl_id, _INSTR_WRITE, payload), wait, 1)
        for s in got:
            if s.dxl_id == dxl_id:
                if s.error:
                    raise DynamixelError(
                        f"servo {dxl_id} reports hardware error {s.error:#04x} "
                        f"while writing {item}={value}")
                return
        raise DynamixelError(f"servo {dxl_id} did not acknowledge {item}={value}")

    def read_positions(self, ids, wait: float = 0.12) -> dict[int, int]:
        """Present position in ticks for each id, read one at a time.

        One at a time rather than a sync read: six round trips at 1 Mbaud cost under a
        millisecond of wire time each, and a per-servo read reports WHICH servo failed.
        A sync read that comes back short just says the batch was short, which is the
        less useful half of the message when a connector works loose.
        """
        out = {}
        for i in ids:
            try:
                out[int(i)] = self.read(int(i), "present_position", wait=wait)
            except DynamixelError:
                continue
        return out
