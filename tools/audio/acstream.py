"""Shared ESP32 <-> PC audio stream protocol.

Mirrors the CAPTURE mode protocol defined in ``src/main.cpp``.

Wire format, little-endian throughout::

    StreamHeader (32 bytes, once)
        char     magic[4]      "ACD1"
        uint16   version
        uint16   header_size
        uint32   sample_rate
        uint16   format         1 = signed 32-bit LE, 24 valid bits
        uint16   channels
        uint16   bits_valid
        uint16   frame_samples
        uint32   reserved x3

    FrameHeader (12 bytes, repeating)
        char     magic[4]      "ACDF"
        uint32   sequence
        uint16   samples
        uint16   dropped       frames lost since the previous frame

    ...followed by ``samples`` int32 little-endian PCM values.

Samples arrive right-aligned: the INMP441's 24-bit word already
shifted down out of the 32-bit I2S slot, so full scale is 2**23.
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass

import numpy as np
import serial

DEFAULT_BAUD = 921600

STREAM_MAGIC = b"ACD1"
FRAME_MAGIC = b"ACDF"

STREAM_HEADER_FORMAT = "<4sHHIHHHHIII"
STREAM_HEADER_SIZE = struct.calcsize(STREAM_HEADER_FORMAT)

FRAME_HEADER_FORMAT = "<4sIHH"
FRAME_HEADER_SIZE = struct.calcsize(FRAME_HEADER_FORMAT)

FORMAT_PCM_S32LE = 1

# The INMP441 delivers 24 bits, so this is digital full scale.
FULL_SCALE = float(1 << 23)

assert STREAM_HEADER_SIZE == 32, STREAM_HEADER_SIZE
assert FRAME_HEADER_SIZE == 12, FRAME_HEADER_SIZE


class ProtocolError(RuntimeError):
    """The device sent something that is not a valid stream."""


@dataclass(frozen=True)
class StreamInfo:
    version: int
    sample_rate: int
    sample_format: int
    channels: int
    bits_valid: int
    frame_samples: int

    @property
    def frame_seconds(self) -> float:
        return self.frame_samples / self.sample_rate

    def describe(self) -> str:
        return (
            f"protocol v{self.version}  "
            f"{self.sample_rate} Hz  "
            f"{self.channels} ch  "
            f"{self.bits_valid}-bit  "
            f"{self.frame_samples} samples/frame "
            f"({self.frame_seconds * 1000:.0f} ms)"
        )


@dataclass
class Frame:
    sequence: int
    dropped: int
    samples: np.ndarray  # int32, right-aligned 24-bit


class AudioStream:
    """Drives the ESP32 into CAPTURE mode and reads PCM frames.

    Use as a context manager::

        with AudioStream("COM13") as stream:
            print(stream.info.describe())
            for frame in stream.frames():
                ...
    """

    def __init__(
        self,
        port: str,
        baud: int = DEFAULT_BAUD,
        *,
        boot_wait: float = 2.0,
        timeout: float = 2.0,
        verbose: bool = True,
    ) -> None:
        self.port = port
        self.baud = baud
        self.boot_wait = boot_wait
        self.timeout = timeout
        self.verbose = verbose

        self.serial: serial.Serial | None = None
        self.info: StreamInfo | None = None

        # Bytes read ahead while scanning for a header. Reads are
        # served from here first, otherwise every scan would throw
        # away the payload it already pulled off the port and the
        # stream would desync on the very next frame.
        self._pending = bytearray()

        # Sequence bookkeeping, so callers can report loss.
        self.expected_sequence: int | None = None
        self.dropped_frames = 0
        self.lost_frames = 0
        self.resyncs = 0

    # ---------------------------------------------------- lifecycle

    def __enter__(self) -> "AudioStream":
        self.open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def open(self) -> None:
        self._log(f"Opening {self.port} at {self.baud} baud...")

        self.serial = serial.Serial(
            self.port,
            self.baud,
            timeout=self.timeout,
        )

        # Opening the port toggles DTR/RTS on most USB-serial
        # adapters, which resets the ESP32. Wait for its banner
        # to finish rather than talking over the bootloader.
        time.sleep(self.boot_wait)

        self._discard_input()

        self.info = self._start_capture()

    def close(self) -> None:
        if self.serial is None:
            return

        try:
            if self.serial.is_open:
                # Ask the device to stop streaming so the next run
                # does not open onto a torrent of stale PCM.
                self.serial.write(b"s")
                self.serial.flush()
                time.sleep(0.1)
                self._discard_input()
        except serial.SerialException:
            pass
        finally:
            self.serial.close()
            self.serial = None

    # ----------------------------------------------------- handshake

    def _start_capture(self) -> StreamInfo:
        assert self.serial is not None

        deadline = time.monotonic() + 10.0
        attempt = 0

        while time.monotonic() < deadline:
            # 's' first, in case a previous run left it streaming.
            self.serial.write(b"sc")
            self.serial.flush()

            attempt += 1
            self._log(f"Requesting CAPTURE mode (attempt {attempt})...")

            # Anything before the magic is the boot banner or leftover
            # TEXT mode output, so it is simply skipped.
            raw = self._scan_for(
                STREAM_MAGIC, STREAM_HEADER_SIZE, timeout=2.0
            )

            if raw is not None:
                return self._parse_stream_header(raw)

        raise ProtocolError(
            "No stream header received. Check that the firmware in "
            f"src/main.cpp is flashed and that {self.baud} baud matches "
            "SERIAL_BAUD."
        )

    def _scan_for(
        self, magic: bytes, size: int, timeout: float
    ) -> bytes | None:
        """Find ``magic`` in the stream and return the header it starts.

        Whatever was read past the header is pushed back, so the caller
        can go straight on to reading the payload.
        """
        assert self.serial is not None

        window = self._pending
        self._pending = bytearray()

        deadline = time.monotonic() + timeout

        while True:
            index = window.find(magic)

            if index >= 0:
                del window[:index]

                while len(window) < size:
                    chunk = self.serial.read(size - len(window))

                    if not chunk:
                        break

                    window += chunk

                if len(window) < size:
                    self._pending = window
                    return None

                header = bytes(window[:size])
                self._pending = window[size:]

                return header

            # Keep only enough tail to catch a magic split across reads.
            keep = len(magic) - 1

            if len(window) > keep:
                del window[: len(window) - keep]

            if time.monotonic() >= deadline:
                self._pending = window
                return None

            chunk = self.serial.read(max(1, self.serial.in_waiting))

            if chunk:
                window += chunk

    @staticmethod
    def _parse_stream_header(raw: bytes) -> StreamInfo:
        (
            magic,
            version,
            header_size,
            sample_rate,
            sample_format,
            channels,
            bits_valid,
            frame_samples,
            _r0,
            _r1,
            _r2,
        ) = struct.unpack(STREAM_HEADER_FORMAT, raw)

        if magic != STREAM_MAGIC:
            raise ProtocolError(f"Bad stream magic: {magic!r}")

        if header_size != STREAM_HEADER_SIZE:
            raise ProtocolError(
                f"Unexpected header size {header_size}, "
                f"this tool speaks {STREAM_HEADER_SIZE}"
            )

        if sample_format != FORMAT_PCM_S32LE:
            raise ProtocolError(
                f"Unsupported sample format {sample_format}"
            )

        if channels != 1:
            raise ProtocolError(
                f"Only mono is supported, device sent {channels} channels"
            )

        return StreamInfo(
            version=version,
            sample_rate=sample_rate,
            sample_format=sample_format,
            channels=channels,
            bits_valid=bits_valid,
            frame_samples=frame_samples,
        )

    # --------------------------------------------------------- frames

    def frames(self):
        """Yield :class:`Frame` objects until the stream is closed."""
        while True:
            frame = self.read_frame()

            if frame is None:
                return

            yield frame

    def read_frame(self) -> Frame | None:
        assert self.serial is not None and self.info is not None

        header = self._read_frame_header()

        if header is None:
            return None

        sequence, dropped, sample_count = header

        payload = self._read_exact(sample_count * 4)

        if len(payload) < sample_count * 4:
            return None

        samples = np.frombuffer(payload, dtype="<i4")

        if self.expected_sequence is not None:
            gap = sequence - self.expected_sequence

            if gap > 0:
                self.lost_frames += gap

        self.expected_sequence = sequence + 1
        self.dropped_frames += dropped

        return Frame(
            sequence=sequence,
            dropped=dropped,
            samples=samples,
        )

    def _read_frame_header(self) -> tuple[int, int, int] | None:
        """Read the next frame header, resynchronising if needed."""
        assert self.serial is not None and self.info is not None

        raw = self._read_exact(FRAME_HEADER_SIZE)

        if len(raw) < FRAME_HEADER_SIZE:
            return None

        if raw[:4] != FRAME_MAGIC:
            # The magic may start inside what we just read, so put it
            # back rather than searching from after it.
            self._pending[:0] = raw

            raw = self._resync()

            if raw is None:
                return None

        magic, sequence, samples, dropped = struct.unpack(
            FRAME_HEADER_FORMAT, raw
        )

        if magic != FRAME_MAGIC:
            return None

        # A corrupt length would make us read for a very long time.
        if samples == 0 or samples > 16384:
            raise ProtocolError(f"Implausible frame length: {samples}")

        return sequence, dropped, samples

    def _resync(self) -> bytes | None:
        """Hunt for the next frame magic after a desync."""
        self.resyncs += 1
        self._log("Lost frame sync, resynchronising...")

        return self._scan_for(FRAME_MAGIC, FRAME_HEADER_SIZE, timeout=5.0)

    def _read_exact(self, count: int) -> bytes:
        """Read exactly ``count`` bytes, or fewer on timeout."""
        assert self.serial is not None

        buffer = bytearray()

        if self._pending:
            take = min(count, len(self._pending))
            buffer += self._pending[:take]
            del self._pending[:take]

        while len(buffer) < count:
            chunk = self.serial.read(count - len(buffer))

            if not chunk:
                break

            buffer += chunk

        return bytes(buffer)

    def _discard_input(self) -> None:
        assert self.serial is not None

        self._pending.clear()
        self.serial.reset_input_buffer()

    # ---------------------------------------------------------- misc

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)


def to_float(samples: np.ndarray) -> np.ndarray:
    """Convert right-aligned 24-bit samples to floats in [-1, 1)."""
    return samples.astype(np.float64) / FULL_SCALE


def dbfs(rms: float) -> float:
    """RMS in raw 24-bit counts -> dBFS. Silence clamps to -120."""
    if rms <= 0.0:
        return -120.0

    return max(-120.0, 20.0 * np.log10(rms / FULL_SCALE))


def list_ports() -> list[str]:
    from serial.tools import list_ports as _list_ports

    return [
        f"{port.device}  {port.description}"
        for port in _list_ports.comports()
    ]
