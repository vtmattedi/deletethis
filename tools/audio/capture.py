"""Record raw PCM from the ESP32 CAPTURE stream into a WAV file.

    python tools/audio/capture.py COM13 --label off --seconds 30

Produces::

    tools/audio/recordings/2026-09-30_091500_off.wav

WAV is preferred over a bare .raw dump because it carries the sample
rate and sample format, so every later tool can load it without being
told what it is looking at.

The 24-bit INMP441 samples are stored as 32-bit PCM, shifted up by 8
bits so that full scale is the full-scale of the container. Nothing is
lost, and ``sample / 2**31`` is then a correct normalised amplitude.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
import wave
from pathlib import Path

import numpy as np

from acstream import (
    DEFAULT_BAUD,
    AudioStream,
    ProtocolError,
    dbfs,
    list_ports,
)

DEFAULT_OUTDIR = Path(__file__).resolve().parent / "recordings"

# Progress is printed roughly this often, not per 32 ms frame.
STATUS_INTERVAL_SECONDS = 1.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record ESP32/INMP441 audio to a WAV file.",
    )

    parser.add_argument(
        "port",
        nargs="?",
        help="Serial port, e.g. COM13 or /dev/ttyUSB0",
    )

    parser.add_argument(
        "--label",
        default="unlabeled",
        help="State being recorded: off, fan, compressor, speech, ...",
    )

    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Stop after this many seconds (default: until Ctrl-C)",
    )

    parser.add_argument(
        "--baud",
        type=int,
        default=DEFAULT_BAUD,
        help=f"Serial baud rate (default: {DEFAULT_BAUD})",
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Directory for recordings (default: tools/audio/recordings)",
    )

    parser.add_argument(
        "--name",
        default=None,
        help="Explicit file name stem, overriding the timestamp+label",
    )

    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="List available serial ports and exit",
    )

    return parser


def output_path(args: argparse.Namespace) -> Path:
    if args.name:
        stem = args.name
    else:
        stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        stem = f"{stamp}_{args.label}"

    return args.outdir / f"{stem}.wav"


def open_wav(path: Path, sample_rate: int) -> wave.Wave_write:
    path.parent.mkdir(parents=True, exist_ok=True)

    handle = wave.open(str(path), "wb")

    handle.setnchannels(1)
    handle.setsampwidth(4)
    handle.setframerate(sample_rate)

    return handle


def to_wav_samples(samples: np.ndarray) -> np.ndarray:
    """Right-aligned 24-bit -> 32-bit container, full scale aligned."""
    clipped = np.clip(samples, -(1 << 23), (1 << 23) - 1)

    return (clipped.astype(np.int64) << 8).astype(np.int32)


def record(args: argparse.Namespace) -> int:
    path = output_path(args)

    try:
        stream = AudioStream(args.port, args.baud)
        stream.open()
    except ProtocolError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"error: could not open {args.port}: {error}", file=sys.stderr)
        return 1

    info = stream.info
    assert info is not None

    print(f"Stream: {info.describe()}")
    print(f"Writing: {path}")

    if args.seconds is not None:
        print(f"Recording {args.seconds:.1f} s. Ctrl-C stops early.")
    else:
        print("Recording until Ctrl-C.")

    handle = open_wav(path, info.sample_rate)

    total_samples = 0
    target_samples = (
        None
        if args.seconds is None
        else int(args.seconds * info.sample_rate)
    )

    started = time.monotonic()
    next_status = started + STATUS_INTERVAL_SECONDS
    interrupted = False

    try:
        for frame in stream.frames():
            samples = frame.samples

            if target_samples is not None:
                remaining = target_samples - total_samples

                if remaining <= 0:
                    break

                samples = samples[:remaining]

            handle.writeframes(to_wav_samples(samples).tobytes())
            total_samples += len(samples)

            now = time.monotonic()

            if now >= next_status:
                next_status = now + STATUS_INTERVAL_SECONDS
                print_status(stream, samples, total_samples, info.sample_rate)

            if target_samples is not None and total_samples >= target_samples:
                break
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted.")
    finally:
        handle.close()
        stream.close()

    duration = total_samples / info.sample_rate

    print()
    print(f"Saved {duration:.2f} s ({total_samples} samples) to {path}")

    if stream.lost_frames or stream.dropped_frames or stream.resyncs:
        print(
            f"Warning: {stream.lost_frames} frames lost in transit, "
            f"{stream.dropped_frames} dropped on the device, "
            f"{stream.resyncs} resyncs."
        )
        print(
            "If this is not occasional, the serial link is the bottleneck: "
            "raise SERIAL_BAUD in src/main.cpp or close other serial tools."
        )

    if duration < 1.0 and not interrupted:
        print("Warning: the recording is suspiciously short.", file=sys.stderr)
        return 1

    return 0


def print_status(
    stream: AudioStream,
    samples: np.ndarray,
    total_samples: int,
    sample_rate: int,
) -> None:
    block = samples.astype(np.float64)
    block -= block.mean()

    rms = float(np.sqrt(np.mean(block * block))) if block.size else 0.0

    print(
        f"  {total_samples / sample_rate:7.1f} s   "
        f"RMS {rms:10.0f}   "
        f"{dbfs(rms):7.2f} dBFS   "
        f"lost {stream.lost_frames}",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_ports:
        ports = list_ports()

        if ports:
            print("\n".join(ports))
        else:
            print("No serial ports found.")

        return 0

    if not args.port:
        parser.error("a serial port is required (or use --list-ports)")

    return record(args)


if __name__ == "__main__":
    raise SystemExit(main())
