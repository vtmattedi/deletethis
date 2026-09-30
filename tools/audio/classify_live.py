"""Live OFF / FAN / COMPRESSOR classification from the ESP32 stream.

    python tools/audio/classify_live.py COM9

    python tools/audio/classify_live.py COM9 \
        --compressor-threshold -48 \
        --fan-mid-threshold -63 \
        --fan-high-threshold -68 \
        --hold-seconds 2

Reads the same 16 kHz mono PCM stream as capture.py, over the same
protocol, with no firmware changes: the ESP32 still just streams audio
and every decision is made here, where a threshold can be changed
without a reflash.

Features are computed on exactly the same terms as analyze.py --
1024-sample windows, 50% overlap, Hamming, spectral band power in dBFS
plus true time-domain RMS -- so a threshold read off an analyze.py
table means the same thing here. ``test_equivalence`` in the tests
below pins that down.

The rules are hierarchical and deliberately simple:

    stage 1   30-80 Hz persistently high     -> COMPRESSOR
    stage 2   500-1k and/or 1k-2k high       -> FAN
              otherwise                      -> OFF

Nothing is published from a single FFT window. Features are smoothed by
a rolling median, and the resulting candidate must hold for a couple of
seconds before it becomes the reported state. That is what stops a door
slam or a printer from being read as a compressor.

The default thresholds are exploratory starting points, not measured
values. Record OFF / FAN / COMPRESSOR, run analyze.py, and replace them.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import get_window

sys.path.insert(0, str(Path(__file__).resolve().parent))

from acstream import (  # noqa: E402
    DEFAULT_BAUD,
    FULL_SCALE,
    AudioStream,
    ProtocolError,
    list_ports,
)
from analyze import (  # noqa: E402
    BANDS,
    DB_FLOOR,
    power_to_db,
)

# The states this classifier can report, in escalation order.
OFF = "OFF"
FAN = "FAN"
COMPRESSOR = "COMPRESSOR"

# Bands the v1 rules need, plus the two diagnostics. Looked up by name
# in analyze.BANDS so the edges cannot drift apart from the offline
# tool; a rename there becomes a KeyError here rather than a silently
# different band.
RULE_BANDS = ["30-80", "500-1k", "1k-2k"]
DIAGNOSTIC_BANDS = ["200-1200"]

DEFAULT_NFFT = 1024
DEFAULT_OVERLAP = 0.5

# Exploratory starting points. See the module docstring.
DEFAULT_COMPRESSOR_THRESHOLD = -48.0
DEFAULT_FAN_MID_THRESHOLD = -63.0
DEFAULT_FAN_HIGH_THRESHOLD = -68.0

DEFAULT_HOLD_SECONDS = 2.0
DEFAULT_MEDIAN_SECONDS = 0.5
DEFAULT_REFRESH_HZ = 5.0


def band_edges(name: str) -> tuple[float, float]:
    for band_name, low, high in BANDS:
        if band_name == name:
            return low, high

    raise KeyError(
        f"analyze.BANDS has no band {name!r}; "
        f"available: {[n for n, _, _ in BANDS]}"
    )


# ---------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------


def window_power(
    block: np.ndarray,
    window: np.ndarray,
) -> np.ndarray:
    """One window -> per-bin power.

    Matches scipy.signal.spectrogram(detrend="constant",
    scaling="spectrum", mode="psd"), which is what analyse_file() uses,
    so one live window equals one column of an offline spectrogram.
    """
    centred = block - block.mean()

    spectrum = np.fft.rfft(centred * window)

    power = (
        spectrum.real ** 2 + spectrum.imag ** 2
    ) / window.sum() ** 2

    # One-sided: every bin but DC, and Nyquist when the length is even,
    # stands in for its mirror image too.
    if block.size % 2 == 0:
        power[1:-1] *= 2.0
    else:
        power[1:] *= 2.0

    return power


@dataclass
class Features:
    """One analysis window."""

    time: float
    rms_db: float
    bands: dict[str, float]


class FeatureExtractor:
    """Turns a stream of frames into overlapping analysis windows."""

    def __init__(
        self,
        sample_rate: int,
        nfft: int = DEFAULT_NFFT,
        overlap: float = DEFAULT_OVERLAP,
    ) -> None:
        self.sample_rate = sample_rate
        self.nfft = nfft
        self.hop = nfft - int(nfft * overlap)

        # Periodic, as scipy.signal.spectrogram uses. np.hamming is
        # symmetric and would give slightly different numbers.
        self.window = get_window("hamming", nfft)

        self.freqs = np.fft.rfftfreq(nfft, 1.0 / sample_rate)

        self.masks = {
            name: (self.freqs >= low) & (self.freqs < high)
            for name, low, high in (
                (name, *band_edges(name))
                for name in RULE_BANDS + DIAGNOSTIC_BANDS
            )
        }

        self.buffer = np.zeros(0, dtype=np.float64)
        self.samples_seen = 0

    def push(self, samples: np.ndarray) -> list[Features]:
        """Add a frame, return every window it completes."""
        block = samples.astype(np.float64) / FULL_SCALE

        self.buffer = (
            block
            if self.buffer.size == 0
            else np.concatenate((self.buffer, block))
        )

        produced: list[Features] = []

        while self.buffer.size >= self.nfft:
            window_samples = self.buffer[: self.nfft]

            produced.append(
                self._analyse(window_samples)
            )

            self.buffer = self.buffer[self.hop :]
            self.samples_seen += self.hop

        return produced

    def _analyse(self, block: np.ndarray) -> Features:
        centred = block - block.mean()

        rms_db = float(
            power_to_db(float(np.mean(centred * centred)))
        )

        power = window_power(block, self.window)

        bands = {}

        for name, mask in self.masks.items():
            if not mask.any():
                bands[name] = DB_FLOOR
                continue

            bands[name] = float(
                power_to_db(float(power[mask].sum()))
            )

        return Features(
            time=self.samples_seen / self.sample_rate,
            rms_db=rms_db,
            bands=bands,
        )


# ---------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------


@dataclass
class Thresholds:
    compressor: float = DEFAULT_COMPRESSOR_THRESHOLD
    fan_mid: float = DEFAULT_FAN_MID_THRESHOLD
    fan_high: float = DEFAULT_FAN_HIGH_THRESHOLD
    fan_require_both: bool = False


def classify(
    bands: dict[str, float],
    thresholds: Thresholds,
) -> str:
    """Hierarchical rules over already-smoothed band levels."""

    # Stage 1. The compressor is the only thing in a room that puts
    # sustained energy this low, so it is tested first and wins
    # outright.
    if bands["30-80"] >= thresholds.compressor:
        return COMPRESSOR

    # Stage 2. Airflow is broadband and sits in the mid and upper mids.
    mid = bands["500-1k"] >= thresholds.fan_mid
    high = bands["1k-2k"] >= thresholds.fan_high

    if thresholds.fan_require_both:
        return FAN if (mid and high) else OFF

    return FAN if (mid or high) else OFF


# ---------------------------------------------------------------------
# Temporal smoothing
# ---------------------------------------------------------------------


@dataclass
class Decision:
    candidate: str
    stable_seconds: float
    state: str | None
    changed: bool
    smoothed: dict[str, float]
    rms_db: float


class Smoother:
    """Rolling median of features, then a hold before publishing.

    Two separate mechanisms, doing two different jobs:

    * the median over ``median_seconds`` removes single-window spikes,
      so one loud transient cannot move the candidate at all;
    * the hold over ``hold_seconds`` means a candidate has to persist
      before it is published, so a genuinely new but short-lived sound
      (someone talking near the mic) never becomes a reported state.
    """

    def __init__(
        self,
        thresholds: Thresholds,
        window_rate: float,
        median_seconds: float = DEFAULT_MEDIAN_SECONDS,
        hold_seconds: float = DEFAULT_HOLD_SECONDS,
        history_seconds: float = 2.0,
    ) -> None:
        self.thresholds = thresholds
        self.hold_seconds = hold_seconds

        median_windows = max(1, round(median_seconds * window_rate))

        self.history: deque[Features] = deque(
            maxlen=max(
                median_windows,
                round(history_seconds * window_rate),
            )
        )

        self.median_windows = median_windows

        self.candidate: str | None = None
        self.candidate_since = 0.0
        self.state: str | None = None

    def update(self, features: Features) -> Decision:
        self.history.append(features)

        recent = list(self.history)[-self.median_windows :]

        smoothed = {
            name: float(
                np.median([item.bands[name] for item in recent])
            )
            for name in features.bands
        }

        rms_db = float(
            np.median([item.rms_db for item in recent])
        )

        candidate = classify(smoothed, self.thresholds)

        if candidate != self.candidate:
            self.candidate = candidate
            self.candidate_since = features.time

        stable = features.time - self.candidate_since

        changed = False

        if candidate != self.state and stable >= self.hold_seconds:
            self.state = candidate
            changed = True

        return Decision(
            candidate=candidate,
            stable_seconds=stable,
            state=self.state,
            changed=changed,
            smoothed=smoothed,
            rms_db=rms_db,
        )

    def band_spread(self, name: str) -> float:
        """Standard deviation of a band over the kept history.

        A compressor is steady, so a high spread in 30-80 alongside a
        high level is a hint that something transient is being read as
        one.
        """
        if len(self.history) < 2:
            return 0.0

        return float(
            np.std([item.bands[name] for item in self.history])
        )


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------


class Display:
    def __init__(
        self,
        refresh_hz: float,
        diagnostics: bool,
        stream_out=sys.stdout,
    ) -> None:
        self.interval = 1.0 / refresh_hz if refresh_hz > 0 else 0.0
        self.diagnostics = diagnostics
        self.out = stream_out

        self.next_refresh = 0.0
        self.interactive = stream_out.isatty()
        self.line_length = 0

    def status(
        self,
        decision: Decision,
        spread: float,
        force: bool = False,
    ) -> None:
        now = time.monotonic()

        if not force and now < self.next_refresh:
            return

        self.next_refresh = now + self.interval

        parts = [
            f"STATE={decision.state or '--'}",
            f"30-80={decision.smoothed['30-80']:.1f}",
            f"500-1k={decision.smoothed['500-1k']:.1f}",
            f"1k-2k={decision.smoothed['1k-2k']:.1f}",
            f"RMS={decision.rms_db:.1f}",
            f"candidate={decision.candidate}",
            f"stable={decision.stable_seconds:.1f}s",
        ]

        if self.diagnostics:
            parts.insert(
                4,
                f"200-1200={decision.smoothed['200-1200']:.1f}",
            )
            parts.append(f"30-80sd={spread:.1f}")

        line = "  ".join(parts)

        if self.interactive:
            padding = max(0, self.line_length - len(line))
            self.out.write("\r" + line + " " * padding)
            self.out.flush()
            self.line_length = len(line)
        else:
            self.out.write(line + "\n")
            self.out.flush()

    def transition(self, previous: str | None, decision: Decision) -> None:
        stamp = time.strftime("%H:%M:%S")

        if self.interactive:
            self.out.write("\r" + " " * self.line_length + "\r")
            self.line_length = 0

        self.out.write(
            f"{stamp}  {previous or '--'} -> {decision.state}"
            f"   30-80={decision.smoothed['30-80']:.1f}"
            f"  500-1k={decision.smoothed['500-1k']:.1f}"
            f"  1k-2k={decision.smoothed['1k-2k']:.1f}\n"
        )
        self.out.flush()

    def finish(self) -> None:
        if self.interactive and self.line_length:
            self.out.write("\n")
            self.out.flush()
            self.line_length = 0


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Live OFF / FAN / COMPRESSOR classification from the "
            "ESP32 audio stream."
        ),
    )

    parser.add_argument(
        "port",
        nargs="?",
        help="Serial port, e.g. COM9",
    )

    parser.add_argument(
        "--baud",
        type=int,
        default=DEFAULT_BAUD,
        help=f"Serial baud rate (default: {DEFAULT_BAUD})",
    )

    parser.add_argument(
        "--nfft",
        type=int,
        default=DEFAULT_NFFT,
        help=f"FFT size (default: {DEFAULT_NFFT}, as analyze.py)",
    )

    parser.add_argument(
        "--overlap",
        type=float,
        default=DEFAULT_OVERLAP,
        help=f"Window overlap (default: {DEFAULT_OVERLAP})",
    )

    thresholds = parser.add_argument_group("thresholds (dBFS)")

    thresholds.add_argument(
        "--compressor-threshold",
        type=float,
        default=DEFAULT_COMPRESSOR_THRESHOLD,
        help=(
            "30-80 Hz level at or above which the state is COMPRESSOR "
            f"(default: {DEFAULT_COMPRESSOR_THRESHOLD:g})"
        ),
    )

    thresholds.add_argument(
        "--fan-mid-threshold",
        type=float,
        default=DEFAULT_FAN_MID_THRESHOLD,
        help=(
            "500-1k Hz level counting towards FAN "
            f"(default: {DEFAULT_FAN_MID_THRESHOLD:g})"
        ),
    )

    thresholds.add_argument(
        "--fan-high-threshold",
        type=float,
        default=DEFAULT_FAN_HIGH_THRESHOLD,
        help=(
            "1k-2k Hz level counting towards FAN "
            f"(default: {DEFAULT_FAN_HIGH_THRESHOLD:g})"
        ),
    )

    thresholds.add_argument(
        "--fan-require",
        choices=["either", "both"],
        default="either",
        help=(
            "Whether one or both fan bands must be over threshold "
            "(default: either)"
        ),
    )

    timing = parser.add_argument_group("smoothing")

    timing.add_argument(
        "--hold-seconds",
        type=float,
        default=DEFAULT_HOLD_SECONDS,
        help=(
            "How long a candidate must persist before it is published "
            f"(default: {DEFAULT_HOLD_SECONDS:g})"
        ),
    )

    timing.add_argument(
        "--median-seconds",
        type=float,
        default=DEFAULT_MEDIAN_SECONDS,
        help=(
            "Rolling median length applied to features "
            f"(default: {DEFAULT_MEDIAN_SECONDS:g})"
        ),
    )

    output = parser.add_argument_group("output")

    output.add_argument(
        "--refresh-hz",
        type=float,
        default=DEFAULT_REFRESH_HZ,
        help=(
            "Status refresh rate, not every window "
            f"(default: {DEFAULT_REFRESH_HZ:g})"
        ),
    )

    output.add_argument(
        "--diagnostics",
        action="store_true",
        help="Also show 200-1200 Hz and the 30-80 Hz rolling spread",
    )

    output.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Append per-window features and state to this CSV",
    )

    output.add_argument(
        "--list-ports",
        action="store_true",
        help="List available serial ports and exit",
    )

    return parser


def print_settings(args: argparse.Namespace, window_rate: float) -> None:
    print(
        f"Windows: {args.nfft} samples, "
        f"{args.overlap:.0%} overlap, Hamming, "
        f"{window_rate:.1f}/s"
    )
    print(
        f"Rules:   COMPRESSOR if 30-80 >= "
        f"{args.compressor_threshold:g} dB"
    )
    print(
        f"         FAN if 500-1k >= {args.fan_mid_threshold:g} "
        f"{'and' if args.fan_require == 'both' else 'or'} "
        f"1k-2k >= {args.fan_high_threshold:g} dB"
    )
    print(
        f"Smooth:  {args.median_seconds:g}s median, "
        f"{args.hold_seconds:g}s hold before publishing"
    )
    print()
    print(
        "These thresholds are exploratory starting points. Check them "
        "against analyze.py output from real recordings."
    )
    print()
    print("Ctrl-C to stop.")
    print()


def run(args: argparse.Namespace) -> int:
    try:
        stream = AudioStream(args.port, args.baud)
        stream.open()
    except ProtocolError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(
            f"error: could not open {args.port}: {error}",
            file=sys.stderr,
        )
        return 1

    info = stream.info
    assert info is not None

    print(f"Stream: {info.describe()}")

    extractor = FeatureExtractor(
        info.sample_rate,
        args.nfft,
        args.overlap,
    )

    window_rate = info.sample_rate / extractor.hop

    print_settings(args, window_rate)

    thresholds = Thresholds(
        compressor=args.compressor_threshold,
        fan_mid=args.fan_mid_threshold,
        fan_high=args.fan_high_threshold,
        fan_require_both=args.fan_require == "both",
    )

    smoother = Smoother(
        thresholds,
        window_rate=window_rate,
        median_seconds=args.median_seconds,
        hold_seconds=args.hold_seconds,
    )

    display = Display(args.refresh_hz, args.diagnostics)

    log_handle = None
    log_writer = None

    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        log_handle = args.log.open("w", newline="", encoding="utf-8")
        log_writer = csv.writer(log_handle)
        log_writer.writerow(
            ["t", "rms_db"]
            + RULE_BANDS
            + DIAGNOSTIC_BANDS
            + ["candidate", "stable_s", "state"]
        )

    time_in_state: dict[str, float] = {}
    last_time: float | None = None

    try:
        for frame in stream.frames():
            for features in extractor.push(frame.samples):
                previous = smoother.state
                decision = smoother.update(features)

                if decision.state is not None and last_time is not None:
                    time_in_state[decision.state] = (
                        time_in_state.get(decision.state, 0.0)
                        + features.time
                        - last_time
                    )

                last_time = features.time

                if decision.changed:
                    display.transition(previous, decision)

                display.status(
                    decision,
                    smoother.band_spread("30-80"),
                )

                if log_writer is not None:
                    log_writer.writerow(
                        [f"{features.time:.3f}", f"{features.rms_db:.2f}"]
                        + [
                            f"{features.bands[name]:.2f}"
                            for name in RULE_BANDS + DIAGNOSTIC_BANDS
                        ]
                        + [
                            decision.candidate,
                            f"{decision.stable_seconds:.2f}",
                            decision.state or "",
                        ]
                    )
    except KeyboardInterrupt:
        pass
    finally:
        display.finish()

        if log_handle is not None:
            log_handle.close()

        stream.close()

    print()

    if time_in_state:
        total = sum(time_in_state.values())

        print("Time in each reported state:")

        for state in (OFF, FAN, COMPRESSOR):
            seconds = time_in_state.get(state, 0.0)

            if seconds <= 0.0:
                continue

            print(
                f"  {state:<11} {seconds:7.1f} s   "
                f"{seconds / total:5.1%}"
            )

        print()

    if stream.lost_frames or stream.dropped_frames:
        print(
            f"Warning: {stream.lost_frames} frames lost, "
            f"{stream.dropped_frames} dropped on the device. "
            "Levels during those gaps are not trustworthy."
        )

    if args.log:
        print(f"Wrote {args.log}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_ports:
        ports = list_ports()
        print("\n".join(ports) if ports else "No serial ports found.")
        return 0

    if not args.port:
        parser.error("a serial port is required (or use --list-ports)")

    if args.nfft < 64 or args.nfft & (args.nfft - 1):
        parser.error("--nfft must be a power of two and at least 64")

    if not 0.0 <= args.overlap < 1.0:
        parser.error("--overlap must be in [0, 1)")

    if args.hold_seconds < 0.0:
        parser.error("--hold-seconds must not be negative")

    if args.median_seconds <= 0.0:
        parser.error("--median-seconds must be positive")

    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
