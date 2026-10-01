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
table means the same thing here. The shared-feature regression tests
pin the original rule inputs down numerically.

The rules are hierarchical and deliberately simple:

    stage 1   30-80 Hz persistently high     -> COMPRESSOR
    stage 2   500-1k and/or 1k-2k high       -> FAN
              otherwise                      -> OFF

Nothing is published from a single FFT window. Features are smoothed by
a rolling median, and the resulting candidate must hold for a couple of
seconds before it becomes the reported state. That is what stops a door
slam or a printer from being read as a compressor.

The defaults were tuned with ``--replay`` against 18 labelled
recordings and reach 99.3% of decided windows, 17/18 files perfect.
COMPRESSOR is exact in all 6 files including speech. The one remaining
error is OFF with talking being called FAN for part of one recording:
speech puts real energy in both fan bands, and level alone cannot
always tell a fan from a voice. Re-run --replay after any rule change.
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
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parent))

from acstream import (  # noqa: E402
    DEFAULT_BAUD,
    FULL_SCALE,
    AudioStream,
    ProtocolError,
    list_ports,
)
from analyze import (  # noqa: E402
    DEFAULT_RECORDINGS,
    DEFAULT_RESULTS,
    format_table,
    infer_labels,
    to_float,
)
from features import (  # noqa: E402
    DEFAULT_NFFT,
    DEFAULT_OVERLAP,
    DIAGNOSTIC_BANDS,
    FEATURE_NAMES,
    RULE_BANDS,
    FeatureExtractor,
    Features,
)

# The states this classifier can report, in escalation order.
OFF = "OFF"
FAN = "FAN"
COMPRESSOR = "COMPRESSOR"

# Measured against the 18 labelled recordings with --replay. See the
# module docstring for what the numbers are worth and where they fail.
#
# COMPRESSOR has an enormous margin: 30-80 sits at -38 dB when the
# compressor runs and at -57..-59 dB in every other condition, so -48
# is 10 dB clear of both sides.
#
# FAN is far tighter. The fan's own 1k-2k level is about -63.7 dB, and
# speech with the AC off reaches -58 dB in peaks, so the usable window
# for the high threshold is only -66..-64 and -65 is its centre. Both
# fan bands must be over threshold, which is what separates a fan from
# someone talking; either-of-two cannot get past 93%.
DEFAULT_COMPRESSOR_THRESHOLD = -48.0
DEFAULT_FAN_MID_THRESHOLD = -62.0
DEFAULT_FAN_HIGH_THRESHOLD = -65.0
DEFAULT_FAN_REQUIRE = "both"

DEFAULT_HOLD_SECONDS = 2.0
DEFAULT_MEDIAN_SECONDS = 0.5
DEFAULT_REFRESH_HZ = 5.0


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

    def reset(self) -> None:
        """Forget the feature history after a gap in the audio.

        The published state is kept: it is the last thing actually
        observed, and a dropout is not evidence that it changed. But
        the candidate has to re-earn its hold, so nothing is published
        on the strength of medians taken across a discontinuity.
        """
        self.history.clear()
        self.candidate = None
        self.candidate_since = 0.0

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

    def gap(self, frames: int, frame_seconds: float) -> None:
        stamp = time.strftime("%H:%M:%S")

        if self.interactive:
            self.out.write("\r" + " " * self.line_length + "\r")
            self.line_length = 0

        self.out.write(
            f"{stamp}  gap: {frames} frame(s), "
            f"{frames * frame_seconds * 1000:.0f} ms missing; "
            "history reset\n"
        )
        self.out.flush()

    def finish(self) -> None:
        if self.interactive and self.line_length:
            self.out.write("\n")
            self.out.flush()
            self.line_length = 0


# ---------------------------------------------------------------------
# Offline replay
#
# Runs the live classifier -- the same FeatureExtractor, the same
# Smoother, the same classify() -- over recorded WAV files, so a rule
# change can be judged against 18 labelled recordings in a second
# instead of by standing in front of the air conditioner.
#
# Nothing here is a second implementation of the rules. If this
# disagrees with the live path, that is a bug in this file.
# ---------------------------------------------------------------------


UNDECIDED = "--"


@dataclass
class ReplayResult:
    path: Path
    expected: str
    interference: str

    predictions: list[str | None]
    features: list[dict[str, float]]
    window_rate: float

    @property
    def decided(self) -> list[str]:
        return [p for p in self.predictions if p is not None]

    @property
    def warmup_seconds(self) -> float:
        for index, prediction in enumerate(self.predictions):
            if prediction is not None:
                return index / self.window_rate

        return len(self.predictions) / self.window_rate

    @property
    def accuracy(self) -> float:
        decided = self.decided

        if not decided:
            return 0.0

        return sum(
            1 for p in decided if p == self.expected
        ) / len(decided)

    @property
    def classified(self) -> str:
        decided = self.decided

        if not decided:
            return UNDECIDED

        return max(set(decided), key=decided.count)


def replay_file(
    path: Path,
    args: argparse.Namespace,
    thresholds: Thresholds,
    frame_samples: int = 512,
) -> ReplayResult:
    sample_rate, raw = wavfile.read(path)

    # to_float normalises to [-1, 1); the extractor expects raw counts
    # and divides by FULL_SCALE itself, so put it back in that domain.
    signal = to_float(raw) * FULL_SCALE

    extractor = FeatureExtractor(
        sample_rate,
        args.nfft,
        args.overlap,
    )

    window_rate = sample_rate / extractor.hop

    smoother = Smoother(
        thresholds,
        window_rate=window_rate,
        median_seconds=args.median_seconds,
        hold_seconds=args.hold_seconds,
    )

    predictions: list[str | None] = []
    feature_rows: list[dict[str, float]] = []

    # Fed in frames the size the ESP32 actually sends, so the window
    # boundaries land exactly where they would live.
    for start in range(0, signal.size - frame_samples + 1, frame_samples):
        chunk = signal[start : start + frame_samples]

        for features in extractor.push(chunk):
            feature_rows.append(features.values)
            predictions.append(smoother.update(features).state)

    labels = infer_labels(path)

    return ReplayResult(
        path=path,
        expected=labels.state.upper(),
        interference=labels.interference,
        predictions=predictions,
        features=feature_rows,
        window_rate=window_rate,
    )


def replay_table(results: list[ReplayResult]) -> str:
    rows = []

    for result in results:
        rows.append(
            [
                result.path.stem,
                result.expected,
                result.classified,
                f"{result.accuracy:.0%}",
                f"{len(result.decided)}",
                f"{result.warmup_seconds:.1f}",
            ]
        )

    return format_table(
        rows,
        ["File", "expected", "classified", "accuracy", "windows", "warmup s"],
    )


def confusion_matrix(results: list[ReplayResult]) -> str:
    states = [OFF, FAN, COMPRESSOR]

    counts = {
        actual: {predicted: 0 for predicted in states}
        for actual in states
    }

    for result in results:
        if result.expected not in counts:
            continue

        for prediction in result.decided:
            if prediction in counts[result.expected]:
                counts[result.expected][prediction] += 1

    rows = []

    for actual in states:
        total = sum(counts[actual].values())

        row = [f"actual {actual}"]

        for predicted in states:
            count = counts[actual][predicted]

            row.append(
                f"{count}"
                if not total
                else f"{count} ({count / total:.0%})"
            )

        rows.append(row)

    return format_table(
        rows,
        ["", f"pred {OFF}", f"pred {FAN}", f"pred {COMPRESSOR}"],
    )


def condition_table(results: list[ReplayResult]) -> str:
    grouped: dict[tuple[str, str], list[ReplayResult]] = {}

    for result in results:
        grouped.setdefault(
            (result.expected, result.interference), []
        ).append(result)

    rows = []

    for (expected, interference), group in sorted(grouped.items()):
        decided = [p for r in group for p in r.decided]

        correct = sum(1 for p in decided if p == expected)

        predictions = {p for p in decided}

        rows.append(
            [
                f"{expected} / {interference}",
                str(len(group)),
                f"{correct / len(decided):.0%}" if decided else "-",
                ", ".join(sorted(predictions)) or "-",
            ]
        )

    return format_table(
        rows,
        ["Condition", "files", "accuracy", "predicted"],
    )


def write_replay_csv(path: Path, results: list[ReplayResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)

        writer.writerow(
            ["file", "expected", "interference", "t"]
            + FEATURE_NAMES
            + ["predicted"]
        )

        for result in results:
            for index, prediction in enumerate(result.predictions):
                writer.writerow(
                    [
                        result.path.stem,
                        result.expected,
                        result.interference,
                        f"{index / result.window_rate:.3f}",
                    ]
                    + [
                        f"{result.features[index][name]:.6g}"
                        for name in FEATURE_NAMES
                    ]
                    + [prediction or ""]
                )


def run_replay(args: argparse.Namespace) -> int:
    paths: list[Path] = []

    for entry in args.replay or [DEFAULT_RECORDINGS]:
        entry = Path(entry)

        if entry.is_dir():
            paths.extend(sorted(entry.glob("*.wav")))
        else:
            paths.append(entry)

    if not paths:
        print(
            f"No recordings found in {DEFAULT_RECORDINGS}.",
            file=sys.stderr,
        )
        return 1

    thresholds = Thresholds(
        compressor=args.compressor_threshold,
        fan_mid=args.fan_mid_threshold,
        fan_high=args.fan_high_threshold,
        fan_require_both=args.fan_require == "both",
    )

    results: list[ReplayResult] = []

    for path in paths:
        if not path.exists():
            print(f"warning: {path} does not exist", file=sys.stderr)
            continue

        try:
            results.append(replay_file(path, args, thresholds))
        except (ValueError, OSError) as error:
            print(f"warning: {path.name}: {error}", file=sys.stderr)

    if not results:
        print("error: nothing could be replayed", file=sys.stderr)
        return 1

    unknown = [r for r in results if r.expected == "UNKNOWN"]

    print()
    print(
        f"Replaying the live classifier over {len(results)} recording(s)"
    )
    print()
    print(
        f"Rules:  COMPRESSOR if 30-80 >= "
        f"{args.compressor_threshold:g}"
    )
    print(
        f"        FAN if 500-1k >= {args.fan_mid_threshold:g} "
        f"{'and' if args.fan_require == 'both' else 'or'} "
        f"1k-2k >= {args.fan_high_threshold:g}"
    )
    print(
        f"Smooth: {args.median_seconds:g}s median, "
        f"{args.hold_seconds:g}s hold"
    )

    print()
    print("Per file")
    print()
    print(replay_table(results))

    print()
    print("Confusion matrix, over decided windows")
    print()
    print(confusion_matrix(results))

    print()
    print("By condition")
    print()
    print(condition_table(results))

    scored = [r for r in results if r.expected != "UNKNOWN"]

    if scored:
        decided = [p for r in scored for p in r.decided]
        correct = sum(
            1
            for r in scored
            for p in r.decided
            if p == r.expected
        )

        perfect = sum(1 for r in scored if r.accuracy >= 0.999)

        print()
        print(
            f"Overall: {correct / len(decided):.1%} of "
            f"{len(decided)} decided windows; "
            f"{perfect}/{len(scored)} files classified perfectly"
        )

    if unknown:
        print()
        print(
            f"Note: {len(unknown)} file(s) have no recognised state in "
            "the name and are listed but not scored: "
            + ", ".join(r.path.stem for r in unknown)
        )

    if args.replay_csv:
        write_replay_csv(args.replay_csv, results)
        print()
        print(f"Wrote {args.replay_csv}")

    print()

    return 0


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
        "--replay",
        nargs="*",
        metavar="PATH",
        default=None,
        help=(
            "Offline mode: run this exact classifier over recorded "
            "WAV files instead of the serial port, and report "
            "per-file accuracy and a confusion matrix. Bare --replay "
            "uses tools/audio/recordings"
        ),
    )

    parser.add_argument(
        "--replay-csv",
        type=Path,
        nargs="?",
        const=DEFAULT_RESULTS / "replay.csv",
        default=None,
        help=(
            "Write the per-window replay verdicts to CSV; bare flag "
            "writes tools/audio/results/replay.csv"
        ),
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
        default=DEFAULT_FAN_REQUIRE,
        help=(
            "Whether one or both fan bands must be over threshold "
            # argparse runs help text through %-expansion, so a literal
            # per cent sign has to be doubled.
            f"(default: {DEFAULT_FAN_REQUIRE}; 'either' cannot get "
            "past 93%% on the recorded data)"
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
        nargs="?",
        const=DEFAULT_RESULTS / "live.csv",
        default=None,
        help=(
            "Write per-window features and state to CSV; bare --log "
            "writes tools/audio/results/live.csv"
        ),
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
            ["t"]
            + FEATURE_NAMES
            + ["candidate", "stable_s", "state"]
        )

    time_in_state: dict[str, float] = {}
    last_time: float | None = None

    # Frames the host never received plus frames the device could not
    # read. Either way there is a hole in the audio.
    missing = stream.lost_frames + stream.dropped_frames
    gaps = 0

    try:
        for frame in stream.frames():
            now_missing = stream.lost_frames + stream.dropped_frames

            if now_missing > missing:
                lost = now_missing - missing
                missing = now_missing
                gaps += 1

                extractor.reset(
                    skip_samples=lost * info.frame_samples
                )
                smoother.reset()

                display.gap(lost, info.frame_samples / info.sample_rate)

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
                        [f"{features.time:.3f}"]
                        + [
                            f"{features.values[name]:.6g}"
                            for name in FEATURE_NAMES
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

    if args.replay is None and not args.port:
        parser.error(
            "a serial port is required "
            "(or use --replay for offline evaluation, or --list-ports)"
        )

    if args.replay is not None and args.port:
        parser.error("--replay reads files, so it takes no serial port")

    if args.nfft < 64 or args.nfft & (args.nfft - 1):
        parser.error("--nfft must be a power of two and at least 64")

    if not 0.0 <= args.overlap < 1.0:
        parser.error("--overlap must be in [0, 1)")

    if args.hold_seconds < 0.0:
        parser.error("--hold-seconds must not be negative")

    if args.median_seconds <= 0.0:
        parser.error("--median-seconds must be positive")

    if args.replay is not None:
        return run_replay(args)

    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
