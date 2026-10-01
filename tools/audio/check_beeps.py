"""Replay recordings through the beep detector and report what it finds.

    python tools/audio/check_beeps.py tools/audio/results/events
    python tools/audio/check_beeps.py tools/audio/recordings
    python tools/audio/check_beeps.py clip.wav --expect clip.wav:12.5,27.7

The inputs can be WAV files, or directories of them. A directory of
saved events is understood: each WAV's JSON supplies the moment the
classifier changed its mind and what the reviewer said was happening, so
the report can say how long before the published transition each beep
was, which is how the beep and the actual change were found to line up.

Two questions, answered separately
----------------------------------
Does it find the beeps that are there?
    Give the times you know about with ``--expect FILE:T1,T2,...`` (clip
    seconds, for the file whose name matches). Every expected time must
    be matched by exactly one detected beep within ``--tolerance``;
    unmatched detections near nothing are "extra", and two detections
    for one expected beep are flagged as duplicates.

Does it invent beeps that are not?
    Point it at audio where nobody pressed the remote. Anything found
    there is a false positive, reported as a rate per hour of audio. The
    recordings made before any command was sent are the natural set.
    Be careful with the event WAVs: the unit beeps whenever it is
    commanded, and many events were *caused* by a command, so a detection
    in them is usually a true one, not a false alarm.

The detector sees exactly what it sees live: the shared extractor's raw
per-window features, not smoothed ones.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluation as ev  # noqa: E402
from backend.config import result_paths  # noqa: E402
from classifier.detectors import BeepConfig  # noqa: E402


def collect(paths: list[Path]) -> list[Path]:
    wavs: list[Path] = []

    for path in paths:
        if path.is_dir():
            wavs.extend(sorted(path.glob("*.wav")))
        elif path.is_file():
            wavs.append(path)
        else:
            raise SystemExit(f"{path} does not exist")

    return wavs


def event_context(wav: Path) -> dict | None:
    """The saved event beside a WAV, if there is one."""
    sidecar = wav.with_suffix(".json")

    if not sidecar.is_file():
        return None

    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    if not isinstance(meta, dict) or "audio" not in meta:
        return None

    review = meta.get("review") or {}
    reviewed = review.get("status") == "reviewed"

    return {
        "transition": float(meta["audio"].get("preSeconds", 0.0)),
        "named": f"{meta.get('from')}->{meta.get('to')}",
        "actual": (
            f"{review.get('actualFrom')}->{review.get('actualTo')}"
            if reviewed and "actualFrom" in review else None
        ),
        "source": meta.get("source", "transition"),
        "moment": _moment(meta.get("id", "")),
    }


def _moment(identifier: str) -> datetime | None:
    try:
        return datetime.strptime(identifier[:17], "%Y-%m-%d_%H%M%S")
    except ValueError:
        return None


def parse_expect(items: list[str]) -> dict[str, list[float]]:
    expected: dict[str, list[float]] = {}

    for item in items:
        name, separator, times = item.rpartition(":")

        if not separator or not name:
            raise SystemExit(f"--expect wants FILE:T1,T2,..., got {item!r}")

        expected.setdefault(Path(name).name, []).extend(
            float(t) for t in times.split(",") if t.strip()
        )

    return expected


def match(expected: list[float], found: list, tolerance: float):
    """Pair expected times with detections; report what is left over."""
    unused = list(range(len(found)))
    rows = []

    for target in expected:
        near = [
            i for i in unused
            if abs(found[i].start_time - target) <= tolerance
        ]

        if not near:
            rows.append((target, None, 0))
            continue

        best = min(near, key=lambda i: abs(found[i].start_time - target))
        duplicates = len(near) - 1

        for i in near:
            unused.remove(i)

        rows.append((target, found[best], duplicates))

    return rows, [found[i] for i in unused]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay recordings through the beep detector.",
    )

    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument(
        "--expect", action="append", default=[], metavar="FILE:T1,T2",
        help="Known beep times in a file, in clip seconds. Repeatable",
    )
    parser.add_argument(
        "--tolerance", type=float, default=0.3,
        help="Seconds a detection may be from an expected beep",
    )
    parser.add_argument(
        "--out", type=Path, default=result_paths("v2").evaluation,
        help="Where beeps.csv is written",
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Only the summary, not every beep")

    defaults = BeepConfig()
    parser.add_argument("--min-contrast", type=float,
                        default=defaults.min_contrast_db, metavar="DB")
    parser.add_argument("--min-level", type=float,
                        default=defaults.min_level_db, metavar="DBFS")
    parser.add_argument("--min-ms", type=float, default=defaults.min_ms)
    parser.add_argument("--max-ms", type=float, default=defaults.max_ms)
    parser.add_argument("--min-hz", type=float, default=defaults.min_hz)
    parser.add_argument("--max-hz", type=float, default=defaults.max_hz)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config = BeepConfig(
        min_contrast_db=args.min_contrast, min_level_db=args.min_level,
        min_ms=args.min_ms, max_ms=args.max_ms,
        min_hz=args.min_hz, max_hz=args.max_hz,
    )

    wavs = collect(args.paths)

    if not wavs:
        print("No WAV files found.", file=sys.stderr)
        return 1

    expected = parse_expect(args.expect)
    cache = ev.FeatureCache(args.cache_dir or args.out / "cache")

    rows = []
    seconds = 0.0
    files_with = 0
    rejected: dict[str, int] = {}
    problems = 0

    print()
    print(f"Detector: contrast >= {config.min_contrast_db:g} dB, level >= "
          f"{config.min_level_db:g} dBFS, {config.min_hz:g}-"
          f"{config.max_hz:g} Hz, {config.min_ms:g}-{config.max_ms:g} ms")
    print()

    for wav in wavs:
        windows = cache.get(SimpleNamespace(id=wav.stem, wav_path=wav))
        found, detector = ev.replay_beeps(windows, config)
        context = event_context(wav)

        duration = len(windows.times) * windows.hop / windows.sample_rate
        seconds += duration
        files_with += bool(found)

        for reason, count in detector.rejected.items():
            rejected[reason] = rejected.get(reason, 0) + count

        for beep in found:
            offset = (
                beep.start_time - context["transition"] if context else None
            )
            rows.append([
                wav.name, f"{beep.start_time:.3f}", f"{beep.end_time:.3f}",
                f"{beep.duration_ms:.0f}", f"{beep.peak_hz:.1f}",
                f"{beep.contrast_db:.1f}", f"{beep.level_db:.1f}",
                "" if offset is None else f"{offset:.3f}",
                (context or {}).get("named", ""),
                (context or {}).get("actual") or "",
                (context or {}).get("source", ""),
            ])

        if args.quiet:
            continue

        label = wav.name

        if context:
            label += (
                f"  [{context['source']}, {context['named']}"
                + (f", reviewed {context['actual']}" if context["actual"] else "")
                + "]"
            )

        if not found and wav.name not in expected:
            continue

        print(label)

        for beep in found:
            where = f"{beep.start_time:7.2f} s"

            if context:
                where += f"  ({beep.start_time - context['transition']:+.2f} s from transition)"

            print(f"    beep at {where}: {beep.duration_ms:3.0f} ms, "
                  f"{beep.peak_hz:6.1f} Hz, {beep.contrast_db:4.1f} dB contrast, "
                  f"{beep.level_db:6.1f} dBFS")

        if wav.name in expected:
            matched, extra = match(expected[wav.name], found, args.tolerance)

            for target, beep, duplicates in matched:
                if beep is None:
                    problems += 1
                    print(f"    MISSED   expected beep at {target:.2f} s")
                else:
                    note = ""
                    if duplicates:
                        problems += 1
                        note = f"  DUPLICATE x{duplicates + 1}"
                    print(f"    ok       expected {target:.2f} s -> found "
                          f"{beep.start_time:.2f} s "
                          f"({beep.start_time - target:+.2f} s){note}")

            for beep in extra:
                problems += 1
                print(f"    EXTRA    beep at {beep.start_time:.2f} s was "
                      "not expected")

    print()
    print(f"{len(wavs)} files, {seconds / 60:.1f} minutes of audio: "
          f"{len(rows)} beeps in {files_with} files")

    if rejected:
        print(f"Tone-like runs rejected: {rejected}")

    if seconds > 0:
        print(f"Rate: {len(rows) / (seconds / 3600):.1f} per hour of audio")

    if expected:
        print(f"Against expectations: "
              f"{'all matched' if not problems else str(problems) + ' problems'}")

    if rows and not args.no_csv:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "beeps.csv"

        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "file", "start_s", "end_s", "duration_ms", "peak_hz",
                "contrast_db", "level_db", "from_transition_s",
                "classifier_named", "reviewed_as", "source",
            ])
            writer.writerows(rows)

        print(f"Wrote {path}")

    print()

    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
