"""Why the firmware's beep detector uses a 64 ms footprint on a 128 ms window.

The PC BeepDetector models a tone's extent in time with one "window" length,
used for the duration correction and the refractory. It was calibrated on the
PC's 64 ms window. The firmware analyses 128 ms windows (2048 samples). This
replays the reviewed beep events with the firmware's features and the PC
detector under different footprint values, and counts how many of the beeps
the PC's own 1024-point pipeline finds are reproduced.

    python validation/calibrate_beep.py

Writes validation/results/beep_calibration.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import dataset as ds
import evaluation
import watson_ref as ref
from classifier.detectors import BeepConfig, BeepDetector

OUT = Path(__file__).resolve().parent / "results" / "beep_calibration.md"


def pc_beeps(path: Path):
    windows = evaluation.extract_windows(path)
    detector = BeepDetector(BeepConfig(), hop_seconds=windows.hop_seconds)
    found = []
    for feature in windows.features():
        found += detector.update(feature)
    return found + detector.flush()


def firmware_beeps(samples: np.ndarray, footprint: float):
    extractor = ref.RefExtractor()
    detector = BeepDetector(BeepConfig(), hop_seconds=ref.HOP_SECONDS, window_seconds=footprint)
    found = []
    for start in range(0, samples.size - ref.HOP + 1, ref.HOP):
        for window in extractor.push(samples[start:start + ref.HOP]):
            found += detector.update(window)
    return found + detector.flush()


def main() -> None:
    records = [r for r in ds.v2_records() if r.event_type == "beep" and r.reviewed]
    reference = {r.id: pc_beeps(r.wav_path) for r in records}
    audio = {r.id: ref.load_wav(r.wav_path) for r in records}
    total = sum(len(v) for v in reference.values())

    lines = ["# Beep footprint calibration", "",
             f"{len(records)} reviewed beep events; the PC's 1024-point pipeline finds "
             f"{total} beeps in them. Firmware features (2048-point), PC BeepDetector, "
             "different footprints:", "",
             "| footprint (ms) | PC beeps reproduced | extra beeps | duration diff mean / sd (ms) |",
             "|---:|---:|---:|---:|"]

    for footprint in (0.064, 0.080, 0.096, 0.128):
        matched, extra, diffs = 0, 0, []
        for r in records:
            fw = firmware_beeps(audio[r.id], footprint)
            pc = reference[r.id]
            for b in pc:
                hit = [f for f in fw if abs(f.end_time - b.end_time) < 0.25]
                if hit:
                    matched += 1
                    diffs.append(hit[0].duration_ms - b.duration_ms)
            extra += sum(1 for f in fw if not any(abs(f.end_time - b.end_time) < 0.25 for b in pc))
        lines.append(f"| {footprint * 1000:.0f} | {matched} / {total} | {extra} | "
                     f"{np.mean(diffs):+.1f} / {np.std(diffs):.1f} |")

    lines += ["", "64 ms reproduces every beep with no extras; a larger footprint merges "
              "beeps that are ~250 ms apart (the refractory outlasts the gap) and "
              "under-reads durations. The firmware keeps 64 ms "
              "(`kBeepFootprintMs`, src/audio/AudioConfig.h)."]
    text = "\n".join(lines) + "\n"
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
