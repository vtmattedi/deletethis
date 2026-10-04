"""Labelled replay: firmware detector path vs the stabilised PC v2 detectors.

Both are run over the same labelled audio (recordings named by state, and the
reviewed v1 transition events) and scored per analysis window against the
same truth, on the windows where each has published a value:

  firmware   the C++ core (validation/host/replay), 2048/512, compressor on
             52-65 Hz with the derived threshold
  PC v2      classifier.v2 defaults at the PC's 1024/512 (30-80 Hz, -38 dB)

For each observation (fan, compressor) it reports recall (truth yes, said
yes) and false-positive rate (truth no, said yes), overall and per
interference. Truth is the historical OFF/FAN/COMPRESSOR label mapped onto the
two observations (compressor implies fan), warm-up and transition guards
excluded, exactly as tools/audio/evaluation.py does.

    python validation/replay_eval.py [--out validation/results/replay_eval.md]
"""

from __future__ import annotations

import argparse
import os
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

import dataset as ds
import evaluation
import watson_ref as ref
from classifier.common import ObservationSmoother
from classifier.v2 import ObservationRules
from features import FeatureExtractor

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "validation" / ".cache"
THRESHOLD = -43.0


def firmware_published(wav: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(window start seconds, fan published, compressor published); -1 unknown."""
    CACHE.mkdir(exist_ok=True)
    raw = CACHE / (wav.stem + ".raw")
    np.round(ref.load_wav(wav)).astype("<i4").tofile(raw)
    done = subprocess.run(
        ["docker", "run", "--rm", "-v", f"{ROOT}:/work", "-w", "/work", "gcc:13",
         "/work/validation/host/replay", f"/work/validation/.cache/{raw.name}",
         f"comp_threshold={THRESHOLD}"],
        capture_output=True, text=True, check=True,
        env={**os.environ, "MSYS_NO_PATHCONV": "1"},
    )
    rows = [list(map(float, l.split(",")[1:])) for l in done.stdout.splitlines()
            if l.startswith("W,")]
    a = np.array(rows)
    return a[:, 0] / 1000.0, a[:, 23].astype(int), a[:, 24].astype(int)


def pc_published(wav: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    windows = evaluation.extract_windows(wav)
    rules = ObservationRules()
    smoother = ObservationSmoother(rules, windows.window_rate)
    fan, comp = [], []
    for feature in windows.features():
        d = smoother.update(feature)
        fan.append(-1 if d.fan_detected is None else int(d.fan_detected))
        comp.append(-1 if d.compressor_detected is None else int(d.compressor_detected))
    return windows.times, np.array(fan), np.array(comp)


class Tally:
    def __init__(self) -> None:
        self.tp = self.fn = self.fp = self.tn = 0

    def add(self, truth: np.ndarray, said: np.ndarray) -> None:
        ok = (truth >= 0) & (said >= 0)
        t, s = truth[ok], said[ok]
        self.tp += int(((t == 1) & (s == 1)).sum())
        self.fn += int(((t == 1) & (s == 0)).sum())
        self.fp += int(((t == 0) & (s == 1)).sum())
        self.tn += int(((t == 0) & (s == 0)).sum())

    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else float("nan")

    def fpr(self) -> float:
        return self.fp / (self.fp + self.tn) if self.fp + self.tn else float("nan")

    def n(self) -> int:
        return self.tp + self.fn + self.fp + self.tn


def truth_at(times: np.ndarray, truth_times: np.ndarray, truth: np.ndarray) -> np.ndarray:
    index = np.clip(np.searchsorted(truth_times, times), 0, truth_times.size - 1)
    return truth[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "validation" / "results" / "replay_eval.md"))
    args = parser.parse_args()

    items = []
    for name, state, interference, path in ds.recording_segments():
        items.append((path, interference, ("rec", state)))
    for r in ds.v1_records():
        items.append((r.wav_path, r.interference_label, ("v1", r)))

    tallies = {
        (who, obs, cat): Tally()
        for who in ("firmware", "pc_v2") for obs in ("fan", "compressor")
        for cat in ["all", "clean", "talking", "printer", "other"]
    }

    for i, (path, interference, source) in enumerate(items, 1):
        pc_t, pc_fan, pc_comp = pc_published(path)
        fw_t, fw_fan, fw_comp = firmware_published(path)

        if source[0] == "rec":
            f_truth, c_truth = ds.truth_columns(pc_t, source[1])
            fan_truth = (pc_t, f_truth)
            comp_truth = (pc_t, c_truth)
        else:
            labels = evaluation.truth_labels(source[1], pc_t)
            obs = evaluation.observation_truth(labels)
            fan_truth = (pc_t, obs["fan"])
            comp_truth = (pc_t, obs["compressor"])

        cat = interference if interference in ("clean", "talking", "printer") else "other"
        for who, t, fan, comp in (("firmware", fw_t, fw_fan, fw_comp),
                                  ("pc_v2", pc_t, pc_fan, pc_comp)):
            for obs, said, (tt, tv) in (("fan", fan, fan_truth),
                                        ("compressor", comp, comp_truth)):
                truth = truth_at(t, tt, tv)
                for c in ("all", cat):
                    tallies[(who, obs, c)].add(truth, said)
        print(f"[{i}/{len(items)}] {path.name}", flush=True)

    lines = ["# Labelled replay: firmware vs PC v2", "",
             f"{len(items)} labelled files (recordings + reviewed v1 events). "
             "Per-window, on windows where the detector has published.", "",
             "| observation | category | detector | windows | recall | false-positive |",
             "|---|---|---|---:|---:|---:|"]
    for obs in ("fan", "compressor"):
        for cat in ("all", "clean", "talking", "printer", "other"):
            for who in ("firmware", "pc_v2"):
                t = tallies[(who, obs, cat)]
                if t.n():
                    lines.append(f"| {obs} | {cat} | {who} | {t.n()} | "
                                 f"{t.recall():.3f} | {t.fpr():.3f} |")
    text = "\n".join(lines) + "\n"
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
