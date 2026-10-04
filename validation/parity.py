"""Firmware-vs-reference parity over the labelled audio.

Runs each file through

  * the firmware's own analysis core, compiled on the host (validation/host),
    fed through the same BlockQueue/drain() path the analysis task uses, and
  * the float64 reference (watson_ref.py): the PC extractor math at 2048/512
    driving the PC ObservationSmoother and BeepDetector,

and compares them window by window:

  features    max |difference| per feature, in dB / Hz
  candidates  fan / compressor candidate agreement
  published   the times each published observation changed
  beeps       count, end time, duration, peak, contrast

    python validation/parity.py            # recordings + reviewed events
    python validation/parity.py --quick    # a small, fast subset
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

import dataset as ds
import watson_ref as ref
from classifier.common import ObservationSmoother
from classifier.detectors import BeepDetector

ROOT = Path(__file__).resolve().parent.parent
HOST = ROOT / "validation" / "host" / "replay"
CACHE = ROOT / "validation" / ".cache"

THRESHOLD = -43.0

# Columns of the "W" lines, in order.
FW_COLUMNS = [
    "t_ms", "rms", "mid", "high", "primary", "lower", "shoulder", "upper",
    "beep_band", "beep_peak", "beep_neighbour", "beep_contrast", "high_std",
    "history_ms", "sm_primary", "sm_lower", "sm_upper", "sm_mid", "sm_high",
    "sm_std", "sm_history", "fan_cand", "comp_cand", "fan_pub", "comp_pub",
]

# Reference key for each firmware column that has a direct counterpart.
REF_KEYS = {
    "rms": "rms", "mid": "500-1k", "high": "1k-2k", "primary": "comp_primary",
    "lower": "comp_lower", "shoulder": "comp_shoulder", "upper": "comp_upper",
    "beep_band": "beep_band_power", "beep_peak": "beep_peak_hz",
    "beep_neighbour": "beep_neighbor_power",
    "beep_contrast": "beep_contrast_db", "high_std": "1k-2k_std",
}
SMOOTHED_KEYS = {
    "sm_primary": "comp_primary", "sm_lower": "comp_lower",
    "sm_upper": "comp_upper", "sm_mid": "500-1k", "sm_high": "1k-2k",
    "sm_std": "1k-2k_std", "sm_history": "temporal_seconds",
}


def write_raw(path: Path) -> Path:
    CACHE.mkdir(exist_ok=True)
    out = CACHE / (path.stem + ".raw")
    samples = ref.load_wav(path)
    np.round(samples).astype("<i4").tofile(out)
    return out


def run_firmware(raw: Path, extra: list[str] | None = None) -> dict:
    cmd = [
        "docker", "run", "--rm", "-v", f"{ROOT}:/work", "-w", "/work",
        "gcc:13", "/work/validation/host/replay",
        f"/work/validation/.cache/{raw.name}",
        f"comp_threshold={THRESHOLD}",
    ] + (extra or [])
    done = subprocess.run(
        cmd, capture_output=True, text=True, check=True,
        env={**__import__("os").environ, "MSYS_NO_PATHCONV": "1"},
    )
    windows, fan, comp, beeps, stats = [], [], [], [], ""
    for line in done.stdout.splitlines():
        kind, _, rest = line.partition(",")
        if kind == "W":
            windows.append([float(x) for x in rest.split(",")])
        elif kind == "FAN":
            fan.append(tuple(float(x) for x in rest.split(",")))
        elif kind == "COMPRESSOR":
            comp.append(tuple(float(x) for x in rest.split(",")))
        elif kind == "BEEP":
            beeps.append([float(x) for x in rest.split(",")])
        elif kind == "STATS":
            stats = rest
    return {"w": np.array(windows), "fan": fan, "comp": comp,
            "beeps": beeps, "stats": stats}


def run_reference(samples: np.ndarray) -> dict:
    rules = ref.FirmwareRules(compressor_threshold=THRESHOLD)
    extractor = ref.RefExtractor()
    smoother = ObservationSmoother(
        rules, window_rate=ref.SAMPLE_RATE / ref.HOP,
        median_seconds=0.5, hold_seconds=2.0, history_seconds=2.0,
    )
    beep = BeepDetector(
        rules.beep, hop_seconds=ref.HOP_SECONDS,
        window_seconds=ref.BEEP_FOOTPRINT_SECONDS,
    )
    rows, fan, comp, beeps = [], [], [], []
    for start in range(0, samples.size - ref.HOP + 1, ref.HOP):
        for w in extractor.push(samples[start:start + ref.HOP]):
            d = smoother.update(w)
            v = w.values
            row = {"t_ms": round(w.time * 1000), "rms": v["rms"]}
            for col, key in REF_KEYS.items():
                row[col] = v[key]
            for col, key in SMOOTHED_KEYS.items():
                row[col] = d.values[key] * (1000.0 if key == "temporal_seconds" else 1.0)
            row["fan_cand"] = int(d.fan_candidate)
            row["comp_cand"] = int(d.compressor_candidate)
            rows.append(row)
            t = round(w.time * 1000)
            if d.fan_changed:
                fan.append((t, int(d.fan_detected)))
            if d.compressor_changed:
                comp.append((t, int(d.compressor_detected)))
            for b in beep.update(w):
                beeps.append([round(b.end_time * 1000), round(b.duration_ms),
                              b.peak_hz, b.contrast_db, b.level_db])
    return {"rows": rows, "fan": fan, "comp": comp, "beeps": beeps,
            "rejected": dict(beep.rejected)}


def compare(name: str, fw: dict, rf: dict, worst: dict) -> list[str]:
    problems = []
    w = fw["w"]
    n_fw, n_ref = len(w), len(rf["rows"])
    if n_fw != n_ref:
        problems.append(f"window count {n_fw} != reference {n_ref}")
        return problems

    cols = {c: i for i, c in enumerate(FW_COLUMNS)}
    for col in list(REF_KEYS) + list(SMOOTHED_KEYS):
        a = w[:, cols[col]]
        b = np.array([r[col] for r in rf["rows"]])
        # A window whose reference level sits on the -120 dB floor carries no
        # information (digital silence in a band); skip it.
        keep = b > -119.0 if col not in ("beep_peak", "high_std", "sm_std",
                                          "sm_history") else np.ones(a.shape, bool)
        if col.startswith("beep_"):
            # Only where there is a tone to measure: in noise-only windows the
            # peak bin is an argmax over noise and flips between two
            # near-equal bins, which says nothing about the detector.
            tone = np.array([r["beep_contrast"] for r in rf["rows"]]) >= 5.0
            keep = keep & tone
        if keep.any():
            d = float(np.max(np.abs(a[keep] - b[keep])))
            worst[col] = max(worst.get(col, 0.0), d)

    for col in ("fan_cand", "comp_cand"):
        a = w[:, cols[col]].astype(int)
        b = np.array([r[col] for r in rf["rows"]])
        miss = int((a != b).sum())
        worst[col] = worst.get(col, 0) + miss
        if miss:
            problems.append(f"{col}: {miss}/{n_fw} windows differ")

    for label, key in (("fan", "fan"), ("compressor", "comp")):
        a = [(int(t), int(v)) for t, v in fw[key]]
        b = [(int(t), int(v)) for t, v in rf[key]]
        if a != b:
            problems.append(f"{label} published changes differ: fw={a} ref={b}")

    fb = fw["beeps"]
    rb = rf["beeps"]
    if len(fb) != len(rb):
        problems.append(f"beep count fw={len(fb)} ref={len(rb)}")
    else:
        for x, y in zip(fb, rb):
            if abs(x[0] - y[0]) > 1 or abs(x[1] - y[1]) > 1:
                problems.append(f"beep timing fw={x} ref={y}")
            worst["beep_peak_hz"] = max(worst.get("beep_peak_hz", 0.0), abs(x[2] - y[2]))
            worst["beep_contrast_db"] = max(worst.get("beep_contrast_db", 0.0), abs(x[3] - y[3]))
            worst["beep_level_db"] = max(worst.get("beep_level_db", 0.0), abs(x[4] - y[4]))
    return problems


def files(quick: bool) -> list[Path]:
    recs = [p for _, _, _, p in ds.recording_segments()]
    v1 = [r.wav_path for r in ds.v1_records()]
    v2 = [r.wav_path for r in ds.v2_records()]
    if quick:
        return recs[::6] + v1[::20] + v2[::8]
    return recs + v1[::4] + v2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if not HOST.exists():
        print("build the host tool first: validation/host/build.sh")
        return 2

    worst: dict = {}
    failures = 0
    paths = files(args.quick)
    for i, path in enumerate(paths, 1):
        samples = ref.load_wav(path)
        raw = write_raw(path)
        fw = run_firmware(raw)
        rf = run_reference(samples)
        problems = compare(path.stem, fw, rf, worst)
        status = "ok" if not problems else "DIFF"
        print(f"[{i:3d}/{len(paths)}] {status:4s} {path.name}  "
              f"windows={len(fw['w'])} beeps={len(fw['beeps'])}")
        for p in problems:
            print("        ", p)
        failures += bool(problems)

    print("\nworst-case differences over all files (firmware - reference):")
    for key in list(REF_KEYS) + list(SMOOTHED_KEYS) + [
        "beep_peak_hz", "beep_contrast_db", "beep_level_db",
    ]:
        if key in worst:
            print(f"  {key:18s} {worst[key]:.5f}")
    print(f"  candidate mismatches: fan={worst.get('fan_cand', 0)} "
          f"compressor={worst.get('comp_cand', 0)} windows")
    print(f"\n{len(paths) - failures}/{len(paths)} files identical in behaviour")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
