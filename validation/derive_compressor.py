"""Derive the firmware's compressor threshold for the 52-65 Hz feature.

The old PC feature was the whole 30-80 Hz band with a -38 dB threshold.
Narrowing the band to 52-65 Hz (around the measured ~58.6 Hz fundamental)
changes the power it integrates, so that number does not carry over. This
recomputes the feature with the firmware's exact math (watson_ref.RefExtractor,
2048-point Hamming, hop 512, then the 0.5 s rolling median the detectors read)
over every labelled window and picks the threshold from the data.

    python validation/derive_compressor.py

Writes validation/results/compressor_derivation.md. Also reports the 30-45 Hz
and 65-80 Hz sideband deltas, to calibrate the (disabled-by-default) guard.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import cache_features as cf

OUT = Path(__file__).resolve().parent / "results" / "compressor_derivation.md"
MEDIAN_WINDOWS = 16          # round(0.5 s * 31.25 windows/s)
SWEEP = np.arange(-60.0, -25.0, 0.5)


def balanced(pred: np.ndarray, truth: np.ndarray) -> float:
    if not (truth == 1).any() or not (truth == 0).any():
        return float("nan")
    return (pred[truth == 1].mean() + (1.0 - pred[truth == 0].mean())) / 2.0


def main() -> None:
    segments = cf.all_segments()
    index = {n: i for i, n in enumerate(segments[0]["names"])}

    truth, group, kind, interference = [], [], [], []
    feature = {k: [] for k in ("comp_primary", "comp_lower", "comp_upper", "comp_legacy")}

    for s in segments:
        smoothed = {k: cf.rolling_median(s["m"][:, index[k]], MEDIAN_WINDOWS) for k in feature}
        keep = s["comp"] >= 0
        truth.append(s["comp"][keep])
        group += [s["group"]] * int(keep.sum())
        kind += [s["kind"]] * int(keep.sum())
        interference += [s["interference"]] * int(keep.sum())
        for k in feature:
            feature[k].append(smoothed[k][keep])

    y = np.concatenate(truth)
    group = np.array(group)
    interference = np.array(interference)
    f = {k: np.concatenate(v) for k, v in feature.items()}
    p = f["comp_primary"]

    lines = ["# Compressor threshold derivation (52-65 Hz primary band)", "",
             f"{len(segments)} labelled files, {len(y)} scored windows "
             f"({int((y == 1).sum())} compressor, {int((y == 0).sum())} not), "
             f"{len(set(group))} time-groups. Feature: dB of the summed 52-65 Hz "
             "power, 2048-point Hamming, median over 0.5 s.", ""]

    lines += ["## Distributions (dBFS, median-smoothed)", "",
              "| feature | compressor ON p5 / p50 / p95 | OFF p50 / p95 / p99 |", "|---|---|---|"]
    for k, label in (("comp_primary", "52-65 Hz (primary)"), ("comp_legacy", "30-80 Hz (old)"),
                     ("comp_lower", "30-45 Hz (lower sideband)"), ("comp_upper", "65-80 Hz (upper sideband)")):
        on, off = f[k][y == 1], f[k][y == 0]
        lines.append(f"| {label} | {np.percentile(on, 5):.1f} / {np.percentile(on, 50):.1f} / "
                     f"{np.percentile(on, 95):.1f} | {np.percentile(off, 50):.1f} / "
                     f"{np.percentile(off, 95):.1f} / {np.percentile(off, 99):.1f} |")

    lines += ["", "## Threshold sweep on the primary band", "",
              "| threshold dB | balanced accuracy | recall | false-positive |", "|---:|---:|---:|---:|"]
    for t in np.arange(-52.0, -34.0, 1.0):
        pred = p >= t
        lines.append(f"| {t:.0f} | {balanced(pred, y):.4f} | {pred[y == 1].mean():.4f} | "
                     f"{pred[y == 0].mean():.4f} |")

    scores = np.array([balanced(p >= t, y) for t in SWEEP])
    best = float(SWEEP[int(np.nanargmax(scores))])
    plateau = SWEEP[scores >= np.nanmax(scores) - 0.005]

    # Leave-one-group-out: choose on all other groups, score on the held-out
    # one. Windows inside a group are near-copies, so splitting by window would
    # flatter any threshold.
    chosen, pred_all = [], np.zeros(y.shape, bool)
    for g in sorted(set(group)):
        train, test = group != g, group == g
        t = float(SWEEP[int(np.nanargmax([balanced(p[train] >= t, y[train]) for t in SWEEP]))])
        chosen.append(t)
        pred_all[test] = p[test] >= t
    logo = balanced(pred_all, y)

    # Rounded to a whole dB, towards the side away from the recall cliff.
    picked = float(np.floor(best))

    lines += ["", "## Choice", "",
              f"* best balanced accuracy at {best:.1f} dB ({np.nanmax(scores):.4f})",
              f"* plateau (within 0.5 % of best): {plateau.min():.1f} ... {plateau.max():.1f} dB",
              f"* leave-one-group-out ({len(chosen)} groups): chosen {min(chosen):.1f} ... "
              f"{max(chosen):.1f} dB, median {np.median(chosen):.1f}; held-out balanced "
              f"accuracy {logo:.4f}",
              f"* **chosen: {picked:.1f} dB** (the best, rounded down to a whole dB, away from the "
              f"recall cliff above {plateau.max():.0f} dB)",
              f"* at {picked:.1f} dB: recall {(p >= picked)[y == 1].mean():.4f}, "
              f"false-positive {(p >= picked)[y == 0].mean():.4f}",
              f"* old rule for comparison, 30-80 Hz at -38 dB: balanced accuracy "
              f"{balanced(f['comp_legacy'] >= -38.0, y):.4f}", "",
              "## By interference at the chosen threshold", "",
              "| interference | ON windows | OFF windows | recall | false-positive |", "|---|---:|---:|---:|---:|"]
    for c in sorted(set(interference)):
        m = interference == c
        on, off = (y[m] == 1), (y[m] == 0)
        pred = p[m] >= picked
        recall = f"{pred[on].mean():.3f}" if on.any() else "-"
        fpr = f"{pred[off].mean():.3f}" if off.any() else "-"
        lines.append(f"| {c} | {int(on.sum())} | {int(off.sum())} | {recall} | {fpr} |")

    detected = (p >= picked)
    tp, fp = detected & (y == 1), detected & (y == 0)
    d_lower, d_upper = p - f["comp_lower"], p - f["comp_upper"]
    lines += ["", "## Sideband deltas for the future guard (primary - sideband, dB)", "",
              "Calibration data only: the guard ships disabled with margins `0,0`.", "",
              "| windows | primary-lower p1 / p5 / p50 | primary-upper p1 / p5 / p50 |", "|---|---|---|"]
    for label, m in (("true compressor, detected", tp), ("not compressor, detected (false positives)", fp)):
        if m.any():
            a, b = d_lower[m], d_upper[m]
            lines.append(f"| {label} ({int(m.sum())}) | {np.percentile(a, 1):.1f} / {np.percentile(a, 5):.1f} / "
                         f"{np.percentile(a, 50):.1f} | {np.percentile(b, 1):.1f} / {np.percentile(b, 5):.1f} / "
                         f"{np.percentile(b, 50):.1f} |")

    text = "\n".join(lines) + "\n"
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
