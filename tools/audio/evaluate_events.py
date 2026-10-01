"""Score the classifier against human-reviewed events.

    python tools/audio/evaluate_events.py tools/audio/results/events
    python tools/audio/evaluate_events.py tools/audio/results/events \\
        --classifier v2

Replays each reviewed event's WAV through the same extractor, rule and
Smoother the live backend uses, and compares the published state with
what the reviewer said was really happening. Ground truth is
``review.actualFrom`` / ``review.actualTo`` and nothing else: the
classifier's own ``from`` / ``to`` are what is being tested, so they can
never be the answer. Unreviewed events and ``UNKNOWN`` labels are
counted and reported but never scored.

The input can be events recorded under any classifier version -- they
are audio plus a human label, evidence rather than output. Results are
written under ``tools/audio/results/v2/evaluation`` regardless, so the
v1 dataset is only ever read.

Files written (all under the output directory):

    summary.json            parameters, dataset, headline metrics
    confusion_v1.csv        window-level confusion matrices
    confusion_v2.csv
    per_event.csv           every event, both classifiers side by side
    per_interference.csv    recall per truth class per interference tag
    feature_separation.csv  which features tell FAN from OFF at all
    candidate_thresholds.csv  every setting the search tried

The threshold search and its validation are v2-only and can be skipped
with ``--no-search``. Read its output with the dataset size in mind: the
labelled set is small and its windows are highly correlated, so the
plateau of good settings is more trustworthy than the single best one,
and the leave-one-group-out figure is the honest estimate of how a
chosen threshold behaves on events it was not chosen from.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluation as ev  # noqa: E402
from backend.config import (  # noqa: E402
    ClassifierConfig,
    load_runtime_config,
    result_paths,
)
from classifier import VERSIONS  # noqa: E402
from classifier.v2 import STABILITY_FEATURES, ThresholdsV2  # noqa: E402
from features import FEATURE_NAMES  # noqa: E402

DEFAULT_V1_EVENTS = result_paths("v1").events

# Settings within this much balanced accuracy of the best are treated
# as equally good when reporting a plateau.
PLATEAU_TOLERANCE = 0.005


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


def load_config(
    version: str, path: Path | None
) -> tuple[ClassifierConfig, str]:
    """The config to evaluate, and where it came from.

    An explicit path wins; otherwise the version's own saved config if
    it exists (the settings the live system actually runs with);
    otherwise the code defaults. A file written for another version is
    an error rather than a guess.
    """
    source = path or result_paths(version).config

    if not source.is_file():
        return ClassifierConfig.for_version(version), "code defaults"

    payload = dict(load_runtime_config(source))
    saved_version = payload.pop("classifierVersion", "v1")

    if saved_version != version:
        raise SystemExit(
            f"{source} is a {saved_version} config, but {version} "
            "was asked for"
        )

    config = ClassifierConfig.for_version(version).patched(payload)

    return config, str(source)


def apply_overrides(
    config: ClassifierConfig, overrides: list[str]
) -> ClassifierConfig:
    patch: dict = {}

    for item in overrides:
        key, separator, value = item.partition("=")

        if not separator:
            raise SystemExit(f"--override wants KEY=VALUE, got {item!r}")

        if key in ("fanRequire", "fanStabilityFeature"):
            patch[key] = value
        else:
            try:
                patch[key] = float(value)
            except ValueError:
                raise SystemExit(f"--override {key}: {value!r} is not a number")

    return config.patched(patch)


# ---------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)

        for row in rows:
            writer.writerow([
                f"{value:.6g}" if isinstance(value, float) else value
                for value in row
            ])


def pct(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:6.1%}"


def table(rows: list[list[str]], header: list[str]) -> str:
    widths = [len(h) for h in header]

    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def line(cells):
        return "  ".join(
            cell.ljust(widths[0]) if i == 0 else cell.rjust(widths[i])
            for i, cell in enumerate(cells)
            for cell in [cells[i]]
        )

    out = [line(header), "  ".join("-" * w for w in widths)]
    out.extend(line(row) for row in rows)

    return "\n".join(out)


def confusion_rows(matrix: np.ndarray) -> list[list]:
    rows = []

    for index, label in enumerate(ev.LABELS):
        total = int(matrix[index].sum())
        rows.append([
            label, total, *[int(x) for x in matrix[index]],
            float(matrix[index, index] / total) if total else "",
        ])

    return rows


CONFUSION_HEADER = [
    "truth", "windows", "pred_OFF", "pred_FAN", "pred_COMPRESSOR",
    "pred_none", "recall",
]


def print_confusion(title: str, matrix: np.ndarray) -> None:
    print(title)
    rows = []

    for index, label in enumerate(ev.LABELS):
        total = int(matrix[index].sum())
        rows.append([
            label, str(total),
            *[
                f"{matrix[index, j] / total:.1%}" if total else "-"
                for j in range(4)
            ],
        ])

    print(table(rows, ["truth", "windows", "OFF", "FAN", "COMPRESSOR",
                       "none"]))
    print()


# ---------------------------------------------------------------------
# Per-event and per-interference
# ---------------------------------------------------------------------


def majority(truth: np.ndarray, predicted: np.ndarray, side: str) -> str:
    """Most common published state over the scored windows of one side."""
    scored = truth != ev.UNSCORED
    codes = predicted[scored]

    if codes.size == 0:
        return ""

    values, counts = np.unique(codes, return_counts=True)
    top = int(values[int(np.argmax(counts))])

    return "none" if top < 0 else ev.LABELS[top]


def per_event_rows(
    events: list[ev.ScoredEvent],
    results: dict[str, ev.Evaluation],
) -> tuple[list[str], list[list]]:
    header = [
        "id", "time", "source", "recorded_by", "recorded_from",
        "recorded_to", "actual_from", "actual_to", "interference",
        "group", "scored_windows",
    ]

    for name in results:
        header += [
            f"{name}_accuracy", f"{name}_majority",
            f"{name}_off_to_fan", f"{name}_fan_to_off",
        ]

    if len(results) == 2:
        header.append("v2_minus_v1_accuracy")

    rows = []

    for index, event in enumerate(events):
        record = event.record
        meta = record.metadata

        row: list = [
            record.id,
            record.moment.isoformat(timespec="seconds"),
            meta.get("source", "transition"),
            record.classifier_version,
            meta.get("from") or "",
            meta.get("to") or "",
            record.actual_from,
            record.actual_to,
            record.interference_label,
            event.group,
            event.scored,
        ]

        accuracies = {}

        for name, result in results.items():
            predicted = result.published[index]
            matrix = ev.confusion(event.truth, predicted)
            total = int(matrix.sum())
            accuracy = float(np.trace(matrix[:, :3]) / total)

            accuracies[name] = accuracy
            row += [
                accuracy,
                majority(event.truth, predicted, "all"),
                int(matrix[0, 1]),
                int(matrix[1, 0]),
            ]

        if len(results) == 2:
            row.append(accuracies["v2"] - accuracies["v1"])

        rows.append(row)

    return header, rows


def per_interference_rows(
    events: list[ev.ScoredEvent],
    results: dict[str, ev.Evaluation],
) -> list[list]:
    rows = []

    for name, result in results.items():
        by_tag: dict[str, np.ndarray] = {}

        for index, event in enumerate(events):
            tag = event.record.interference_label
            matrix = ev.confusion(event.truth, result.published[index])
            by_tag[tag] = by_tag.get(tag, 0) + matrix

        for tag in sorted(by_tag):
            matrix = by_tag[tag]

            for label_index, label in enumerate(ev.LABELS):
                total = int(matrix[label_index].sum())

                if not total:
                    continue

                rows.append([
                    name, tag, label, total,
                    *[int(x) for x in matrix[label_index]],
                    float(matrix[label_index, label_index] / total),
                ])

    return rows


PER_INTERFERENCE_HEADER = [
    "classifier", "interference", "truth", "windows", "pred_OFF",
    "pred_FAN", "pred_COMPRESSOR", "pred_none", "recall",
]


# ---------------------------------------------------------------------
# Which features separate FAN from OFF
# ---------------------------------------------------------------------


def best_split(low: np.ndarray, high: np.ndarray) -> tuple[float, float]:
    """Threshold maximising balanced accuracy, 'low' below it.

    ``low`` and ``high`` are samples of the class expected to sit below
    and above. Returns (threshold, balanced accuracy).
    """
    values = np.concatenate([low, high])
    candidates = np.unique(np.percentile(values, np.linspace(1, 99, 99)))

    best = (float("nan"), 0.0)

    for threshold in candidates:
        score = ((low <= threshold).mean() + (high > threshold).mean()) / 2

        if score > best[1]:
            best = (float(threshold), float(score))

    return best


def feature_separation(
    fast: ev.FastEvaluator, rule: ThresholdsV2
) -> list[list]:
    """Rank features by how well they separate FAN from OFF.

    Restricted to the windows that matter: those where v2 already sees
    fan *energy* and no compressor. Anything the energy test rejects is
    not a candidate for the gate to fix.
    """
    mid = fast.column("500-1k") >= rule.fan_mid
    high = fast.column("1k-2k") >= rule.fan_high
    energy = (mid & high) if rule.fan_require_both else (mid | high)
    energy &= fast.column("30-80") < rule.compressor

    truth = np.concatenate(fast.truth)

    fan = energy & (truth == ev.LABEL_CODE["FAN"])
    off = energy & (truth == ev.LABEL_CODE["OFF"])

    rows = []

    for name in FEATURE_NAMES:
        values = fast.column(name)
        a, b = values[fan], values[off]

        if a.size < 2 or b.size < 2:
            continue

        # AUC as the Mann-Whitney statistic: P(random FAN window sits
        # above a random OFF window). 0.5 is no information.
        ranks = rankdata(np.concatenate([a, b]))
        auc = (ranks[: a.size].sum() - a.size * (a.size + 1) / 2) / (
            a.size * b.size
        )

        # Direction: a gate keeps FAN, so say which side FAN lives on.
        fan_lower = auc < 0.5
        low, high_ = (a, b) if fan_lower else (b, a)
        threshold, balanced = best_split(low, high_)

        rows.append([
            name,
            "lower" if fan_lower else "higher",
            float(max(auc, 1 - auc)),
            threshold,
            balanced,
            float(np.median(a)),
            float(np.median(b)),
            int(a.size),
            int(b.size),
        ])

    rows.sort(key=lambda row: row[2], reverse=True)

    return rows


FEATURE_SEPARATION_HEADER = [
    "feature", "fan_is", "auc", "best_threshold", "balanced_accuracy",
    "fan_median", "off_median", "fan_windows", "off_windows",
]


# ---------------------------------------------------------------------
# Threshold search
# ---------------------------------------------------------------------


def with_rule(rule: ThresholdsV2, **changes) -> ThresholdsV2:
    values = {
        "compressor": rule.compressor,
        "fan_mid": rule.fan_mid,
        "fan_high": rule.fan_high,
        "fan_require_both": rule.fan_require_both,
        "stability_feature": rule.stability_feature,
        "stability_threshold": rule.stability_threshold,
        "stability_min_seconds": rule.stability_min_seconds,
    }
    values.update(changes)

    return ThresholdsV2(**values)


class Search:
    """Try v2 settings and keep every result."""

    def __init__(
        self,
        fast: ev.FastEvaluator,
        events: list[ev.ScoredEvent],
    ) -> None:
        self.fast = fast
        self.events = events
        self.groups = np.array([e.group for e in events])
        self.rows: list[list] = []
        self.per_event: dict[tuple, np.ndarray] = {}

    def run(self, stage: str, rule: ThresholdsV2) -> dict:
        per_event = self.fast.confusion(rule)
        total = per_event.sum(axis=0)
        summary = ev.metrics(total)

        accuracy = np.array([
            np.trace(m[:, :3]) / max(m.sum(), 1) for m in per_event
        ])

        key = (
            stage, rule.stability_feature, rule.stability_threshold,
            rule.compressor, rule.fan_mid, rule.fan_high,
        )
        self.per_event[key] = per_event

        row = {
            "stage": stage,
            "feature": rule.stability_feature,
            "stability_threshold": rule.stability_threshold,
            "compressor": rule.compressor,
            "fan_mid": rule.fan_mid,
            "fan_high": rule.fan_high,
            "balanced": summary["balancedAccuracy"],
            "recall_off": summary["recall"]["OFF"],
            "recall_fan": summary["recall"]["FAN"],
            "recall_compressor": summary["recall"]["COMPRESSOR"],
            "off_to_fan": summary["offToFan"],
            "off_to_compressor": summary["offToCompressor"],
            "fan_to_off": summary["fanToOff"],
            "compressor_misses": summary["compressorMisses"],
            "event_accuracy": float(accuracy.mean()),
        }
        self.rows.append(row)

        return row

    def table(self) -> tuple[list[str], list[list]]:
        header = list(self.rows[0])

        return header, [
            [
                "" if value is None else value
                for value in row.values()
            ]
            for row in self.rows
        ]


def threshold_grid(values: np.ndarray, points: int = 33) -> np.ndarray:
    """A grid over the range a feature actually takes."""
    low, high = np.percentile(values, [2, 98])
    grid = np.linspace(low, high, points)

    return np.unique(np.round(grid, 3))


def plateau(rows: list[dict], tolerance: float) -> dict:
    """Settings within ``tolerance`` of the best, along one sweep."""
    ordered = sorted(rows, key=lambda r: r["stability_threshold"])
    best = max(r["balanced"] for r in ordered)

    good = [
        r["stability_threshold"] for r in ordered
        if r["balanced"] >= best - tolerance
    ]

    return {
        "best": next(
            r["stability_threshold"] for r in ordered
            if r["balanced"] == best
        ),
        "bestBalanced": best,
        "plateau": [min(good), max(good)],
        "recommended": float(np.median(good)),
        "tolerance": tolerance,
    }


def leave_one_group_out(
    search: Search,
    sweep: list[dict],
    feature: str,
) -> dict:
    """Pick the threshold without the held-out group, then test on it.

    Only the stability threshold is chosen here; everything else is
    held at the configured values. So the gap between this and the
    in-sample best measures how much choosing that one number from this
    data flatters it -- which is the figure to believe about the gate.
    """
    thresholds = sorted({r["stability_threshold"] for r in sweep})

    stack = np.stack([
        search.per_event[
            ("stability", feature, t, sweep[0]["compressor"],
             sweep[0]["fan_mid"], sweep[0]["fan_high"])
        ]
        for t in thresholds
    ])                                        # (T, E, 3, 4)

    groups = search.groups
    held_out_total = np.zeros((3, 4), dtype=np.int64)
    chosen: list[float] = []

    def balanced(matrix: np.ndarray) -> float:
        recall = [
            matrix[i, i] / matrix[i].sum() if matrix[i].sum() else np.nan
            for i in range(3)
        ]
        return float(np.nanmean(recall))

    for group in np.unique(groups):
        train = groups != group
        scores = [balanced(stack[t][train].sum(axis=0))
                  for t in range(len(thresholds))]

        # Several thresholds often tie on a plateau; take the middle
        # one, as the recommendation would.
        best = max(scores)
        tied = [i for i, s in enumerate(scores) if s >= best - 1e-12]
        pick = tied[len(tied) // 2]

        chosen.append(thresholds[pick])
        held_out_total += stack[pick][~train].sum(axis=0)

    summary = ev.metrics(held_out_total)
    summary["groups"] = int(len(np.unique(groups)))
    summary["thresholdsChosen"] = {
        "min": float(min(chosen)),
        "median": float(np.median(chosen)),
        "max": float(max(chosen)),
    }

    return summary


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------


def plot_sweep(
    path: Path,
    sweep: list[dict],
    feature: str,
    chosen: float,
    v1_summary: dict | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(sweep, key=lambda r: r["stability_threshold"])
    x = [r["stability_threshold"] for r in ordered]

    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True
    )

    top.plot(x, [r["recall_fan"] for r in ordered], label="FAN recall",
             color="#1f9d55")
    top.plot(x, [r["recall_off"] for r in ordered], label="OFF recall",
             color="#3b7dd8")
    top.plot(x, [r["balanced"] for r in ordered], label="balanced",
             color="black", linestyle="--")

    if v1_summary is not None:
        top.axhline(v1_summary["recall"]["FAN"], color="#1f9d55",
                    linestyle=":", alpha=0.6, label="v1 FAN recall")
        top.axhline(v1_summary["recall"]["OFF"], color="#3b7dd8",
                    linestyle=":", alpha=0.6, label="v1 OFF recall")

    top.set_ylabel("window-level")
    top.set_ylim(0, 1.02)
    top.set_title(f"Stationarity gate: {feature}")
    top.legend(ncol=3, fontsize=8)
    top.grid(alpha=0.3)

    bottom.plot(x, [r["off_to_fan"] for r in ordered],
                label="OFF called FAN", color="#d64545")
    bottom.plot(x, [r["fan_to_off"] for r in ordered],
                label="FAN called OFF", color="#d2691e")
    bottom.set_ylabel("windows")
    bottom.set_xlabel(f"{feature} threshold (dB)")
    bottom.legend(fontsize=8)
    bottom.grid(alpha=0.3)

    for axis in (top, bottom):
        axis.axvline(chosen, color="grey", linestyle="-.", alpha=0.7)

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=130)
    plt.close(figure)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score the classifier against reviewed events.",
    )

    parser.add_argument(
        "events",
        nargs="*",
        type=Path,
        help=(
            "Directories of event JSON/WAV pairs (default: the v1 "
            "events, plus the v2 events if there are any)"
        ),
    )

    parser.add_argument(
        "--classifier",
        choices=(*VERSIONS, "both"),
        default="both",
        help="Which classifier to score (default: both, side by side)",
    )

    parser.add_argument(
        "--v1-config", type=Path, default=None,
        help="API-shaped JSON for v1 (default: the saved v1 config)",
    )
    parser.add_argument(
        "--v2-config", type=Path, default=None,
        help="API-shaped JSON for v2 (default: the saved v2 config)",
    )
    parser.add_argument(
        "--override", action="append", default=[], metavar="KEY=VALUE",
        help=(
            "Change one v2 setting, in its API spelling, e.g. "
            "fanStabilityThreshold=1.8. Repeatable"
        ),
    )

    parser.add_argument(
        "--out", type=Path, default=result_paths("v2").evaluation,
        help="Output directory (default: tools/audio/results/v2/evaluation)",
    )

    parser.add_argument(
        "--warmup", type=float, default=ev.DEFAULT_WARMUP_SECONDS,
        help="Seconds skipped at the start of every replay",
    )
    parser.add_argument(
        "--guard", type=float, default=ev.DEFAULT_GUARD_SECONDS,
        help="Seconds skipped either side of a real transition",
    )
    parser.add_argument(
        "--group-gap", type=float, default=ev.DEFAULT_GROUP_GAP_SECONDS,
        help="Events this close in time share a validation group",
    )

    parser.add_argument(
        "--no-search", action="store_true",
        help="Skip the threshold search and its validation",
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Do not write plots",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Where extracted features are cached (default: <out>/cache)",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Re-extract features instead of using the cache",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    started = time.perf_counter()

    directories = list(args.events)

    if not directories:
        directories = [DEFAULT_V1_EVENTS]
        v2_events = result_paths("v2").events

        if v2_events.is_dir():
            directories.append(v2_events)

    records = ev.load_events(directories)

    if not records:
        print(f"No events found in: {', '.join(map(str, directories))}",
              file=sys.stderr)
        return 1

    counts = ev.dataset_counts(records)

    print()
    print(f"Events: {counts['events']} found, {counts['labelled']} "
          f"reviewed and scored "
          f"({counts['steady']} steady, {counts['transitions']} real "
          f"transitions)")
    print(f"        {counts['unreviewed']} unreviewed and "
          f"{counts['reviewedUnknown']} reviewed-UNKNOWN are counted but "
          "never scored")
    print(f"        final state: {counts['byFinalState']}")
    print(f"        interference: {counts['byInterference']}")

    if counts["labelled"] == 0:
        print("\nNothing to score: no event has a usable review.",
              file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)

    cache = ev.FeatureCache(
        None if args.no_cache else (args.cache_dir or args.out / "cache")
    )
    events = ev.prepare(
        records, cache, args.warmup, args.guard, args.group_gap
    )

    groups = len({e.group for e in events})
    scored_windows = sum(e.scored for e in events)

    print(f"\nReplayed {len(events)} events: {scored_windows} scored "
          f"windows in {groups} groups "
          f"(features: {cache.hits} cached, {cache.misses} extracted)")
    print(f"Skipping {args.warmup:g}s warm-up and {args.guard:g}s either "
          f"side of real transitions; groups are events within "
          f"{args.group_gap:g}s of each other.")

    wanted = (
        list(VERSIONS) if args.classifier == "both" else [args.classifier]
    )

    configs: dict[str, ClassifierConfig] = {}
    sources: dict[str, str] = {}

    for version in wanted:
        path = args.v1_config if version == "v1" else args.v2_config
        config, source = load_config(version, path)

        if version == "v2" and args.override:
            config = apply_overrides(config, args.override)
            source += f" + overrides {args.override}"

        configs[version] = config
        sources[version] = source

    results: dict[str, ev.Evaluation] = {}
    matrices: dict[str, np.ndarray] = {}
    candidate_matrices: dict[str, np.ndarray] = {}

    for version in wanted:
        config = configs[version]

        results[version] = ev.evaluate(
            version, events, config.rule(),
            config.median_seconds, config.hold_seconds,
        )

        matrices[version] = results[version].matrices(events).sum(axis=0)
        candidate_matrices[version] = results[version].matrices(
            events, "candidate"
        ).sum(axis=0)

        print(f"\n=== {version}: {sources[version]}")
        api = config.to_api()
        print("    " + ", ".join(
            f"{k}={v}" for k, v in api.items()
            if k not in ("classifierVersion",)
        ))
        print_confusion(
            f"Published state, window level ({version})", matrices[version]
        )

        summary = ev.metrics(matrices[version])
        print(f"    balanced accuracy {summary['balancedAccuracy']:.3f}   "
              f"OFF->FAN {summary['offToFan']}   "
              f"OFF->COMPRESSOR {summary['offToCompressor']}   "
              f"FAN->OFF {summary['fanToOff']}   "
              f"COMPRESSOR missed {summary['compressorMisses']}")

        write_csv(
            args.out / f"confusion_{version}.csv",
            CONFUSION_HEADER,
            confusion_rows(matrices[version]),
        )

    # ---- reference: v1 at its code default, to show why -38 matters ----
    reference = None

    if "v1" in wanted:
        default = ClassifierConfig.for_version("v1")

        if default.rule() != configs["v1"].rule():
            baseline = ev.evaluate(
                "v1-default", events, default.rule(),
                default.median_seconds, default.hold_seconds,
            )
            reference = ev.metrics(
                baseline.matrices(events).sum(axis=0)
            )
            reference["compressorThreshold"] = default.compressor_threshold

    header, rows = per_event_rows(events, results)
    write_csv(args.out / "per_event.csv", header, rows)
    write_csv(
        args.out / "per_interference.csv",
        PER_INTERFERENCE_HEADER,
        per_interference_rows(events, results),
    )

    print("\nBy interference (published state; recall per truth class)")
    shown = per_interference_rows(events, results)
    print(table(
        [
            [r[0], r[1], r[2], str(r[3]), f"{r[8]:.1%}"]
            for r in shown
        ],
        ["classifier", "interference", "truth", "windows", "recall"],
    ))

    # ---- search and validation (v2) ----------------------------------------
    search_summary = None

    if "v2" in wanted and not args.no_search:
        config = configs["v2"]
        rule = config.rule()

        fast = ev.FastEvaluator(events, config.median_seconds,
                                config.hold_seconds)

        checked = ev.verify_fast_path(
            events, rule, config.median_seconds, config.hold_seconds,
            limit=min(20, len(events)),
        )

        search = Search(fast, events)
        separation = feature_separation(fast, rule)
        write_csv(
            args.out / "feature_separation.csv",
            FEATURE_SEPARATION_HEADER, separation,
        )

        print("\nFeatures that separate FAN from OFF, among windows where "
              "v2 sees fan energy and no compressor (top 8 by AUC)")
        print(table(
            [
                [r[0], r[1], f"{r[2]:.3f}", f"{r[3]:.3g}",
                 f"{r[4]:.3f}", f"{r[5]:.3g}", f"{r[6]:.3g}"]
                for r in separation[:8]
            ],
            ["feature", "fan is", "AUC", "best thr", "bal.acc",
             "fan med", "off med"],
        ))

        sweeps: dict[str, list[dict]] = {}

        for feature in STABILITY_FEATURES:
            values = fast.column(feature)
            sweeps[feature] = [
                search.run(
                    "stability",
                    with_rule(rule, stability_feature=feature,
                              stability_threshold=float(t)),
                )
                for t in threshold_grid(values)
            ]

        # Compressor and fan-band sweeps at the configured gate.
        for threshold in np.arange(-50.0, -25.5, 1.0):
            search.run("compressor", with_rule(
                rule, compressor=float(threshold)))

        for mid in np.arange(-66.0, -55.5, 1.0):
            for high in np.arange(-69.0, -58.5, 1.0):
                search.run("fan_bands", with_rule(
                    rule, fan_mid=float(mid), fan_high=float(high)))

        # A joint look around the configured setting.
        for comp in rule.compressor + np.arange(-2, 2.5, 1.0):
            for mid in rule.fan_mid + np.arange(-2, 2.5, 1.0):
                for high in rule.fan_high + np.arange(-2, 2.5, 1.0):
                    search.run("joint", with_rule(
                        rule, compressor=float(comp),
                        fan_mid=float(mid), fan_high=float(high)))

        header, rows = search.table()
        write_csv(args.out / "candidate_thresholds.csv", header, rows)

        v1_summary = ev.metrics(matrices["v1"]) if "v1" in matrices else None

        print(f"\nThreshold search: {len(search.rows)} settings tried "
              f"(fast path checked against the real Smoother on "
              f"{checked} events)")
        print("\nStationarity gate by feature (others held at the "
              "configured values)")

        summary_rows = []
        feature_results = {}

        for feature, sweep in sweeps.items():
            found = plateau(sweep, PLATEAU_TOLERANCE)
            feature_results[feature] = found
            best = next(
                r for r in sweep
                if r["stability_threshold"] == found["best"]
            )
            summary_rows.append([
                feature,
                f"{found['best']:.3g}",
                f"{found['plateau'][0]:.3g}..{found['plateau'][1]:.3g}",
                f"{found['bestBalanced']:.3f}",
                pct(best["recall_fan"]),
                str(best["off_to_fan"]),
                str(best["fan_to_off"]),
            ])

        print(table(
            summary_rows,
            ["feature", "best thr", "plateau", "balanced", "FAN recall",
             "OFF->FAN", "FAN->OFF"],
        ))

        feature = rule.stability_feature
        sweep = sweeps[feature]
        found = feature_results[feature]

        validation = leave_one_group_out(search, sweep, feature)

        print(f"\nGate on {feature}: best {found['best']:.3g} dB, "
              f"plateau {found['plateau'][0]:.3g}..{found['plateau'][1]:.3g}"
              f", recommended {found['recommended']:.3g} dB")
        print(f"Leave-one-group-out ({validation['groups']} groups): "
              f"balanced accuracy {validation['balancedAccuracy']:.3f} "
              f"vs in-sample best {found['bestBalanced']:.3f}; chose "
              f"thresholds {validation['thresholdsChosen']['min']:.3g}.."
              f"{validation['thresholdsChosen']['max']:.3g}")

        compressor_rows = [
            r for r in search.rows if r["stage"] == "compressor"
        ]
        best_compressor = max(compressor_rows, key=lambda r: r["balanced"])

        print(f"Compressor threshold (gate as configured): best "
              f"{best_compressor['compressor']:g} dB "
              f"(balanced {best_compressor['balanced']:.3f}); configured "
              f"{rule.compressor:g} dB")

        search_summary = {
            "settingsTried": len(search.rows),
            "fastPathVerifiedEvents": checked,
            "gate": {
                "feature": feature,
                **found,
                "configured": rule.stability_threshold,
            },
            "byFeature": feature_results,
            "leaveOneGroupOut": validation,
            "compressor": {
                "configured": rule.compressor,
                "best": best_compressor["compressor"],
                "bestBalanced": best_compressor["balanced"],
            },
            "topSeparatingFeatures": [
                {"feature": r[0], "auc": r[2], "fanIs": r[1]}
                for r in separation[:5]
            ],
        }

        if not args.no_plots:
            plot_path = (
                result_paths("v2").plots / "fan_stability_threshold.png"
                if args.out == result_paths("v2").evaluation
                else args.out / "fan_stability_threshold.png"
            )
            plot_sweep(plot_path, sweep, feature, found["recommended"],
                       v1_summary)
            print(f"Plot: {plot_path}")

    # ---- summary ---------------------------------------------------------------
    summary = {
        "generatedAt": datetime.now().isoformat(timespec="seconds"),
        "directories": [str(d) for d in directories],
        "parameters": {
            "warmupSeconds": args.warmup,
            "guardSeconds": args.guard,
            "groupGapSeconds": args.group_gap,
        },
        "dataset": {
            **counts,
            "scoredEvents": len(events),
            "scoredWindows": scored_windows,
            "groups": groups,
        },
        "classifiers": {
            version: {
                "config": configs[version].to_api(),
                "configSource": sources[version],
                "published": ev.metrics(matrices[version]),
                "candidate": ev.metrics(candidate_matrices[version]),
            }
            for version in wanted
        },
        "reference": (
            {"v1CodeDefault": reference} if reference is not None else None
        ),
        "search": search_summary,
        "seconds": round(time.perf_counter() - started, 1),
    }

    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, default=float) + "\n",
        encoding="utf-8",
    )

    if len(wanted) == 2:
        one, two = (ev.metrics(matrices[v]) for v in ("v1", "v2"))

        def change(before: int, after: int) -> str:
            if not before:
                return "n/a"
            return f"{(after - before) / before:+.0%}"

        print("\nv1 -> v2, window level, published state")
        print(f"  OFF -> FAN          {one['offToFan']:6d} -> "
              f"{two['offToFan']:6d}  ({change(one['offToFan'], two['offToFan'])})")
        print(f"  FAN recall          {pct(one['recall']['FAN'])} -> "
              f"{pct(two['recall']['FAN'])}")
        print(f"  FAN -> OFF          {one['fanToOff']:6d} -> "
              f"{two['fanToOff']:6d}")
        print(f"  COMPRESSOR recall   {pct(one['recall']['COMPRESSOR'])} -> "
              f"{pct(two['recall']['COMPRESSOR'])}")
        print(f"  balanced accuracy   {one['balancedAccuracy']:.3f} -> "
              f"{two['balancedAccuracy']:.3f}")

    print(f"\nWrote {args.out}")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
