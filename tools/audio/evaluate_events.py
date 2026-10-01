"""Score the classifiers against human-reviewed events.

    python tools/audio/evaluate_events.py tools/audio/results/events
    python tools/audio/evaluate_events.py tools/audio/results/events \\
        --classifier v2

Replays each reviewed event's WAV through the same extractor, detectors
and smoothing the live backend uses, and compares what was published
with what the reviewer said was really happening. Ground truth is
``review.actualFrom`` / ``review.actualTo`` and nothing else: the
classifier's own ``from`` / ``to`` are what is being tested, so they can
never be the answer. Unreviewed events and ``UNKNOWN`` labels are
counted and reported but never scored.

Two independent yes/no questions
--------------------------------
Classifier v2 does not report one state; it reports whether the fan is
running and whether the compressor is running, separately. So that is
how it is scored: each observation has its own confusion matrix and its
own errors (a fan reported that was not there, a compressor missed that
was). v1's single state is folded onto the same two questions so the two
can be compared on equal terms.

The reviewed events are labelled with the old three-way scheme, so the
yes/no truth is derived from it by one stated assumption -- the
compressor runs only while the fan does. See ``evaluation.py``. It is a
bridge for the historical reviews, not the v2 review schema.

The input can be events recorded under any classifier version -- they
are audio plus a human label, evidence rather than output. Results are
written under ``tools/audio/results/v2/evaluation`` regardless, so the
v1 dataset is only ever read.

Files written (all under the output directory):

    summary.json              parameters, dataset, headline metrics
    confusion_v1.csv          window-level confusion, per observation
    confusion_v2.csv
    per_event.csv             every event, both classifiers side by side
    per_interference.csv      recall per truth value per interference tag
    feature_separation.csv    which features tell yes from no, per target
    candidate_thresholds.csv  every setting the search tried
    beeps.csv                 (via check_beeps.py) every beep found

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
from classifier.detectors import (  # noqa: E402
    STABILITY_FEATURES,
    CompressorConfig,
    FanConfig,
)
from features import FEATURE_NAMES  # noqa: E402

DEFAULT_V1_EVENTS = result_paths("v1").events

# Settings within this much balanced accuracy of the best are treated
# as equally good when reporting a plateau.
PLATEAU_TOLERANCE = 0.005

TRUTH_NAME = {0: "absent", 1: "present"}


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
        )

    out = [line(header), "  ".join("-" * w for w in widths)]
    out.extend(line(row) for row in rows)

    return "\n".join(out)


CONFUSION_HEADER = [
    "observation", "truth", "windows", "pred_no", "pred_yes",
    "pred_nothing_yet", "correct",
]


def confusion_rows(matrices: dict[str, np.ndarray]) -> list[list]:
    rows = []

    for observation, matrix in matrices.items():
        for truth in (0, 1):
            total = int(matrix[truth].sum())
            rows.append([
                observation, TRUTH_NAME[truth], total,
                *[int(x) for x in matrix[truth]],
                float(matrix[truth, truth] / total) if total else "",
            ])

    return rows


def print_confusions(title: str, matrices: dict[str, np.ndarray]) -> None:
    print(title)
    rows = []

    for observation, matrix in matrices.items():
        for truth in (0, 1):
            total = int(matrix[truth].sum())
            rows.append([
                observation, TRUTH_NAME[truth], str(total),
                *[
                    f"{matrix[truth, j] / total:.1%}" if total else "-"
                    for j in range(3)
                ],
            ])

    print(table(rows, ["observation", "really", "windows", "said no",
                       "said yes", "nothing yet"]))
    print()


def observation_metrics(
    matrices: dict[str, np.ndarray]
) -> dict[str, dict]:
    return {
        observation: ev.metrics(matrix)
        for observation, matrix in matrices.items()
    }


def print_headline(label: str, summary: dict[str, dict]) -> None:
    rows = []

    for observation, m in summary.items():
        rows.append([
            observation,
            pct(m["recall"]), pct(m["specificity"]),
            pct(m["precision"]),
            f"{m['balancedAccuracy']:.3f}"
            if m["balancedAccuracy"] is not None else "n/a",
            str(m["falsePositive"]), str(m["falseNegative"]),
        ])

    print(table(rows, [label, "recall", "specificity", "precision",
                       "balanced", "false yes", "false no"]))
    print()


# ---------------------------------------------------------------------
# Per-event and per-interference
# ---------------------------------------------------------------------


def per_event_rows(
    events: list[ev.ScoredEvent],
    runs: dict[str, ev.ObservationRun],
) -> tuple[list[str], list[list]]:
    header = [
        "id", "time", "source", "recorded_by", "recorded_from",
        "recorded_to", "actual_from", "actual_to", "interference",
        "group", "scored_windows",
    ]

    for name in runs:
        for observation in ev.OBSERVATIONS:
            header += [
                f"{name}_{observation}_false_yes",
                f"{name}_{observation}_false_no",
            ]

    compare = len(runs) == 2

    if compare:
        for observation in ev.OBSERVATIONS:
            header.append(f"v2_minus_v1_{observation}_errors")

    matrices = {
        name: ev.run_confusions(events, run) for name, run in runs.items()
    }

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

        errors: dict[str, dict[str, int]] = {}

        for name in runs:
            errors[name] = {}

            for observation in ev.OBSERVATIONS:
                matrix = matrices[name][observation][index]
                false_yes = int(matrix[0, 1])
                false_no = int(matrix[1, 0] + matrix[1, 2] + matrix[0, 2])

                errors[name][observation] = false_yes + false_no
                row += [false_yes, false_no]

        if compare:
            for observation in ev.OBSERVATIONS:
                row.append(
                    errors["v2"][observation] - errors["v1"][observation]
                )

        rows.append(row)

    return header, rows


PER_INTERFERENCE_HEADER = [
    "classifier", "observation", "interference", "really", "windows",
    "pred_no", "pred_yes", "pred_nothing_yet", "correct",
]


def per_interference_rows(
    events: list[ev.ScoredEvent],
    runs: dict[str, ev.ObservationRun],
) -> list[list]:
    rows = []

    for name, run in runs.items():
        matrices = ev.run_confusions(events, run)

        for observation in ev.OBSERVATIONS:
            by_tag: dict[str, np.ndarray] = {}

            for index, event in enumerate(events):
                tag = event.record.interference_label
                by_tag[tag] = (
                    by_tag.get(tag, 0) + matrices[observation][index]
                )

            for tag in sorted(by_tag):
                for truth in (0, 1):
                    matrix = by_tag[tag]
                    total = int(matrix[truth].sum())

                    if not total:
                        continue

                    rows.append([
                        name, observation, tag, TRUTH_NAME[truth], total,
                        *[int(x) for x in matrix[truth]],
                        float(matrix[truth, truth] / total),
                    ])

    return rows


# ---------------------------------------------------------------------
# Which features separate present from absent
# ---------------------------------------------------------------------


def best_split(low: np.ndarray, high: np.ndarray) -> tuple[float, float]:
    """Threshold maximising balanced accuracy, 'low' below it."""
    values = np.concatenate([low, high])
    candidates = np.unique(np.percentile(values, np.linspace(1, 99, 99)))

    best = (float("nan"), 0.0)

    for threshold in candidates:
        score = ((low <= threshold).mean() + (high > threshold).mean()) / 2

        if score > best[1]:
            best = (float(threshold), float(score))

    return best


def separation(
    fast: ev.FastEvaluator,
    target: str,
    restrict: np.ndarray,
) -> list[list]:
    """Rank features by how well they separate present from absent."""
    truth = np.concatenate(fast.truth[target])

    present = restrict & (truth == 1)
    absent = restrict & (truth == 0)

    rows = []

    for name in FEATURE_NAMES:
        values = fast.column(name)
        a, b = values[present], values[absent]

        if a.size < 2 or b.size < 2:
            continue

        # AUC as the Mann-Whitney statistic: P(a random present window
        # sits above a random absent one). 0.5 is no information.
        ranks = rankdata(np.concatenate([a, b]))
        auc = (ranks[: a.size].sum() - a.size * (a.size + 1) / 2) / (
            a.size * b.size
        )

        present_lower = auc < 0.5
        low, high = (a, b) if present_lower else (b, a)
        threshold, balanced = best_split(low, high)

        rows.append([
            target, name,
            "lower" if present_lower else "higher",
            float(max(auc, 1 - auc)), threshold, balanced,
            float(np.median(a)), float(np.median(b)),
            int(a.size), int(b.size),
        ])

    rows.sort(key=lambda row: row[3], reverse=True)

    return rows


FEATURE_SEPARATION_HEADER = [
    "target", "feature", "present_is", "auc", "best_threshold",
    "balanced_accuracy", "present_median", "absent_median",
    "present_windows", "absent_windows",
]


# ---------------------------------------------------------------------
# Threshold search
# ---------------------------------------------------------------------


def threshold_grid(values: np.ndarray, points: int = 33) -> np.ndarray:
    """A grid over the range a feature actually takes."""
    low, high = np.percentile(values, [2, 98])

    return np.unique(np.round(np.linspace(low, high, points), 3))


def with_fan(config: FanConfig, **changes) -> FanConfig:
    values = {
        "mid_threshold": config.mid_threshold,
        "high_threshold": config.high_threshold,
        "require_both": config.require_both,
        "stability_feature": config.stability_feature,
        "stability_threshold": config.stability_threshold,
        "stability_min_seconds": config.stability_min_seconds,
    }
    values.update(changes)

    return FanConfig(**values)


class Search:
    """Try detector settings and keep every result.

    The detectors are independent, so a fan setting is judged only on the
    fan question and a compressor setting only on the compressor one.
    """

    HEADER = [
        "stage", "observation", "setting", "value", "feature",
        "fan_mid", "fan_high", "compressor",
        "balanced", "recall", "specificity", "false_yes", "false_no",
        "event_accuracy",
    ]

    def __init__(self, fast: ev.FastEvaluator) -> None:
        self.fast = fast
        self.groups = np.array([e.group for e in fast.events])
        self.rows: list[list] = []
        self.stacks: dict[tuple, np.ndarray] = {}

    def record(
        self,
        stage: str,
        observation: str,
        per_event: np.ndarray,
        key: tuple,
        *,
        setting: str,
        value: float,
        feature: str = "",
        fan: FanConfig | None = None,
        compressor: CompressorConfig | None = None,
    ) -> dict:
        summary = ev.metrics(per_event.sum(axis=0))

        accuracy = np.array([
            (m[0, 0] + m[1, 1]) / max(m.sum(), 1) for m in per_event
        ])

        self.stacks[(stage,) + key] = per_event

        row = {
            "stage": stage,
            "observation": observation,
            "setting": setting,
            "value": value,
            "feature": feature,
            "balanced": summary["balancedAccuracy"],
            "recall": summary["recall"],
            "specificity": summary["specificity"],
            "false_yes": summary["falsePositive"],
            "false_no": summary["falseNegative"],
            "event_accuracy": float(accuracy.mean()),
        }

        self.rows.append([
            stage, observation, setting, value, feature,
            fan.mid_threshold if fan else "",
            fan.high_threshold if fan else "",
            compressor.threshold if compressor else "",
            summary["balancedAccuracy"], summary["recall"],
            summary["specificity"], summary["falsePositive"],
            summary["falseNegative"], float(accuracy.mean()),
        ])

        return row

    def fan_stability(self, base: FanConfig, feature: str, value: float):
        config = with_fan(
            base, stability_feature=feature, stability_threshold=value
        )

        return self.record(
            "fan_stability", "fan", self.fast.fan_confusion(config),
            (feature, value), setting="stability_threshold", value=value,
            feature=feature, fan=config,
        )

    def fan_bands(self, base: FanConfig, mid: float, high: float):
        config = with_fan(base, mid_threshold=mid, high_threshold=high)

        return self.record(
            "fan_bands", "fan", self.fast.fan_confusion(config),
            (mid, high), setting="bands", value=mid, fan=config,
        )

    def compressor(self, threshold: float):
        config = CompressorConfig(threshold=threshold)

        return self.record(
            "compressor", "compressor",
            self.fast.compressor_confusion(config),
            (threshold,), setting="threshold", value=threshold,
            compressor=config,
        )

    def table(self) -> tuple[list[str], list[list]]:
        return self.HEADER, self.rows


def plateau(rows: list[dict], tolerance: float) -> dict:
    """Settings within ``tolerance`` of the best, along one sweep."""
    ordered = sorted(rows, key=lambda r: r["value"])
    best = max(r["balanced"] for r in ordered)

    good = [
        r["value"] for r in ordered if r["balanced"] >= best - tolerance
    ]

    return {
        "best": next(r["value"] for r in ordered if r["balanced"] == best),
        "bestBalanced": best,
        "plateau": [min(good), max(good)],
        "recommended": float(np.median(good)),
        "tolerance": tolerance,
    }


def leave_one_group_out(
    stack: np.ndarray,
    groups: np.ndarray,
    values: list[float],
) -> dict:
    """Pick the setting without the held-out group, then test on it.

    ``stack`` is the per-event confusion for each candidate value,
    shape (values, events, 2, 3). Only the one setting is chosen here;
    everything else is held at the configured values. So the gap between
    this and the in-sample best measures how much choosing that one
    number from this data flatters it -- which is the figure to believe.
    """
    held_out_total = np.zeros((2, 3), dtype=np.int64)
    chosen: list[float] = []

    def balanced(matrix: np.ndarray) -> float:
        return ev.metrics(matrix)["balancedAccuracy"] or 0.0

    for group in np.unique(groups):
        train = groups != group
        scores = [
            balanced(stack[t][train].sum(axis=0))
            for t in range(len(values))
        ]

        # Several values often tie on a plateau; take the middle one,
        # as the recommendation would.
        best = max(scores)
        tied = [i for i, s in enumerate(scores) if s >= best - 1e-12]
        pick = tied[len(tied) // 2]

        chosen.append(values[pick])
        held_out_total += stack[pick][~train].sum(axis=0)

    summary = ev.metrics(held_out_total)
    summary["groups"] = int(len(np.unique(groups)))
    summary["valuesChosen"] = {
        "min": float(min(chosen)),
        "median": float(np.median(chosen)),
        "max": float(max(chosen)),
    }

    return summary


# ---------------------------------------------------------------------
# Beeps in the reviewed events
# ---------------------------------------------------------------------


def beep_summary(
    records: list[ev.EventRecord],
    cache: ev.FeatureCache,
    config,
) -> dict:
    """Beeps found across every event, by what the reviewer said.

    Includes unreviewed events and v2 observation events: a beep needs no
    review to be found. Grouped by the reviewed transition because the
    interesting question is whether a power command always beeps.
    """
    by_kind: dict[str, dict] = {}
    offsets: dict[str, list[float]] = {}
    total = 0
    seconds = 0.0

    for record in records:
        windows = cache.get(record)
        found, _ = ev.replay_beeps(windows, config)

        seconds += len(windows.times) * windows.hop_seconds
        total += len(found)

        if record.labelled:
            kind = f"{record.actual_from}->{record.actual_to}"
        else:
            kind = (
                f"{record.metadata.get('from')}->"
                f"{record.metadata.get('to')} (not reviewed)"
            )

        bucket = by_kind.setdefault(kind, {"events": 0, "withBeep": 0})
        bucket["events"] += 1
        bucket["withBeep"] += bool(found)

        for beep in found:
            offsets.setdefault(kind, []).append(
                beep.start_time - record.pre_seconds
            )

    # How far before the published transition the nearest beep fell,
    # for the events whose state really changed.
    near = {}

    for kind, values in offsets.items():
        close = [v for v in values if -3.2 < v < -2.0]

        if close:
            near[kind] = {
                "count": len(close),
                "medianSeconds": float(np.median(close)),
                "minSeconds": float(min(close)),
                "maxSeconds": float(max(close)),
            }

    return {
        "beeps": total,
        "minutesOfAudio": seconds / 60.0,
        "perHour": total / (seconds / 3600.0) if seconds else 0.0,
        "byTransition": dict(sorted(by_kind.items())),
        "beforePublishedTransition": near,
    }


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

    ordered = sorted(sweep, key=lambda r: r["value"])
    x = [r["value"] for r in ordered]

    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True
    )

    top.plot(x, [r["recall"] for r in ordered],
             label="fan found when running", color="#1f9d55")
    top.plot(x, [r["specificity"] for r in ordered],
             label="no fan reported when not running", color="#3b7dd8")
    top.plot(x, [r["balanced"] for r in ordered], label="balanced",
             color="black", linestyle="--")

    if v1_summary is not None:
        top.axhline(v1_summary["recall"], color="#1f9d55",
                    linestyle=":", alpha=0.6, label="v1 recall")
        top.axhline(v1_summary["specificity"], color="#3b7dd8",
                    linestyle=":", alpha=0.6, label="v1 specificity")

    top.set_ylabel("window-level")
    top.set_ylim(0, 1.02)
    top.set_title(f"Fan detector: stationarity gate on {feature}")
    top.legend(ncol=2, fontsize=8)
    top.grid(alpha=0.3)

    bottom.plot(x, [r["false_yes"] for r in ordered],
                label="fan reported, not running", color="#d64545")
    bottom.plot(x, [r["false_no"] for r in ordered],
                label="fan running, not reported", color="#d2691e")
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
        description="Score the classifiers against reviewed events.",
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
        "--no-beeps", action="store_true",
        help="Skip the beep summary",
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

    if counts["observationEvents"]:
        print(f"        {counts['observationEvents']} v2 observation "
              "events are not scored here (they review each observation "
              "separately)")

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
    print("Truth for the yes/no questions is derived from the old labels "
          "assuming the compressor runs only while the fan does.")

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

    runs: dict[str, ev.ObservationRun] = {}
    published: dict[str, dict[str, np.ndarray]] = {}
    candidate: dict[str, dict[str, np.ndarray]] = {}

    for version in wanted:
        config = configs[version]

        if version == "v1":
            runs[version] = ev.run_v1(
                version, events, config.rule(),
                config.median_seconds, config.hold_seconds,
            )
        else:
            runs[version] = ev.run_v2(
                version, events, config.rule(),
                config.median_seconds, config.hold_seconds,
            )

        per_event = ev.run_confusions(events, runs[version])
        published[version] = {
            o: m.sum(axis=0) for o, m in per_event.items()
        }
        candidate[version] = {
            o: m.sum(axis=0)
            for o, m in ev.run_confusions(
                events, runs[version], "candidate"
            ).items()
        }

        print(f"\n=== {version}: {sources[version]}")
        api = config.to_api()
        print("    " + ", ".join(
            f"{k}={v}" for k, v in api.items()
            if k != "classifierVersion" and not k.startswith("beep")
        ))
        print_confusions(
            f"Published, window level ({version})", published[version]
        )
        print_headline(version, observation_metrics(published[version]))

        write_csv(
            args.out / f"confusion_{version}.csv", CONFUSION_HEADER,
            confusion_rows(published[version]),
        )

    # ---- reference: v1 at its code default, to show why -38 matters ----
    reference = None

    if "v1" in wanted:
        default = ClassifierConfig.for_version("v1")

        if default.rule() != configs["v1"].rule():
            baseline = ev.run_v1(
                "v1-default", events, default.rule(),
                default.median_seconds, default.hold_seconds,
            )
            reference = {
                "compressorThreshold": default.compressor_threshold,
                **observation_metrics({
                    o: m.sum(axis=0)
                    for o, m in ev.run_confusions(events, baseline).items()
                }),
            }

    header, rows = per_event_rows(events, runs)
    write_csv(args.out / "per_event.csv", header, rows)

    interference_rows = per_interference_rows(events, runs)
    write_csv(
        args.out / "per_interference.csv", PER_INTERFERENCE_HEADER,
        interference_rows,
    )

    print("By interference (published; share of windows answered "
          "correctly)")
    print(table(
        [
            [r[0], r[1], r[2], r[3], str(r[4]), f"{r[8]:.1%}"]
            for r in interference_rows
        ],
        ["classifier", "observation", "interference", "really",
         "windows", "correct"],
    ))

    # ---- search and validation (v2) ----------------------------------------
    search_summary = None

    if "v2" in wanted and not args.no_search:
        config = configs["v2"]
        rules = config.rule()

        fast = ev.FastEvaluator(events, config.median_seconds,
                                config.hold_seconds)

        checked = ev.verify_fast_path(
            events, rules, config.median_seconds, config.hold_seconds,
            limit=min(20, len(events)),
        )

        search = Search(fast)

        # -- which features say anything at all ---------------------------
        fan_energy = (
            (fast.column("500-1k") >= rules.fan.mid_threshold)
            & (fast.column("1k-2k") >= rules.fan.high_threshold)
            if rules.fan.require_both else
            (fast.column("500-1k") >= rules.fan.mid_threshold)
            | (fast.column("1k-2k") >= rules.fan.high_threshold)
        )

        everywhere = np.ones(len(fast.matrix), dtype=bool)
        fan_separation = separation(fast, "fan", fan_energy)
        compressor_separation = separation(
            fast, "compressor", everywhere
        )

        write_csv(
            args.out / "feature_separation.csv", FEATURE_SEPARATION_HEADER,
            fan_separation + compressor_separation,
        )

        print("\nFeatures that tell a running fan from none, among the "
              "windows that already show fan energy (top 6 by AUC)")
        print(table(
            [
                [r[1], r[2], f"{r[3]:.3f}", f"{r[4]:.3g}",
                 f"{r[5]:.3f}", f"{r[6]:.3g}", f"{r[7]:.3g}"]
                for r in fan_separation[:6]
            ],
            ["feature", "fan is", "AUC", "best thr", "bal.acc",
             "fan med", "none med"],
        ))
        print("\nFeatures that tell a running compressor from none "
              "(top 4 by AUC)")
        print(table(
            [
                [r[1], r[2], f"{r[3]:.3f}", f"{r[4]:.3g}",
                 f"{r[5]:.3f}", f"{r[6]:.3g}", f"{r[7]:.3g}"]
                for r in compressor_separation[:4]
            ],
            ["feature", "comp is", "AUC", "best thr", "bal.acc",
             "comp med", "none med"],
        ))

        # -- the sweeps -----------------------------------------------------
        sweeps: dict[str, list[dict]] = {}

        for feature in STABILITY_FEATURES:
            sweeps[feature] = [
                search.fan_stability(rules.fan, feature, float(value))
                for value in threshold_grid(fast.column(feature))
            ]

        for mid in np.arange(-66.0, -55.5, 1.0):
            for high in np.arange(-69.0, -58.5, 1.0):
                search.fan_bands(rules.fan, float(mid), float(high))

        compressor_sweep = [
            search.compressor(float(value))
            for value in np.arange(-50.0, -25.5, 1.0)
        ]

        header, rows = search.table()
        write_csv(args.out / "candidate_thresholds.csv", header, rows)

        v1_fan = (
            observation_metrics(published["v1"])["fan"]
            if "v1" in published else None
        )

        print(f"\nThreshold search: {len(search.rows)} settings tried "
              f"(fast path checked against the real smoother on "
              f"{checked} events)")
        print("\nFan stationarity gate by feature (everything else held "
              "at the configured values)")

        summary_rows = []
        feature_results = {}

        for feature, sweep in sweeps.items():
            found = plateau(sweep, PLATEAU_TOLERANCE)
            feature_results[feature] = found
            best = next(r for r in sweep if r["value"] == found["best"])
            summary_rows.append([
                feature,
                f"{found['best']:.3g}",
                f"{found['plateau'][0]:.3g}..{found['plateau'][1]:.3g}",
                f"{found['bestBalanced']:.3f}",
                pct(best["recall"]), pct(best["specificity"]),
                str(best["false_yes"]), str(best["false_no"]),
            ])

        print(table(
            summary_rows,
            ["feature", "best thr", "plateau", "balanced", "fan found",
             "no-fan ok", "false yes", "false no"],
        ))

        feature = rules.fan.stability_feature
        sweep = sweeps[feature]
        found = feature_results[feature]
        values = sorted({r["value"] for r in sweep})

        fan_validation = leave_one_group_out(
            np.stack([
                search.stacks[("fan_stability", feature, value)]
                for value in values
            ]),
            search.groups, values,
        )

        compressor_values = sorted({r["value"] for r in compressor_sweep})
        compressor_found = plateau(compressor_sweep, PLATEAU_TOLERANCE)
        compressor_validation = leave_one_group_out(
            np.stack([
                search.stacks[("compressor", value)]
                for value in compressor_values
            ]),
            search.groups, compressor_values,
        )

        print(f"\nFan gate on {feature}: best {found['best']:.3g} dB, "
              f"plateau {found['plateau'][0]:.3g}.."
              f"{found['plateau'][1]:.3g}, recommended "
              f"{found['recommended']:.3g} dB; configured "
              f"{rules.fan.stability_threshold:g} dB")
        print(f"  leave-one-group-out ({fan_validation['groups']} "
              f"groups): balanced {fan_validation['balancedAccuracy']:.3f}"
              f" vs in-sample best {found['bestBalanced']:.3f}; chose "
              f"{fan_validation['valuesChosen']['min']:.3g}.."
              f"{fan_validation['valuesChosen']['max']:.3g}")
        print(f"Compressor threshold: best {compressor_found['best']:g} "
              f"dB, plateau {compressor_found['plateau'][0]:g}.."
              f"{compressor_found['plateau'][1]:g}; configured "
              f"{rules.compressor.threshold:g} dB")
        print(f"  leave-one-group-out: balanced "
              f"{compressor_validation['balancedAccuracy']:.3f} vs "
              f"in-sample best {compressor_found['bestBalanced']:.3f}")

        search_summary = {
            "settingsTried": len(search.rows),
            "fastPathVerifiedEvents": checked,
            "fan": {
                "feature": feature,
                **found,
                "configured": rules.fan.stability_threshold,
                "leaveOneGroupOut": fan_validation,
                "byFeature": feature_results,
            },
            "compressor": {
                **compressor_found,
                "configured": rules.compressor.threshold,
                "leaveOneGroupOut": compressor_validation,
            },
            "topSeparatingFeatures": {
                "fan": [
                    {"feature": r[1], "auc": r[3], "presentIs": r[2]}
                    for r in fan_separation[:5]
                ],
                "compressor": [
                    {"feature": r[1], "auc": r[3], "presentIs": r[2]}
                    for r in compressor_separation[:5]
                ],
            },
        }

        if not args.no_plots:
            plot_path = (
                result_paths("v2").plots / "fan_stability_threshold.png"
                if args.out == result_paths("v2").evaluation
                else args.out / "fan_stability_threshold.png"
            )
            plot_sweep(plot_path, sweep, feature, found["recommended"],
                       v1_fan)
            print(f"Plot: {plot_path}")

    # ---- beeps across every event -----------------------------------------
    beeps = None

    if "v2" in wanted and not args.no_beeps:
        beeps = beep_summary(records, cache, configs["v2"].beep_config())

        print(f"\nBeeps: {beeps['beeps']} found in "
              f"{beeps['minutesOfAudio']:.0f} minutes of event audio")
        print(table(
            [
                [kind, str(v["events"]), str(v["withBeep"])]
                for kind, v in beeps["byTransition"].items()
            ],
            ["reviewed as", "events", "with a beep"],
        ))

        for kind, v in beeps["beforePublishedTransition"].items():
            print(f"  {kind}: {v['count']} beeps "
                  f"{-v['medianSeconds']:.2f} s before the published "
                  f"transition (range {-v['maxSeconds']:.2f}.."
                  f"{-v['minSeconds']:.2f} s)")

    # ---- summary ---------------------------------------------------------------
    summary = {
        "generatedAt": datetime.now().isoformat(timespec="seconds"),
        "directories": [str(d) for d in directories],
        "parameters": {
            "warmupSeconds": args.warmup,
            "guardSeconds": args.guard,
            "groupGapSeconds": args.group_gap,
            "truthAssumption": (
                "compressor runs only while the fan does: "
                "OFF=(no,no) FAN=(yes,no) COMPRESSOR=(yes,yes)"
            ),
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
                "published": observation_metrics(published[version]),
                "candidate": observation_metrics(candidate[version]),
            }
            for version in wanted
        },
        "reference": (
            {"v1CodeDefault": reference} if reference is not None else None
        ),
        "search": search_summary,
        "beeps": beeps,
        "seconds": round(time.perf_counter() - started, 1),
    }

    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, default=float) + "\n",
        encoding="utf-8",
    )

    if len(wanted) == 2:
        one = observation_metrics(published["v1"])
        two = observation_metrics(published["v2"])

        def change(before: int, after: int) -> str:
            if not before:
                return "n/a"
            return f"{(after - before) / before:+.0%}"

        print("\nv1 -> v2, window level, published")

        for observation in ev.OBSERVATIONS:
            a, b = one[observation], two[observation]
            print(f"  {observation}:")
            print(f"    reported but not running  {a['falsePositive']:6d} -> "
                  f"{b['falsePositive']:6d}  "
                  f"({change(a['falsePositive'], b['falsePositive'])})")
            print(f"    running but not reported  {a['falseNegative']:6d} -> "
                  f"{b['falseNegative']:6d}  "
                  f"({change(a['falseNegative'], b['falseNegative'])})")
            print(f"    recall                    {pct(a['recall'])} -> "
                  f"{pct(b['recall'])}")
            print(f"    specificity               "
                  f"{pct(a['specificity'])} -> {pct(b['specificity'])}")
            print(f"    balanced accuracy         "
                  f"{a['balancedAccuracy']:.3f} -> "
                  f"{b['balancedAccuracy']:.3f}")

    print(f"\nWrote {args.out}")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
