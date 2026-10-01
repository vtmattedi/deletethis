"""Does any learned model beat the simple detectors? An analysis tool.

    python tools/audio/compare_models.py tools/audio/results/events

Fits logistic regression, shallow decision trees and a small random
forest to the same reviewed events the detectors are scored on, one
yes/no question at a time -- *is the fan running*, *is the compressor
running* -- and compares them with the v1 and v2 rules under identical
conditions. It exists to answer one question honestly -- is a model
worth the complexity? -- and the expected answer is usually no.

This is analysis only. Nothing in the live backend imports scikit-learn
or this file, and a model that scores well does not become a detector
because of it. If a shallow tree comes close to the best model, the
right move is to read its splits and fold them into the deterministic
detector, which is cheap to run, easy to explain and can be tuned from
the UI.

How the comparison is kept fair
-------------------------------
* Same inputs. Models see exactly what the detectors see: the median-
  smoothed feature values for each window.
* Same decision path. Window predictions go through the same
  publication hold as the detectors, so every number is "published",
  directly comparable with ``evaluate_events.py``.
* Grouped validation. Windows within an event are near-copies and
  neighbouring events overlap in time, so folds are split by *group*
  (events within ``--group-gap`` seconds of each other), never by
  window. A model is always scored on audio from groups it never saw.
  Splitting by window would let it memorise the room and report a
  score it cannot repeat.
* The detectors are not fitted here. v1 and v2 are scored as
  configured, so the models are the ones being asked to earn their place.

The yes/no truth comes from the old three-way reviews under the same
stated assumption as ``evaluate_events.py``: the compressor runs only
while the fan does.

With a few hundred labelled seconds and a handful of interference
tags, differences of a point or two are noise. Read the per-interference
table before the headline.

Needs scikit-learn (``pip install -r tools/audio/requirements-analysis.txt``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate_events as ee  # noqa: E402
import evaluation as ev  # noqa: E402
from backend.config import result_paths  # noqa: E402
from features import FEATURE_NAMES  # noqa: E402

# A model has to beat the detector by this much balanced accuracy, under
# grouped validation, before it is worth discussing.
MATERIAL_GAIN = 0.02

# A tree within this of the best model is preferred: it can be read,
# and turned into a rule.
TREE_TOLERANCE = 0.01


def require_sklearn():
    try:
        import sklearn  # noqa: F401
    except ImportError:
        raise SystemExit(
            "compare_models.py needs scikit-learn, which the backend "
            "deliberately does not depend on:\n"
            "    pip install -r tools/audio/requirements-analysis.txt"
        )


def model_zoo(seed: int = 0) -> dict:
    """The candidates. Small on purpose: this is a small dataset."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    zoo = {
        "logistic": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=seed
            ),
        ),
    }

    for depth in (1, 2, 3, 4):
        zoo[f"tree_depth{depth}"] = lambda depth=depth: (
            DecisionTreeClassifier(
                max_depth=depth,
                # Leaves must rest on real evidence, not a few windows.
                min_samples_leaf=200,
                class_weight="balanced",
                random_state=seed,
            )
        )

    zoo["forest"] = lambda: RandomForestClassifier(
        n_estimators=60,
        max_depth=6,
        min_samples_leaf=100,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=1,
    )

    return zoo


class Dataset:
    """Every window of every scored event, as a model sees it."""

    def __init__(
        self,
        events: list[ev.ScoredEvent],
        median_seconds: float,
        hold_seconds: float,
    ) -> None:
        self.events = events
        self.hold_seconds = hold_seconds

        self.fast = ev.FastEvaluator(events, median_seconds, hold_seconds)

        index = {name: i for i, name in enumerate(self.fast.names)}
        self.columns = [index[name] for name in FEATURE_NAMES]
        self.X = self.fast.matrix[:, self.columns]

        # One yes/no truth per question, over every window.
        self.truth = {
            observation: np.concatenate(self.fast.truth[observation])
            for observation in ev.OBSERVATIONS
        }
        self.event_of = np.concatenate([
            np.full(b - a, i)
            for i, (a, b) in enumerate(self.fast.bounds)
        ])
        self.groups = np.array([e.group for e in events])

    def published(self, codes: np.ndarray) -> list[np.ndarray]:
        """Window predictions through each event's publication hold."""
        return [
            ev.hold_published(codes[a:b], rate, self.hold_seconds)
            for (a, b), rate in zip(self.fast.bounds, self.fast.rates)
        ]

    def confusions(
        self, observation: str, published: list[np.ndarray]
    ) -> np.ndarray:
        return np.stack([
            ev.confusion(truth, state)
            for truth, state in zip(self.fast.truth[observation], published)
        ])


def grouped_folds(groups: np.ndarray, folds: int):
    """Event-index folds in which no group is on both sides."""
    from sklearn.model_selection import GroupKFold

    count = min(folds, len(np.unique(groups)))
    splitter = GroupKFold(n_splits=count)
    indices = np.arange(len(groups))

    for train, test in splitter.split(indices, groups=groups):
        # The invariant everything else relies on.
        assert not set(groups[train]) & set(groups[test])
        yield train, test


def cross_validate(
    dataset: Dataset, observation: str, make_model, folds: int
) -> np.ndarray:
    """Out-of-fold yes/no predictions for every window."""
    truth = dataset.truth[observation]
    predicted = np.full(len(truth), -1, dtype=np.int8)

    for train_events, test_events in grouped_folds(dataset.groups, folds):
        train = np.isin(dataset.event_of, train_events) & (
            truth != ev.UNSCORED
        )
        test = np.isin(dataset.event_of, test_events)

        labels = truth[train]

        if len(np.unique(labels)) < 2:
            continue                    # nothing to learn from this fold

        model = make_model()
        model.fit(dataset.X[train], labels)
        predicted[test] = model.predict(dataset.X[test])

    return predicted


def row(name: str, matrix: np.ndarray, kind: str) -> dict:
    summary = ev.metrics(matrix)

    return {
        "model": name,
        "kind": kind,
        "balanced": summary["balancedAccuracy"],
        "recall": summary["recall"],
        "specificity": summary["specificity"],
        "false_yes": summary["falsePositive"],
        "false_no": summary["falseNegative"],
    }


def tree_rules(dataset: Dataset, observation: str, depth: int) -> str:
    """The splits of a tree fitted on everything, for reading."""
    from sklearn.tree import DecisionTreeClassifier, export_text

    truth = dataset.truth[observation]
    scored = truth != ev.UNSCORED

    tree = DecisionTreeClassifier(
        max_depth=depth, min_samples_leaf=200, class_weight="balanced",
        random_state=0,
    ).fit(dataset.X[scored], truth[scored])

    return export_text(
        tree, feature_names=list(FEATURE_NAMES),
        class_names=["absent", "present"],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare learned models with the v1/v2 detectors.",
    )
    parser.add_argument("events", nargs="*", type=Path)
    parser.add_argument("--v2-config", type=Path, default=None)
    parser.add_argument("--v1-config", type=Path, default=None)
    parser.add_argument(
        "--out", type=Path, default=result_paths("v2").evaluation,
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--warmup", type=float,
                        default=ev.DEFAULT_WARMUP_SECONDS)
    parser.add_argument("--guard", type=float,
                        default=ev.DEFAULT_GUARD_SECONDS)
    parser.add_argument("--group-gap", type=float,
                        default=ev.DEFAULT_GROUP_GAP_SECONDS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require_sklearn()

    directories = list(args.events) or [ee.DEFAULT_V1_EVENTS]
    v2_events = result_paths("v2").events

    if not args.events and v2_events.is_dir():
        directories.append(v2_events)

    records = ev.load_events(directories)
    cache = ev.FeatureCache(args.cache_dir or args.out / "cache")
    events = ev.prepare(records, cache, args.warmup, args.guard,
                        args.group_gap)

    if not events:
        print("No reviewed events to learn from.", file=sys.stderr)
        return 1

    v2, v2_source = ee.load_config("v2", args.v2_config)
    v1, v1_source = ee.load_config("v1", args.v1_config)

    dataset = Dataset(events, v2.median_seconds, v2.hold_seconds)
    groups = len(np.unique(dataset.groups))
    scored = int((dataset.truth["fan"] != ev.UNSCORED).sum())

    print()
    print(f"{len(events)} events, {scored} scored windows, {groups} groups; "
          f"{min(args.folds, groups)}-fold validation split by group")
    print(f"Detectors as configured: v1 from {v1_source}, v2 from "
          f"{v2_source}")

    v1_run = ev.run_v1("v1", events, v1.rule(), v1.median_seconds,
                       v1.hold_seconds)
    v1_matrices = ev.run_confusions(events, v1_run)

    rules = v2.rule()
    v2_published = dataset.fast.published(rules)

    results: dict[str, list[tuple[str, np.ndarray, str]]] = {}
    tags = [e.record.interference_label for e in events]
    per_tag: dict[tuple, np.ndarray] = {}
    rows: dict[str, list[dict]] = {}

    zoo = model_zoo()

    for observation in ev.OBSERVATIONS:
        entries = [
            ("v1 rule", v1_matrices[observation], "rule"),
            ("v2 detector",
             dataset.confusions(observation, v2_published[observation]),
             "rule"),
        ]

        for name, make in zoo.items():
            codes = cross_validate(dataset, observation, make, args.folds)
            entries.append((
                name,
                dataset.confusions(observation, dataset.published(codes)),
                "model",
            ))

        results[observation] = entries
        rows[observation] = [
            row(name, matrix.sum(axis=0), kind)
            for name, matrix, kind in entries
        ]

        for name, matrix, _ in entries:
            for tag in sorted(set(tags)):
                mask = np.array([t == tag for t in tags])
                per_tag[(observation, name, tag)] = matrix[mask].sum(axis=0)

    # ---- report ------------------------------------------------------------
    verdicts = {}
    best_models = {}

    for observation in ev.OBSERVATIONS:
        print(f"\n=== {observation}: published, window level, "
              "out-of-fold for models")
        print(ee.table(
            [
                [
                    r["model"], f"{r['balanced']:.3f}",
                    ee.pct(r["recall"]), ee.pct(r["specificity"]),
                    str(r["false_yes"]), str(r["false_no"]),
                ]
                for r in rows[observation]
            ],
            ["model", "balanced", "recall", "specificity", "false yes",
             "false no"],
        ))

        by_name = {r["model"]: r for r in rows[observation]}
        detector = by_name["v2 detector"]["balanced"]
        models = [r for r in rows[observation] if r["kind"] == "model"]
        best = max(models, key=lambda r: r["balanced"])
        trees = [r for r in models if r["model"].startswith("tree")]
        best_tree = max(trees, key=lambda r: r["balanced"])

        gain = best["balanced"] - detector
        tree_gap = best["balanced"] - best_tree["balanced"]

        print(f"\nBest model: {best['model']} at {best['balanced']:.3f}; "
              f"v2 detector {detector:.3f} (difference {gain:+.3f})")

        if gain < MATERIAL_GAIN:
            verdict = (
                f"No model beats the {observation} detector by "
                f"{MATERIAL_GAIN:.2f}. Keep the detector: it is as good "
                "on this data and far simpler."
            )
        elif tree_gap <= TREE_TOLERANCE:
            verdict = (
                f"{best['model']} beats the {observation} detector, but "
                f"{best_tree['model']} is within {TREE_TOLERANCE:.2f} of "
                "it. Read the tree's splits and fold them into the "
                "detector rather than deploying a model."
            )
        else:
            verdict = (
                f"{best['model']} beats the {observation} detector and "
                "no tree matches it. Worth investigating, but check it "
                "on events recorded after this dataset before trusting "
                "the gap."
            )

        print(verdict)

        verdicts[observation] = {
            "best": best["model"],
            "gainOverDetector": gain,
            "verdict": verdict,
        }
        best_models[observation] = best

    print("\nBy interference: balanced accuracy over the truth values "
          "present")

    for observation in ev.OBSERVATIONS:
        names = [r["model"] for r in rows[observation]]
        table_rows = []

        for tag in sorted(set(tags)):
            cells = [tag]

            for name in names:
                result = ev.metrics(per_tag[(observation, name, tag)])
                cells.append(
                    "-" if result["balancedAccuracy"] is None
                    else f"{result['balancedAccuracy']:.2f}"
                )

            table_rows.append(cells)

        print(f"\n{observation}")
        print(ee.table(table_rows, ["interference"] + names))

    rules_text = {
        observation: tree_rules(dataset, observation, 3)
        for observation in ev.OBSERVATIONS
    }

    for observation, text in rules_text.items():
        print(f"\nDepth-3 tree for {observation}, fitted on everything "
              f"(for reading, not scored):\n{text}")

    # ---- artifacts ----------------------------------------------------------
    args.out.mkdir(parents=True, exist_ok=True)

    header = ["observation"] + list(rows["fan"][0])
    ee.write_csv(
        args.out / "models.csv", header,
        [
            [observation] + [
                r[h] if r[h] is not None else "" for h in header[1:]
            ]
            for observation in ev.OBSERVATIONS
            for r in rows[observation]
        ],
    )

    per_tag_rows = []

    for (observation, name, tag), matrix in per_tag.items():
        for truth in (0, 1):
            total = int(matrix[truth].sum())

            if total:
                per_tag_rows.append([
                    observation, name, tag, ee.TRUTH_NAME[truth], total,
                    float(matrix[truth, truth] / total),
                ])

    ee.write_csv(
        args.out / "models_per_interference.csv",
        ["observation", "model", "interference", "really", "windows",
         "correct"],
        per_tag_rows,
    )

    (args.out / "tree_rules.txt").write_text(
        "\n".join(
            f"=== {observation}\n{text}"
            for observation, text in rules_text.items()
        ),
        encoding="utf-8",
    )

    (args.out / "models.json").write_text(
        json.dumps({
            "events": len(events),
            "scoredWindows": scored,
            "groups": groups,
            "folds": min(args.folds, groups),
            "materialGain": MATERIAL_GAIN,
            "rows": rows,
            "verdicts": verdicts,
        }, indent=2, default=float) + "\n",
        encoding="utf-8",
    )

    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
