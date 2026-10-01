"""The model comparison is analysis only, and must be leak-free."""

import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools" / "audio"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluation as ev  # noqa: E402
import synthetic  # noqa: E402
from test_evaluation import START, write_event  # noqa: E402

try:
    import sklearn  # noqa: F401
except ImportError:                                  # pragma: no cover
    sklearn = None

if sklearn is not None:
    import compare_models  # noqa: E402


@unittest.skipIf(sklearn is None, "scikit-learn is analysis-only")
class CompareModelsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        events = cls.root / "events"
        events.mkdir()

        # Two of each scenario an hour apart, so there are enough groups
        # for real folds and each fold has something to learn from.
        for repeat in range(2):
            for index, name in enumerate(synthetic.FIXTURES):
                expected = synthetic.EXPECTED[name]
                write_event(
                    events,
                    START + timedelta(
                        hours=repeat * 20 + index, minutes=repeat
                    ),
                    samples=synthetic.make(name, seed=100 * repeat + 7),
                    reviewed=(expected, expected),
                )

        records = ev.load_events([events])
        cls.events = ev.prepare(records, ev.FeatureCache(None))
        cls.events_dir = events

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def dataset(self):
        return compare_models.Dataset(self.events, 0.5, 2.0)

    def test_folds_never_put_a_group_on_both_sides(self):
        dataset = self.dataset()
        seen = []

        for train, test in compare_models.grouped_folds(dataset.groups, 5):
            self.assertFalse(
                set(dataset.groups[train]) & set(dataset.groups[test])
            )
            seen.extend(test.tolist())

        # Every event is held out exactly once.
        self.assertEqual(sorted(seen), list(range(len(self.events))))

    def test_windows_of_one_event_are_never_split(self):
        # The leak the comparison exists to avoid: training on one part
        # of an event and testing on another.
        dataset = self.dataset()

        for train_events, test_events in compare_models.grouped_folds(
            dataset.groups, 5
        ):
            in_train = np.isin(dataset.event_of, train_events)
            in_test = np.isin(dataset.event_of, test_events)

            self.assertFalse((in_train & in_test).any())

            for event_index in range(len(self.events)):
                window = dataset.event_of == event_index
                self.assertIn(
                    int(in_train[window].sum()), (0, int(window.sum()))
                )

    def test_published_path_matches_the_evaluator(self):
        # Predictions must reach the score through the same hold the
        # detector uses, or "model vs rule" compares different things.
        from backend.config import ClassifierConfig

        rules = ClassifierConfig.for_version("v2").rule()
        dataset = self.dataset()
        candidates = dataset.fast.candidates(rules)
        theirs = dataset.fast.published(rules)

        for observation in ev.OBSERVATIONS:
            ours = dataset.published(candidates[observation])

            for left, right in zip(ours, theirs[observation]):
                self.assertTrue(np.array_equal(left, right))

    def test_models_are_binary_per_observation(self):
        dataset = self.dataset()

        for observation in ev.OBSERVATIONS:
            labels = set(np.unique(dataset.truth[observation]).tolist())
            self.assertLessEqual(labels, {ev.UNSCORED, 0, 1})
            self.assertEqual(len(dataset.truth[observation]),
                             len(dataset.X))

    def test_the_comparison_runs_and_writes_its_artifacts(self):
        out = self.root / "models"

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = compare_models.main([
                str(self.events_dir), "--out", str(out), "--folds", "4",
                "--v1-config", str(self.root / "a.json"),
                "--v2-config", str(self.root / "b.json"),
            ])

        self.assertEqual(code, 0)

        for name in ("models.csv", "models.json",
                     "models_per_interference.csv", "tree_rules.txt"):
            self.assertTrue((out / name).is_file(), name)

        summary = json.loads((out / "models.json").read_text())

        self.assertEqual(summary["folds"], 4)
        self.assertEqual(summary["materialGain"], 0.02)

        self.assertEqual(set(summary["rows"]), {"fan", "compressor"})

        for observation, rows in summary["rows"].items():
            names = {row["model"] for row in rows}
            self.assertTrue(
                {"v1 rule", "v2 detector", "logistic", "forest"} <= names,
                observation,
            )
            self.assertIn("tree_depth3", names)

            # Clean scenarios are easy: nothing should be worse than
            # chance.
            for row in rows:
                if row["model"].startswith("tree_depth1"):
                    continue        # a stump, deliberately degenerate
                self.assertGreater(row["balanced"], 0.5,
                                   (observation, row["model"]))

            self.assertIn(observation, summary["verdicts"])

        self.assertIn("|---", (out / "tree_rules.txt").read_text())

    def test_the_backend_never_imports_the_analysis_tools(self):
        for path in (TOOLS / "backend").glob("*.py"):
            text = path.read_text(encoding="utf-8")

            self.assertNotIn("sklearn", text, path.name)
            self.assertNotIn("compare_models", text, path.name)

        requirements = (TOOLS / "requirements.txt").read_text()
        self.assertNotIn("scikit-learn", requirements)


if __name__ == "__main__":
    unittest.main()
