"""Replaying reviewed events: ground truth, grouping, scoring, the CLI."""

import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools" / "audio"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate_events  # noqa: E402
import evaluation as ev  # noqa: E402
import synthetic  # noqa: E402
from backend.config import ClassifierConfig  # noqa: E402
from classifier.v2 import ThresholdsV2  # noqa: E402

START = datetime(2026, 10, 1, 12, 0, 0)


def write_event(
    directory: Path,
    moment: datetime,
    fixture: str | None = None,
    samples: np.ndarray | None = None,
    reviewed=("OFF", "OFF"),
    interference=(),
    recorded=("FAN", "COMPRESSOR"),
    pre=12.0,
    seconds=24,
    version=None,
):
    """A saved event: WAV, and JSON with whatever review is asked for."""
    identifier = f"{moment:%Y-%m-%d_%H%M%S}_{recorded[0]}_to_{recorded[1]}"
    wav = directory / f"{identifier}.wav"

    if samples is None:
        samples = synthetic.make(fixture, seconds=seconds)

    synthetic.write_wav(wav, samples)

    metadata = {
        "id": identifier,
        "time": moment.isoformat(timespec="seconds"),
        # What the live classifier *thought* -- never the answer.
        "from": recorded[0],
        "to": recorded[1],
        "source": "transition",
        "audio": {
            "file": wav.name,
            "sampleRate": synthetic.SAMPLE_RATE,
            "seconds": len(samples) / synthetic.SAMPLE_RATE,
            "preSeconds": pre,
            "postSeconds": len(samples) / synthetic.SAMPLE_RATE - pre,
        },
        "classifierConfig": {},
    }

    if version:
        metadata["classifierVersion"] = version

    if reviewed is None:
        metadata["review"] = {"status": "unreviewed"}
    else:
        metadata["review"] = {
            "status": "reviewed",
            "classificationCorrect": False,
            "actualFrom": reviewed[0],
            "actualTo": reviewed[1],
            "interference": list(interference),
            "notes": "",
        }

    (directory / f"{identifier}.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    return identifier


class TruthTests(unittest.TestCase):
    def record(self, directory, **kwargs):
        identifier = write_event(directory, START, fixture="off_clean",
                                 **kwargs)
        return ev.parse_event(directory / f"{identifier}.json")

    def test_steady_event_is_labelled_after_the_warmup(self):
        with tempfile.TemporaryDirectory() as d:
            record = self.record(Path(d), reviewed=("FAN", "FAN"))
            times = np.arange(0, 24, 0.032)
            truth = ev.truth_labels(record, times, warmup=5.0, guard=4.0)

        self.assertTrue((truth[times < 5.0] == ev.UNSCORED).all())
        self.assertTrue((truth[times >= 5.0] == ev.LABEL_CODE["FAN"]).all())

    def test_transition_is_labelled_either_side_with_a_guard(self):
        with tempfile.TemporaryDirectory() as d:
            record = self.record(Path(d), reviewed=("FAN", "COMPRESSOR"),
                                 pre=12.0)
            times = np.arange(0, 24, 0.032)
            truth = ev.truth_labels(record, times, warmup=5.0, guard=4.0)

        fan, comp = ev.LABEL_CODE["FAN"], ev.LABEL_CODE["COMPRESSOR"]

        self.assertTrue((truth[times < 5.0] == ev.UNSCORED).all())
        self.assertTrue(
            (truth[(times >= 5.0) & (times < 8.0)] == fan).all()
        )
        # Four seconds either side of the moment (12 s) are not scored.
        middle = (times > 8.0) & (times < 16.0)
        self.assertTrue((truth[middle] == ev.UNSCORED).all())
        self.assertTrue((truth[times > 16.0] == comp).all())

    def test_the_classifiers_own_from_and_to_are_never_the_answer(self):
        with tempfile.TemporaryDirectory() as d:
            record = self.record(
                Path(d), reviewed=("OFF", "OFF"),
                recorded=("COMPRESSOR", "COMPRESSOR"),
            )
            times = np.arange(0, 24, 0.032)
            truth = ev.truth_labels(record, times)

        scored = truth[truth != ev.UNSCORED]
        self.assertTrue((scored == ev.LABEL_CODE["OFF"]).all())

    def test_unreviewed_and_unknown_events_carry_no_label(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            times = np.arange(0, 24, 0.032)

            for index, reviewed in enumerate(
                (None, ("UNKNOWN", "UNKNOWN"), ("OFF", "UNKNOWN"))
            ):
                identifier = write_event(
                    directory, START + timedelta(minutes=index),
                    fixture="off_clean", reviewed=reviewed,
                )
                record = ev.parse_event(directory / f"{identifier}.json")

                self.assertFalse(record.labelled, reviewed)
                self.assertTrue(
                    (ev.truth_labels(record, times) == ev.UNSCORED).all()
                )

    def test_dataset_counts_report_but_do_not_score_the_rest(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)

            for index, (review, tag) in enumerate((
                (("OFF", "OFF"), ("talking",)),
                (("FAN", "FAN"), ()),
                (("FAN", "COMPRESSOR"), ()),
                (None, ()),
                (("UNKNOWN", "UNKNOWN"), ()),
            )):
                write_event(
                    directory, START + timedelta(minutes=10 * index),
                    fixture="off_clean", reviewed=review, interference=tag,
                    seconds=6, pre=3.0,
                )

            counts = ev.dataset_counts(ev.load_events([directory]))

        self.assertEqual(counts["events"], 5)
        self.assertEqual(counts["labelled"], 3)
        self.assertEqual(counts["unreviewed"], 1)
        self.assertEqual(counts["reviewedUnknown"], 1)
        self.assertEqual(counts["steady"], 2)
        self.assertEqual(counts["transitions"], 1)
        self.assertEqual(counts["byInterference"], {"clean": 2, "talking": 1})

    def test_events_missing_their_audio_are_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            identifier = write_event(directory, START, fixture="off_clean",
                                     seconds=6, pre=3.0)
            (directory / f"{identifier}.wav").unlink()

            self.assertEqual(ev.load_events([directory]), [])

    def test_old_events_without_a_version_count_as_v1(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            old = write_event(directory, START, fixture="off_clean",
                              seconds=6, pre=3.0)
            new = write_event(directory, START + timedelta(hours=1),
                              fixture="off_clean", seconds=6, pre=3.0,
                              version="v2")
            records = {
                r.id: r for r in ev.load_events([directory])
            }

        self.assertEqual(records[old].classifier_version, "v1")
        self.assertEqual(records[new].classifier_version, "v2")


class GroupingTests(unittest.TestCase):
    def records(self, offsets, seconds=30):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)

            for offset in offsets:
                write_event(
                    directory, START + timedelta(seconds=offset),
                    fixture="off_clean", seconds=6, pre=3.0,
                    reviewed=("OFF", "OFF"),
                )

            records = ev.load_events([directory])

        return records

    def test_overlapping_events_share_a_group(self):
        # Three events a few seconds apart overlap heavily; one is far.
        records = self.records([0, 4, 9, 4000])
        groups = ev.group_events(records, gap_seconds=0)

        ids = [r.id for r in records]
        self.assertEqual(groups[ids[0]], groups[ids[1]])
        self.assertEqual(groups[ids[1]], groups[ids[2]])
        self.assertNotEqual(groups[ids[2]], groups[ids[3]])

    def test_gap_joins_near_misses_and_chains(self):
        # 100 s apart: separate with no gap, one group with a 120 s gap,
        # and a chain stays one group however long it runs.
        records = self.records([0, 100, 200, 300, 400])
        ids = [r.id for r in records]

        apart = ev.group_events(records, gap_seconds=0)
        self.assertEqual(len(set(apart.values())), 5)

        chained = ev.group_events(records, gap_seconds=120)
        self.assertEqual(len(set(chained.values())), 1)

    def test_every_event_is_in_exactly_one_group(self):
        records = self.records([0, 5, 500, 505, 1000])
        groups = ev.group_events(records, gap_seconds=60)

        self.assertEqual(set(groups), {r.id for r in records})
        self.assertEqual(len(set(groups.values())), 3)

    def test_validation_never_trains_on_the_group_it_tests(self):
        """Leave-one-group-out picks each threshold without that group.

        Two groups prefer opposite thresholds. If the held-out group
        leaked into the choice, each would pick its own favourite; it
        must pick the *other* group's instead.
        """
        # per-event confusions, shape (3 classes, 4): only recall on FAN
        # differs. Threshold A suits group 0, threshold B suits group 1.
        def matrix(fan_right, fan_wrong):
            m = np.zeros((3, 4), dtype=np.int64)
            m[0, 0] = 100                      # OFF always right
            m[2, 2] = 100                      # COMPRESSOR always right
            m[1, 1], m[1, 0] = fan_right, fan_wrong
            return m

        sweep = [
            {"stability_threshold": 1.0, "compressor": -38.0,
             "fan_mid": -62.0, "fan_high": -65.0},
            {"stability_threshold": 2.0, "compressor": -38.0,
             "fan_mid": -62.0, "fan_high": -65.0},
        ]
        per_event = {
            ("stability", "1k-2k_std", 1.0, -38.0, -62.0, -65.0):
                np.stack([matrix(100, 0), matrix(0, 100)]),
            ("stability", "1k-2k_std", 2.0, -38.0, -62.0, -65.0):
                np.stack([matrix(0, 100), matrix(100, 0)]),
        }
        search = SimpleNamespace(
            groups=np.array([0, 1]), per_event=per_event,
        )

        result = evaluate_events.leave_one_group_out(
            search, sweep, "1k-2k_std"
        )

        # Group 0 held out -> trained on group 1 -> picks 2.0, which is
        # wrong for group 0. Likewise the other way. Every held-out FAN
        # window is therefore wrong: the estimate is honestly bad, not
        # flattered by seeing its own answer.
        self.assertEqual(result["recall"]["FAN"], 0.0)
        self.assertEqual(result["groups"], 2)
        self.assertEqual(result["thresholdsChosen"]["min"], 1.0)
        self.assertEqual(result["thresholdsChosen"]["max"], 2.0)


class ScoringTests(unittest.TestCase):
    def test_confusion_and_metrics(self):
        truth = np.array([0, 0, 0, 0, 1, 1, 2, 2, -1], dtype=np.int8)
        predicted = np.array([0, 0, 1, 2, 1, 0, 2, -1, 1], dtype=np.int8)

        matrix = ev.confusion(truth, predicted)

        self.assertEqual(matrix.tolist(), [
            [2, 1, 1, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ])
        self.assertEqual(int(matrix.sum()), 8)       # the -1 is unscored

        result = ev.metrics(matrix)
        self.assertEqual(result["offToFan"], 1)
        self.assertEqual(result["offToCompressor"], 1)
        self.assertEqual(result["fanToOff"], 1)
        self.assertEqual(result["compressorMisses"], 1)
        self.assertEqual(result["unpublished"], 1)
        self.assertAlmostEqual(result["recall"]["OFF"], 0.5)
        self.assertAlmostEqual(
            result["balancedAccuracy"], (0.5 + 0.5 + 0.5) / 3
        )

    def test_hold_publishes_after_the_candidate_has_persisted(self):
        # 10 windows/s and a 1 s hold: a candidate is published on its
        # 11th consecutive window (the Smoother needs the span between
        # first and last to reach the hold).
        rate = 10.0

        steady = np.zeros(20, dtype=np.int8)
        published = ev.hold_published(steady, rate, hold_seconds=1.0)

        self.assertTrue((published[:10] == -1).all())
        self.assertEqual(int(published[10]), 0)

        # A short run never gets there; a long one does; a later blip
        # does not displace it.
        candidates = np.array([0] * 5 + [1] * 30 + [0] * 3, dtype=np.int8)
        published = ev.hold_published(candidates, rate, hold_seconds=1.0)

        self.assertTrue((published[:15] == -1).all())     # OFF too short
        self.assertEqual(int(published[14]), -1)
        self.assertEqual(int(published[15]), 1)           # FAN earned
        self.assertEqual(int(published[-1]), 1)           # blip ignored

        # No hold: the candidate is the state.
        instant = ev.hold_published(candidates, rate, hold_seconds=0.0)
        self.assertEqual(int(instant[1]), 0)


class FixtureReplayTests(unittest.TestCase):
    """Curated stand-ins for the scenarios the classifier must separate."""

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        directory = Path(cls.temporary.name)

        for index, name in enumerate(synthetic.FIXTURES):
            expected = synthetic.EXPECTED[name]
            tag = {"off_talking": "talking", "off_tv": "tv",
                   "off_alarm": "other",
                   "off_impulsive": "other"}.get(name)
            write_event(
                directory, START + timedelta(hours=index), fixture=name,
                reviewed=(expected, expected),
                interference=(tag,) if tag else (),
            )

        records = ev.load_events([directory])
        cls.events = ev.prepare(records, ev.FeatureCache(None))
        cls.by_fixture = {
            name: event
            for name, event in zip(synthetic.FIXTURES, cls.events)
        }

        cls.v1 = ClassifierConfig().patched({"compressorThreshold": -38.0})
        cls.v2 = ClassifierConfig.for_version("v2")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def replay(self, config, name):
        event = self.by_fixture[name]
        candidate, state = ev.replay_exact(
            event.windows, config.rule(), config.median_seconds,
            config.hold_seconds,
        )
        scored = event.truth != ev.UNSCORED
        return candidate[scored], state[scored]

    def test_every_fixture_is_scored(self):
        self.assertEqual(len(self.events), len(synthetic.FIXTURES))

        for event in self.events:
            self.assertGreater(event.scored, 400)

    def test_clean_scenarios_end_in_the_right_state_under_both(self):
        for name in ("off_clean", "fan_clean", "compressor"):
            expected = ev.LABEL_CODE[synthetic.EXPECTED[name]]

            for config in (self.v1, self.v2):
                _, state = self.replay(config, name)

                self.assertTrue(
                    (state == expected).all(),
                    f"{name} under {config.version}: "
                    f"{np.unique(state, return_counts=True)}",
                )

    def test_clean_fan_is_stationary_so_v2_keeps_it(self):
        candidate, _ = self.replay(self.v2, "fan_clean")

        self.assertGreater((candidate == 1).mean(), 0.95)

    def test_v2_rejects_the_interference_that_makes_v1_flap(self):
        for name in ("off_talking", "off_tv", "off_alarm"):
            v1_candidate, v1_state = self.replay(self.v1, name)
            v2_candidate, v2_state = self.replay(self.v2, name)

            # These put real power in both fan bands, so v1 keeps
            # calling FAN -- on and off, never long enough to publish.
            self.assertGreater((v1_candidate == 1).mean(), 0.2, name)
            self.assertLess((v1_state == 1).mean(), 0.5, name)

            # v2 sees that none of it is steady.
            self.assertLess((v2_candidate == 1).mean(), 0.03, name)
            self.assertTrue((v2_state == 0).all(), name)

    def test_impulsive_noise_is_not_a_fan_to_either(self):
        for config in (self.v1, self.v2):
            candidate, state = self.replay(config, "off_impulsive")

            self.assertLess((candidate == 1).mean(), 0.02)
            self.assertTrue((state == 0).all())

    def test_compressor_is_unaffected_by_the_gate(self):
        for name in synthetic.FIXTURES:
            v1_candidate, _ = self.replay(self.v1, name)
            v2_candidate, _ = self.replay(self.v2, name)

            self.assertEqual(
                (v1_candidate == 2).tolist(), (v2_candidate == 2).tolist(),
                name,
            )

    def test_with_the_gate_open_v2_is_v1(self):
        open_gate = ClassifierConfig.for_version("v2").patched({
            "fanStabilityThreshold": 100.0,
            "fanStabilityMinSeconds": 0.0,
            "compressorThreshold": -38.0,
        })

        for name in synthetic.FIXTURES:
            for left, right in zip(
                self.replay(self.v1, name), self.replay(open_gate, name)
            ):
                self.assertTrue(np.array_equal(left, right), name)

    def test_fast_path_agrees_with_the_real_smoother(self):
        for hold in (0.0, 1.0, 2.0, 3.5):
            for median in (0.1, 0.5, 1.0):
                checked = ev.verify_fast_path(
                    self.events, self.v2.rule(), median, hold,
                )
                self.assertEqual(checked, len(self.events))

    def test_cache_returns_identical_features_and_notices_changes(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            identifier = write_event(directory, START, fixture="fan_clean",
                                     seconds=8, pre=4.0,
                                     reviewed=("FAN", "FAN"))
            record = ev.parse_event(directory / f"{identifier}.json")

            cache = ev.FeatureCache(directory / "cache")
            first = cache.get(record)
            second = cache.get(record)

            self.assertEqual((cache.misses, cache.hits), (1, 1))
            self.assertTrue(np.array_equal(first.matrix, second.matrix))
            self.assertEqual(first.names, second.names)

            # Replacing the audio invalidates the entry.
            synthetic.write_wav(record.wav_path,
                                synthetic.make("off_clean", seconds=8))
            third = cache.get(record)

            self.assertEqual(cache.misses, 2)
            self.assertFalse(np.array_equal(first.matrix, third.matrix))


class EvaluateEventsCliTests(unittest.TestCase):
    def test_end_to_end_writes_every_artifact(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            events = root / "events"
            out = root / "evaluation"
            events.mkdir()

            for index, name in enumerate(synthetic.FIXTURES):
                expected = synthetic.EXPECTED[name]
                write_event(
                    events, START + timedelta(hours=index), fixture=name,
                    reviewed=(expected, expected),
                    interference=("talking",) if "talking" in name else (),
                )

            # One unreviewed and one UNKNOWN event, which must not count.
            write_event(events, START + timedelta(hours=20),
                        fixture="fan_clean", reviewed=None)
            write_event(events, START + timedelta(hours=21),
                        fixture="fan_clean",
                        reviewed=("UNKNOWN", "UNKNOWN"))

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = evaluate_events.main([
                    str(events), "--out", str(out),
                    # Hermetic: do not read the developer's saved configs.
                    "--v1-config", str(root / "none1.json"),
                    "--v2-config", str(root / "none2.json"),
                ])

            self.assertEqual(code, 0)

            for name in (
                "summary.json", "confusion_v1.csv", "confusion_v2.csv",
                "per_event.csv", "per_interference.csv",
                "feature_separation.csv", "candidate_thresholds.csv",
                "fan_stability_threshold.png",
            ):
                self.assertTrue((out / name).is_file(), name)

            summary = json.loads((out / "summary.json").read_text())

            dataset = summary["dataset"]
            self.assertEqual(dataset["events"], len(synthetic.FIXTURES) + 2)
            self.assertEqual(dataset["labelled"], len(synthetic.FIXTURES))
            self.assertEqual(dataset["unreviewed"], 1)
            self.assertEqual(dataset["reviewedUnknown"], 1)
            self.assertEqual(dataset["scoredEvents"], len(synthetic.FIXTURES))

            v1, v2 = (summary["classifiers"][v] for v in ("v1", "v2"))
            self.assertEqual(v1["config"]["classifierVersion"], "v1")
            self.assertEqual(v2["config"]["classifierVersion"], "v2")

            # v2 should not be worse than v1 on this set, and it must
            # remove the interference-driven FAN candidates.
            self.assertGreaterEqual(
                v2["published"]["balancedAccuracy"],
                v1["published"]["balancedAccuracy"],
            )
            self.assertLessEqual(
                v2["published"]["offToFan"], v1["published"]["offToFan"]
            )
            self.assertEqual(v2["published"]["offToFan"], 0)

            self.assertGreater(summary["search"]["settingsTried"], 100)
            self.assertGreaterEqual(
                summary["search"]["fastPathVerifiedEvents"], 1
            )

            with (out / "per_event.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), len(synthetic.FIXTURES))
            # Groups are present, and truth is the review, not "from/to".
            self.assertTrue(all(row["group"] != "" for row in rows))
            self.assertEqual(
                {row["actual_to"] for row in rows},
                {"OFF", "FAN", "COMPRESSOR"},
            )

            # The input events were only ever read.
            self.assertEqual(
                len(list(events.glob("*.json"))),
                len(synthetic.FIXTURES) + 2,
            )

    def test_single_classifier_and_override(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            events = root / "events"
            events.mkdir()
            write_event(events, START, fixture="fan_clean",
                        reviewed=("FAN", "FAN"))

            with contextlib.redirect_stdout(io.StringIO()):
                code = evaluate_events.main([
                    str(events), "--out", str(root / "out"),
                    "--classifier", "v2", "--no-search", "--no-plots",
                    "--v2-config", str(root / "none.json"),
                    "--override", "fanStabilityThreshold=1.5",
                ])

            summary = json.loads(
                (root / "out" / "summary.json").read_text()
            )

        self.assertEqual(code, 0)
        self.assertEqual(list(summary["classifiers"]), ["v2"])
        self.assertEqual(
            summary["classifiers"]["v2"]["config"]["fanStabilityThreshold"],
            1.5,
        )
        self.assertIsNone(summary["search"])

    def test_nothing_reviewed_is_an_error_not_a_zero_score(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            events = root / "events"
            events.mkdir()
            write_event(events, START, fixture="fan_clean", reviewed=None)

            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                code = evaluate_events.main([
                    str(events), "--out", str(root / "out"),
                    "--v1-config", str(root / "a.json"),
                    "--v2-config", str(root / "b.json"),
                ])

        self.assertNotEqual(code, 0)

    def test_a_config_for_the_other_version_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "v1.json"
            path.write_text(json.dumps({"compressorThreshold": -41.0}))

            with self.assertRaises(SystemExit):
                evaluate_events.load_config("v2", path)

            config, _ = evaluate_events.load_config("v1", path)
            self.assertEqual(config.compressor_threshold, -41.0)


if __name__ == "__main__":
    unittest.main()
