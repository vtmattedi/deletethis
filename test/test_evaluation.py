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
        """Leave-one-group-out picks each value without that group.

        Two groups prefer opposite values. If the held-out group leaked
        into the choice, each would pick its own favourite; it must pick
        the *other* group's instead.
        """
        def matrix(right, wrong):
            # truth yes: ``right`` called yes, ``wrong`` called no;
            # truth no: always right.
            m = np.zeros((2, 3), dtype=np.int64)
            m[0, 0] = 100
            m[1, 1], m[1, 0] = right, wrong
            return m

        # value A suits group 0, value B suits group 1
        stack = np.stack([
            np.stack([matrix(100, 0), matrix(0, 100)]),
            np.stack([matrix(0, 100), matrix(100, 0)]),
        ])

        result = evaluate_events.leave_one_group_out(
            stack, np.array([0, 1]), [1.0, 2.0]
        )

        # Group 0 held out -> chosen on group 1 -> picks 2.0, wrong for
        # group 0; likewise the other way. Every held-out "yes" window is
        # wrong: the estimate is honestly bad, not flattered.
        self.assertEqual(result["recall"], 0.0)
        self.assertEqual(result["groups"], 2)
        self.assertEqual(result["valuesChosen"]["min"], 1.0)
        self.assertEqual(result["valuesChosen"]["max"], 2.0)


class ScoringTests(unittest.TestCase):
    def test_confusion_and_metrics(self):
        # truth: no no no no yes yes yes -1(unscored)
        truth = np.array([0, 0, 0, 0, 1, 1, 1, -1], dtype=np.int8)
        # said:  no no yes none yes no none yes
        predicted = np.array([0, 0, 1, -1, 1, 0, -1, 1], dtype=np.int8)

        matrix = ev.confusion(truth, predicted)

        self.assertEqual(matrix.tolist(), [
            [2, 1, 1],
            [1, 1, 1],
        ])
        self.assertEqual(int(matrix.sum()), 7)       # the -1 is unscored

        result = ev.metrics(matrix)
        self.assertEqual(result["falsePositive"], 1)
        self.assertEqual(result["falseNegative"], 1)
        self.assertEqual(result["unpublished"], 2)
        self.assertAlmostEqual(result["recall"], 1 / 3)
        self.assertAlmostEqual(result["specificity"], 2 / 4)
        self.assertAlmostEqual(result["precision"], 1 / 2)
        self.assertAlmostEqual(
            result["balancedAccuracy"], (1 / 3 + 1 / 2) / 2
        )

    def test_compatibility_truth_is_the_documented_mapping(self):
        old = np.array([0, 1, 2, -1], dtype=np.int8)
        seen = ev.observation_truth(old)

        self.assertEqual(seen["fan"].tolist(), [0, 1, 1, -1])
        self.assertEqual(seen["compressor"].tolist(), [0, 0, 1, -1])

        # v1's published state is folded the same way; "nothing yet"
        # stays nothing-yet.
        folded = ev.legacy_observations(old)
        self.assertEqual(folded["fan"].tolist(), [0, 1, 1, -1])
        self.assertEqual(folded["compressor"].tolist(), [0, 0, 1, -1])

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
        """(candidate, published) per observation, scored windows only."""
        event = self.by_fixture[name]

        if config.version == "v1":
            run = ev.run_v1(
                "v1", [event], config.rule(), config.median_seconds,
                config.hold_seconds,
            )
        else:
            run = ev.run_v2(
                "v2", [event], config.rule(), config.median_seconds,
                config.hold_seconds,
            )

        scored = event.truth != ev.UNSCORED

        return (
            {o: run.candidate[o][0][scored] for o in ev.OBSERVATIONS},
            {o: run.published[o][0][scored] for o in ev.OBSERVATIONS},
        )

    def test_every_fixture_is_scored(self):
        self.assertEqual(len(self.events), len(synthetic.FIXTURES))

        for event in self.events:
            self.assertGreater(event.scored, 400)

    def test_clean_scenarios_publish_the_right_observations_under_both(self):
        expected = {
            "off_clean": (0, 0), "fan_clean": (1, 0), "compressor": (1, 1),
        }

        for name, (fan, compressor) in expected.items():
            for config in (self.v1, self.v2):
                _, published = self.replay(config, name)

                self.assertTrue(
                    (published["fan"] == fan).all()
                    and (published["compressor"] == compressor).all(),
                    f"{name} under {config.version}",
                )

    def test_v2_reports_fan_and_compressor_together_on_the_compressor(self):
        candidate, _ = self.replay(self.v2, "compressor")

        # Not a single state: both observations are independently true.
        self.assertGreater((candidate["compressor"] == 1).mean(), 0.95)
        self.assertGreater((candidate["fan"] == 1).mean(), 0.9)

    def test_clean_fan_is_stationary_so_v2_keeps_it(self):
        candidate, _ = self.replay(self.v2, "fan_clean")

        self.assertGreater((candidate["fan"] == 1).mean(), 0.95)

    def test_v2_rejects_the_interference_that_makes_v1_flap(self):
        for name in ("off_talking", "off_tv", "off_alarm"):
            v1_candidate, v1_published = self.replay(self.v1, name)
            v2_candidate, v2_published = self.replay(self.v2, name)

            self.assertGreater((v1_candidate["fan"] == 1).mean(), 0.2, name)
            self.assertLess((v1_published["fan"] == 1).mean(), 0.5, name)

            self.assertLess((v2_candidate["fan"] == 1).mean(), 0.03, name)
            self.assertTrue((v2_published["fan"] == 0).all(), name)

    def test_impulsive_noise_is_not_a_fan_to_either(self):
        for config in (self.v1, self.v2):
            candidate, published = self.replay(config, "off_impulsive")

            self.assertLess((candidate["fan"] == 1).mean(), 0.02)
            self.assertTrue((published["fan"] == 0).all())

    def test_compressor_is_unaffected_by_the_fan_gate(self):
        for name in synthetic.FIXTURES:
            v1_candidate, _ = self.replay(self.v1, name)
            v2_candidate, _ = self.replay(self.v2, name)

            self.assertEqual(
                v1_candidate["compressor"].tolist(),
                v2_candidate["compressor"].tolist(), name,
            )

    def test_with_the_gate_open_v2_fan_is_v1s_fan_evidence(self):
        open_gate = ClassifierConfig.for_version("v2").patched({
            "fanStabilityThreshold": 100.0,
            "fanStabilityMinSeconds": 0.0,
            "compressorThreshold": -38.0,
        })

        for name in ("off_clean", "fan_clean", "off_talking", "off_tv"):
            v1_candidate, _ = self.replay(self.v1, name)
            v2_candidate, _ = self.replay(open_gate, name)

            # v1 reports FAN for fan evidence only when there is no
            # compressor; elsewhere the two views agree.
            self.assertEqual(
                v1_candidate["fan"].tolist(), v2_candidate["fan"].tolist(),
                name,
            )

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
            self.assertIn(
                "COMPRESSOR=(yes,yes)",
                summary["parameters"]["truthAssumption"],
            )

            # Each observation is scored on its own.
            self.assertEqual(set(v2["published"]), {"fan", "compressor"})

            fan1, fan2 = v1["published"]["fan"], v2["published"]["fan"]
            self.assertGreaterEqual(
                fan2["balancedAccuracy"], fan1["balancedAccuracy"]
            )
            self.assertLessEqual(
                fan2["falsePositive"], fan1["falsePositive"]
            )
            self.assertEqual(fan2["falsePositive"], 0)

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
            self.assertIn("v2_fan_false_yes", rows[0])
            self.assertIn("v2_compressor_false_no", rows[0])

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
