"""v2 events: types, review schema, command context, where they are written."""

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "audio"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.app import _bulk_review_fields  # noqa: E402
from backend.config import (  # noqa: E402
    RESULTS_V2,
    AppConfig,
    ClassifierConfig,
    EventConfig,
)
from backend.events import EventRecorder  # noqa: E402
from backend.history import HistoryStore  # noqa: E402
from backend.models import (  # noqa: E402
    CommandMarker,
    EventBulkReviewRequest,
    review_for_event,
)
from pydantic import ValidationError  # noqa: E402

V2 = ClassifierConfig.for_version("v2")


def meta(event_type, to=None):
    return {
        "id": "x", "classifierVersion": "v2", "eventType": event_type,
        "from": None if to is None else not to, "to": to,
    }


class RecorderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.recorder = EventRecorder(
            10, EventConfig(pre_seconds=1, post_seconds=0.2), self.directory,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, **kwargs):
        self.recorder.push(np.arange(10, dtype=np.int32))
        identifier = self.recorder.start(
            from_state=kwargs.pop("from_state", False),
            to_state=kwargs.pop("to_state", True),
            stable_seconds=2.0, stream_time=10.0,
            features={"rms": -30.0}, decision_window={},
            classifier_config=V2, **kwargs,
        )
        self.recorder.push(np.arange(2, dtype=np.int32))
        return identifier

    def test_observation_events_are_named_and_typed(self):
        cases = [
            ("fan", "FAN_ON", True), ("fan", "FAN_OFF", False),
            ("compressor", "COMPRESSOR_ON", True),
            ("compressor", "COMPRESSOR_OFF", False),
        ]
        for event_type, suffix, value in cases:
            identifier = self.write(
                from_state=not value, to_state=value,
                event_type=event_type, identifier_suffix=suffix,
                extra={"observations": {"fan": True, "compressor": value}},
            )
            self.assertTrue(identifier.endswith("_" + suffix), identifier)

            event = self.recorder.get(identifier)
            self.assertEqual(event["eventType"], event_type)
            self.assertEqual(event["to"], value)
            self.assertEqual(event["classifierVersion"], "v2")
            self.assertEqual(event["observations"]["compressor"], value)
            self.assertEqual(event["review"], {"status": "unreviewed"})

    def test_a_beep_event_carries_its_measurements_and_no_transition(self):
        identifier = self.write(
            from_state=None, to_state=None, event_type="beep",
            identifier_suffix="BEEP",
            extra={
                "beep": {"peakHz": 4120.0, "durationMs": 150},
                "observations": {"fan": True, "compressor": False},
            },
        )
        event = self.recorder.get(identifier)

        self.assertTrue(identifier.endswith("_BEEP"))
        self.assertEqual(event["eventType"], "beep")
        self.assertIsNone(event["to"])
        self.assertEqual(event["beep"]["peakHz"], 4120.0)

    def test_the_event_file_is_self_contained_on_disk(self):
        identifier = self.write(event_type="fan", identifier_suffix="FAN_ON")

        on_disk = json.loads((self.directory / f"{identifier}.json").read_text())
        self.assertEqual(on_disk["eventType"], "fan")
        self.assertTrue((self.directory / f"{identifier}.wav").is_file())
        # Nothing was written anywhere else.
        self.assertEqual(
            {p.parent for p in self.directory.glob("*.*")}, {self.directory}
        )

    def test_search_finds_event_types_and_observations(self):
        self.write(event_type="compressor", identifier_suffix="COMPRESSOR_ON",
                   extra={"observations": {"fan": True, "compressor": True}})

        self.assertEqual(self.recorder.list_events(search="compressor")["total"], 1)

    def test_old_v1_events_get_no_event_type(self):
        self.recorder.push(np.arange(10, dtype=np.int32))
        identifier = self.recorder.start(
            from_state="FAN", to_state="COMPRESSOR", stable_seconds=2.0,
            stream_time=10.0, features={"rms": -30.0}, decision_window={},
            classifier_config=ClassifierConfig(),
        )
        self.recorder.push(np.arange(2, dtype=np.int32))
        event = self.recorder.get(identifier)

        self.assertNotIn("eventType", event)
        self.assertEqual(event["classifierVersion"], "v1")
        self.assertIn("_FAN_to_COMPRESSOR", identifier)


class CommandContextTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.history = HistoryStore(self.directory / "h.db", 3600)
        self.recorder = EventRecorder(
            10, EventConfig(pre_seconds=1, post_seconds=0.2), self.directory,
        )
        self.recorder.command_lookup = self.history.commands_between

    def tearDown(self):
        self.history.close()
        self.temporary.cleanup()

    def beep_event(self):
        self.recorder.push(np.arange(10, dtype=np.int32))
        identifier = self.recorder.start(
            from_state=None, to_state=None, stable_seconds=0.0,
            stream_time=10.0, features={}, decision_window={},
            classifier_config=V2, event_type="beep", identifier_suffix="BEEP",
        )
        self.recorder.push(np.arange(2, dtype=np.int32))
        return self.recorder.get(identifier)

    def test_a_nearby_command_is_attached(self):
        self.history.record_command("POWER", True, 9.0, "from remote")
        event = self.beep_event()

        context = event["commandContext"]
        self.assertEqual(context["command"], "POWER")
        self.assertTrue(context["expectedBeep"])
        self.assertEqual(context["sentAt"], 9.0)
        self.assertLess(abs(context["secondsFromEvent"]), 5)
        self.assertEqual(context["note"], "from remote")

    def test_a_command_that_arrives_late_is_attached_to_the_saved_event(self):
        event = self.beep_event()
        self.assertNotIn("commandContext", event)

        marker = self.history.record_command("POWER", True, 9.0, delay=2.0)
        attached = self.recorder.attach_late_command(marker)

        self.assertEqual(attached, [event["id"]])
        context = self.recorder.get(event["id"])["commandContext"]
        self.assertEqual(context["command"], "POWER")
        # Dated at the estimated send time, not the arrival time.
        self.assertLess(marker["time"], time.time() - 1.5)
        # An event that already has one is left alone.
        again = self.history.record_command("OTHER", True, 9.5)
        self.assertEqual(self.recorder.attach_late_command(again), [])

    def test_no_command_means_no_context_not_an_empty_one(self):
        self.assertNotIn("commandContext", self.beep_event())

    def test_a_command_long_ago_is_not_attached(self):
        self.history.record_command("POWER", True, 1.0)
        # Rewind it far outside the window.
        self.history.connection.execute("UPDATE commands SET t = t - 3600")
        self.history.connection.commit()

        self.assertNotIn("commandContext", self.beep_event())

    def test_the_nearest_of_several_wins(self):
        self.history.record_command("FAR", True, 1.0)
        self.history.connection.execute("UPDATE commands SET t = t - 8")
        self.history.connection.commit()
        time.sleep(0.01)
        self.history.record_command("NEAR", True, 2.0)

        self.assertEqual(self.beep_event()["commandContext"]["command"], "NEAR")

    def test_the_context_can_be_set_and_removed_by_hand(self):
        event = self.beep_event()

        set_ = self.recorder.annotate(event["id"], {
            "commandContext": {"command": "TEMP_UP", "sentAt": None,
                               "expectedBeep": True, "note": ""},
        })
        self.assertEqual(set_["commandContext"]["command"], "TEMP_UP")
        on_disk = json.loads(
            (self.directory / f"{event['id']}.json").read_text())
        self.assertEqual(on_disk["commandContext"]["command"], "TEMP_UP")

        gone = self.recorder.annotate(event["id"], {"commandContext": None})
        self.assertNotIn("commandContext", gone)
        self.assertIsNone(self.recorder.annotate("missing", {"a": 1}))

    def test_a_command_does_not_change_what_was_detected(self):
        # The marker is data about the world, never an input to Watson.
        with_command = self.beep_event()
        self.assertEqual(with_command["eventType"], "beep")
        self.assertNotIn("acknowledged", with_command)
        self.assertNotIn("success", with_command)


class ReviewSchemaTests(unittest.TestCase):
    def test_fan_correct_derives_the_actual_value_from_the_event(self):
        fields = review_for_event(meta("fan", to=True), {"correct": True})
        self.assertEqual(fields["actualValue"], True)

        fields = review_for_event(meta("fan", to=False), {"correct": True})
        self.assertEqual(fields["actualValue"], False)

    def test_wrong_means_the_opposite_value(self):
        # Binary: FAN_ON wrong = the fan was off; COMPRESSOR_OFF wrong = on.
        for event_type, to in (("fan", True), ("fan", False),
                               ("compressor", True)):
            fields = review_for_event(meta(event_type, to=to),
                                      {"correct": False})
            self.assertEqual(fields["actualValue"], not to)
            self.assertFalse(fields["correct"])

    def test_contradictions_are_refused(self):
        with self.assertRaises(ValueError):
            review_for_event(
                meta("fan", to=True), {"correct": True, "actualValue": False}
            )
        with self.assertRaises(ValueError):
            review_for_event(
                meta("beep"), {"correct": True, "actualBeep": False}
            )

    def test_a_beep_is_reviewed_as_a_beep(self):
        self.assertTrue(
            review_for_event(meta("beep"), {"correct": True})["actualBeep"])
        self.assertFalse(
            review_for_event(meta("beep"), {"correct": False})["actualBeep"])

    def test_fields_that_belong_to_another_type_are_refused(self):
        with self.assertRaises(ValueError):
            review_for_event(meta("beep"), {"correct": True, "actualValue": True})
        with self.assertRaises(ValueError):
            review_for_event(meta("fan", to=True),
                             {"correct": True, "actualBeep": True})

    def test_a_v1_shaped_review_is_not_a_v2_review(self):
        with self.assertRaises(ValidationError):
            review_for_event(meta("fan", to=True), {
                "classificationCorrect": True,
                "actualFrom": "OFF", "actualTo": "FAN",
            })

    def test_v1_events_keep_the_v1_schema(self):
        v1 = {"id": "y", "from": "FAN", "to": "COMPRESSOR"}

        fields = review_for_event(v1, {
            "classificationCorrect": False, "actualFrom": "FAN",
            "actualTo": "FAN",
        })
        self.assertEqual(fields["actualTo"], "FAN")

        with self.assertRaises(ValidationError):
            review_for_event(v1, {"correct": True})

    def test_interference_and_notes_survive(self):
        fields = review_for_event(meta("fan", to=True), {
            "correct": False, "actualValue": False,
            "interference": ["tv", "talking"], "notes": "tv on",
        })
        self.assertEqual(fields["interference"], ["tv", "talking"])
        self.assertEqual(fields["notes"], "tv on")


class OutcomeReviewTests(unittest.TestCase):
    """Three outcomes for a reported boolean transition X -> Y."""

    def event(self, origin, target, kind="fan"):
        return {**meta(kind, to=target), "from": origin}

    def test_correct_is_the_reported_transition(self):
        f = review_for_event(self.event(False, True), {"outcome": "correct"})
        self.assertEqual((f["actualFrom"], f["actualTo"]), (False, True))
        self.assertTrue(f["correct"])

    def test_no_transition_stayed_true_or_false(self):
        # OFF -> ON that really stayed ON, or stayed OFF.
        for outcome, value in (("stayed_true", True), ("stayed_false", False)):
            f = review_for_event(self.event(False, True), {"outcome": outcome})
            self.assertEqual((f["actualFrom"], f["actualTo"]), (value, value))
            self.assertEqual(f["actualValue"], value)
            self.assertFalse(f["correct"])

        # ON -> OFF likewise.
        f = review_for_event(self.event(True, False), {"outcome": "stayed_true"})
        self.assertEqual((f["actualFrom"], f["actualTo"]), (True, True))

    def test_the_old_binary_form_maps_onto_outcomes(self):
        f = review_for_event(self.event(False, True), {"correct": False})
        self.assertEqual(f["outcome"], "stayed_false")
        f = review_for_event(self.event(True, False), {"correct": False})
        self.assertEqual(f["outcome"], "stayed_true")
        f = review_for_event(self.event(True, False), {"correct": True})
        self.assertEqual(f["outcome"], "correct")

    def test_contradictions_and_missing_verdicts_are_refused(self):
        ev = self.event(False, True)
        for body in ({}, {"outcome": "correct", "correct": False},
                     {"outcome": "stayed_true", "correct": True},
                     {"outcome": "stayed_true", "actualValue": False}):
            with self.assertRaises(ValueError, msg=body):
                review_for_event(ev, body)

    def test_a_beep_has_no_transition_outcomes(self):
        with self.assertRaises(ValueError):
            review_for_event(meta("beep"), {"outcome": "stayed_true"})

    def test_bulk_outcomes(self):
        request = EventBulkReviewRequest(ids=["a"], outcome="stayed_false")
        self.assertFalse(request.verdict)
        f = _bulk_review_fields(self.event(False, True), request)
        self.assertEqual(f["outcome"], "stayed_false")

        with self.assertRaises(ValueError):
            _bulk_review_fields(meta("beep"), request)

        ok = EventBulkReviewRequest(ids=["a"], outcome="correct")
        self.assertTrue(_bulk_review_fields(meta("beep"), ok)["actualBeep"])
        with self.assertRaises(ValidationError):
            EventBulkReviewRequest(ids=["a"], outcome="correct", correct=True)


class ManualReviewTests(unittest.TestCase):
    def manual(self, fan, compressor):
        return {**meta("manual"),
                "observations": {"fan": fan, "compressor": compressor}}

    def test_a_manual_review_says_what_was_happening(self):
        fields = review_for_event(self.manual(True, False), {
            "actualFan": True, "actualCompressor": False, "actualBeep": True,
        })
        self.assertEqual(fields["actualBeep"], True)
        self.assertTrue(fields["correct"])        # detectors agreed

        fields = review_for_event(self.manual(True, False), {
            "actualFan": True, "actualCompressor": True, "actualBeep": False,
        })
        self.assertFalse(fields["correct"])       # compressor was missed

    def test_all_three_answers_are_needed(self):
        with self.assertRaises(ValueError):
            review_for_event(self.manual(True, True), {"actualFan": True})

    def test_the_fields_belong_to_manual_events_only(self):
        with self.assertRaises(ValueError):
            review_for_event(meta("fan", to=True),
                             {"correct": True, "actualFan": True})
        with self.assertRaises(ValueError):
            review_for_event(meta("beep"), {"actualBeep": True})


class BulkReviewTests(unittest.TestCase):
    def test_exactly_one_schema_per_request(self):
        with self.assertRaises(ValidationError):
            EventBulkReviewRequest(ids=["a"])
        with self.assertRaises(ValidationError):
            EventBulkReviewRequest(
                ids=["a"], classificationCorrect=True, correct=True)

        self.assertTrue(EventBulkReviewRequest(ids=["a"], correct=True).verdict)
        self.assertFalse(
            EventBulkReviewRequest(
                ids=["a"], classificationCorrect=False,
                actualFrom="OFF", actualTo="OFF").verdict)

    def test_v1_wrong_still_needs_labels(self):
        with self.assertRaises(ValidationError):
            EventBulkReviewRequest(ids=["a"], classificationCorrect=False)

    def test_one_request_reviews_a_mix_of_event_types(self):
        request = EventBulkReviewRequest(ids=["a", "b"], correct=True)

        fan = _bulk_review_fields(meta("fan", to=True), request)
        beep = _bulk_review_fields(meta("beep"), request)

        self.assertEqual(fan["actualValue"], True)
        self.assertEqual(beep["actualBeep"], True)

    def test_a_wrong_verdict_is_derived_per_event(self):
        request = EventBulkReviewRequest(ids=["a"], correct=False)

        self.assertEqual(
            _bulk_review_fields(meta("fan", to=True), request)["actualValue"],
            False)
        self.assertEqual(
            _bulk_review_fields(meta("compressor", to=False),
                                request)["actualValue"], True)
        self.assertFalse(
            _bulk_review_fields(meta("beep"), request)["actualBeep"])


class MarkerAndStorageTests(unittest.TestCase):
    def test_command_markers_are_validated(self):
        CommandMarker(command="POWER")
        with self.assertRaises(ValidationError):
            CommandMarker(command="")
        with self.assertRaises(ValidationError):
            CommandMarker(command="POWER", acknowledged=True)

    def test_v2_events_live_only_under_results_v2(self):
        settings = AppConfig.for_version("v2")

        self.assertEqual(settings.events_dir, RESULTS_V2 / "events")
        self.assertNotEqual(settings.events_dir.parent.name, "results")

    def test_history_columns_include_the_observations(self):
        from backend.history import COLUMNS

        for name in ("fan_detected", "compressor_detected", "beep_count"):
            self.assertIn(name, COLUMNS)


if __name__ == "__main__":
    unittest.main()
