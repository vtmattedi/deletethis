import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


TOOLS = Path(__file__).resolve().parents[1] / "tools" / "audio"
sys.path.insert(0, str(TOOLS))

from backend.config import ClassifierConfig, EventConfig  # noqa: E402
from backend.events import EventRecorder  # noqa: E402
from backend.models import EventDeleteRequest, ReviewPatch  # noqa: E402
from pydantic import ValidationError  # noqa: E402


class EventRecorderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.recorder = EventRecorder(
            sample_rate=10,
            config=EventConfig(pre_seconds=1, post_seconds=0.2),
            directory=self.directory,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def write_event(self, source="transition"):
        self.recorder.push(np.arange(10, dtype=np.int32))
        identifier = self.recorder.start(
            from_state="FAN",
            to_state="COMPRESSOR",
            stable_seconds=2.5,
            stream_time=10.0,
            features={"rms": -30.0},
            decision_window={},
            classifier_config=ClassifierConfig(),
            source=source,
        )
        self.recorder.push(np.arange(2, dtype=np.int32))
        return identifier

    def test_review_search_pagination_and_delete(self):
        identifier = self.write_event(source="manual")
        metadata = self.recorder.get(identifier)
        self.assertEqual(metadata["review"], {"status": "unreviewed"})
        self.assertEqual(metadata["source"], "manual")

        updated = self.recorder.review(identifier, {
            "classificationCorrect": False,
            "actualFrom": "FAN",
            "actualTo": "FAN",
            "interference": ["printer"],
            "notes": "calibration spike",
        })
        self.assertEqual(updated["from"], "FAN")
        self.assertEqual(updated["to"], "COMPRESSOR")
        self.assertEqual(updated["review"]["status"], "reviewed")

        on_disk = json.loads(
            (self.directory / f"{identifier}.json").read_text()
        )
        self.assertEqual(on_disk["review"]["actualTo"], "FAN")
        self.assertEqual(
            self.recorder.list_events(search="incorrect")["total"], 1
        )
        self.assertEqual(
            self.recorder.list_events(search="printer")["total"], 1
        )

        self.assertTrue(self.recorder.delete(identifier))
        self.assertFalse((self.directory / f"{identifier}.json").exists())
        self.assertFalse((self.directory / f"{identifier}.wav").exists())
        self.assertFalse(self.recorder.delete(identifier))

    def test_old_json_is_migrated_to_unreviewed(self):
        path = self.directory / "old.json"
        path.write_text(json.dumps({"id": "old"}), encoding="utf-8")
        recorder = EventRecorder(10, EventConfig(), self.directory)
        self.assertEqual(
            recorder.get("old")["review"], {"status": "unreviewed"}
        )
        self.assertEqual(
            json.loads(path.read_text())["review"],
            {"status": "unreviewed"},
        )

    def test_review_requires_verdict_and_actual_labels(self):
        with self.assertRaises(ValidationError):
            ReviewPatch.model_validate({
                "classificationCorrect": False,
                "actualFrom": "FAN",
            })
        with self.assertRaises(ValidationError):
            ReviewPatch.model_validate({
                "classificationCorrect": False,
                "actualFrom": "FAN",
                "actualTo": "AMBIGUOUS",
            })
        review = ReviewPatch.model_validate({
            "classificationCorrect": False,
            "actualFrom": "UNKNOWN",
            "actualTo": "FAN",
        })
        self.assertEqual(review.interference, [])

    def test_bulk_delete_request_requires_a_bounded_nonempty_list(self):
        self.assertEqual(
            EventDeleteRequest.model_validate({"ids": ["one", "two"]}).ids,
            ["one", "two"],
        )
        with self.assertRaises(ValidationError):
            EventDeleteRequest.model_validate({"ids": []})
        with self.assertRaises(ValidationError):
            EventDeleteRequest.model_validate({"ids": ["x"] * 101})


if __name__ == "__main__":
    unittest.main()
