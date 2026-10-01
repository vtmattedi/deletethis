import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np


TOOLS = Path(__file__).resolve().parents[1] / "tools" / "audio"
sys.path.insert(0, str(TOOLS))

from backend.config import AppConfig  # noqa: E402
from backend.stream import StreamService  # noqa: E402


class LiveAudioTests(unittest.TestCase):
    def test_listener_receives_exact_little_endian_pcm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = StreamService(AppConfig(
                history_path=root / "history.db",
                events_dir=root / "events",
            ))
            received = []
            service.add_audio_listener(received.append)
            samples = np.array([-(1 << 23), -1, 0, 1, (1 << 23) - 1], dtype=np.int32)
            service._publish_audio(samples)
            service.remove_audio_listener(received.append)
            service._publish_audio(samples)
            service.close()

        self.assertEqual(len(received), 1)
        decoded = np.frombuffer(received[0], dtype="<i4")
        np.testing.assert_array_equal(decoded, samples)

    def test_short_gap_preserves_transition_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = StreamService(AppConfig(
                history_path=root / "history.db",
                events_dir=root / "events",
            ))
            service.classifier = Mock()
            service._previous_state = "FAN"
            stream = SimpleNamespace(lost_frames=2, dropped_frames=0)
            info = SimpleNamespace(sample_rate=16000, frame_samples=512)

            missing = service._check_gap(stream, info, 0)

            self.assertEqual(missing, 2)
            self.assertEqual(service._previous_state, "FAN")
            service.classifier.reset.assert_called_once_with(
                skip_samples=1024
            )
            service.close()

    def test_large_gap_forgets_transition_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = StreamService(AppConfig(
                history_path=root / "history.db",
                events_dir=root / "events",
            ))
            service.classifier = Mock()
            service._previous_state = "FAN"
            stream = SimpleNamespace(lost_frames=10, dropped_frames=0)
            info = SimpleNamespace(sample_rate=16000, frame_samples=512)

            service._check_gap(stream, info, 0)

            self.assertIsNone(service._previous_state)
            service.close()


if __name__ == "__main__":
    unittest.main()
