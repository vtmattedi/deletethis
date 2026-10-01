import sys
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
