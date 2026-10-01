import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.io import wavfile
from scipy.signal import get_window


TOOLS = Path(__file__).resolve().parents[1] / "tools" / "audio"
sys.path.insert(0, str(TOOLS))

from acstream import FULL_SCALE  # noqa: E402
from analyze import analyse_file  # noqa: E402
from backend.history import HISTORY_ADDITIONS, HistoryStore  # noqa: E402
from backend.classifier import ClassifierService  # noqa: E402
from backend.config import ClassifierConfig  # noqa: E402
from classify_live import Thresholds, classify  # noqa: E402
from features import (  # noqa: E402
    BANDS,
    FEATURE_NAMES,
    TEMPORAL_FEATURES,
    FeatureExtractor,
    power_to_db,
)


class SharedFeatureTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.samples = rng.integers(
            -(1 << 20), 1 << 20, size=4096, dtype=np.int32
        )

    def test_original_rule_features_are_numerically_unchanged(self):
        actual = FeatureExtractor(16000).push(self.samples)[0]
        block = self.samples[:1024].astype(np.float64) / FULL_SCALE
        centred = block - block.mean()
        expected_rms = float(power_to_db(np.mean(centred * centred)))

        window = get_window("hamming", 1024)
        spectrum = np.fft.rfft(centred * window)
        power = (spectrum.real ** 2 + spectrum.imag ** 2) / window.sum() ** 2
        power[1:-1] *= 2.0
        freqs = np.fft.rfftfreq(1024, 1 / 16000)

        self.assertAlmostEqual(actual.rms_db, expected_rms, places=12)
        edges = {name: (low, high) for name, low, high in BANDS}
        for name in ("30-80", "500-1k", "1k-2k"):
            low, high = edges[name]
            mask = (freqs >= low) & (freqs < high)
            expected = float(power_to_db(power[mask].sum()))
            self.assertAlmostEqual(actual.bands[name], expected, places=12)

        thresholds = Thresholds(fan_require_both=True)
        original = classify(actual.bands, thresholds)
        changed_diagnostics = dict(actual.bands)
        for name in changed_diagnostics.keys() - {
            "30-80", "500-1k", "1k-2k"
        }:
            changed_diagnostics[name] = 1e9
        self.assertEqual(classify(changed_diagnostics, thresholds), original)

    def test_offline_analysis_uses_the_shared_feature_rows(self):
        expected = FeatureExtractor(16000).push(self.samples)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fan_talking_01.wav"
            wavfile.write(path, 16000, (self.samples.astype(np.int64) << 8).astype(np.int32))
            analysis = analyse_file(path, 1024, 0.5)

        self.assertEqual(set(analysis.feature_values), set(FEATURE_NAMES))
        for name in FEATURE_NAMES:
            np.testing.assert_allclose(
                analysis.feature(name),
                [row.values[name] for row in expected],
                rtol=1e-12,
                atol=1e-12,
            )

    def test_temporal_features_are_present_and_finite(self):
        rows = FeatureExtractor(16000).push(self.samples)
        self.assertGreater(len(rows), 1)
        for name in TEMPORAL_FEATURES:
            self.assertTrue(np.isfinite(rows[-1].values[name]), name)
        self.assertGreater(rows[-1].values["rms_std"], 0.0)

    def test_live_snapshot_and_decision_summary_expose_new_features(self):
        service = ClassifierService(16000, ClassifierConfig())
        service.push(self.samples)
        snapshot = service.current()
        summary = service.decision_window()
        for name in (
            "spectral_centroid", "spectral_flatness", "spectral_flux",
            "500-1k_minus_rms", "500-1k_std",
        ):
            self.assertIn(name, snapshot.features)
            self.assertIn(name, summary)


class HistoryMigrationTests(unittest.TestCase):
    def test_compact_history_writes_new_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            store = HistoryStore(Path(directory) / "history.db")
            snapshot = SimpleNamespace(
                state="FAN",
                candidate="FAN",
                stable_seconds=2.0,
                features={
                    "rms": -40.0,
                    "30-80": -60.0,
                    "500-1k": -55.0,
                    "1k-2k": -62.0,
                    "200-1200": -53.0,
                    "500-1k_std": 1.25,
                    "1k-2k_std": 0.75,
                    "spectral_flux": 0.04,
                    "spectral_flatness": 0.2,
                },
            )
            health = SimpleNamespace(lost_frames=0, device_dropped=0)
            self.assertTrue(store.maybe_record(snapshot, health, now=100.0))
            data = store.query(99.0, 101.0)
            store.close()

        self.assertEqual(data["columns"]["band_500_1k_std"], [1.25])
        self.assertEqual(data["columns"]["band_1k_2k_std"], [0.75])
        self.assertEqual(data["columns"]["spectral_flux"], [0.04])
        self.assertEqual(data["columns"]["spectral_flatness"], [0.2])

    def test_existing_database_gains_compact_diagnostic_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.db"
            connection = sqlite3.connect(path)
            connection.execute("CREATE TABLE history (t REAL PRIMARY KEY)")
            connection.execute(
                "CREATE TABLE connection "
                "(t REAL PRIMARY KEY, connected INTEGER, detail TEXT)"
            )
            connection.close()

            store = HistoryStore(path)
            columns = {
                row[1] for row in store.connection.execute(
                    "PRAGMA table_info(history)"
                )
            }
            store.close()

        self.assertTrue(set(HISTORY_ADDITIONS).issubset(columns))


if __name__ == "__main__":
    unittest.main()
