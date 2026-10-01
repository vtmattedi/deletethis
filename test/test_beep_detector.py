"""The beep detector: what it accepts, what it turns down, and how often."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "audio"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import synthetic  # noqa: E402
from classifier.detectors.beep import BeepConfig, BeepDetector  # noqa: E402
from features import FeatureExtractor  # noqa: E402

RATE = synthetic.SAMPLE_RATE
SECONDS = 4.0


def room(seconds=SECONDS, sigma=0.0004, seed=3):
    return synthetic.with_noise(np.zeros(int(seconds * RATE)), sigma, seed)


def detect(signal, config=None, flush=True):
    """Beeps found in a signal, and the detector that found them."""
    counts = synthetic.to_counts(signal).astype(np.int64)
    extractor = FeatureExtractor(RATE)
    detector = BeepDetector(
        config or BeepConfig(),
        hop_seconds=extractor.hop / RATE,
        window_seconds=extractor.nfft / RATE,
    )
    events = []

    for start in range(0, counts.size - 511, 512):
        for window in extractor.push(counts[start:start + 512]):
            events.extend(detector.update(window))

    if flush:
        events.extend(detector.flush())

    return events, detector


def with_tone(base, start, duration, **kwargs):
    return base + synthetic.tone(base.size, start, duration, **kwargs)


class AcceptTests(unittest.TestCase):
    def test_the_units_beep_is_found_once_with_its_measurements(self):
        signal = with_tone(room(), 1.5, 0.150)
        events, detector = detect(signal)

        self.assertEqual(len(events), 1)
        beep = events[0]

        self.assertAlmostEqual(beep.peak_hz, 4120.0, delta=15)
        self.assertGreater(beep.contrast_db, 15.0)
        self.assertAlmostEqual(beep.duration_ms, 150, delta=50)
        # Placed where it really is, to within a window or two.
        self.assertAlmostEqual(beep.start_time, 1.5, delta=0.1)
        self.assertGreater(beep.end_time, beep.start_time)
        self.assertEqual(detector.count, 1)
        self.assertIs(detector.last, beep)

    def test_a_beep_over_a_loud_room_is_still_found(self):
        base = room(sigma=0.004)             # ten times the usual floor
        events, _ = detect(with_tone(base, 1.5, 0.150, amplitude=0.02))

        self.assertEqual(len(events), 1)

    def test_one_physical_beep_is_one_event(self):
        # A long-ish beep spans many overlapping windows. It must not be
        # reported once per window.
        for duration in (0.08, 0.15, 0.25):
            events, _ = detect(with_tone(room(), 1.5, duration))
            self.assertEqual(len(events), 1, duration)

    def test_two_beeps_are_two_events(self):
        signal = with_tone(with_tone(room(), 1.0, 0.150), 2.5, 0.150)
        events, detector = detect(signal)

        self.assertEqual(len(events), 2)
        self.assertEqual(detector.count, 2)
        self.assertGreater(events[1].start_time - events[0].end_time, 0.5)

    def test_a_gap_shorter_than_the_tolerance_is_one_beep(self):
        signal = with_tone(with_tone(room(), 1.0, 0.08), 1.14, 0.08)
        events, _ = detect(signal)

        self.assertEqual(len(events), 1)

    def test_a_rebound_right_after_a_beep_is_not_a_second_one(self):
        # Gap longer than the tolerance but inside the refractory time.
        signal = with_tone(with_tone(room(), 1.0, 0.10), 1.21, 0.10)
        events, _ = detect(signal)
        relaxed, _ = detect(signal, BeepConfig(refractory_ms=0.0))

        self.assertEqual(len(events), 1)
        self.assertEqual(len(relaxed), 2)

    def test_a_beep_can_end_the_recording(self):
        # Reported on flush, not lost.
        signal = with_tone(room(2.0), 1.82, 0.15)
        events, _ = detect(signal, flush=False)
        flushed, _ = detect(signal, flush=True)

        self.assertLessEqual(len(events), len(flushed))
        self.assertEqual(len(flushed), 1)

    def test_it_is_reported_live_not_only_at_the_end(self):
        counts = synthetic.to_counts(
            with_tone(room(6.0), 1.0, 0.15)
        ).astype(np.int64)
        extractor = FeatureExtractor(RATE)
        detector = BeepDetector(
            BeepConfig(), hop_seconds=extractor.hop / RATE,
            window_seconds=extractor.nfft / RATE,
        )
        reported_at = None

        for start in range(0, counts.size - 511, 512):
            for window in extractor.push(counts[start:start + 512]):
                if detector.update(window) and reported_at is None:
                    reported_at = window.time

        self.assertIsNotNone(reported_at)
        self.assertLess(reported_at, 2.0)         # well before the end

    def test_to_api_is_json_ready(self):
        events, _ = detect(with_tone(room(), 1.5, 0.15))
        api = events[0].to_api()

        self.assertEqual(
            set(api),
            {"streamSeconds", "startSeconds", "endSeconds", "durationMs",
             "peakHz", "contrastDb", "levelDb"},
        )


class RejectTests(unittest.TestCase):
    def assertNone(self, signal, reason=None, config=None):
        events, detector = detect(signal, config)
        self.assertEqual(events, [], detector.stats())

        if reason:
            self.assertGreater(detector.rejected[reason], 0, reason)

    def test_silence(self):
        self.assertNone(room())

    def test_too_short(self):
        self.assertNone(with_tone(room(), 1.5, 0.02), "too_short")

    def test_too_long(self):
        self.assertNone(with_tone(room(), 1.0, 0.6), "too_long")

    def test_wrong_frequency(self):
        for hz in (1000.0, 3000.0, 3700.0, 4000.0, 4010.0, 4230.0, 4500.0):
            self.assertNone(with_tone(room(), 1.5, 0.15, hz=hz))

    def test_a_strong_tone_just_outside_the_window_is_not_the_unit(self):
        # Loud enough to stand out from its neighbours, so only the
        # frequency window turns it down.
        for hz in (4035.0, 4195.0):
            signal = with_tone(room(), 1.5, 0.15, hz=hz, amplitude=0.03)
            self.assertNone(signal)

        # Same loudness inside the window is accepted.
        events, _ = detect(
            with_tone(room(), 1.5, 0.15, hz=4120.0, amplitude=0.03)
        )
        self.assertEqual(len(events), 1)

    def test_too_quiet_to_stand_out(self):
        self.assertNone(
            with_tone(room(), 1.5, 0.15, amplitude=0.0004), "weak"
        )

    def test_broadband_noise_in_the_band_is_not_a_tone(self):
        rng = np.random.default_rng(5)
        burst = np.zeros(int(SECONDS * RATE))
        first = int(1.5 * RATE)
        burst[first:first + int(0.15 * RATE)] = 0.02 * rng.standard_normal(
            int(0.15 * RATE)
        )
        self.assertNone(room() + burst)

    def test_a_wandering_pitch_is_not_the_units_tone(self):
        n = int(SECONDS * RATE)
        first, length = int(1.0 * RATE), int(0.25 * RATE)
        t = np.arange(length) / RATE
        # sweeps 4.06 -> 4.17 kHz inside the window
        phase = 2 * np.pi * (4060 * t + 0.5 * (110 / 0.25) * t * t)
        sweep = np.zeros(n)
        sweep[first:first + length] = 0.0036 * np.sin(phase)

        events, detector = detect(room() + sweep)

        self.assertEqual(events, [])

    def test_the_level_floor_is_enforced(self):
        config = BeepConfig(min_level_db=-30.0)

        self.assertNone(with_tone(room(), 1.5, 0.15), config=config)


class StateTests(unittest.TestCase):
    def test_reset_forgets_a_run_in_progress(self):
        counts = synthetic.to_counts(
            with_tone(room(3.0), 1.0, 0.15)
        ).astype(np.int64)
        extractor = FeatureExtractor(RATE)
        detector = BeepDetector(
            BeepConfig(), hop_seconds=extractor.hop / RATE,
            window_seconds=extractor.nfft / RATE,
        )

        found = []
        for index, start in enumerate(range(0, counts.size - 511, 512)):
            for window in extractor.push(counts[start:start + 512]):
                found.extend(detector.update(window))
                # Cut the stream in the middle of the beep.
                if 1.03 < window.time < 1.07:
                    detector.reset()

        self.assertLessEqual(len(found), 1)
        # Whatever happened, nothing half-formed leaks out afterwards.
        self.assertEqual(detector.flush(), [])

    def test_apply_config_keeps_the_counters(self):
        _, detector = detect(with_tone(room(), 1.5, 0.15))
        self.assertEqual(detector.count, 1)

        detector.apply_config(BeepConfig(min_contrast_db=20.0))

        self.assertEqual(detector.count, 1)
        self.assertEqual(detector.config.min_contrast_db, 20.0)

    def test_a_stricter_setting_turns_the_same_beep_down(self):
        signal = with_tone(room(), 1.5, 0.15)

        found, _ = detect(signal)
        strict, _ = detect(signal, BeepConfig(min_contrast_db=60.0))

        self.assertEqual(len(found), 1)
        self.assertEqual(strict, [])

    def test_the_detector_does_not_know_about_commands(self):
        # It takes features and nothing else: no command, no expected
        # beep, no notion of acknowledgement.
        import inspect

        params = inspect.signature(BeepDetector.update).parameters
        self.assertEqual(list(params), ["self", "features"])


if __name__ == "__main__":
    unittest.main()
