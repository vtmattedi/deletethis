"""v2 observations: fan and compressor are independent, each on its own hold."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "audio"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import synthetic  # noqa: E402
from classifier.common import HoldTimer, ObservationSmoother  # noqa: E402
from classifier.v2 import ObservationRules  # noqa: E402
from features import FeatureExtractor  # noqa: E402


def windows_of(*parts):
    """Feature windows for fixtures played one after another."""
    samples = np.concatenate([
        synthetic.to_counts(synthetic.make(name, seconds=seconds))
        for name, seconds in parts
    ]).astype(np.int64)

    extractor = FeatureExtractor(16000)
    out = []

    for start in range(0, samples.size - 511, 512):
        out += extractor.push(samples[start:start + 512])

    return out


def smoother(hold=2.0, median=0.5):
    extractor = FeatureExtractor(16000)

    return ObservationSmoother(
        ObservationRules(), window_rate=16000 / extractor.hop,
        median_seconds=median, hold_seconds=hold,
    )


class HoldTimerTests(unittest.TestCase):
    def test_publishes_after_the_candidate_has_persisted(self):
        timer = HoldTimer(2.0)

        self.assertEqual(timer.update(True, 0.0), (False, 0.0))
        self.assertIsNone(timer.published)
        self.assertEqual(timer.update(True, 1.0), (False, 1.0))

        changed, stable = timer.update(True, 2.0)
        self.assertTrue(changed)
        self.assertEqual(stable, 2.0)
        self.assertIs(timer.published, True)

        # Staying put is not a change.
        self.assertFalse(timer.update(True, 3.0)[0])

    def test_a_blip_restarts_the_clock(self):
        timer = HoldTimer(2.0)
        timer.update(True, 0.0)
        timer.update(True, 2.0)
        self.assertIs(timer.published, True)

        timer.update(False, 3.0)          # blip
        timer.update(True, 3.5)
        self.assertFalse(timer.update(True, 5.0)[0])
        self.assertIs(timer.published, True)

        timer.update(False, 6.0)
        self.assertFalse(timer.update(False, 7.0)[0])
        self.assertTrue(timer.update(False, 8.0)[0])
        self.assertIs(timer.published, False)

    def test_two_timers_do_not_share_state(self):
        fan, compressor = HoldTimer(2.0), HoldTimer(2.0)

        for t in (0.0, 1.0, 2.0, 3.0):
            fan.update(True, t)
            compressor.update(False, t)

        self.assertIs(fan.published, True)
        self.assertIs(compressor.published, False)

        # The fan flipping later leaves the compressor's clock alone.
        compressor.update(True, 3.0)
        fan.update(False, 3.5)
        self.assertTrue(compressor.update(True, 5.0)[0])
        self.assertIs(fan.published, True)       # still waiting out its hold


class ObservationSmootherTests(unittest.TestCase):
    def run_all(self, parts, **kwargs):
        s = smoother(**kwargs)
        return [(w, s.update(w)) for w in windows_of(*parts)]

    def test_nothing_is_published_until_held(self):
        steps = self.run_all([("fan_clean", 6)])

        self.assertIsNone(steps[0][1].fan_detected)
        self.assertIsNone(steps[0][1].compressor_detected)

    def test_off(self):
        last = self.run_all([("off_clean", 8)])[-1][1]

        self.assertIs(last.fan_detected, False)
        self.assertIs(last.compressor_detected, False)

    def test_fan_alone(self):
        last = self.run_all([("fan_clean", 8)])[-1][1]

        self.assertIs(last.fan_detected, True)
        self.assertIs(last.compressor_detected, False)

    def test_compressor_reports_both_fan_and_compressor(self):
        last = self.run_all([("compressor", 8)])[-1][1]

        self.assertIs(last.fan_detected, True)
        self.assertIs(last.compressor_detected, True)
        self.assertEqual(last.legacy_state(), "COMPRESSOR")

    def test_compressor_starting_and_stopping_never_touches_the_fan(self):
        steps = self.run_all(
            [("fan_clean", 8), ("compressor", 8), ("fan_clean", 8)]
        )
        fan_changes = [d for _, d in steps if d.fan_changed]
        compressor_changes = [d for _, d in steps if d.compressor_changed]

        # The fan was found once, at the start, and never lost.
        self.assertEqual([d.fan_detected for d in fan_changes], [True])
        # The compressor: first "no" at start, then on, then off.
        self.assertEqual(
            [d.compressor_detected for d in compressor_changes],
            [False, True, False],
        )
        self.assertIs(steps[-1][1].fan_detected, True)

    def test_holds_are_independent_in_both_directions(self):
        # Fan's hold long, compressor's short: with one timer per
        # observation they must publish at different moments.
        s = smoother(hold=0.5)
        s.fan.hold_seconds = 4.0

        steps = [s.update(w) for w in windows_of(("compressor", 10))]
        fan_at = next(d.fan_stable_seconds for d in steps if d.fan_changed)
        comp_at = next(
            d.compressor_stable_seconds for d in steps if d.compressor_changed
        )

        self.assertGreaterEqual(fan_at, 4.0)
        self.assertLess(comp_at, 4.0)

    def test_both_detectors_read_one_shared_history(self):
        s = smoother()
        windows = windows_of(("fan_clean", 3))

        for window in windows:
            decision = s.update(window)

        self.assertEqual(len(s.history), min(len(windows), s.history.maxlen))
        # One smoothed set serves both observations.
        self.assertIn("30-80", decision.smoothed)
        self.assertIn("1k-2k_std", decision.values)

    def test_reset_forgets_history_and_holds_but_keeps_what_was_published(self):
        s = smoother()

        for window in windows_of(("fan_clean", 6)):
            s.update(window)

        self.assertIs(s.fan.published, True)

        s.reset()

        self.assertEqual(len(s.history), 0)
        self.assertIs(s.fan.published, True)

    def test_interference_is_not_a_fan_but_does_not_hide_the_compressor(self):
        last = self.run_all([("off_talking", 10)])[-1][1]

        self.assertIs(last.fan_detected, False)
        self.assertIs(last.compressor_detected, False)


if __name__ == "__main__":
    unittest.main()
