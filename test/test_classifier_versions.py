"""Classifier v1/v2: rules, configuration, storage isolation, the API."""

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools" / "audio"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import synthetic  # noqa: E402
from backend import config as config_module  # noqa: E402
from backend.app import create_app, load_saved_config  # noqa: E402
from backend.classifier import ClassifierService  # noqa: E402
from backend.config import (  # noqa: E402
    RESULTS,
    RESULTS_V2,
    AppConfig,
    ClassifierConfig,
    EventConfig,
    result_paths,
)
from backend.events import EventRecorder, event_timeline  # noqa: E402
from backend.models import ConfigPatch  # noqa: E402
from classifier import v1 as rules_v1  # noqa: E402
from classifier import v2 as rules_v2  # noqa: E402
from classifier.common import COMPRESSOR, FAN, OFF, STATES, Smoother  # noqa: E402
from classifier.v1 import Thresholds, classify  # noqa: E402
from classifier.v2 import (  # noqa: E402
    STABILITY_FEATURES,
    TEMPORAL_FILL,
    ThresholdsV2,
    classify_v2,
    classify_v2_codes,
)
from features import (  # noqa: E402
    TEMPORAL_FEATURES,
    FeatureExtractor,
)
from features import TEMPORAL_FILL as FEATURES_TEMPORAL_FILL  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from pydantic import ValidationError  # noqa: E402


def values(**overrides):
    """A complete feature set for the v2 rule, FAN-like by default."""
    base = {
        "30-80": -60.0,
        "500-1k": -55.0,
        "1k-2k": -55.0,
        "1k-2k_std": 1.0,
        "500-1k_std": 1.0,
        TEMPORAL_FILL: 2.0,
    }
    base.update(overrides)
    return base


class V2RuleTests(unittest.TestCase):
    def setUp(self):
        self.rule = ThresholdsV2()

    def test_compressor_wins_over_fan_evidence(self):
        state = classify_v2(values(**{"30-80": -30.0}), self.rule)
        self.assertEqual(state, COMPRESSOR)

        # ...even when the fan evidence is perfect and stationary.
        state = classify_v2(
            values(**{"30-80": -30.0, "1k-2k_std": 0.1}), self.rule
        )
        self.assertEqual(state, COMPRESSOR)

    def test_fan_energy_and_low_std_is_fan(self):
        self.assertEqual(classify_v2(values(), self.rule), FAN)

    def test_fan_energy_with_high_std_is_off(self):
        noisy = values(**{"1k-2k_std": self.rule.stability_threshold + 5})
        self.assertEqual(classify_v2(noisy, self.rule), OFF)

    def test_low_energy_and_low_std_is_off(self):
        quiet = values(**{"500-1k": -80.0, "1k-2k": -80.0})
        self.assertEqual(classify_v2(quiet, self.rule), OFF)

    def test_stability_is_inclusive_at_the_threshold(self):
        edge = values(**{"1k-2k_std": self.rule.stability_threshold})
        self.assertEqual(classify_v2(edge, self.rule), FAN)

    def test_gate_does_not_trust_a_short_history(self):
        # A std over a couple of windows is near zero: everything looks
        # stationary. The gate must not promote it to FAN.
        fresh = values(**{TEMPORAL_FILL: 0.2, "1k-2k_std": 0.0})
        self.assertEqual(classify_v2(fresh, self.rule), OFF)

        ready = values(**{
            TEMPORAL_FILL: self.rule.stability_min_seconds,
            "1k-2k_std": 0.0,
        })
        self.assertEqual(classify_v2(ready, self.rule), FAN)

    def test_missing_or_nan_stability_is_never_fan(self):
        nan = values(**{"1k-2k_std": float("nan")})
        self.assertEqual(classify_v2(nan, self.rule), OFF)

        missing = values()
        del missing["1k-2k_std"]
        with self.assertRaises(KeyError):
            classify_v2(missing, self.rule)

    def test_either_mode_needs_only_one_fan_band(self):
        one_band = values(**{"500-1k": -80.0})
        both = ThresholdsV2(fan_require_both=True)
        either = ThresholdsV2(fan_require_both=False)

        self.assertEqual(classify_v2(one_band, both), OFF)
        self.assertEqual(classify_v2(one_band, either), FAN)

    def test_alternative_stability_feature(self):
        rule = ThresholdsV2(
            stability_feature="500-1k_std", stability_threshold=2.0
        )
        self.assertEqual(
            classify_v2(values(**{"500-1k_std": 5.0, "1k-2k_std": 0.1}),
                        rule),
            OFF,
        )
        self.assertEqual(
            classify_v2(values(**{"500-1k_std": 1.0, "1k-2k_std": 9.0}),
                        rule),
            FAN,
        )

    def test_unknown_stability_feature_is_rejected(self):
        with self.assertRaises(ValueError):
            ThresholdsV2(stability_feature="30-80")

    def test_array_path_matches_scalar_path(self):
        # The search runs this rule over arrays; live runs it on
        # scalars. They must be the same function, so they agree.
        rng = np.random.default_rng(0)
        count = 2000
        table = {
            "30-80": rng.uniform(-70, -25, count),
            "500-1k": rng.uniform(-75, -45, count),
            "1k-2k": rng.uniform(-75, -45, count),
            "1k-2k_std": rng.uniform(0, 12, count),
            TEMPORAL_FILL: rng.uniform(0, 3, count),
        }

        for rule in (
            ThresholdsV2(),
            ThresholdsV2(fan_require_both=False, stability_threshold=3.0),
        ):
            codes = classify_v2_codes(table, rule)

            for index in range(0, count, 7):
                row = {name: float(col[index]) for name, col in table.items()}
                self.assertEqual(
                    STATES[int(codes[index])], classify_v2(row, rule)
                )

    def test_names_agree_with_the_feature_extractor(self):
        self.assertEqual(TEMPORAL_FILL, FEATURES_TEMPORAL_FILL)
        self.assertTrue(set(STABILITY_FEATURES) <= set(TEMPORAL_FEATURES))
        # Spelled out twice in the API model; they must not drift.
        self.assertEqual(
            set(ConfigPatch.model_fields["fanStabilityFeature"]
                .annotation.__args__[0].__args__),
            set(STABILITY_FEATURES),
        )

    def test_the_extractor_reports_how_much_history_backs_the_std(self):
        extractor = FeatureExtractor(16000)
        samples = synthetic.to_counts(
            synthetic.make("fan_clean", seconds=4)
        ).astype(np.int32)

        windows = extractor.push(samples)
        fill = [w.diagnostics[TEMPORAL_FILL] for w in windows]

        # Grows from one window's worth, then saturates at the 2 s span.
        self.assertAlmostEqual(fill[0], 512 / 16000)
        self.assertEqual(fill, sorted(fill))
        self.assertAlmostEqual(fill[-1], 2.0, places=1)

        # Bookkeeping, not a measurement: not a dataset column.
        from features import FEATURE_NAMES
        self.assertNotIn(TEMPORAL_FILL, FEATURE_NAMES)


class LegacySmoother:
    """The smoother exactly as it was before classifier versions.

    Kept here, verbatim in behaviour, so that "v1 is unchanged" is a
    checked property rather than a promise: the refactored Smoother
    driving the v1 rule has to reproduce this window for window.
    """

    def __init__(self, thresholds, window_rate, median_seconds=0.5,
                 hold_seconds=2.0, history_seconds=2.0):
        from collections import deque

        self.thresholds = thresholds
        self.hold_seconds = hold_seconds
        self.median_windows = max(1, round(median_seconds * window_rate))
        self.history = deque(
            maxlen=max(self.median_windows,
                       round(history_seconds * window_rate))
        )
        self.candidate = None
        self.candidate_since = 0.0
        self.state = None

    def update(self, features):
        self.history.append(features)
        recent = list(self.history)[-self.median_windows:]
        smoothed = {
            name: float(np.median([item.bands[name] for item in recent]))
            for name in features.bands
        }
        rms_db = float(np.median([item.rms_db for item in recent]))
        candidate = classify(smoothed, self.thresholds)

        if candidate != self.candidate:
            self.candidate = candidate
            self.candidate_since = features.time

        stable = features.time - self.candidate_since
        changed = False
        if candidate != self.state and stable >= self.hold_seconds:
            self.state = candidate
            changed = True

        return candidate, stable, self.state, changed, smoothed, rms_db


class V1UnchangedTests(unittest.TestCase):
    def test_code_defaults_are_pinned(self):
        self.assertEqual(rules_v1.DEFAULT_COMPRESSOR_THRESHOLD, -48.0)
        self.assertEqual(rules_v1.DEFAULT_FAN_MID_THRESHOLD, -62.0)
        self.assertEqual(rules_v1.DEFAULT_FAN_HIGH_THRESHOLD, -65.0)
        self.assertEqual(rules_v1.DEFAULT_FAN_REQUIRE, "both")

        bare = ClassifierConfig()
        self.assertEqual(bare.version, "v1")
        self.assertEqual(bare.compressor_threshold, -48.0)

    def test_v1_ignores_everything_except_three_bands(self):
        rule = Thresholds(compressor=-38, fan_mid=-62, fan_high=-65,
                          fan_require_both=True)
        noisy = values(**{"1k-2k_std": 99.0, TEMPORAL_FILL: 0.0})
        self.assertEqual(rule.decide(noisy), FAN)

    def test_refactored_smoother_reproduces_the_old_one(self):
        rule = Thresholds(compressor=-38, fan_mid=-62, fan_high=-65,
                          fan_require_both=True)

        for name in synthetic.FIXTURES:
            samples = synthetic.to_counts(
                synthetic.make(name, seconds=14)
            ).astype(np.int64)

            extractor = FeatureExtractor(16000)
            rate = 16000 / extractor.hop
            new = Smoother(rule, rate, 0.5, 2.0)
            old = LegacySmoother(rule, rate, 0.5, 2.0)

            windows = []
            for start in range(0, samples.size - 511, 512):
                windows += extractor.push(samples[start:start + 512])

            self.assertGreater(len(windows), 300)

            for window in windows:
                decision = new.update(window)
                expected = old.update(window)

                self.assertEqual(decision.candidate, expected[0], name)
                self.assertEqual(decision.stable_seconds, expected[1], name)
                self.assertEqual(decision.state, expected[2], name)
                self.assertEqual(decision.changed, expected[3], name)
                self.assertEqual(decision.smoothed, expected[4], name)
                self.assertEqual(decision.rms_db, expected[5], name)


class ConfigVersionTests(unittest.TestCase):
    def test_v1_reports_only_v1_fields(self):
        api = ClassifierConfig().to_api()

        self.assertEqual(api["classifierVersion"], "v1")
        for name in config_module.V2_API_FIELDS:
            self.assertNotIn(name, api)

    def test_v2_serialises_its_extra_fields(self):
        config = ClassifierConfig.for_version("v2")
        api = config.to_api()

        self.assertEqual(api["classifierVersion"], "v2")
        self.assertEqual(api["fanStabilityFeature"], "1k-2k_std")
        self.assertEqual(
            api["fanStabilityThreshold"],
            rules_v2.DEFAULT_FAN_STABILITY_THRESHOLD,
        )
        self.assertEqual(
            api["fanStabilityMinSeconds"],
            rules_v2.DEFAULT_FAN_STABILITY_MIN_SECONDS,
        )
        self.assertIsInstance(config.rule(), ThresholdsV2)
        self.assertIsInstance(ClassifierConfig().rule(), Thresholds)

    def test_v2_defaults_keep_the_compressor_threshold_in_use(self):
        v2 = ClassifierConfig.for_version("v2")

        self.assertEqual(v2.compressor_threshold, -38.0)
        self.assertEqual(v2.fan_mid_threshold, -62.0)
        self.assertEqual(v2.fan_high_threshold, -65.0)
        self.assertEqual(v2.fan_require, "both")

    def test_patch_applies_and_round_trips_v2_fields(self):
        config = ClassifierConfig.for_version("v2").patched({
            "fanStabilityThreshold": 3.5,
            "fanStabilityFeature": "500-1k_std",
            "fanStabilityMinSeconds": 0.5,
        })
        api = config.to_api()

        self.assertEqual(api["fanStabilityThreshold"], 3.5)
        self.assertEqual(api["fanStabilityFeature"], "500-1k_std")

        again = ClassifierConfig.from_api(api)
        self.assertEqual(again, config)

    def test_version_cannot_be_patched_at_runtime(self):
        with self.assertRaises(ValidationError):
            ConfigPatch(classifierVersion="v1")

        # And patched() ignores it too: it changes where results live.
        config = ClassifierConfig.for_version("v2").patched(
            {"classifierVersion": "v1"}
        )
        self.assertEqual(config.version, "v2")

    def test_patch_validates_the_new_fields(self):
        with self.assertRaises(ValidationError):
            ConfigPatch(fanStabilityThreshold=0)
        with self.assertRaises(ValidationError):
            ConfigPatch(fanStabilityFeature="30-80")

        self.assertEqual(
            ConfigPatch(fanStabilityThreshold=4.0).changes(),
            {"fanStabilityThreshold": 4.0},
        )

    def test_old_events_without_a_version_are_v1(self):
        recorded = {
            "compressorThreshold": -38.0,
            "fanMidThreshold": -62.0,
            "fanHighThreshold": -65.0,
            "fanRequire": "both",
            "medianSeconds": 0.5,
            "holdSeconds": 2.0,
        }
        config = ClassifierConfig.from_api(recorded)

        self.assertEqual(config.version, "v1")
        self.assertEqual(config.compressor_threshold, -38.0)
        self.assertIsInstance(config.rule(), Thresholds)

    def test_unknown_version_is_rejected(self):
        with self.assertRaises(ValueError):
            ClassifierConfig(version="v3")
        with self.assertRaises(ValueError):
            result_paths("v3")


class StorageIsolationTests(unittest.TestCase):
    def test_v2_writes_under_results_v2(self):
        paths = result_paths("v2")
        settings = AppConfig.for_version("v2")

        self.assertEqual(paths.root, RESULTS / "v2")
        self.assertEqual(settings.events_dir, RESULTS_V2 / "events")
        self.assertEqual(settings.history_path, RESULTS_V2 / "audio.db")
        self.assertEqual(settings.config_path, RESULTS_V2 / "config.json")
        self.assertEqual(paths.evaluation, RESULTS_V2 / "evaluation")
        self.assertEqual(settings.classifier.version, "v2")

    def test_v1_keeps_its_original_locations(self):
        settings = AppConfig.for_version("v1")

        self.assertEqual(settings.events_dir, RESULTS / "events")
        self.assertEqual(settings.history_path, RESULTS / "audio.db")
        self.assertEqual(settings.config_path, RESULTS / "config.json")
        # And a bare AppConfig() is v1, exactly as before versions.
        self.assertEqual(AppConfig().classifier.version, "v1")
        self.assertEqual(AppConfig().events_dir, RESULTS / "events")

    def test_v2_refuses_the_v1_evidence_directories(self):
        v1 = result_paths("v1")

        for field, path in (
            ("events_dir", v1.events),
            ("history_path", v1.history),
            ("config_path", v1.config),
        ):
            with self.assertRaises(ValueError, msg=field):
                AppConfig.for_version("v2", **{field: path})

    def test_v1_refuses_the_v2_directories(self):
        with self.assertRaises(ValueError):
            AppConfig(events_dir=RESULTS_V2 / "events")
        with self.assertRaises(ValueError):
            AppConfig(config_path=RESULTS_V2 / "config.json")

    def test_other_locations_are_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            AppConfig.for_version(
                "v2", events_dir=root / "e", history_path=root / "h.db",
                config_path=root / "c.json",
            )

    def test_existing_v1_events_are_never_touched_by_a_v2_config(self):
        # The v2 results folder lives inside results/ but is a sibling
        # of the v1 events folder, not a parent or child of it.
        self.assertNotIn(
            RESULTS / "events", (RESULTS_V2 / "events").parents
        )
        self.assertNotIn(
            RESULTS_V2, (RESULTS / "events").parents
        )


class EventMetadataTests(unittest.TestCase):
    def record(self, config):
        with tempfile.TemporaryDirectory() as directory:
            recorder = EventRecorder(
                10, EventConfig(pre_seconds=1, post_seconds=0.2),
                Path(directory),
            )
            recorder.push(np.arange(10, dtype=np.int32))
            identifier = recorder.start(
                from_state="FAN", to_state="COMPRESSOR",
                stable_seconds=2.0, stream_time=5.0,
                features={"rms": -30.0}, decision_window={},
                classifier_config=config,
            )
            recorder.push(np.arange(2, dtype=np.int32))

            on_disk = json.loads(
                (Path(directory) / f"{identifier}.json").read_text()
            )

        return on_disk

    def test_v2_event_records_its_version_and_config(self):
        event = self.record(ClassifierConfig.for_version("v2"))

        self.assertEqual(event["classifierVersion"], "v2")
        self.assertEqual(event["classifierConfig"]["classifierVersion"], "v2")
        self.assertIn("fanStabilityThreshold", event["classifierConfig"])
        self.assertEqual(event["review"], {"status": "unreviewed"})

    def test_v1_event_records_v1(self):
        event = self.record(ClassifierConfig())

        self.assertEqual(event["classifierVersion"], "v1")
        self.assertNotIn("fanStabilityThreshold", event["classifierConfig"])

    def test_timeline_uses_the_recorded_version(self):
        with tempfile.TemporaryDirectory() as directory:
            wav = Path(directory) / "fan.wav"
            synthetic.write_wav(wav, synthetic.make("fan_clean", seconds=6))

            for config in (ClassifierConfig.for_version("v2"),
                           ClassifierConfig()):
                timeline = event_timeline(wav, config, pre_seconds=3.0)

                self.assertEqual(
                    timeline["classifierVersion"], config.version
                )
                self.assertGreater(timeline["count"], 100)
                self.assertIn("1k-2k_std", timeline["columns"])


class ServiceSnapshotTests(unittest.TestCase):
    def test_live_snapshot_exposes_the_v2_inputs(self):
        service = ClassifierService(
            16000, ClassifierConfig.for_version("v2")
        )
        samples = synthetic.to_counts(
            synthetic.make("fan_clean", seconds=5)
        ).astype(np.int32)

        for start in range(0, samples.size - 511, 512):
            service.push(samples[start:start + 512])

        api = service.current().to_api()

        self.assertEqual(api["classifierVersion"], "v2")
        for name in ("1k-2k_std", "500-1k_std", "2k-4k_std",
                     "spectral_flux"):
            self.assertIn(name, api["features"])

        self.assertEqual(api["state"], "FAN")

    def test_a_running_service_cannot_change_version(self):
        service = ClassifierService(16000, ClassifierConfig())

        with self.assertRaises(ValueError):
            service.apply_config(ClassifierConfig.for_version("v2"))


class VersionedApiTests(unittest.TestCase):
    def make_app(self, root, version):
        settings = AppConfig.for_version(
            version,
            events_dir=root / "events",
            history_path=root / "history.db",
            config_path=root / "config.json",
        )
        return settings, create_app(settings)

    @staticmethod
    def endpoint(app, path, method):
        return next(
            route.endpoint for route in app.routes
            if getattr(route, "path", None) == path
            and method in getattr(route, "methods", set())
        )

    def test_patch_persists_the_version_and_stability_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings, app = self.make_app(root, "v2")
            patch = self.endpoint(app, "/api/config", "PATCH")

            result = asyncio.run(patch(ConfigPatch(
                fanStabilityThreshold=4.25, holdSeconds=3.0,
            )))
            saved = json.loads(settings.config_path.read_text())
            app.state.service.close()

        self.assertEqual(result["fanStabilityThreshold"], 4.25)
        self.assertEqual(saved["classifierVersion"], "v2")
        self.assertEqual(saved["fanStabilityThreshold"], 4.25)
        self.assertEqual(saved["holdSeconds"], 3.0)

    def test_stability_settings_are_refused_on_a_v1_run(self):
        with tempfile.TemporaryDirectory() as directory:
            settings, app = self.make_app(Path(directory), "v1")
            patch = self.endpoint(app, "/api/config", "PATCH")

            with self.assertRaises(HTTPException) as caught:
                asyncio.run(patch(ConfigPatch(fanStabilityThreshold=3.0)))

            app.state.service.close()

        self.assertEqual(caught.exception.status_code, 422)

    def test_status_and_config_report_the_version(self):
        with tempfile.TemporaryDirectory() as directory:
            for version in ("v1", "v2"):
                root = Path(directory) / version
                settings, app = self.make_app(root, version)

                status = asyncio.run(
                    self.endpoint(app, "/api/status", "GET")()
                )
                config = asyncio.run(
                    self.endpoint(app, "/api/config", "GET")()
                )
                app.state.service.close()

                self.assertEqual(status["classifierVersion"], version)
                self.assertEqual(config["classifierVersion"], version)

    def test_a_saved_file_is_never_reinterpreted_across_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"

            # No version key: written before versions existed -> v1.
            old = {"compressorThreshold": -41.0, "holdSeconds": 3.0}
            path.write_text(json.dumps(old), encoding="utf-8")

            v1 = AppConfig(config_path=path)
            self.assertEqual(load_saved_config(v1), old)
            self.assertEqual(v1.classifier.compressor_threshold, -41.0)

            # The same file offered to a v2 run is refused, not guessed.
            v2 = AppConfig.for_version("v2", config_path=path)
            with self.assertRaises(ValueError) as caught:
                load_saved_config(v2)
            self.assertIn("v1", str(caught.exception))
            self.assertEqual(v2.classifier.compressor_threshold, -38.0)

            # A v2 file loads into v2...
            path.write_text(json.dumps({
                "classifierVersion": "v2", "fanStabilityThreshold": 3.0,
            }), encoding="utf-8")
            self.assertEqual(
                load_saved_config(v2), {"fanStabilityThreshold": 3.0}
            )
            self.assertEqual(v2.classifier.fan_stability_threshold, 3.0)

            # ...and is refused by v1.
            with self.assertRaises(ValueError):
                load_saved_config(AppConfig(config_path=path))

    def test_v2_only_keys_in_a_v1_file_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"fanStabilityThreshold": 3.0}))

            with self.assertRaises(ValueError):
                load_saved_config(AppConfig(config_path=path))


if __name__ == "__main__":
    unittest.main()
