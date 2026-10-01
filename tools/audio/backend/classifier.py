"""Classification service: the shared extractor and Smoother, live.

This is a holder, not a classifier. Features come from the one
FeatureExtractor, the decision from the rule of the configured
classifier version, and the hold from the one Smoother -- the same
three pieces offline replay and event timelines use, so live and replay
cannot diverge.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from classifier.common import (
    Decision,
    ObservationDecision,
    ObservationSmoother,
    Smoother,
)
from classifier.detectors import BeepDetector, BeepEvent, fan_energy
from classifier.detectors import fan_stationary
from features import (
    DIAGNOSTIC_BANDS,
    FEATURE_NAMES,
    RULE_BANDS,
    FeatureExtractor,
    Features,
)

from .config import ClassifierConfig

BAND_NAMES = RULE_BANDS + DIAGNOSTIC_BANDS


@dataclass
class Snapshot:
    """What the browser is told. Cheap to build, cheap to send."""

    state: str | None
    candidate: str | None
    stable_seconds: float
    features: dict[str, float]
    stream_time: float
    classifier_version: str = "v1"

    def to_api(self) -> dict:
        return {
            "classifierVersion": self.classifier_version,
            "state": self.state,
            "candidate": self.candidate,
            "stableSeconds": round(self.stable_seconds, 2),
            "features": {
                name: round(value, 4)
                for name, value in self.features.items()
            },
            "streamSeconds": round(self.stream_time, 2),
        }


class ClassifierService:
    """Feeds PCM frames through the classifier, one frame at a time.

    Called from the stream thread. The lock only guards config swaps
    and snapshot reads, both of which are rare compared with frames.
    """

    def __init__(
        self,
        sample_rate: int,
        config: ClassifierConfig,
        nfft: int = 1024,
        overlap: float = 0.5,
    ) -> None:
        self.sample_rate = sample_rate
        self.nfft = nfft
        self.overlap = overlap

        self.lock = threading.RLock()
        self.config = config

        self.extractor = FeatureExtractor(sample_rate, nfft, overlap)
        self.window_rate = sample_rate / self.extractor.hop

        self.smoother = self._build_smoother(config)

        self.snapshot = Snapshot(
            state=None,
            candidate=None,
            stable_seconds=0.0,
            features={name: 0.0 for name in FEATURE_NAMES}
            | {name: -120.0 for name in ["rms", *BAND_NAMES]},
            stream_time=0.0,
            classifier_version=config.version,
        )

    def _build_smoother(self, config: ClassifierConfig) -> Smoother:
        return Smoother(
            config.rule(),
            window_rate=self.window_rate,
            median_seconds=config.median_seconds,
            hold_seconds=config.hold_seconds,
        )

    # ------------------------------------------------------ frames

    def push(
        self, samples: np.ndarray
    ) -> list[tuple[Features, Decision]]:
        """Process one PCM frame, returning every window it completed."""
        produced: list[tuple[Features, Decision]] = []

        with self.lock:
            for features in self.extractor.push(samples):
                decision = self.smoother.update(features)

                produced.append((features, decision))

            if produced:
                features, decision = produced[-1]

                self.snapshot = Snapshot(
                    state=decision.state,
                    candidate=decision.candidate,
                    stable_seconds=decision.stable_seconds,
                    # What the rule was given, not the raw last window:
                    # the smoothed values override the raw ones, so the
                    # UI and a recorded transition show the inputs the
                    # decision was actually made from.
                    features=dict(features.values) | decision.values,
                    stream_time=features.time,
                    classifier_version=self.config.version,
                )

        return produced

    # ------------------------------------------------------- state

    def reset(self, skip_samples: int = 0) -> None:
        """Discard everything built across a discontinuity."""
        with self.lock:
            self.extractor.reset(skip_samples=skip_samples)
            self.smoother.reset()

    def apply_config(self, config: ClassifierConfig) -> None:
        """Change the rules live.

        The published state is kept -- it is the last thing actually
        observed, and moving a threshold is not an observation. But
        the rolling median and the candidate are cleared and the hold
        has to be earned again, so nothing is published on the
        strength of history gathered under the old rules.
        """
        if config.version != self.config.version:
            raise ValueError(
                f"cannot switch classifier from {self.config.version} "
                f"to {config.version} while running: results are "
                "stored per version, so that needs a restart"
            )

        with self.lock:
            published = self.smoother.state

            if self.config.affects_smoothing(config):
                # The deques are sized from these, so rebuild.
                self.smoother = self._build_smoother(config)
            else:
                self.smoother.thresholds = config.rule()
                self.smoother.reset()

            self.smoother.state = published
            self.config = config

    def current(self) -> Snapshot:
        with self.lock:
            return self.snapshot

    def decision_window(self) -> dict[str, dict[str, float]]:
        """Statistics over the smoother's retained history.

        This is what the rule actually saw when it decided, which is
        the useful thing to keep alongside a transition.
        """
        with self.lock:
            history = list(self.smoother.history)

        if not history:
            return {}

        summary: dict[str, dict[str, float]] = {}

        for name in FEATURE_NAMES:
            values = np.array(
                [item.values[name] for item in history],
                dtype=np.float64,
            )

            summary[name] = {
                "median": round(float(np.median(values)), 6),
                "std": round(float(np.std(values)), 6),
                "min": round(float(values.min()), 6),
                "max": round(float(values.max()), 6),
            }

        return summary


# ---------------------------------------------------------------------
# Classifier v2: independent observations
# ---------------------------------------------------------------------


@dataclass
class ObservationSnapshot:
    """What the browser is told under v2.

    Three separate things -- fan, compressor, beep -- and no combined
    state. ``legacy_*`` is the old OFF / FAN / COMPRESSOR folded back
    out of the two booleans for views that still want it; it is a
    convenience for display and history, not an output of Watson.
    """

    fan_detected: bool | None
    compressor_detected: bool | None
    fan_candidate: bool | None
    compressor_candidate: bool | None
    fan_stable_seconds: float
    compressor_stable_seconds: float
    fan_energy: bool | None
    fan_stationary: bool | None
    features: dict[str, float]
    stream_time: float
    last_beep: dict | None = None
    beep_count: int = 0
    classifier_version: str = "v2"

    # -- the old single-state view, for history and legacy panels ------

    @property
    def state(self) -> str | None:
        if self.fan_detected is None or self.compressor_detected is None:
            return None

        if self.compressor_detected:
            return "COMPRESSOR"

        return "FAN" if self.fan_detected else "OFF"

    @property
    def candidate(self) -> str | None:
        if self.fan_candidate is None or self.compressor_candidate is None:
            return None

        if self.compressor_candidate:
            return "COMPRESSOR"

        return "FAN" if self.fan_candidate else "OFF"

    @property
    def stable_seconds(self) -> float:
        # How long the legacy view has held: the shorter of the two.
        return min(self.fan_stable_seconds, self.compressor_stable_seconds)

    def to_api(self) -> dict:
        return {
            "classifierVersion": self.classifier_version,
            "observations": {
                "fan": self.fan_detected,
                "compressor": self.compressor_detected,
            },
            "candidates": {
                "fan": self.fan_candidate,
                "compressor": self.compressor_candidate,
            },
            "stableSeconds": {
                "fan": round(self.fan_stable_seconds, 2),
                "compressor": round(self.compressor_stable_seconds, 2),
            },
            # Why the fan detector says what it says.
            "fanEvidence": {
                "energy": self.fan_energy,
                "stationary": self.fan_stationary,
            },
            "lastBeep": self.last_beep,
            "beepCount": self.beep_count,
            "legacyState": self.state,
            "legacyCandidate": self.candidate,
            "features": {
                name: round(value, 4)
                for name, value in self.features.items()
            },
            "streamSeconds": round(self.stream_time, 2),
        }


@dataclass
class ObservationStep:
    """Everything one analysis window produced."""

    features: Features
    decision: ObservationDecision
    beeps: list[BeepEvent]


class ObservationService:
    """Live v2: independent fan, compressor and beep observations.

    One shared FeatureExtractor. The smoothed features go to the two
    stateless detectors, each with its own hold. The beep detector reads
    the *raw* window instead, because half a second of median would erase
    a tone a quarter of that long.
    """

    def __init__(
        self,
        sample_rate: int,
        config: ClassifierConfig,
        nfft: int = 1024,
        overlap: float = 0.5,
    ) -> None:
        if config.version != "v2":
            raise ValueError("ObservationService runs classifier v2")

        self.sample_rate = sample_rate
        self.nfft = nfft
        self.overlap = overlap

        self.lock = threading.RLock()
        self.config = config

        self.extractor = FeatureExtractor(sample_rate, nfft, overlap)
        self.window_rate = sample_rate / self.extractor.hop

        self.smoother = self._build_smoother(config)
        self.beeps = BeepDetector(
            config.beep_config(),
            hop_seconds=self.extractor.hop / sample_rate,
        )

        self.snapshot = ObservationSnapshot(
            fan_detected=None,
            compressor_detected=None,
            fan_candidate=None,
            compressor_candidate=None,
            fan_stable_seconds=0.0,
            compressor_stable_seconds=0.0,
            fan_energy=None,
            fan_stationary=None,
            features={name: 0.0 for name in FEATURE_NAMES}
            | {name: -120.0 for name in ["rms", *BAND_NAMES]},
            stream_time=0.0,
        )

    def _build_smoother(self, config: ClassifierConfig):
        return ObservationSmoother(
            config.rule(),
            window_rate=self.window_rate,
            median_seconds=config.median_seconds,
            hold_seconds=config.hold_seconds,
        )

    # ------------------------------------------------------ frames

    def push(self, samples: np.ndarray) -> list[ObservationStep]:
        """Process one PCM frame, returning every window it completed."""
        steps: list[ObservationStep] = []

        with self.lock:
            for features in self.extractor.push(samples):
                decision = self.smoother.update(features)
                beeps = self.beeps.update(features)

                steps.append(ObservationStep(features, decision, beeps))

            if steps:
                self._publish(steps[-1])

        return steps

    def _publish(self, step: ObservationStep) -> None:
        decision = step.decision
        rule = self.smoother.rules

        # The two pieces of evidence behind the fan observation, so the
        # UI can say whether it is the energy or the stability that is
        # missing. Evaluated on the same smoothed values the detector saw.
        self.snapshot = ObservationSnapshot(
            fan_detected=decision.fan_detected,
            compressor_detected=decision.compressor_detected,
            fan_candidate=decision.fan_candidate,
            compressor_candidate=decision.compressor_candidate,
            fan_stable_seconds=decision.fan_stable_seconds,
            compressor_stable_seconds=decision.compressor_stable_seconds,
            fan_energy=bool(fan_energy(decision.values, rule.fan)),
            fan_stationary=bool(fan_stationary(decision.values, rule.fan)),
            features=dict(step.features.values) | decision.values,
            stream_time=step.features.time,
            last_beep=(
                self.beeps.last.to_api() if self.beeps.last else None
            ),
            beep_count=self.beeps.count,
        )

    # ------------------------------------------------------- state

    def reset(self, skip_samples: int = 0) -> None:
        """Discard everything built across a discontinuity.

        A tone seen on both sides of a gap is not one tone we can
        measure, so the beep detector forgets its run too.
        """
        with self.lock:
            self.extractor.reset(skip_samples=skip_samples)
            self.smoother.reset()
            self.beeps.reset()

    def apply_config(self, config: ClassifierConfig) -> None:
        """Change the rules live.

        Published observations are kept -- moving a threshold is not an
        observation -- but the rolling median and both candidates are
        cleared and each hold has to be earned again.
        """
        if config.version != self.config.version:
            raise ValueError(
                f"cannot switch classifier from {self.config.version} "
                f"to {config.version} while running: results are "
                "stored per version, so that needs a restart"
            )

        with self.lock:
            fan = self.smoother.fan.published
            compressor = self.smoother.compressor.published

            if self.config.affects_smoothing(config):
                self.smoother = self._build_smoother(config)
            else:
                self.smoother.rules = config.rule()
                self.smoother.reset()

            self.smoother.fan.published = fan
            self.smoother.compressor.published = compressor

            self.beeps.apply_config(config.beep_config())
            self.config = config

    def current(self) -> ObservationSnapshot:
        with self.lock:
            return self.snapshot

    def decision_window(self) -> dict[str, dict[str, float]]:
        """Statistics over the smoother's retained history.

        What the detectors saw when they decided, which is the useful
        thing to keep alongside a change.
        """
        with self.lock:
            history = list(self.smoother.history)

        if not history:
            return {}

        summary: dict[str, dict[str, float]] = {}

        for name in FEATURE_NAMES:
            values = np.array(
                [item.values[name] for item in history],
                dtype=np.float64,
            )

            summary[name] = {
                "median": round(float(np.median(values)), 6),
                "std": round(float(np.std(values)), 6),
                "min": round(float(values.min()), 6),
                "max": round(float(values.max()), 6),
            }

        return summary


def make_classifier_service(sample_rate: int, config: ClassifierConfig):
    """The live service for a config's classifier version."""
    if config.version == "v2":
        return ObservationService(sample_rate, config)

    return ClassifierService(sample_rate, config)
