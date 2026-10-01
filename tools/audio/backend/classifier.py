"""Classification, using the existing implementation unchanged.

This is a holder for classify_live's FeatureExtractor and Smoother,
not a second classifier. Everything a rule does is still decided by
classify_live.classify(), so live and replay cannot diverge.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from classify_live import (
    DIAGNOSTIC_BANDS,
    FEATURE_NAMES,
    RULE_BANDS,
    Decision,
    FeatureExtractor,
    Features,
    Smoother,
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

    def to_api(self) -> dict:
        return {
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
        )

    def _build_smoother(self, config: ClassifierConfig) -> Smoother:
        return Smoother(
            config.thresholds(),
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
                    features=features.values
                    | dict(decision.smoothed)
                    | {"rms": decision.rms_db},
                    stream_time=features.time,
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
        with self.lock:
            published = self.smoother.state

            if self.config.affects_smoothing(config):
                # The deques are sized from these, so rebuild.
                self.smoother = self._build_smoother(config)
            else:
                self.smoother.thresholds = config.thresholds()
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
