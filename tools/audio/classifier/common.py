"""Pieces every classifier version shares.

Three layers, deliberately separate:

    FeatureExtractor      windows -> features         (features.py)
    rule                  smoothed features -> state  (v1.py / v2.py)
    Smoother              median, then publication hold

A rule never sees raw windows and never keeps state. The Smoother
never knows what a rule means. That is what lets ``holdSeconds`` and a
stationarity test coexist without being confused for one another: the
first stops a short-lived candidate being *published*, the second says
what the signal *is*.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, Protocol

import numpy as np

if TYPE_CHECKING:  # features.py imports acstream; keep this layer light
    from features import Features

# The states a classifier can report, in escalation order.
OFF = "OFF"
FAN = "FAN"
COMPRESSOR = "COMPRESSOR"
STATES = (OFF, FAN, COMPRESSOR)

DEFAULT_HOLD_SECONDS = 2.0
DEFAULT_MEDIAN_SECONDS = 0.5


def median_values(recent: "list[Features]", features: "Features") -> dict:
    """Median of every feature over the given windows.

    The median is taken over *everything* a window carries, not just the
    bands, because a rule may depend on any of it. Shared by every
    smoother here so that "the smoothed features" means one thing.
    """
    names = list(features.values)
    matrix = np.array(
        [[item.values[name] for name in names] for item in recent],
        dtype=np.float64,
    )

    return dict(zip(names, np.median(matrix, axis=0).tolist()))


class Rule(Protocol):
    """Anything that turns smoothed feature values into a state."""

    def decide(self, values: Mapping[str, float]) -> str: ...


@dataclass
class Decision:
    candidate: str
    stable_seconds: float
    state: str | None
    changed: bool
    # Median-smoothed band levels, as before: what the UI shows.
    smoothed: dict[str, float]
    rms_db: float
    # The complete smoothed feature set the rule was given: bands,
    # rms and every diagnostic, temporal ones included.
    values: dict[str, float] = field(default_factory=dict)


class Smoother:
    """Rolling median of features, then a hold before publishing.

    Two separate mechanisms, doing two different jobs:

    * the median over ``median_seconds`` removes single-window spikes,
      so one loud transient cannot move the candidate at all;
    * the hold over ``hold_seconds`` means a candidate has to persist
      before it is published, so a genuinely new but short-lived sound
      (someone talking near the mic) never becomes a reported state.

    The median is taken over *every* feature in the window, not just
    the bands, because a rule may depend on any of them. v1 reads three
    bands and ignores the rest, so its decisions are unchanged.
    """

    def __init__(
        self,
        thresholds: Rule,
        window_rate: float,
        median_seconds: float = DEFAULT_MEDIAN_SECONDS,
        hold_seconds: float = DEFAULT_HOLD_SECONDS,
        history_seconds: float = 2.0,
    ) -> None:
        # Named `thresholds` for the callers that predate versions; it
        # is whatever rule object the classifier version supplies.
        self.thresholds = thresholds
        self.hold_seconds = hold_seconds

        median_windows = max(1, round(median_seconds * window_rate))

        self.history: deque[Features] = deque(
            maxlen=max(
                median_windows,
                round(history_seconds * window_rate),
            )
        )

        self.median_windows = median_windows

        self.candidate: str | None = None
        self.candidate_since = 0.0
        self.state: str | None = None

    def reset(self) -> None:
        """Forget the feature history after a gap in the audio.

        The published state is kept: it is the last thing actually
        observed, and a dropout is not evidence that it changed. But
        the candidate has to re-earn its hold, so nothing is published
        on the strength of medians taken across a discontinuity.
        """
        self.history.clear()
        self.candidate = None
        self.candidate_since = 0.0

    def update(self, features: Features) -> Decision:
        self.history.append(features)

        recent = list(self.history)[-self.median_windows :]

        values = median_values(recent, features)

        smoothed = {name: values[name] for name in features.bands}
        rms_db = values["rms"]

        candidate = self.thresholds.decide(values)

        if candidate != self.candidate:
            self.candidate = candidate
            self.candidate_since = features.time

        stable = features.time - self.candidate_since

        changed = False

        if candidate != self.state and stable >= self.hold_seconds:
            self.state = candidate
            changed = True

        return Decision(
            candidate=candidate,
            stable_seconds=stable,
            state=self.state,
            changed=changed,
            smoothed=smoothed,
            rms_db=rms_db,
            values=values,
        )

    def band_spread(self, name: str) -> float:
        """Standard deviation of a band over the kept history.

        A compressor is steady, so a high spread in 30-80 alongside a
        high level is a hint that something transient is being read as
        one.
        """
        if len(self.history) < 2:
            return 0.0

        return float(
            np.std([item.bands[name] for item in self.history])
        )


class HoldTimer:
    """Publishes a boolean once its candidate has held long enough.

    The hold in ``Smoother``, for one observation. Each observation owns
    one, so a change in the fan can neither reset nor delay the
    compressor: they are separate facts with separate evidence.
    """

    def __init__(self, hold_seconds: float) -> None:
        self.hold_seconds = hold_seconds
        self.candidate: bool | None = None
        self.since = 0.0
        self.published: bool | None = None

    def reset(self) -> None:
        """Re-earn the hold after a gap. The published value stays: it
        is the last thing actually observed."""
        self.candidate = None
        self.since = 0.0

    def update(self, candidate: bool, time: float) -> tuple[bool, float]:
        """Returns (published value just changed, seconds held so far)."""
        if candidate != self.candidate:
            self.candidate = candidate
            self.since = time

        stable = time - self.since

        if candidate != self.published and stable >= self.hold_seconds:
            self.published = candidate

            return True, stable

        return False, stable


@dataclass
class ObservationDecision:
    """Both observations for one window, each with its own timer."""

    fan_candidate: bool
    fan_detected: bool | None
    fan_stable_seconds: float
    fan_changed: bool

    compressor_candidate: bool
    compressor_detected: bool | None
    compressor_stable_seconds: float
    compressor_changed: bool

    # The one set of median-smoothed features both detectors read.
    values: dict[str, float] = field(default_factory=dict)
    # Median-smoothed band levels, what the UI shows.
    smoothed: dict[str, float] = field(default_factory=dict)
    rms_db: float = 0.0

    def legacy_state(self) -> str | None:
        """OFF / FAN / COMPRESSOR from the published observations.

        None until both have been published. A display convenience; see
        ``AcousticObservations.legacy_state``.
        """
        if self.fan_detected is None or self.compressor_detected is None:
            return None

        if self.compressor_detected:
            return COMPRESSOR

        return FAN if self.fan_detected else OFF

    def legacy_candidate(self) -> str:
        if self.compressor_candidate:
            return COMPRESSOR

        return FAN if self.fan_candidate else OFF


class ObservationSmoother:
    """Median-smooth the features once, then run independent observations.

        features -> median (one history, shared)
                      |-> fan detector        -> hold timer -> fan_detected
                      '-> compressor detector -> hold timer -> compressor_detected

    The feature history is not duplicated: both detectors read the same
    smoothed values. Everything after that is separate, so nothing one
    observation does can change when the other is published.
    """

    def __init__(
        self,
        rules,
        window_rate: float,
        median_seconds: float = DEFAULT_MEDIAN_SECONDS,
        hold_seconds: float = DEFAULT_HOLD_SECONDS,
        history_seconds: float = 2.0,
    ) -> None:
        self.rules = rules
        self.hold_seconds = hold_seconds

        median_windows = max(1, round(median_seconds * window_rate))

        self.history: deque[Features] = deque(
            maxlen=max(
                median_windows,
                round(history_seconds * window_rate),
            )
        )
        self.median_windows = median_windows

        self.fan = HoldTimer(hold_seconds)
        self.compressor = HoldTimer(hold_seconds)

    def reset(self) -> None:
        """Forget the feature history after a gap in the audio.

        Published observations are kept; each has to re-earn its hold.
        """
        self.history.clear()
        self.fan.reset()
        self.compressor.reset()

    def update(self, features: Features) -> ObservationDecision:
        self.history.append(features)

        recent = list(self.history)[-self.median_windows :]
        values = median_values(recent, features)

        seen = self.rules.observe(values)
        time = features.time

        fan_changed, fan_stable = self.fan.update(seen.fan_detected, time)
        compressor_changed, compressor_stable = self.compressor.update(
            seen.compressor_detected, time
        )

        return ObservationDecision(
            fan_candidate=seen.fan_detected,
            fan_detected=self.fan.published,
            fan_stable_seconds=fan_stable,
            fan_changed=fan_changed,
            compressor_candidate=seen.compressor_detected,
            compressor_detected=self.compressor.published,
            compressor_stable_seconds=compressor_stable,
            compressor_changed=compressor_changed,
            values=values,
            smoothed={name: values[name] for name in features.bands},
            rms_db=values["rms"],
        )
