"""Classifier v2: independent acoustic observations.

v1 collapsed what it heard into one state, OFF, FAN or COMPRESSOR. That
cannot say "fan and compressor both running", and it makes the
compressor suppress the fan: a compressor that is on stops the fan being
looked at at all. v2 reports what it hears as separate facts instead:

    fan_detected          bool      is the fan running?
    compressor_detected   bool      is the compressor running?
    beep                  event     did the unit just beep?

Each is decided by its own detector (``detectors/``), smoothed and held
on its own timer, and none can suppress another. ``fan=true,
compressor=true`` and ``fan=true, compressor=false`` are both ordinary.

Watson does not infer whether the air conditioner is "on". It does not
know that a beep means a command was accepted, or that a compressor
starting means a request succeeded. The controller sends the commands
and keeps the history, so it is the one place those facts can be
combined; Watson's job is to report the raw observations accurately.

The legacy state
----------------
``AcousticObservations.legacy_state`` folds the two booleans back into
OFF / FAN / COMPRESSOR for old views and for comparison with v1. It
assumes a running compressor implies a turning fan, which is true of this
air conditioner and is the same assumption the v1 labels make. It is a
display convenience, not a v2 output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .common import COMPRESSOR, FAN, OFF
from .detectors import (
    DEFAULT_COMPRESSOR_THRESHOLD,
    DEFAULT_FAN_HIGH_THRESHOLD,
    DEFAULT_FAN_MID_THRESHOLD,
    DEFAULT_FAN_REQUIRE,
    DEFAULT_FAN_STABILITY_FEATURE,
    DEFAULT_FAN_STABILITY_MIN_SECONDS,
    DEFAULT_FAN_STABILITY_THRESHOLD,
    STABILITY_FEATURES,
    TEMPORAL_FILL,
    BeepConfig,
    CompressorConfig,
    FanConfig,
    detect_compressor,
    detect_fan,
    fan_energy,
    fan_stationary,
)


@dataclass(frozen=True)
class AcousticObservations:
    """What the stateless detectors say about one moment."""

    fan_detected: bool
    compressor_detected: bool

    def legacy_state(self) -> str:
        """OFF / FAN / COMPRESSOR, for old views. Not a v2 output."""
        if self.compressor_detected:
            return COMPRESSOR

        return FAN if self.fan_detected else OFF


@dataclass(frozen=True)
class ObservationRules:
    """The stateless detectors of v2 and their settings.

    The beep detector is stateful and lives apart (it reads raw windows,
    not smoothed ones); its settings travel with these only so that one
    object describes the whole of v2.
    """

    fan: FanConfig = FanConfig()
    compressor: CompressorConfig = CompressorConfig()
    beep: BeepConfig = BeepConfig()

    def observe(self, values: Mapping[str, float]) -> AcousticObservations:
        """Run both detectors on one smoothed feature set.

        Neither result is allowed to depend on the other.
        """
        return AcousticObservations(
            fan_detected=bool(detect_fan(values, self.fan)),
            compressor_detected=bool(
                detect_compressor(values, self.compressor)
            ),
        )

    def observe_arrays(self, values: Mapping[str, object]):
        """The same detectors over arrays: (fan, compressor) booleans.

        The threshold search runs this over every window of every event
        at once. It is the very code ``observe`` runs on one window.
        """
        return (
            detect_fan(values, self.fan),
            detect_compressor(values, self.compressor),
        )


__all__ = [
    "DEFAULT_COMPRESSOR_THRESHOLD",
    "DEFAULT_FAN_HIGH_THRESHOLD",
    "DEFAULT_FAN_MID_THRESHOLD",
    "DEFAULT_FAN_REQUIRE",
    "DEFAULT_FAN_STABILITY_FEATURE",
    "DEFAULT_FAN_STABILITY_MIN_SECONDS",
    "DEFAULT_FAN_STABILITY_THRESHOLD",
    "STABILITY_FEATURES",
    "TEMPORAL_FILL",
    "AcousticObservations",
    "BeepConfig",
    "CompressorConfig",
    "FanConfig",
    "ObservationRules",
    "detect_compressor",
    "detect_fan",
    "fan_energy",
    "fan_stationary",
]
