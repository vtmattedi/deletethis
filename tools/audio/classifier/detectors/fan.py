"""Is the fan running?

Two tests, and both must pass:

    energy       500-1k and 1k-2k are both above their thresholds
    stationary   the fan spectrum is holding still

Energy alone cannot tell a fan from speech, a printer or a television:
all of them put real power in those two bands. What a fan does that they
do not is stay put. Over two seconds its band level barely moves, where
speech and machinery keep changing. That is measured by the ``*_std``
features -- the standard deviation of a band's level over the last two
seconds -- and the gate is ``feature <= stability_threshold``.

It also needs the history to back that number. A standard deviation over
two windows is near zero, which would make anything look stationary, so
the gate does not trust a feature until ``stability_min_seconds`` of
history stand behind it.

This is not ``holdSeconds``. The hold stops a short-lived candidate being
*published*; stationarity says what the signal *is*. A printer that runs
for ten seconds outlasts any hold short enough to be useful.

The detector is independent of the compressor. A running compressor does
not make the fan "not detected" or "detected": it is evaluated on its own
evidence, like any other moment.

Known cost: a stationarity gate tolerates a steady fan but not a steady
fan *under* speech or television, whose variation it inherits. See the
README for what that cost measured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .arrays import scalarise

# The temporal features a stationarity gate may use, all of which are
# "smaller means steadier". features.TEMPORAL_FEATURES is the source of
# truth; this is the subset that makes sense as an upper bound.
STABILITY_FEATURES = (
    "1k-2k_std",
    "500-1k_std",
    "2k-4k_std",
    "rms_std",
    "spectral_flux_median",
    "spectral_flux_std",
)

# Bookkeeping value the extractor publishes alongside the std features:
# seconds of history they were computed over. A string here rather than
# an import, so this module stays free of the DSP layer; a test pins it
# to features.TEMPORAL_FILL.
TEMPORAL_FILL = "temporal_seconds"

DEFAULT_FAN_MID_THRESHOLD = -62.0
DEFAULT_FAN_HIGH_THRESHOLD = -65.0
DEFAULT_FAN_REQUIRE = "both"

DEFAULT_FAN_STABILITY_FEATURE = "1k-2k_std"

# dB. Chosen by replaying the reviewed events (88 events, 53k scored
# windows, 63 time-groups; tools/audio/evaluate_events.py), not guessed.
# Starting guesses of 1.2-1.5 dB cost 20-30% of real fan windows, and
# 2.0 dB still costs about 7%. The measured behaviour is:
#
#   below ~2 dB    detection collapses (a real fan moves a little)
#   ~4 .. ~6 dB    false detections stay flat at their floor while
#                  recall climbs to 98.6%: a tighter gate buys nothing
#   above ~6.1 dB  false detections grow again as noise gets through
#
# so the best balanced accuracy sits on a plateau of 4.9 .. 7.0 dB and
# 6.0 is its centre. Choosing it from 62 of 63 groups and testing on the
# held-out one (leave-one-group-out) gives the same balanced accuracy as
# the in-sample best, so the figure is not an artefact of tuning it on
# the data it is scored on. Re-run the evaluation when the dataset grows
# or the microphone moves: this is a property of the installation.
DEFAULT_FAN_STABILITY_THRESHOLD = 6.0

# A std over two samples is meaningless and near zero, so the gate will
# not call anything stationary until it has this much history behind it.
DEFAULT_FAN_STABILITY_MIN_SECONDS = 1.0


@dataclass(frozen=True)
class FanConfig:
    mid_threshold: float = DEFAULT_FAN_MID_THRESHOLD
    high_threshold: float = DEFAULT_FAN_HIGH_THRESHOLD
    require_both: bool = DEFAULT_FAN_REQUIRE == "both"
    stability_feature: str = DEFAULT_FAN_STABILITY_FEATURE
    stability_threshold: float = DEFAULT_FAN_STABILITY_THRESHOLD
    stability_min_seconds: float = DEFAULT_FAN_STABILITY_MIN_SECONDS

    def __post_init__(self) -> None:
        if self.stability_feature not in STABILITY_FEATURES:
            raise ValueError(
                f"unknown stability feature {self.stability_feature!r}; "
                f"choose one of {', '.join(STABILITY_FEATURES)}"
            )


def fan_energy(features: Mapping[str, object], config: FanConfig):
    """Both fan bands (or either, if configured) are above threshold."""
    mid = np.asarray(features["500-1k"]) >= config.mid_threshold
    high = np.asarray(features["1k-2k"]) >= config.high_threshold

    return scalarise((mid & high) if config.require_both else (mid | high))


def fan_stationary(features: Mapping[str, object], config: FanConfig):
    """The fan spectrum is holding still, and we have history to say so.

    A missing or NaN stability value compares False, so an unknown
    spectrum is never promoted to a fan.
    """
    steady = (
        np.asarray(features[config.stability_feature])
        <= config.stability_threshold
    )
    informed = (
        np.asarray(features[TEMPORAL_FILL]) >= config.stability_min_seconds
    )

    return scalarise(steady & informed)


def detect_fan(features: Mapping[str, object], config: FanConfig):
    """Fan energy, and a fan spectrum that is holding still."""
    return scalarise(
        np.asarray(fan_energy(features, config))
        & np.asarray(fan_stationary(features, config))
    )
