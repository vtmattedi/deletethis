"""Classifier v2: v1 plus a stationarity gate on FAN.

    30-80 Hz at or above ``compressor``                      -> COMPRESSOR
    fan energy in both bands
      AND the fan spectrum is stationary                     -> FAN
    otherwise                                                -> OFF

The COMPRESSOR rule is v1's, unchanged. What v2 adds is the second half
of the FAN test. Fan *energy* alone cannot tell a fan from a voice, a
printer or a television: all of them put real power in 500-1k and
1k-2k. What a fan does that they do not is stay put. Over two seconds
its band level barely moves, where speech and machinery keep changing.
That is measured by the ``*_std`` features -- the standard deviation of
a band's level over the last two seconds -- and the gate is simply
``feature <= fan_stability_threshold``.

Stationarity and ``holdSeconds`` are different tools and neither
substitutes for the other. The hold stops a short-lived candidate being
*published*; stationarity says what the signal *is*. A hold alone can
only wait, and a printer that runs for ten seconds outlasts any hold
that is short enough to be useful.

One decision function, used everywhere. ``classify_v2_codes`` is written
on numpy booleans so that it accepts either scalars (the live path, one
window at a time) or whole arrays (the threshold search, every window of
every event at once) and executes the same lines for both. There is no
second copy of the rule for the evaluation tools to drift away from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .common import COMPRESSOR, FAN, OFF, STATES

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
# seconds of history they were computed over. Duplicated as a string
# here (rather than imported) so this module stays free of the DSP
# layer; a test pins it to features.TEMPORAL_FILL.
TEMPORAL_FILL = "temporal_seconds"

# --- defaults ------------------------------------------------------
#
# COMPRESSOR is -38, not v1's code default of -48. The live system has
# run at -38 since the dataset was being collected (91 of the first 125
# events were recorded under it), because in the newer sessions the
# OFF-state 30-80 level has a median of -48.7 dB: a -48 threshold calls
# 34% of OFF windows COMPRESSOR. It is the same rule with the value the
# system actually runs; see the evaluation under results/v2/evaluation.
DEFAULT_COMPRESSOR_THRESHOLD = -38.0
DEFAULT_FAN_MID_THRESHOLD = -62.0
DEFAULT_FAN_HIGH_THRESHOLD = -65.0
DEFAULT_FAN_REQUIRE = "both"

DEFAULT_FAN_STABILITY_FEATURE = "1k-2k_std"

# dB. Chosen by replaying the reviewed events (88 events, 53k scored
# windows, 63 time-groups; tools/audio/evaluate_events.py), not guessed.
# Starting guesses of 1.2-1.5 dB cost 20-30% of real FAN windows, and
# 2.0 dB still costs about 7%. The measured behaviour is:
#
#   below ~2 dB    FAN recall collapses (a real fan moves a little)
#   ~4 .. ~6 dB    OFF->FAN stays flat at its floor while FAN recall
#                  climbs to 98.6%: a tighter gate buys nothing here
#   above ~6.1 dB  OFF->FAN grows again as noise gets through
#
# so the best balanced accuracy sits on a plateau of 4.9 .. 7.0 dB and
# 6.0 is its centre. Choosing it from 62 of 63 groups and testing on the
# held-out one (leave-one-group-out) gives the same balanced accuracy
# as the in-sample best, so the figure is not an artefact of tuning it
# on the data it is scored on. Re-run the evaluation when the dataset
# grows or the microphone moves: this is a property of the installation.
DEFAULT_FAN_STABILITY_THRESHOLD = 6.0

# A std over two samples is meaningless and near zero, so the gate will
# not call anything stationary until it has this much history behind it.
DEFAULT_FAN_STABILITY_MIN_SECONDS = 1.0


@dataclass
class ThresholdsV2:
    compressor: float = DEFAULT_COMPRESSOR_THRESHOLD
    fan_mid: float = DEFAULT_FAN_MID_THRESHOLD
    fan_high: float = DEFAULT_FAN_HIGH_THRESHOLD
    fan_require_both: bool = DEFAULT_FAN_REQUIRE == "both"
    stability_feature: str = DEFAULT_FAN_STABILITY_FEATURE
    stability_threshold: float = DEFAULT_FAN_STABILITY_THRESHOLD
    stability_min_seconds: float = DEFAULT_FAN_STABILITY_MIN_SECONDS

    def __post_init__(self) -> None:
        if self.stability_feature not in STABILITY_FEATURES:
            raise ValueError(
                f"unknown stability feature {self.stability_feature!r}; "
                f"choose one of {', '.join(STABILITY_FEATURES)}"
            )

    def decide(self, values: Mapping[str, float]) -> str:
        return classify_v2(values, self)


def classify_v2_codes(
    features: Mapping[str, object],
    rule: ThresholdsV2,
):
    """State codes (index into ``STATES``) for scalar or array features.

    ``features`` maps feature names to a float or an array of floats.
    The result has the same shape: 0 = OFF, 1 = FAN, 2 = COMPRESSOR.
    """
    compressor = np.asarray(features["30-80"]) >= rule.compressor

    mid = np.asarray(features["500-1k"]) >= rule.fan_mid
    high = np.asarray(features["1k-2k"]) >= rule.fan_high
    energy = (mid & high) if rule.fan_require_both else (mid | high)

    # A missing or NaN stability value compares False, so an unknown
    # spectrum is never promoted to FAN.
    stationary = (
        np.asarray(features[rule.stability_feature])
        <= rule.stability_threshold
    ) & (
        np.asarray(features[TEMPORAL_FILL])
        >= rule.stability_min_seconds
    )

    return np.where(compressor, 2, np.where(energy & stationary, 1, 0))


def classify_v2(
    features: Mapping[str, float],
    rule: ThresholdsV2,
) -> str:
    """Decide one window from its complete smoothed feature set."""
    return STATES[int(classify_v2_codes(features, rule))]


# Re-exported for callers that want the state names with the rule.
__all__ = [
    "COMPRESSOR",
    "FAN",
    "OFF",
    "ThresholdsV2",
    "classify_v2",
    "classify_v2_codes",
]
