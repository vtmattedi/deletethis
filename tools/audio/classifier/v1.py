"""Classifier v1: thresholds on three band levels.

    30-80 Hz at or above ``compressor``          -> COMPRESSOR
    500-1k and 1k-2k at or above their levels    -> FAN
    otherwise                                    -> OFF

Frozen. v2 exists so this never has to change: it is the baseline the
collected dataset is judged against, and the rule the events recorded
before v2 were classified by.

The numbers below are the *code* defaults, tuned on the first 18
recordings. They are not what the live system ran with afterwards:
most later events were recorded at ``compressor = -38``, because at
-48 the OFF-state 30-80 level in the newer sessions (median -48.7 dB)
sits right on the threshold. A saved config overrides these.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .common import COMPRESSOR, FAN, OFF

# COMPRESSOR had a ~10 dB margin in the original recordings: 30-80 sat
# at -38 dB with the compressor running and -57..-59 dB otherwise.
#
# FAN is far tighter. The fan's own 1k-2k level is about -63.7 dB, and
# speech with the AC off reaches -58 dB in peaks, so the usable window
# for the high threshold is only -66..-64 and -65 is its centre. Both
# fan bands must be over threshold, which is what separates a fan from
# someone talking; either-of-two cannot get past 93%.
DEFAULT_COMPRESSOR_THRESHOLD = -48.0
DEFAULT_FAN_MID_THRESHOLD = -62.0
DEFAULT_FAN_HIGH_THRESHOLD = -65.0
DEFAULT_FAN_REQUIRE = "both"


@dataclass
class Thresholds:
    compressor: float = DEFAULT_COMPRESSOR_THRESHOLD
    fan_mid: float = DEFAULT_FAN_MID_THRESHOLD
    fan_high: float = DEFAULT_FAN_HIGH_THRESHOLD
    fan_require_both: bool = False

    def decide(self, values: Mapping[str, float]) -> str:
        return classify(values, self)


def classify(
    bands: Mapping[str, float],
    thresholds: Thresholds,
) -> str:
    """Hierarchical rules over already-smoothed band levels."""

    # Stage 1. The compressor is the only thing in a room that puts
    # sustained energy this low, so it is tested first and wins
    # outright.
    if bands["30-80"] >= thresholds.compressor:
        return COMPRESSOR

    # Stage 2. Airflow is broadband and sits in the mid and upper mids.
    mid = bands["500-1k"] >= thresholds.fan_mid
    high = bands["1k-2k"] >= thresholds.fan_high

    if thresholds.fan_require_both:
        return FAN if (mid and high) else OFF

    return FAN if (mid or high) else OFF
