"""Independent acoustic detectors.

Three observations, each its own module and each testable alone:

    compressor   is the compressor running?        stateless
    fan          is the fan running?               stateless
    beep         did the unit just beep?           a small state machine

Watson reports what it hears. It does not infer whether the air
conditioner is on or whether a command worked: "fan and compressor both
on" and "fan on, compressor off" are both ordinary, and neither detector
suppresses the other. The controller combines them with its own record
of what it sent.
"""

from .beep import BeepConfig, BeepDetector, BeepEvent
from .compressor import (
    DEFAULT_COMPRESSOR_THRESHOLD,
    CompressorConfig,
    detect_compressor,
)
from .fan import (
    DEFAULT_FAN_HIGH_THRESHOLD,
    DEFAULT_FAN_MID_THRESHOLD,
    DEFAULT_FAN_REQUIRE,
    DEFAULT_FAN_STABILITY_FEATURE,
    DEFAULT_FAN_STABILITY_MIN_SECONDS,
    DEFAULT_FAN_STABILITY_THRESHOLD,
    STABILITY_FEATURES,
    TEMPORAL_FILL,
    FanConfig,
    detect_fan,
    fan_energy,
    fan_stationary,
)

__all__ = [
    "BeepConfig",
    "BeepDetector",
    "BeepEvent",
    "CompressorConfig",
    "DEFAULT_COMPRESSOR_THRESHOLD",
    "DEFAULT_FAN_HIGH_THRESHOLD",
    "DEFAULT_FAN_MID_THRESHOLD",
    "DEFAULT_FAN_REQUIRE",
    "DEFAULT_FAN_STABILITY_FEATURE",
    "DEFAULT_FAN_STABILITY_MIN_SECONDS",
    "DEFAULT_FAN_STABILITY_THRESHOLD",
    "FanConfig",
    "STABILITY_FEATURES",
    "TEMPORAL_FILL",
    "detect_compressor",
    "detect_fan",
    "fan_energy",
    "fan_stationary",
]
