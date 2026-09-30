"""Runtime configuration.

One authoritative ClassifierConfig. The API speaks camelCase because
the browser does; everything inside is snake_case because Python is.
The mapping lives here and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from classify_live import (
    DEFAULT_COMPRESSOR_THRESHOLD,
    DEFAULT_FAN_HIGH_THRESHOLD,
    DEFAULT_FAN_MID_THRESHOLD,
    DEFAULT_FAN_REQUIRE,
    DEFAULT_HOLD_SECONDS,
    DEFAULT_MEDIAN_SECONDS,
    Thresholds,
)

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent

RESULTS = TOOLS / "results"
EVENTS_DIR = RESULTS / "events"
HISTORY_DB = RESULTS / "audio.db"
RECORDINGS = TOOLS / "recordings"
WEB = TOOLS / "web"

DEFAULT_TARGET = "192.168.1.50:3333"
DEFAULT_HTTP_HOST = "127.0.0.1"
DEFAULT_HTTP_PORT = 8000

# Live snapshots to the browser. The classifier runs at the full
# window rate (~31/s); this is only how often the UI is told.
DEFAULT_LIVE_HZ = 5.0


@dataclass
class ClassifierConfig:
    compressor_threshold: float = DEFAULT_COMPRESSOR_THRESHOLD
    fan_mid_threshold: float = DEFAULT_FAN_MID_THRESHOLD
    fan_high_threshold: float = DEFAULT_FAN_HIGH_THRESHOLD
    fan_require: str = DEFAULT_FAN_REQUIRE
    median_seconds: float = DEFAULT_MEDIAN_SECONDS
    hold_seconds: float = DEFAULT_HOLD_SECONDS

    def thresholds(self) -> Thresholds:
        return Thresholds(
            compressor=self.compressor_threshold,
            fan_mid=self.fan_mid_threshold,
            fan_high=self.fan_high_threshold,
            fan_require_both=self.fan_require == "both",
        )

    def to_api(self) -> dict:
        return {
            "compressorThreshold": self.compressor_threshold,
            "fanMidThreshold": self.fan_mid_threshold,
            "fanHighThreshold": self.fan_high_threshold,
            "fanRequire": self.fan_require,
            "medianSeconds": self.median_seconds,
            "holdSeconds": self.hold_seconds,
        }

    def patched(self, patch: dict) -> "ClassifierConfig":
        """A copy with the given camelCase fields replaced."""
        mapping = {
            "compressorThreshold": "compressor_threshold",
            "fanMidThreshold": "fan_mid_threshold",
            "fanHighThreshold": "fan_high_threshold",
            "fanRequire": "fan_require",
            "medianSeconds": "median_seconds",
            "holdSeconds": "hold_seconds",
        }

        changes = {
            mapping[key]: value
            for key, value in patch.items()
            if key in mapping and value is not None
        }

        return replace(self, **changes)

    def affects_smoothing(self, other: "ClassifierConfig") -> bool:
        """Do the two differ in a way that resizes the history?"""
        return (
            self.median_seconds != other.median_seconds
            or self.hold_seconds != other.hold_seconds
        )


@dataclass
class EventConfig:
    pre_seconds: float = 15.0
    post_seconds: float = 15.0

    def to_api(self) -> dict:
        return {
            "eventPreSeconds": self.pre_seconds,
            "eventPostSeconds": self.post_seconds,
        }

    def patched(self, patch: dict) -> "EventConfig":
        mapping = {
            "eventPreSeconds": "pre_seconds",
            "eventPostSeconds": "post_seconds",
        }

        changes = {
            mapping[key]: value
            for key, value in patch.items()
            if key in mapping and value is not None
        }

        return replace(self, **changes)


@dataclass
class AppConfig:
    target: str = DEFAULT_TARGET
    http_host: str = DEFAULT_HTTP_HOST
    http_port: int = DEFAULT_HTTP_PORT
    live_hz: float = DEFAULT_LIVE_HZ

    classifier: ClassifierConfig = field(
        default_factory=ClassifierConfig
    )

    events: EventConfig = field(default_factory=EventConfig)

    events_dir: Path = EVENTS_DIR
    history_path: Path = HISTORY_DB

    def to_api(self) -> dict:
        payload = self.classifier.to_api()
        payload.update(self.events.to_api())

        return payload
