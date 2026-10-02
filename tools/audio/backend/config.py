"""Runtime configuration.

One authoritative ClassifierConfig. The API speaks camelCase because
the browser does; everything inside is snake_case because Python is.
The mapping lives here and nowhere else.

Classifier versions
-------------------
A config belongs to one classifier version, and the version decides
where results live:

    v1   tools/audio/results/        (the original dataset)
    v2   tools/audio/results/v2/     (everything generated under v2)

The two never share a directory. v1's events are the evidence v2 is
judged against, so a v2 run that quietly added its own events to the
same folder would contaminate the baseline it is compared with.
``AppConfig`` enforces that rather than trusting everyone to remember.

The library default is v1, so code that builds a bare ``AppConfig()``
behaves exactly as it always has. Choosing v2 is an explicit act, made
once at the entry point (``AppConfig.for_version``).
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path

from classifier import VERSIONS
from classifier import v1 as rules_v1
from classifier import v2 as rules_v2
from classifier.common import (
    DEFAULT_HOLD_SECONDS,
    DEFAULT_MEDIAN_SECONDS,
)
from classifier.detectors import beep as beep_defaults
from classifier.detectors import BeepConfig, CompressorConfig, FanConfig

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent

RESULTS = TOOLS / "results"
RESULTS_V2 = RESULTS / "v2"
RECORDINGS = TOOLS / "recordings"
WEB = TOOLS / "web"

# v1's locations, kept under their original names for existing callers.
EVENTS_DIR = RESULTS / "events"
HISTORY_DB = RESULTS / "audio.db"
RUNTIME_CONFIG = RESULTS / "config.json"

DEFAULT_TARGET = "192.168.1.50:3333"
DEFAULT_HTTP_HOST = "127.0.0.1"
DEFAULT_HTTP_PORT = 8000

# Live snapshots to the browser. The classifier runs at the full
# window rate (~31/s); this is only how often the UI is told.
DEFAULT_LIVE_HZ = 5.0

DEFAULT_VERSION = "v1"


@dataclass(frozen=True)
class ResultPaths:
    """Everything one classifier version writes."""

    root: Path
    events: Path
    history: Path
    config: Path
    evaluation: Path
    plots: Path


def result_paths(version: str) -> ResultPaths:
    if version == "v1":
        root = RESULTS
    elif version == "v2":
        root = RESULTS_V2
    else:
        raise ValueError(f"unknown classifier version {version!r}")

    return ResultPaths(
        root=root,
        events=root / "events",
        history=root / "audio.db",
        config=root / "config.json",
        evaluation=root / "evaluation",
        plots=root / "plots",
    )


# The v2-only fields, as the API spells them.
V2_API_FIELDS = (
    "fanStabilityFeature",
    "fanStabilityThreshold",
    "fanStabilityMinSeconds",
    "beepMinContrastDb",
    "beepEdgeContrastDb",
    "beepMinLevelDb",
    "beepMinMs",
    "beepMaxMs",
    "beepMinHz",
    "beepMaxHz",
    "beepMaxPeakSpreadHz",
)


@dataclass
class ClassifierConfig:
    # v1's values: a bare ClassifierConfig() is the v1 classifier, as
    # it was before versions existed.
    version: str = "v1"
    compressor_threshold: float = rules_v1.DEFAULT_COMPRESSOR_THRESHOLD
    fan_mid_threshold: float = rules_v1.DEFAULT_FAN_MID_THRESHOLD
    fan_high_threshold: float = rules_v1.DEFAULT_FAN_HIGH_THRESHOLD
    fan_require: str = rules_v1.DEFAULT_FAN_REQUIRE
    median_seconds: float = DEFAULT_MEDIAN_SECONDS
    hold_seconds: float = DEFAULT_HOLD_SECONDS

    # v2 only. Present on a v1 config so patching is uniform, but
    # neither used by the v1 rule nor reported by its to_api().
    fan_stability_feature: str = rules_v2.DEFAULT_FAN_STABILITY_FEATURE
    fan_stability_threshold: float = (
        rules_v2.DEFAULT_FAN_STABILITY_THRESHOLD
    )
    fan_stability_min_seconds: float = (
        rules_v2.DEFAULT_FAN_STABILITY_MIN_SECONDS
    )

    # v2 only: the beep detector. It reads raw windows, so none of the
    # smoothing settings above apply to it.
    beep_min_contrast_db: float = beep_defaults.DEFAULT_MIN_CONTRAST_DB
    beep_edge_contrast_db: float = beep_defaults.DEFAULT_EDGE_CONTRAST_DB
    beep_min_level_db: float = beep_defaults.DEFAULT_MIN_LEVEL_DB
    beep_min_ms: float = beep_defaults.DEFAULT_MIN_MS
    beep_max_ms: float = beep_defaults.DEFAULT_MAX_MS
    beep_min_hz: float = beep_defaults.DEFAULT_MIN_HZ
    beep_max_hz: float = beep_defaults.DEFAULT_MAX_HZ
    beep_max_peak_spread_hz: float = (
        beep_defaults.DEFAULT_MAX_PEAK_SPREAD_HZ
    )

    def __post_init__(self) -> None:
        if self.version not in VERSIONS:
            raise ValueError(
                f"unknown classifier version {self.version!r}"
            )

    @classmethod
    def for_version(cls, version: str) -> "ClassifierConfig":
        """The defaults of one version."""
        if version == "v2":
            return cls(
                version="v2",
                compressor_threshold=(
                    rules_v2.DEFAULT_COMPRESSOR_THRESHOLD
                ),
                fan_mid_threshold=rules_v2.DEFAULT_FAN_MID_THRESHOLD,
                fan_high_threshold=rules_v2.DEFAULT_FAN_HIGH_THRESHOLD,
                fan_require=rules_v2.DEFAULT_FAN_REQUIRE,
            )

        return cls(version=version)

    @classmethod
    def from_api(
        cls, payload: dict, version: str | None = None
    ) -> "ClassifierConfig":
        """Rebuild a config from API-shaped settings.

        Used for the config recorded inside an event. Events written
        before versions existed carry no ``classifierVersion``, and
        those were all classified by v1.
        """
        chosen = version or payload.get("classifierVersion") or "v1"

        return cls.for_version(chosen).patched(payload)

    def beep_config(self) -> BeepConfig:
        return BeepConfig(
            min_contrast_db=self.beep_min_contrast_db,
            edge_contrast_db=self.beep_edge_contrast_db,
            min_level_db=self.beep_min_level_db,
            min_ms=self.beep_min_ms,
            max_ms=self.beep_max_ms,
            min_hz=self.beep_min_hz,
            max_hz=self.beep_max_hz,
            max_peak_spread_hz=self.beep_max_peak_spread_hz,
        )

    def rule(self):
        """The decision rule for this version.

        v1 is one rule returning one state. v2 is a set of independent
        detectors; this is its stateless half (fan and compressor), and
        ``beep_config`` describes the stateful one.
        """
        if self.version == "v2":
            return rules_v2.ObservationRules(
                fan=FanConfig(
                    mid_threshold=self.fan_mid_threshold,
                    high_threshold=self.fan_high_threshold,
                    require_both=self.fan_require == "both",
                    stability_feature=self.fan_stability_feature,
                    stability_threshold=self.fan_stability_threshold,
                    stability_min_seconds=self.fan_stability_min_seconds,
                ),
                compressor=CompressorConfig(
                    threshold=self.compressor_threshold
                ),
                beep=self.beep_config(),
            )

        return rules_v1.Thresholds(
            compressor=self.compressor_threshold,
            fan_mid=self.fan_mid_threshold,
            fan_high=self.fan_high_threshold,
            fan_require_both=self.fan_require == "both",
        )

    # Older name, still used by callers that predate versions.
    def thresholds(self):
        return self.rule()

    def to_api(self) -> dict:
        payload = {
            "classifierVersion": self.version,
            "compressorThreshold": self.compressor_threshold,
            "fanMidThreshold": self.fan_mid_threshold,
            "fanHighThreshold": self.fan_high_threshold,
            "fanRequire": self.fan_require,
            "medianSeconds": self.median_seconds,
            "holdSeconds": self.hold_seconds,
        }

        if self.version == "v2":
            payload.update({
                "fanStabilityFeature": self.fan_stability_feature,
                "fanStabilityThreshold": self.fan_stability_threshold,
                "fanStabilityMinSeconds": (
                    self.fan_stability_min_seconds
                ),
                "beepMinContrastDb": self.beep_min_contrast_db,
                "beepEdgeContrastDb": self.beep_edge_contrast_db,
                "beepMinLevelDb": self.beep_min_level_db,
                "beepMinMs": self.beep_min_ms,
                "beepMaxMs": self.beep_max_ms,
                "beepMinHz": self.beep_min_hz,
                "beepMaxHz": self.beep_max_hz,
                "beepMaxPeakSpreadHz": self.beep_max_peak_spread_hz,
            })

        return payload

    def patched(self, patch: dict) -> "ClassifierConfig":
        """A copy with the given camelCase fields replaced.

        ``classifierVersion`` is deliberately not patchable: changing
        version changes where results are stored, which is a restart,
        not a setting.
        """
        mapping = {
            "compressorThreshold": "compressor_threshold",
            "fanMidThreshold": "fan_mid_threshold",
            "fanHighThreshold": "fan_high_threshold",
            "fanRequire": "fan_require",
            "medianSeconds": "median_seconds",
            "holdSeconds": "hold_seconds",
            "fanStabilityFeature": "fan_stability_feature",
            "fanStabilityThreshold": "fan_stability_threshold",
            "fanStabilityMinSeconds": "fan_stability_min_seconds",
            "beepMinContrastDb": "beep_min_contrast_db",
            "beepEdgeContrastDb": "beep_edge_contrast_db",
            "beepMinLevelDb": "beep_min_level_db",
            "beepMinMs": "beep_min_ms",
            "beepMaxMs": "beep_max_ms",
            "beepMinHz": "beep_min_hz",
            "beepMaxHz": "beep_max_hz",
            "beepMaxPeakSpreadHz": "beep_max_peak_spread_hz",
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
    # What --target / AUDIO_TARGET said at launch. A target saved from
    # the web UI only holds while this is unchanged, so editing .env and
    # restarting compose still takes effect.
    launch_target: str = ""
    http_host: str = DEFAULT_HTTP_HOST
    http_port: int = DEFAULT_HTTP_PORT
    live_hz: float = DEFAULT_LIVE_HZ

    classifier: ClassifierConfig = field(
        default_factory=ClassifierConfig
    )

    events: EventConfig = field(default_factory=EventConfig)

    events_dir: Path = EVENTS_DIR
    history_path: Path = HISTORY_DB
    config_path: Path = RUNTIME_CONFIG

    def __post_init__(self) -> None:
        if not self.launch_target:
            self.launch_target = self.target
        self.check_isolation()

    @classmethod
    def for_version(cls, version: str, **overrides) -> "AppConfig":
        """A config for one classifier version, with its own results."""
        paths = result_paths(version)

        values = {
            "classifier": ClassifierConfig.for_version(version),
            "events_dir": paths.events,
            "history_path": paths.history,
            "config_path": paths.config,
        }
        values.update(overrides)

        return cls(**values)

    def check_isolation(self) -> None:
        """Refuse to let one version write into the other's results.

        The v1 events are the labelled evidence v2 is evaluated on. A
        v2 run that added its own events to that folder would change
        the baseline it is being compared with, and nothing would show
        it. So this is checked, not documented.
        """
        v1_paths = result_paths("v1")
        v2_root = RESULTS_V2.resolve()

        targets = {
            "events": Path(self.events_dir),
            "history": Path(self.history_path),
            "config": Path(self.config_path),
        }
        v1_stores = {
            "events": v1_paths.events,
            "history": v1_paths.history,
            "config": v1_paths.config,
        }

        for label, path in targets.items():
            resolved = path.resolve()

            if self.classifier.version == "v2":
                if resolved == v1_stores[label].resolve():
                    raise ValueError(
                        f"classifier v2 must not write its {label} "
                        f"into the v1 results ({resolved}); use "
                        f"{result_paths('v2').root}"
                    )
            elif v2_root == resolved or v2_root in resolved.parents:
                raise ValueError(
                    f"classifier v1 must not write its {label} into "
                    f"the v2 results ({resolved})"
                )

    @property
    def target_path(self) -> Path:
        """Where a stream address chosen in the web UI is remembered."""
        return Path(self.config_path).with_name("target.json")

    def to_api(self) -> dict:
        payload = self.classifier.to_api()
        payload.update(self.events.to_api())

        return payload


def load_saved_target(path: Path, launch_target: str) -> str | None:
    """The stream address saved from the web UI, if it still applies.

    It was chosen against a particular launch target. If that has since
    changed (e.g. AUDIO_TARGET edited in .env), the operator has spoken
    more recently than the browser did, so the saved address is dropped.
    """
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    target = payload.get("target") if isinstance(payload, dict) else None
    if not isinstance(target, str) or not target:
        raise ValueError("saved target must be {\"target\": \"host:port\"}")
    if payload.get("launchTarget") != launch_target:
        return None
    return target


def load_runtime_config(path: Path) -> dict:
    """Load the saved API-shaped settings, or defaults when absent."""
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("saved config must be a JSON object")
    return payload


def write_runtime_config(path: Path, payload: dict) -> None:
    """Atomically replace the persisted runtime settings (or target)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
