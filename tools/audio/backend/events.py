"""Transition event recording.

A published state change is the interesting moment, and the seconds
either side of it are the evidence. So the last 15 s of raw PCM are
always in memory; when a transition happens that pre-roll is claimed
and the next 15 s are collected on top of it.

The WAV is the raw source of truth and is never altered -- no gain,
no filtering. The JSON is a compact summary of what the classifier
believed at that moment and under which settings, not a dump of
every window.

What counts as an event depends on the classifier version. Under v1 it
is a change of the single published state. Under v2, which reports
independent observations, an event is one of:

    fan on / fan off                 (FAN_ON, FAN_OFF)
    compressor on / compressor off   (COMPRESSOR_ON, COMPRESSOR_OFF)
    a beep                           (BEEP)
    a manual capture                 (MANUAL)

Each is named for what happened, carries ``eventType``, and records the
observations as they stood. The schemas differ but v1 events stay
readable: v2 only adds keys.

One thing to know about the pre-roll: a transition is published
hold_seconds + roughly median_seconds AFTER the audio actually
changed, so that much of the pre-roll is already the new state. The
default 15 s against a 2.5 s decision leaves plenty of genuine
"before", but raising holdSeconds towards eventPreSeconds eats it,
and at eventPreSeconds <= holdSeconds there is no before at all.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
import wave
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import ClassifierConfig, EventConfig


def to_wav_samples(samples: np.ndarray) -> np.ndarray:
    """Right-aligned 24-bit -> 32-bit container, full scale aligned.

    The same convention capture.py writes, so every existing tool can
    read an event WAV without being told anything about it.
    """
    clipped = np.clip(samples, -(1 << 23), (1 << 23) - 1)

    return (clipped.astype(np.int64) << 8).astype(np.int32)


def event_timeline(
    wav_path: Path,
    config: ClassifierConfig,
    pre_seconds: float,
    nfft: int = 1024,
    overlap: float = 0.5,
) -> dict:
    """Recompute the feature timeline from an event's audio.

    Dispatches on the classifier version recorded with the event.
    """
    if config.version == "v2":
        return _observation_timeline(
            wav_path, config, pre_seconds, nfft, overlap
        )

    return _state_timeline(wav_path, config, pre_seconds, nfft, overlap)


def _observation_timeline(
    wav_path: Path,
    config: ClassifierConfig,
    pre_seconds: float,
    nfft: int = 1024,
    overlap: float = 0.5,
) -> dict:
    """The v2 timeline: fan and compressor over time, and the beeps.

    The smoother starts cold, so the published observations near the
    left edge are still warming up; the candidates and the features are
    exact. Beeps use the raw windows, as they do live.
    """
    from scipy.io import wavfile

    from acstream import FULL_SCALE
    from analyze import to_float
    from classifier.common import ObservationSmoother
    from classifier.detectors import BeepDetector
    from features import FEATURE_NAMES, FeatureExtractor

    sample_rate, raw = wavfile.read(wav_path)
    signal = to_float(raw) * FULL_SCALE

    extractor = FeatureExtractor(sample_rate, nfft, overlap)
    hop_seconds = extractor.hop / sample_rate

    smoother = ObservationSmoother(
        config.rule(),
        window_rate=sample_rate / extractor.hop,
        median_seconds=config.median_seconds,
        hold_seconds=config.hold_seconds,
    )
    beeps = BeepDetector(
        config.beep_config(),
        hop_seconds=hop_seconds,
        window_seconds=nfft / sample_rate,
    )

    columns: dict[str, list] = {
        "t": [],
        "fan": [],
        "compressor": [],
        "fanCandidate": [],
        "compressorCandidate": [],
        # The old single state, derived, so the existing strip works.
        "state": [],
        "candidate": [],
        "beepContrast": [],
    }

    for name in FEATURE_NAMES:
        columns[name] = []

    found = []
    chunk = 512

    for start in range(0, signal.size - chunk + 1, chunk):
        for features in extractor.push(signal[start : start + chunk]):
            decision = smoother.update(features)
            found.extend(beeps.update(features))

            columns["t"].append(round(features.time - pre_seconds, 3))
            columns["fan"].append(decision.fan_detected)
            columns["compressor"].append(decision.compressor_detected)
            columns["fanCandidate"].append(decision.fan_candidate)
            columns["compressorCandidate"].append(
                decision.compressor_candidate
            )
            columns["state"].append(decision.legacy_state())
            columns["candidate"].append(decision.legacy_candidate())
            columns["beepContrast"].append(
                round(features.diagnostics["beep_contrast_db"], 2)
            )

            for name in FEATURE_NAMES:
                columns[name].append(round(decision.values[name], 6))

    found.extend(beeps.flush())

    return {
        "classifierVersion": "v2",
        "sampleRate": sample_rate,
        "nfft": nfft,
        "overlap": overlap,
        "preSeconds": pre_seconds,
        "count": len(columns["t"]),
        "columns": columns,
        "beeps": [
            {
                **beep.to_api(),
                # Relative to the event, like the time axis.
                "startSeconds": round(beep.start_time - pre_seconds, 3),
                "endSeconds": round(beep.end_time - pre_seconds, 3),
            }
            for beep in found
        ],
    }


def _state_timeline(
    wav_path: Path,
    config: ClassifierConfig,
    pre_seconds: float,
    nfft: int = 1024,
    overlap: float = 0.5,
) -> dict:
    """Recompute the v1 feature timeline from an event's audio.

    Derived on demand rather than stored, so the JSON stays a decision
    summary and the WAV stays the only copy of the evidence. It also
    means an old event can be re-examined under different thresholds
    without having recorded anything extra.

    Times are relative to the transition: negative is before it.

    The smoother starts cold here, so the published state near the
    left edge is warming up and will not match what was live at the
    time -- live had history from before the pre-roll. The candidate
    and the band levels are exact; only the held state has to catch
    up.
    """
    from scipy.io import wavfile

    from acstream import FULL_SCALE
    from analyze import to_float
    from classifier.common import Smoother
    from features import FEATURE_NAMES, FeatureExtractor

    sample_rate, raw = wavfile.read(wav_path)

    signal = to_float(raw) * FULL_SCALE

    extractor = FeatureExtractor(sample_rate, nfft, overlap)

    smoother = Smoother(
        config.rule(),
        window_rate=sample_rate / extractor.hop,
        median_seconds=config.median_seconds,
        hold_seconds=config.hold_seconds,
    )

    columns: dict[str, list] = {
        "t": [],
        "state": [],
        "candidate": [],
    }

    for name in FEATURE_NAMES:
        columns[name] = []

    chunk = 512

    for start in range(0, signal.size - chunk + 1, chunk):
        for features in extractor.push(signal[start : start + chunk]):
            decision = smoother.update(features)

            columns["t"].append(
                round(features.time - pre_seconds, 3)
            )
            columns["state"].append(decision.state)
            columns["candidate"].append(decision.candidate)
            # The values the rule decided from, so the picture shows
            # the same smoothed inputs the classifier used.
            for name in FEATURE_NAMES:
                columns[name].append(round(decision.values[name], 6))

    return {
        "classifierVersion": config.version,
        "sampleRate": sample_rate,
        "nfft": nfft,
        "overlap": overlap,
        "preSeconds": pre_seconds,
        "count": len(columns["t"]),
        "columns": columns,
    }


@dataclass
class PendingEvent:
    """A capture in progress, still collecting its post-roll."""

    identifier: str
    started: datetime

    from_state: object
    to_state: object

    stable_seconds: float
    stream_time: float

    features: dict[str, float]
    decision_window: dict[str, dict[str, float]]
    classifier_config: dict
    source: str = "transition"

    # v2: what kind of event, and anything specific to it (the
    # observations at the time, a beep's measurements).
    event_type: str | None = None
    extra: dict = field(default_factory=dict)

    chunks: list[np.ndarray] = field(default_factory=list)
    collected: int = 0
    wanted: int = 0

    @property
    def done(self) -> bool:
        return self.collected >= self.wanted


class EventRecorder:
    """Rolling PCM buffer plus any number of in-flight captures.

    Several captures can overlap: a second transition during the tail
    of the first starts its own event, and both finish independently.
    """

    def __init__(
        self,
        sample_rate: int | None,
        config: EventConfig,
        directory: Path,
    ) -> None:
        self.sample_rate = sample_rate
        self.config = config
        self.directory = directory

        self.lock = threading.Lock()

        self.history: deque[np.ndarray] = deque()
        self.history_samples = 0

        self.pending: list[PendingEvent] = []
        self.written: list[dict] = []

        # Set by the stream service: given a span of wall-clock time,
        # the commands the operator said they sent in it.
        self.command_lookup = None

        self.directory.mkdir(parents=True, exist_ok=True)

        self._load_existing()

    def set_sample_rate(self, sample_rate: int) -> None:
        """Set the device rate once known, before accepting PCM."""
        with self.lock:
            if self.sample_rate == sample_rate:
                return
            if self.pending or self.history:
                raise RuntimeError("cannot change rate during capture")
            self.sample_rate = sample_rate

    def _load_existing(self) -> None:
        """Index events already on disk.

        The files are the record, not this list, so a restart must not
        make earlier transitions disappear from the UI.
        """
        for path in sorted(self.directory.glob("*.json")):
            try:
                metadata = json.loads(
                    path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                continue

            if isinstance(metadata, dict) and "id" in metadata:
                if "review" not in metadata:
                    metadata["review"] = {"status": "unreviewed"}
                    try:
                        self._write_json_atomic(path, metadata)
                    except OSError:
                        # It can still be viewed even on read-only media.
                        pass
                self.written.append(metadata)

    # ------------------------------------------------------ config

    def apply_config(self, config: EventConfig) -> None:
        with self.lock:
            self.config = config
            self._trim()

    @property
    def pre_samples(self) -> int:
        return int(self.config.pre_seconds * (self.sample_rate or 0))

    # --------------------------------------------------------- PCM

    def push(self, samples: np.ndarray) -> list[dict]:
        """Add a frame. Returns metadata for any event just completed."""
        if self.sample_rate is None:
            raise RuntimeError("sample rate is not known yet")
        finished: list[dict] = []

        with self.lock:
            block = np.array(samples, dtype=np.int32, copy=True)

            self.history.append(block)
            self.history_samples += block.size

            self._trim()

            for event in self.pending:
                if event.done:
                    continue

                event.chunks.append(block)
                event.collected += block.size

            still_pending = []

            for event in self.pending:
                if event.done:
                    finished.append(event)
                else:
                    still_pending.append(event)

            self.pending = still_pending

        # Written outside the lock: a WAV is a couple of megabytes and
        # there is no reason to hold up the stream thread's bookkeeping
        # while the disk works.
        return [self._write(event) for event in finished]

    def _trim(self) -> None:
        limit = self.pre_samples

        while (
            self.history
            and self.history_samples - self.history[0].size >= limit
        ):
            self.history_samples -= self.history.popleft().size

    def discard_history(self) -> None:
        """Forget buffered audio across a gap or a reconnect.

        Audio either side of a dropout was never contiguous, and
        splicing it into an event would put a step in the evidence.
        """
        with self.lock:
            self.history.clear()
            self.history_samples = 0

    # ----------------------------------------------------- capture

    def start(
        self,
        from_state: object,
        to_state: object,
        stable_seconds: float,
        stream_time: float,
        features: dict[str, float],
        decision_window: dict[str, dict[str, float]],
        classifier_config: ClassifierConfig,
        source: str = "transition",
        *,
        event_type: str | None = None,
        identifier_suffix: str | None = None,
        extra: dict | None = None,
    ) -> str:
        """Begin capturing an event.

        v1 events are named ``FROM_to_TO``. v2 events pass
        ``identifier_suffix`` (``FAN_ON``, ``BEEP``, ...) and
        ``event_type`` instead, and anything particular to the event
        in ``extra``, which is merged into its metadata.
        """
        if self.sample_rate is None:
            raise RuntimeError("sample rate is not known yet")
        now = datetime.now()

        if identifier_suffix:
            base_identifier = f"{now:%Y-%m-%d_%H%M%S}_{identifier_suffix}"
        else:
            base_identifier = (
                f"{now:%Y-%m-%d_%H%M%S}_"
                f"{from_state or 'NONE'}_to_{to_state}"
            )

        with self.lock:
            identifier = base_identifier
            existing = {
                item.get("id") for item in self.written
            } | {item.identifier for item in self.pending}
            suffix = 2
            while identifier in existing:
                identifier = f"{base_identifier}_{suffix}"
                suffix += 1

            pre = list(self.history)

            event = PendingEvent(
                identifier=identifier,
                started=now,
                from_state=from_state,
                to_state=to_state,
                stable_seconds=stable_seconds,
                stream_time=stream_time,
                features=dict(features),
                decision_window=decision_window,
                classifier_config=classifier_config.to_api(),
                source=source,
                event_type=event_type,
                extra=dict(extra or {}),
                chunks=pre,
                collected=0,
                wanted=int(
                    self.config.post_seconds * self.sample_rate
                ),
            )

            self.pending.append(event)

        return identifier

    # ------------------------------------------------------- write

    def _write(self, event: PendingEvent) -> dict:
        assert self.sample_rate is not None
        samples = (
            np.concatenate(event.chunks)
            if event.chunks
            else np.zeros(0, dtype=np.int32)
        )

        wav_path = self.directory / f"{event.identifier}.wav"
        json_path = self.directory / f"{event.identifier}.json"

        with wave.open(str(wav_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(4)
            handle.setframerate(self.sample_rate)
            handle.writeframes(to_wav_samples(samples).tobytes())

        pre_actual = max(0, samples.size - event.collected)

        v2_keys = {}

        if event.event_type:
            v2_keys = {"eventType": event.event_type, **event.extra}

            context = self._command_context(event)

            if context:
                v2_keys["commandContext"] = context

        metadata = {
            "id": event.identifier,
            # Which classifier produced this event's from/to, and under
            # which settings. Events recorded before versions existed
            # carry neither key and were all classified by v1.
            "classifierVersion": event.classifier_config.get(
                "classifierVersion", "v1"
            ),
            "time": event.started.isoformat(timespec="seconds"),
            "from": event.from_state,
            "to": event.to_state,
            "source": event.source,
            "candidateHeldSeconds": round(event.stable_seconds, 2),
            "streamSeconds": round(event.stream_time, 2),
            "audio": {
                "file": wav_path.name,
                "sampleRate": self.sample_rate,
                "seconds": round(samples.size / self.sample_rate, 2),
                "preSeconds": round(
                    pre_actual / self.sample_rate, 2
                ),
                "postSeconds": round(
                    event.collected / self.sample_rate, 2
                ),
            },
            "featuresAtTransition": {
                name: round(value, 6)
                for name, value in event.features.items()
            },
            "decisionWindow": event.decision_window,
            "classifierConfig": event.classifier_config,
            **v2_keys,
            "review": {"status": "unreviewed"},
        }

        self._write_json_atomic(json_path, metadata)

        with self.lock:
            self.written.append(metadata)

        return metadata

    # How far from an event a command can be and still be its context.
    COMMAND_WINDOW_SECONDS = 10.0

    def _command_context(self, event: PendingEvent) -> dict | None:
        """The command the operator recorded nearest this event, if any.

        Matched on wall-clock time, by distance from the moment the event
        was triggered: the stream clock restarts with the process, so it
        cannot be compared across a restart. Watson only attaches the
        fact; whether it explains the event is the reader's (or the
        controller's) call.
        """
        if self.command_lookup is None:
            return None

        trigger = event.started.timestamp()
        window = self.COMMAND_WINDOW_SECONDS
        commands = self.command_lookup(trigger - window, trigger + window)

        if not commands:
            return None

        nearest = min(commands, key=lambda c: abs(c["time"] - trigger))

        return {
            "command": nearest["command"],
            # Stream seconds, like the beep's own timestamps.
            "sentAt": nearest["streamSeconds"],
            "expectedBeep": nearest["expectedBeep"],
            "secondsFromEvent": round(nearest["time"] - trigger, 2),
            "note": nearest.get("note", ""),
        }

    def list_events(
        self,
        page: int = 1,
        page_size: int = 20,
        search: str = "",
    ) -> dict:
        with self.lock:
            events = list(reversed(self.written))

        query = search.strip().casefold()
        if query:
            events = [
                event for event in events
                if query in self._search_text(event)
            ]

        total = len(events)
        start = (page - 1) * page_size
        return {
            "events": events[start : start + page_size],
            "page": page,
            "pageSize": page_size,
            "total": total,
            "pages": (total + page_size - 1) // page_size,
        }

    @staticmethod
    def _search_text(metadata: dict) -> str:
        review = metadata.get("review", {})
        result = (
            "unreviewed" if review.get("status") != "reviewed"
            else "correct" if review.get("classificationCorrect")
            else "incorrect"
        )
        observations = metadata.get("observations") or {}
        values = [
            metadata.get("id"), metadata.get("time"),
            metadata.get("from"), metadata.get("to"),
            metadata.get("eventType"),
            *(
                f"{name}:{'on' if state else 'off'}"
                for name, state in observations.items()
                if state is not None
            ),
            metadata.get("source"), review.get("status"),
            review.get("actualFrom"), review.get("actualTo"),
            review.get("notes"),
            result,
            *(review.get("interference") or []),
        ]
        return " ".join(str(value) for value in values if value).casefold()

    def get(self, identifier: str) -> dict | None:
        with self.lock:
            for metadata in reversed(self.written):
                if metadata.get("id") == identifier:
                    return metadata

        return None

    def review(self, identifier: str, review: dict) -> dict | None:
        """Replace the human review and atomically persist the JSON."""
        updated, missing = self.review_many({identifier: review})
        return None if missing else updated[0]

    def review_many(
        self, reviews: dict[str, dict]
    ) -> tuple[list[dict], list[str]]:
        """Atomically replace each JSON file after validating all ids.

        Every event is resolved before the first write, so a stale or
        invalid selection cannot partially review the events that still
        exist. Each individual JSON replacement remains atomic.
        """
        with self.lock:
            by_id = {
                item.get("id"): item for item in self.written
                if item.get("id") in reviews
            }
            missing = [
                identifier for identifier in reviews
                if identifier not in by_id
            ]
            if missing:
                return [], missing

            pending = []
            for identifier, review in reviews.items():
                metadata = by_id[identifier]
                updated_review = {"status": "reviewed", **review}
                updated = {**metadata, "review": updated_review}
                path = self._event_path(identifier, ".json")
                if path is None:
                    return [], [identifier]
                pending.append((metadata, updated_review, updated, path))

            for _metadata, _review, updated, path in pending:
                self._write_json_atomic(path, updated)
            for metadata, updated_review, _updated, _path in pending:
                metadata["review"] = updated_review

            return [dict(metadata) for metadata, *_rest in pending], []

    def annotate(self, identifier: str, fields: dict) -> dict | None:
        """Set (or, with None, remove) top-level keys on an event.

        For facts about the event that are not part of the review, such
        as the command that was sent near it. The JSON is replaced
        atomically, like a review.
        """
        with self.lock:
            metadata = next(
                (
                    item for item in self.written
                    if item.get("id") == identifier
                ),
                None,
            )

            if metadata is None:
                return None

            path = self._event_path(identifier, ".json")

            if path is None:
                return None

            updated = dict(metadata)

            for key, value in fields.items():
                if value is None:
                    updated.pop(key, None)
                else:
                    updated[key] = value

            self._write_json_atomic(path, updated)

            metadata.clear()
            metadata.update(updated)

            return dict(metadata)

    def delete(self, identifier: str) -> bool:
        """Delete one completed event's self-contained JSON/WAV pair."""
        with self.lock:
            index = next(
                (
                    i for i, item in enumerate(self.written)
                    if item.get("id") == identifier
                ),
                None,
            )
            if index is None:
                return False

            paths = [
                self._event_path(identifier, suffix)
                for suffix in (".json", ".wav")
            ]
            if any(path is None for path in paths):
                return False
            for path in paths:
                try:
                    assert path is not None
                    path.unlink()
                except FileNotFoundError:
                    pass
            self.written.pop(index)
            return True

    @staticmethod
    def _write_json_atomic(path: Path, metadata: dict) -> None:
        temporary = path.with_name(
            f".{path.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            temporary.write_text(
                json.dumps(metadata, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def wav_path(self, identifier: str) -> Path | None:
        """Resolve an id to a file, refusing anything outside the dir.

        The id arrives from a URL, so it must not be able to name a
        path of its own.
        """
        path = self._event_path(identifier, ".wav")
        if path is None:
            return None
        return path if path.is_file() else None

    def _event_path(self, identifier: str, suffix: str) -> Path | None:
        path = (self.directory / f"{identifier}{suffix}").resolve()
        return path if self.directory.resolve() in path.parents else None

    def status(self) -> dict:
        with self.lock:
            return {
                "buffered": round(
                    self.history_samples / self.sample_rate, 1
                ) if self.sample_rate else 0.0,
                "pending": len(self.pending),
                "written": len(self.written),
            }
