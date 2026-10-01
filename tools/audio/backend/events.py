"""Transition event recording.

A published state change is the interesting moment, and the seconds
either side of it are the evidence. So the last 15 s of raw PCM are
always in memory; when a transition happens that pre-roll is claimed
and the next 15 s are collected on top of it.

The WAV is the raw source of truth and is never altered -- no gain,
no filtering. The JSON is a compact summary of what the classifier
believed at that moment and under which settings, not a dump of
every window.

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
    from classify_live import (
        FEATURE_NAMES,
        FeatureExtractor,
        Smoother,
    )

    sample_rate, raw = wavfile.read(wav_path)

    signal = to_float(raw) * FULL_SCALE

    extractor = FeatureExtractor(sample_rate, nfft, overlap)

    smoother = Smoother(
        config.thresholds(),
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
            values = features.values
            for name in FEATURE_NAMES:
                value = (
                    decision.rms_db if name == "rms"
                    else decision.smoothed[name]
                    if name in decision.smoothed
                    else values[name]
                )
                columns[name].append(round(value, 6))

    return {
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

    from_state: str | None
    to_state: str

    stable_seconds: float
    stream_time: float

    features: dict[str, float]
    decision_window: dict[str, dict[str, float]]
    classifier_config: dict
    source: str = "transition"

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
        from_state: str | None,
        to_state: str,
        stable_seconds: float,
        stream_time: float,
        features: dict[str, float],
        decision_window: dict[str, dict[str, float]],
        classifier_config: ClassifierConfig,
        source: str = "transition",
    ) -> str:
        if self.sample_rate is None:
            raise RuntimeError("sample rate is not known yet")
        now = datetime.now()

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

        metadata = {
            "id": event.identifier,
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
            "review": {"status": "unreviewed"},
        }

        self._write_json_atomic(json_path, metadata)

        with self.lock:
            self.written.append(metadata)

        return metadata

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
        values = [
            metadata.get("id"), metadata.get("time"),
            metadata.get("from"), metadata.get("to"),
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
