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
import threading
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
        sample_rate: int,
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

    # ------------------------------------------------------ config

    def apply_config(self, config: EventConfig) -> None:
        with self.lock:
            self.config = config
            self._trim()

    @property
    def pre_samples(self) -> int:
        return int(self.config.pre_seconds * self.sample_rate)

    # --------------------------------------------------------- PCM

    def push(self, samples: np.ndarray) -> list[dict]:
        """Add a frame. Returns metadata for any event just completed."""
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
    ) -> str:
        now = datetime.now()

        identifier = (
            f"{now:%Y-%m-%d_%H%M%S}_"
            f"{from_state or 'NONE'}_to_{to_state}"
        )

        with self.lock:
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
                name: round(value, 1)
                for name, value in event.features.items()
            },
            "decisionWindow": event.decision_window,
            "classifierConfig": event.classifier_config,
        }

        json_path.write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )

        with self.lock:
            self.written.append(metadata)

        return metadata

    def recent(self, limit: int = 50) -> list[dict]:
        with self.lock:
            return list(reversed(self.written[-limit:]))

    def status(self) -> dict:
        with self.lock:
            return {
                "buffered": round(
                    self.history_samples / self.sample_rate, 1
                ),
                "pending": len(self.pending),
                "written": len(self.written),
            }
