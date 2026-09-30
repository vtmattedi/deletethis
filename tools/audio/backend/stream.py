"""The one connection to the ESP32.

Nothing else in the backend opens a socket. The firmware serves a
single client, so a second connection here would not fail politely --
it would be refused, and whichever component lost the race would
simply see no audio.

Runs on its own thread because acstream is blocking. That thread does
the whole pipeline for each frame: extract, classify, record. All of
it is cheap compared with the 32 ms a frame represents, and keeping it
in one place means there is no ordering to get wrong.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from acstream import AudioStream, ProtocolError

from .classifier import ClassifierService
from .config import AppConfig
from .events import EventRecorder

RECONNECT_DELAY_SECONDS = 2.0


@dataclass
class StreamHealth:
    connected: bool = False
    sample_rate: int = 0
    frame_samples: int = 0

    lost_frames: int = 0
    device_dropped: int = 0
    resyncs: int = 0
    gaps: int = 0

    connections: int = 0
    last_error: str = ""
    connected_since: float = 0.0

    def to_api(self) -> dict:
        return {
            "connected": self.connected,
            "sampleRate": self.sample_rate,
            "frameSamples": self.frame_samples,
            "lostFrames": self.lost_frames,
            "deviceDropped": self.device_dropped,
            "resyncs": self.resyncs,
            "gaps": self.gaps,
            "connections": self.connections,
            "lastError": self.last_error,
            "uptimeSeconds": (
                round(time.monotonic() - self.connected_since, 1)
                if self.connected
                else 0.0
            ),
        }


class StreamService(threading.Thread):
    def __init__(self, config: AppConfig) -> None:
        super().__init__(daemon=True, name="stream")

        self.config = config
        self.stop_event = threading.Event()

        self.lock = threading.Lock()
        self.health = StreamHealth()

        # Built on the first connection, once the device has told us
        # its sample rate rather than us assuming one.
        self.classifier: ClassifierService | None = None
        self.events: EventRecorder | None = None

        self.ready = threading.Event()

        # Last published state, for naming transitions.
        self._previous_state: str | None = None

    # ------------------------------------------------------- loop

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                self._session()
            except (ProtocolError, OSError) as error:
                self._fail(str(error))
            except Exception as error:      # never kill the thread
                self._fail(f"{type(error).__name__}: {error}")

            if self.stop_event.wait(RECONNECT_DELAY_SECONDS):
                return

    def _session(self) -> None:
        stream = AudioStream(
            self.config.target,
            verbose=False,
            boot_wait=0.0,
        )

        stream.open()

        info = stream.info
        assert info is not None

        self._on_connected(info)

        missing = 0

        try:
            for frame in stream.frames():
                if self.stop_event.is_set():
                    return

                missing = self._check_gap(stream, info, missing)

                self._consume(frame.samples)

                self._update_counters(stream)
        finally:
            stream.close()

            with self.lock:
                self.health.connected = False

            print("stream: disconnected")

    # -------------------------------------------------- pipeline

    def _consume(self, samples) -> None:
        assert self.classifier is not None
        assert self.events is not None

        produced = self.classifier.push(samples)

        # Raw PCM, exactly as received, before anything touches it.
        written = self.events.push(samples)

        for metadata in written:
            print(
                f"event: wrote {metadata['id']} "
                f"({metadata['audio']['seconds']:.1f}s)"
            )

        for _features, decision in produced:
            if not decision.changed:
                continue

            self._on_transition(decision)

    def _on_transition(self, decision) -> None:
        assert self.classifier is not None
        assert self.events is not None

        previous = self._previous_state

        # The first publication is the classifier settling, not a
        # transition: nothing changed, we simply did not know yet.
        # Recording it would put a spurious event on disk every time
        # the backend starts, with a pre-roll of whatever silence
        # happened to be buffered.
        if previous is None:
            self._previous_state = decision.state

            print(f"state: settled on {decision.state}")

            return

        if previous == decision.state:
            return

        snapshot = self.classifier.current()

        identifier = self.events.start(
            from_state=previous,
            to_state=decision.state,
            stable_seconds=decision.stable_seconds,
            stream_time=snapshot.stream_time,
            features=snapshot.features,
            decision_window=self.classifier.decision_window(),
            classifier_config=self.classifier.config,
        )

        print(
            f"state: {previous} -> {decision.state}  (held "
            f"{decision.stable_seconds:.1f}s)  event {identifier}"
        )

        self._previous_state = decision.state

    # -------------------------------------------------- bookkeeping

    def _on_connected(self, info) -> None:
        with self.lock:
            self.health.connected = True
            self.health.sample_rate = info.sample_rate
            self.health.frame_samples = info.frame_samples
            self.health.connections += 1
            self.health.last_error = ""
            self.health.connected_since = time.monotonic()

        if self.classifier is None:
            self.classifier = ClassifierService(
                info.sample_rate,
                self.config.classifier,
            )

        if self.events is None:
            self.events = EventRecorder(
                info.sample_rate,
                self.config.events,
                self.config.events_dir,
            )

        # A reconnect is a discontinuity like any other.
        self.classifier.reset()
        self.events.discard_history()

        self.ready.set()

        print(
            f"stream: connected to {self.config.target} "
            f"({info.sample_rate} Hz, "
            f"{info.frame_samples} samples/frame)"
        )

    def _check_gap(self, stream, info, missing: int) -> int:
        now_missing = stream.lost_frames + stream.dropped_frames

        if now_missing <= missing:
            return missing

        lost = now_missing - missing

        assert self.classifier is not None
        assert self.events is not None

        self.classifier.reset(
            skip_samples=lost * info.frame_samples
        )
        self.events.discard_history()

        with self.lock:
            self.health.gaps += 1

        print(
            f"stream: gap of {lost} frame(s), "
            f"{lost * info.frame_samples / info.sample_rate * 1000:.0f}"
            " ms; history reset"
        )

        return now_missing

    def _update_counters(self, stream) -> None:
        with self.lock:
            self.health.lost_frames = stream.lost_frames
            self.health.device_dropped = stream.dropped_frames
            self.health.resyncs = stream.resyncs

    def _fail(self, message: str) -> None:
        with self.lock:
            self.health.connected = False
            self.health.last_error = message

        print(f"stream: {message}; retrying")

    # ------------------------------------------------------- API

    def status(self) -> dict:
        with self.lock:
            payload = {"stream": self.health.to_api()}

        if self.classifier is not None:
            payload.update(self.classifier.current().to_api())

        if self.events is not None:
            payload["events"] = self.events.status()

        return payload

    def stop(self) -> None:
        self.stop_event.set()
