"""Did the air conditioner just beep?

A beep is an *event*, not a state. The unit acknowledges a remote command
with a short pure tone, and what Watson reports is "a tone happened at
this moment", never "the AC is on" and never "the command worked". The
controller owns that interpretation: it knows what it sent and when.

What a beep looks like in the recorded events:

    frequency   4118-4123 Hz, tight (parabolic-interpolated peak)
    duration    64-160 ms as seen through 64 ms windows
    contrast    15.8-29.5 dB above the spectrum on either side
    level       -75 to -48 dBFS: varies by tens of dB with distance

Level varies far more than contrast, which is why contrast decides. A
door slam or a voice can be much louder than a beep and still have no
narrow peak at all.

Two contrast thresholds, because a beep has an envelope
-------------------------------------------------------
A beep's edge windows are weaker than its middle: the tone only partly
fills them. One threshold therefore either misses the edges (and
shortens the beep until it looks too brief) or is set so low that noise
passes. So the *peak* contrast decides whether a run is a tone at all
(``min_contrast_db``), and a lower *edge* threshold (``edge_contrast_db``)
only decides how long it lasted. A run that never reaches the peak
threshold is rejected as weak, however long it is.

Smoothing would destroy it
--------------------------
The fan and compressor detectors work on features median-smoothed over
half a second. A beep lasts about a quarter of that, so the same smoothing
would erase it. This detector therefore reads each window's *raw* beep
features, and keeps its own small state machine in time:

    IDLE --valid window--> TONE --enough invalid windows--> finish
                            |  valid windows extend the run
                            +--> too long: remember it, emit nothing

A fixed pitch is part of the definition
--------------------------------------
The unit's tone comes from a piezo and does not move: across a beep's
windows the peak stays within a few hertz, and even its weak edge
windows span under 30 Hz. A loud impact ringing through a resonance can
also produce a narrow peak with high contrast, but its pitch wanders
over a hundred hertz as it decays. ``max_peak_spread_hz`` rejects runs
whose peak moves more than that. (Frequency alone cannot separate them:
a genuine but weak beep in a noisy room estimated at 4100 Hz is only a
few hertz from the impact that triggered the one false alarm found.)

A run is judged when it ends: too weak, too short or too long are
rejected, anything else becomes one ``BeepEvent``. A dropout of a single
window inside a tone is bridged, since a beep's envelope wobbles, and a
short refractory period after an event stops the decaying tail of one
beep from being counted as another.

Timing resolution is one hop (32 ms at the default settings) and the
windows are twice that long, so start and end are good to about a window.

Duration is corrected for that. A window flags a tone if even part of it
falls inside, so counting flagged windows overstates a beep by about one
window length: 150 ms reads as 192 ms, and a 30 ms blip as 64 ms, which
would slip past a 60 ms minimum. The estimate used here is the distance
between the *centres* of the first and last flagged window, which for
the 50%-overlap windows used here is simply last start minus first start.
It is good to about +-30 ms.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # keep this layer free of the DSP imports
    from features import Features

# Provisional until tuned against more beeps: set from the 26 tone-like
# runs in the recorded events. Every real beep had contrast >= 15.8 dB
# and lasted >= 64 ms; the look-alikes were single 32 ms windows at
# 12-13.5 dB, plus one 128 ms run at 13.4 dB that was loud broadband
# noise rather than a tone. The margin on contrast is about 2 dB.
DEFAULT_MIN_CONTRAST_DB = 15.0
DEFAULT_EDGE_CONTRAST_DB = 10.0
DEFAULT_MIN_LEVEL_DB = -80.0
DEFAULT_MIN_MS = 60.0
DEFAULT_MAX_MS = 300.0
DEFAULT_MIN_HZ = 4050.0
DEFAULT_MAX_HZ = 4180.0
DEFAULT_MAX_PEAK_SPREAD_HZ = 50.0
DEFAULT_MAX_GAP_MS = 40.0
DEFAULT_REFRACTORY_MS = 80.0


@dataclass(frozen=True)
class BeepConfig:
    min_contrast_db: float = DEFAULT_MIN_CONTRAST_DB
    edge_contrast_db: float = DEFAULT_EDGE_CONTRAST_DB
    min_level_db: float = DEFAULT_MIN_LEVEL_DB
    min_ms: float = DEFAULT_MIN_MS
    max_ms: float = DEFAULT_MAX_MS
    min_hz: float = DEFAULT_MIN_HZ
    max_hz: float = DEFAULT_MAX_HZ
    max_peak_spread_hz: float = DEFAULT_MAX_PEAK_SPREAD_HZ
    max_gap_ms: float = DEFAULT_MAX_GAP_MS
    refractory_ms: float = DEFAULT_REFRACTORY_MS

    def __post_init__(self) -> None:
        if self.edge_contrast_db > self.min_contrast_db:
            raise ValueError(
                "the edge contrast cannot exceed the peak contrast"
            )
        if not 0 < self.min_ms < self.max_ms:
            raise ValueError("beep durations need 0 < min_ms < max_ms")
        if not self.min_hz < self.max_hz:
            raise ValueError("beep frequencies need min_hz < max_hz")


@dataclass(frozen=True)
class BeepEvent:
    """One detected beep. Times are seconds on the stream clock."""

    stream_time: float      # the tone's end, on the stream clock
    start_time: float       # estimated onset
    end_time: float
    duration_ms: float
    peak_hz: float
    contrast_db: float
    level_db: float

    def to_api(self) -> dict:
        return {
            "streamSeconds": round(self.stream_time, 3),
            "startSeconds": round(self.start_time, 3),
            "endSeconds": round(self.end_time, 3),
            "durationMs": round(self.duration_ms),
            "peakHz": round(self.peak_hz, 1),
            "contrastDb": round(self.contrast_db, 1),
            "levelDb": round(self.level_db, 1),
        }


@dataclass
class _Run:
    """A tone being followed."""

    start: float
    last: float
    peaks: list[float] = field(default_factory=list)
    contrast: float = -math.inf
    level: float = -math.inf
    missing: int = 0
    windows: int = 0
    too_long: bool = False


class BeepDetector:
    """Turns a stream of windows into beep events.

    ``hop_seconds`` is the spacing of windows (``extractor.hop /
    sample_rate``); it is what turns a count of windows into time.
    """

    def __init__(
        self,
        config: BeepConfig | None = None,
        hop_seconds: float = 512 / 16000,
        window_seconds: float | None = None,
    ) -> None:
        self.config = config or BeepConfig()
        self.hop = hop_seconds
        # The analysis window; 50% overlap unless told otherwise.
        self.window = window_seconds or 2.0 * hop_seconds

        # Invalid windows tolerated inside a tone before it is over.
        self._gap_windows = max(
            0, round(self.config.max_gap_ms / 1000.0 / hop_seconds)
        )

        self.count = 0
        self.rejected: Counter[str] = Counter()
        self.last: BeepEvent | None = None

        self._run: _Run | None = None
        self._blocked_until = -math.inf

    # ----------------------------------------------------------- control

    def reset(self) -> None:
        """Forget a tone in progress, e.g. across a hole in the audio.

        A tone seen on both sides of a gap is not one tone we can
        measure. Counters and the last event are kept: they describe
        what has happened, not what is happening.
        """
        self._run = None
        self._blocked_until = -math.inf

    def apply_config(self, config: BeepConfig) -> None:
        """Change the settings. What has happened is kept; a tone in
        progress is dropped, since it was being judged by the old ones."""
        count, rejected, last = self.count, self.rejected, self.last

        self.__init__(config, self.hop, self.window)

        self.count, self.rejected, self.last = count, rejected, last

    def stats(self) -> dict:
        return {
            "count": self.count,
            "rejected": dict(self.rejected),
            "last": self.last.to_api() if self.last else None,
        }

    # -------------------------------------------------------------- feed

    def valid(self, diagnostics) -> bool:
        """Could this one window be part of a beep tone?

        Deliberately generous (the edge contrast): whether the run is a
        tone is decided on its peak, when it ends.
        """
        config = self.config

        return bool(
            diagnostics["beep_contrast_db"] >= config.edge_contrast_db
            and diagnostics["beep_band_power"] >= config.min_level_db
            and config.min_hz <= diagnostics["beep_peak_hz"] <= config.max_hz
        )

    def update(self, features: "Features") -> list[BeepEvent]:
        """Consume one window; return any beeps that just finished."""
        time = features.time
        valid = self.valid(features.diagnostics)
        run = self._run

        if run is None:
            if valid and time >= self._blocked_until:
                self._run = _Run(start=time, last=time)
                self._extend(self._run, features.diagnostics, time)

            return []

        if valid:
            self._extend(run, features.diagnostics, time)

            return []

        run.missing += 1

        if run.missing <= self._gap_windows:
            return []

        return self._finish()

    def flush(self) -> list[BeepEvent]:
        """End of the audio: whatever tone is in progress is over."""
        return self._finish() if self._run is not None else []

    # ----------------------------------------------------------- internals

    def _extend(self, run: _Run, diagnostics, time: float) -> None:
        run.last = time
        run.missing = 0
        run.windows += 1
        run.peaks.append(float(diagnostics["beep_peak_hz"]))
        run.contrast = max(run.contrast, float(diagnostics["beep_contrast_db"]))
        run.level = max(run.level, float(diagnostics["beep_band_power"]))

        if self._duration_ms(run) > self.config.max_ms:
            run.too_long = True

    def _duration_ms(self, run: _Run) -> float:
        """The tone's length, corrected for the window overhang.

        The flagged windows cover the tone plus up to one window length
        of overhang, which is the hop's worth of time each window adds
        beyond the first, less the window itself.
        """
        span = run.last - run.start + self.hop - (self.window - self.hop)

        return max(0.0, span) * 1000.0

    def _finish(self) -> list[BeepEvent]:
        run = self._run
        assert run is not None
        self._run = None

        duration = self._duration_ms(run)

        # The tone sits in the middle of the flagged windows.
        middle = (run.start + run.last + self.window) / 2.0
        start = middle - duration / 2000.0
        end = middle + duration / 2000.0

        self._blocked_until = (
            run.last + self.window + self.config.refractory_ms / 1000.0
        )

        if run.too_long:
            self.rejected["too_long"] += 1
            return []

        if run.contrast < self.config.min_contrast_db:
            # Tone-like for a while, but never stood out enough.
            self.rejected["weak"] += 1
            return []

        if duration < self.config.min_ms:
            self.rejected["too_short"] += 1
            return []

        if max(run.peaks) - min(run.peaks) > self.config.max_peak_spread_hz:
            # A narrow peak whose pitch wanders: not a piezo tone.
            self.rejected["unstable_pitch"] += 1
            return []

        event = BeepEvent(
            stream_time=end,
            start_time=start,
            end_time=end,
            duration_ms=duration,
            peak_hz=float(np.median(run.peaks)),
            contrast_db=run.contrast,
            level_db=run.level,
        )

        self.count += 1
        self.last = event

        return [event]
