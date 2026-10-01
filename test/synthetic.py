"""Deterministic stand-ins for the situations the classifier must tell apart.

The real recordings are private audio of a home, and ``*.wav`` is
gitignored, so tests cannot ship them. These are shaped to the *features*
that matter instead -- band level and how steadily it sits there -- and
are seeded so every run is identical.

Levels are chosen against the classifier's actual defaults. White noise
of standard deviation s puts roughly 10*log10(s**2 * width/8000) + 1.3 dB
into a band ``width`` Hz wide, so:

    s = 0.0043   -> 1k-2k near -55 dB, 500-1k near -58 dB   (fan energy)
    s = 0.0003   -> 1k-2k near -78 dB                       (quiet room)

The point of each fixture is its *temporal* character:

    steady fan      constant level           -> stationary
    speech          bursts with pauses       -> level swings by 20+ dB
    television      level wandering about    -> swings of several dB
    alarm           gated broadband bursts   -> on/off
    impulsive       clicks over a quiet bed  -> spikes
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from scipy.signal import butter, lfilter

SAMPLE_RATE = 16000
SECONDS = 24
FULL_SCALE_24 = float(1 << 23)

FIXTURES = (
    "off_clean",
    "fan_clean",
    "compressor",
    "off_talking",
    "off_tv",
    "off_alarm",
    "off_impulsive",
)

# What a correct classifier publishes once warmed up.
EXPECTED = {
    "off_clean": "OFF",
    "fan_clean": "FAN",
    "compressor": "COMPRESSOR",
    "off_talking": "OFF",
    "off_tv": "OFF",
    "off_alarm": "OFF",
    "off_impulsive": "OFF",
}


def _band(rng, n, low, high, sigma):
    """Band-limited noise with the given overall standard deviation."""
    b, a = butter(4, [low / (SAMPLE_RATE / 2), high / (SAMPLE_RATE / 2)],
                  btype="band")
    x = lfilter(b, a, rng.standard_normal(n))

    return sigma * x / np.std(x)


def _envelope(rng, n, pieces):
    """Piecewise-constant gain: a list of (seconds, gain) repeated."""
    out = np.empty(n)
    position = 0
    index = 0

    while position < n:
        seconds, gain = pieces[index % len(pieces)]
        length = int(seconds * SAMPLE_RATE)
        out[position : position + length] = gain
        position += length
        index += 1

    return out


def make(name: str, seconds: int = SECONDS, seed: int = 7) -> np.ndarray:
    """Float samples in [-1, 1) for one fixture."""
    rng = np.random.default_rng(seed + FIXTURES.index(name))
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE

    quiet = 0.0003 * rng.standard_normal(n)

    if name == "off_clean":
        return quiet

    if name == "fan_clean":
        return 0.0043 * rng.standard_normal(n) + quiet

    if name == "compressor":
        return (
            0.0043 * rng.standard_normal(n)
            + _band(rng, n, 30, 80, 0.04)
            + quiet
        )

    if name == "off_talking":
        # Syllable bursts and pauses over a quiet room.
        gain = _envelope(rng, n, [(0.7, 1.0), (0.5, 0.0), (1.1, 1.0),
                                  (0.9, 0.0), (0.4, 1.0), (1.3, 0.0)])
        speech = _band(rng, n, 250, 3400, 0.012) * gain

        return speech + quiet

    if name == "off_tv":
        # Programme material: the level wanders by several dB every
        # fraction of a second, with no pauses.
        gain = _envelope(
            rng, n,
            [(0.35, g) for g in (0.15, 2.4, 0.4, 3.0, 0.25, 1.8, 0.2, 2.6)],
        )

        return _band(rng, n, 150, 6000, 0.004) * gain + quiet

    if name == "off_alarm":
        # A warbling alarm is broadband, not a pure tone, so it excites
        # both fan bands; it is on 0.8 s and off 0.6 s.
        gate = _envelope(rng, n, [(0.8, 1.0), (0.6, 0.0)])

        return _band(rng, n, 400, 2500, 0.012) * gate + quiet

    if name == "off_impulsive":
        # Clicks and thumps over the quiet room.
        x = quiet.copy()
        for centre in rng.integers(0, n - 800, size=int(seconds * 1.5)):
            x[centre : centre + 600] += (
                rng.standard_normal(600) * 0.05 * np.hanning(600)
            )

        return x

    raise KeyError(name)


def to_counts(samples: np.ndarray) -> np.ndarray:
    """Float samples to the extractor's raw 24-bit-domain counts."""
    return np.clip(samples * FULL_SCALE_24, -(1 << 23), (1 << 23) - 1)


def write_wav(path: Path, samples: np.ndarray) -> None:
    """The format event WAVs use: 32-bit PCM with the 24 bits on top."""
    counts = np.clip(samples * FULL_SCALE_24, -(1 << 23), (1 << 23) - 1)
    data = (counts.astype(np.int64) << 8).astype(np.int32)

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(4)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(data.tobytes())


# ---------------------------------------------------------------------
# Beeps
# ---------------------------------------------------------------------

# The unit's acknowledgement tone as recorded: a pure tone near 4.12 kHz
# of about 150 ms, around -56 dBFS in its band, over whatever the room
# is doing. 20*log10(A/sqrt(2)) = -56 gives A of about 0.0036.
BEEP_HZ = 4120.0
BEEP_AMPLITUDE = 0.0036


def tone(
    n: int,
    start: float,
    duration: float,
    hz: float = BEEP_HZ,
    amplitude: float = BEEP_AMPLITUDE,
    attack: float = 0.004,
    decay: float = 0.004,
) -> np.ndarray:
    """A tone burst with short ramps so it does not click."""
    out = np.zeros(n)
    first = int(start * SAMPLE_RATE)
    length = int(duration * SAMPLE_RATE)
    last = min(n, first + length)

    if last <= first:
        return out

    t = np.arange(last - first) / SAMPLE_RATE
    envelope = np.ones(last - first)

    ramp_in = max(1, int(attack * SAMPLE_RATE))
    ramp_out = max(1, int(decay * SAMPLE_RATE))
    envelope[:ramp_in] = np.linspace(0, 1, ramp_in)
    envelope[-ramp_out:] = np.minimum(
        envelope[-ramp_out:], np.linspace(1, 0, ramp_out)
    )

    out[first:last] = amplitude * envelope * np.sin(2 * np.pi * hz * t)

    return out


def with_noise(signal: np.ndarray, sigma: float = 0.0004, seed: int = 3):
    """The room under the signal."""
    rng = np.random.default_rng(seed)

    return signal + sigma * rng.standard_normal(signal.size)
