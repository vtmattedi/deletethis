"""Shared spectral and temporal audio feature extraction.

Both live classification and offline analysis import this module.  The
classifier still consumes only its original three band levels; the
additional values are diagnostics and dataset columns.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.signal import get_window

from acstream import FULL_SCALE

DB_FLOOR = -120.0
DEFAULT_NFFT = 1024
DEFAULT_OVERLAP = 0.5
DEFAULT_TEMPORAL_SECONDS = 2.0

PEAK_MIN_HZ = 100.0
PEAK_MAX_HZ = 4000.0
SPECTRAL_MIN_HZ = 30.0
SPECTRAL_MAX_HZ = 4000.0

# 1100-1200 is retained for compatibility with existing analysis output.
BANDS: list[tuple[str, float, float]] = [
    ("30-80", 30.0, 80.0),
    ("80-200", 80.0, 200.0),
    ("200-500", 200.0, 500.0),
    ("500-1k", 500.0, 1000.0),
    ("1k-2k", 1000.0, 2000.0),
    ("2k-4k", 2000.0, 4000.0),
    ("1100-1200", 1100.0, 1200.0),
    ("200-1200", 200.0, 1200.0),
    ("1500-4000", 1500.0, 4000.0),
]
BAND_NAMES = [name for name, _, _ in BANDS]

RULE_BANDS = ["30-80", "500-1k", "1k-2k"]
DIAGNOSTIC_BANDS = [name for name in BAND_NAMES if name not in RULE_BANDS]

SPECTRAL_FEATURES = [
    "peak_hz",
    "spectral_centroid",
    "spectral_flatness",
    "spectral_crest",
    "spectral_flux",
]

RELATIVE_FEATURES = [
    "500-1k_minus_rms",
    "1k-2k_minus_rms",
    "2k-4k_minus_rms",
    "500-1k_minus_200-500",
    "1k-2k_minus_500-1k",
    "2k-4k_minus_500-1k",
    "1500-4000_minus_200-1200",
]

TEMPORAL_FEATURES = [
    "rms_std",
    "500-1k_std",
    "1k-2k_std",
    "2k-4k_std",
    "spectral_flux_median",
    "spectral_flux_std",
]

FEATURE_NAMES = (
    ["rms"]
    + BAND_NAMES
    + SPECTRAL_FEATURES
    + RELATIVE_FEATURES
    + TEMPORAL_FEATURES
)

# Bookkeeping rather than a measurement, so deliberately NOT part of
# FEATURE_NAMES (which fixes the columns of every CSV and event JSON).
# It is how many seconds of history the *_std values above are
# computed over. Right after a start or an audio gap that is a handful
# of windows, and a standard deviation of two samples is near zero:
# every signal looks perfectly stationary. A rule that trusts those
# values has to be able to tell.
TEMPORAL_FILL = "temporal_seconds"


def power_to_db(power: np.ndarray | float) -> np.ndarray | float:
    """Mean-square power to dBFS."""
    amplitude = np.sqrt(np.maximum(power, 0.0))
    return np.maximum(
        20.0 * np.log10(np.maximum(amplitude, 1e-12)),
        DB_FLOOR,
    )


def window_power(block: np.ndarray, window: np.ndarray) -> np.ndarray:
    """Return one-sided per-bin spectrum power for one window."""
    centred = block - block.mean()
    spectrum = np.fft.rfft(centred * window)
    power = (
        spectrum.real ** 2 + spectrum.imag ** 2
    ) / window.sum() ** 2

    if block.size % 2 == 0:
        power[1:-1] *= 2.0
    else:
        power[1:] *= 2.0
    return power


@dataclass
class Features:
    """All values derived from one overlapping analysis window."""

    time: float
    rms_db: float
    bands: dict[str, float]
    diagnostics: dict[str, float]

    @cached_property
    def values(self) -> dict[str, float]:
        # Cached: the smoother reads this for every window in its
        # median span, so building it once matters.
        return {"rms": self.rms_db, **self.bands, **self.diagnostics}


class FeatureExtractor:
    """Turn raw 24-bit-domain PCM frames into shared features."""

    def __init__(
        self,
        sample_rate: int,
        nfft: int = DEFAULT_NFFT,
        overlap: float = DEFAULT_OVERLAP,
        temporal_seconds: float = DEFAULT_TEMPORAL_SECONDS,
    ) -> None:
        self.sample_rate = sample_rate
        self.nfft = nfft
        self.hop = nfft - int(nfft * overlap)
        self.window = get_window("hamming", nfft)
        self.freqs = np.fft.rfftfreq(nfft, 1.0 / sample_rate)
        self.masks = {
            name: (self.freqs >= low) & (self.freqs < high)
            for name, low, high in BANDS
        }
        self.peak_mask = (
            (self.freqs >= PEAK_MIN_HZ)
            & (self.freqs <= PEAK_MAX_HZ)
        )
        self.spectral_mask = (
            (self.freqs >= SPECTRAL_MIN_HZ)
            & (self.freqs <= SPECTRAL_MAX_HZ)
        )
        history_windows = max(
            1, round(temporal_seconds * sample_rate / self.hop)
        )
        self.temporal_history: deque[dict[str, float]] = deque(
            maxlen=history_windows
        )
        self.previous_spectrum: np.ndarray | None = None
        self.buffer = np.zeros(0, dtype=np.float64)
        self.samples_seen = 0

    def reset(self, skip_samples: int = 0) -> None:
        """Drop all state that cannot safely span an audio gap."""
        self.samples_seen += self.buffer.size + skip_samples
        self.buffer = np.zeros(0, dtype=np.float64)
        self.temporal_history.clear()
        self.previous_spectrum = None

    def push(self, samples: np.ndarray) -> list[Features]:
        block = samples.astype(np.float64) / FULL_SCALE
        self.buffer = (
            block if self.buffer.size == 0
            else np.concatenate((self.buffer, block))
        )
        produced: list[Features] = []
        while self.buffer.size >= self.nfft:
            produced.append(self._analyse(self.buffer[: self.nfft]))
            self.buffer = self.buffer[self.hop :]
            self.samples_seen += self.hop
        return produced

    def _analyse(self, block: np.ndarray) -> Features:
        centred = block - block.mean()
        rms_db = float(power_to_db(float(np.mean(centred * centred))))
        power = window_power(block, self.window)

        bands = {
            name: (
                float(power_to_db(float(power[mask].sum())))
                if mask.any() else DB_FLOOR
            )
            for name, mask in self.masks.items()
        }

        focus_power = power[self.spectral_mask]
        focus_freqs = self.freqs[self.spectral_mask]
        total = float(focus_power.sum())
        epsilon = np.finfo(np.float64).tiny

        centroid = (
            float(np.sum(focus_freqs * focus_power) / total)
            if total > epsilon else 0.0
        )
        flatness = float(
            np.exp(np.mean(np.log(focus_power + epsilon)))
            / (np.mean(focus_power) + epsilon)
        ) if focus_power.size and total > epsilon else 0.0
        crest = float(
            np.max(focus_power) / (np.mean(focus_power) + epsilon)
        ) if focus_power.size and total > epsilon else 0.0

        magnitude = np.sqrt(focus_power)
        normalized = magnitude / (float(magnitude.sum()) + epsilon)
        flux = (
            float(np.sqrt(np.sum((normalized - self.previous_spectrum) ** 2)))
            if self.previous_spectrum is not None else 0.0
        )
        self.previous_spectrum = normalized

        peak_power = power[self.peak_mask]
        peak_freqs = self.freqs[self.peak_mask]
        peak_hz = (
            float(peak_freqs[int(np.argmax(peak_power))])
            if peak_freqs.size else 0.0
        )

        diagnostics = {
            "peak_hz": peak_hz,
            "spectral_centroid": centroid,
            "spectral_flatness": flatness,
            "spectral_crest": crest,
            "spectral_flux": flux,
            "500-1k_minus_rms": bands["500-1k"] - rms_db,
            "1k-2k_minus_rms": bands["1k-2k"] - rms_db,
            "2k-4k_minus_rms": bands["2k-4k"] - rms_db,
            "500-1k_minus_200-500": (
                bands["500-1k"] - bands["200-500"]
            ),
            "1k-2k_minus_500-1k": bands["1k-2k"] - bands["500-1k"],
            "2k-4k_minus_500-1k": bands["2k-4k"] - bands["500-1k"],
            "1500-4000_minus_200-1200": (
                bands["1500-4000"] - bands["200-1200"]
            ),
        }

        current = {"rms": rms_db, **bands, **diagnostics}
        self.temporal_history.append(current)
        history = list(self.temporal_history)

        def spread(name: str) -> float:
            return float(np.std([item[name] for item in history]))

        flux_values = [item["spectral_flux"] for item in history]
        diagnostics.update({
            TEMPORAL_FILL: len(history) * self.hop / self.sample_rate,
            "rms_std": spread("rms"),
            "500-1k_std": spread("500-1k"),
            "1k-2k_std": spread("1k-2k"),
            "2k-4k_std": spread("2k-4k"),
            "spectral_flux_median": float(np.median(flux_values)),
            "spectral_flux_std": float(np.std(flux_values)),
        })

        return Features(
            time=self.samples_seen / self.sample_rate,
            rms_db=rms_db,
            bands=bands,
            diagnostics=diagnostics,
        )
