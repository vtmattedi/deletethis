"""Float64 reference for the Watson firmware's feature and detector math.

The firmware (src/audio/) runs the PC v2 detectors on a 2048-point /
512-hop analysis instead of the PC's 1024 / 512. This module is that same
analysis written in numpy, built on the PC tools' own helpers (imported,
never modified), so it is:

* the source of the golden vectors the firmware is checked against,
* the place the compressor threshold was derived from, using the exact
  feature definition the firmware computes,
* a replay harness that drives the *PC* ObservationSmoother and
  BeepDetector with the firmware's features, i.e. what the firmware
  detector path is supposed to reproduce.

Nothing under tools/ is changed. It is imported read-only.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import get_window

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools" / "audio"

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from acstream import FULL_SCALE  # noqa: E402
from classifier.common import ObservationSmoother  # noqa: E402
from classifier.detectors import (  # noqa: E402
    BeepConfig,
    BeepDetector,
    FanConfig,
    fan_energy,
    fan_stationary,
)
from features import (  # noqa: E402
    DB_FLOOR,
    Features,
    TEMPORAL_FILL,
    beep_features,
    power_to_db,
    window_power,
)

SAMPLE_RATE = 16000
NFFT = 2048
HOP = 512
HOP_SECONDS = HOP / SAMPLE_RATE
WINDOW_SECONDS = NFFT / SAMPLE_RATE
HISTORY_SECONDS = 2.0

# Compressor guard / feature regions. [low, high) like every band here.
COMPRESSOR_BANDS = {
    "comp_lower": (30.0, 45.0),      # lower sideband
    "comp_shoulder": (45.0, 52.0),   # diagnostic
    "comp_primary": (52.0, 65.0),    # v0 feature
    "comp_upper": (65.0, 80.0),      # upper sideband
    "comp_legacy": (30.0, 80.0),     # the old PC feature, for comparison
}
FAN_BANDS = {"500-1k": (500.0, 1000.0), "1k-2k": (1000.0, 2000.0)}

BEEP_TONE_HZ = (4050.0, 4180.0)
BEEP_LOWER_HZ = (3850.0, 4000.0)
BEEP_UPPER_HZ = (4230.0, 4380.0)
BEEP_SEARCH_HZ = (3850.0, 4380.0)

# The PC BeepDetector models a tone's footprint in time with one
# "window" length, used for the duration correction and the refractory.
# Its calibration (and the reviewed beeps) is for the PC's 64 ms window.
# A 128 ms Hamming window flags only about that much more of a tone (its
# outer quarters carry almost no weight), so the firmware keeps the PC's
# 64 ms footprint rather than the 128 ms analysis length: with it, all 69
# PC-detected beeps in the reviewed beep events are reproduced and bursts
# of beeps 250 ms apart stay separate. With the true 128 ms they
# under-read as too_short. See validation/calibrate_beep.py.
BEEP_FOOTPRINT_SECONDS = 0.064

# Firmware default for acoustic:compressor:threshold_db lives in
# derive_compressor.py's output and src/audio/DetectorConfig.h.


def band_mask(freqs: np.ndarray, edges: tuple[float, float]) -> np.ndarray:
    return (freqs >= edges[0]) & (freqs < edges[1])


class RefExtractor:
    """Firmware-equivalent analysis: 2048 Hamming, hop 512, float64."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        nfft: int = NFFT,
        hop: int = HOP,
        history_seconds: float = HISTORY_SECONDS,
    ) -> None:
        self.sample_rate = sample_rate
        self.nfft = nfft
        self.hop = hop
        self.window = get_window("hamming", nfft)
        self.freqs = np.fft.rfftfreq(nfft, 1.0 / sample_rate)
        self.bin_hz = sample_rate / nfft

        self.masks = {
            name: band_mask(self.freqs, edges)
            for name, edges in {**COMPRESSOR_BANDS, **FAN_BANDS}.items()
        }
        self.tone = band_mask(self.freqs, BEEP_TONE_HZ)
        self.lower = band_mask(self.freqs, BEEP_LOWER_HZ)
        self.upper = band_mask(self.freqs, BEEP_UPPER_HZ)
        self.search = np.where(band_mask(self.freqs, BEEP_SEARCH_HZ))[0]

        # round() on purpose: the PC extractor sizes its history this way.
        self.history_windows = max(
            1, round(history_seconds * sample_rate / hop)
        )
        self.history: list[float] = []
        self.buffer = np.zeros(0)
        self.samples_seen = 0

    def reset(self, skip_samples: int = 0) -> None:
        self.samples_seen += self.buffer.size + skip_samples
        self.buffer = np.zeros(0)
        self.history.clear()

    def push(self, samples: np.ndarray) -> list[Features]:
        block = samples.astype(np.float64) / FULL_SCALE
        self.buffer = np.concatenate((self.buffer, block))
        produced = []

        while self.buffer.size >= self.nfft:
            produced.append(self.analyse(self.buffer[: self.nfft]))
            self.buffer = self.buffer[self.hop:]
            self.samples_seen += self.hop

        return produced

    def analyse(self, block: np.ndarray) -> Features:
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

        diagnostics = beep_features(
            power, self.freqs, self.tone, self.lower, self.upper,
            self.search, self.bin_hz,
        )

        self.history.append(bands["1k-2k"])
        del self.history[: -self.history_windows]

        diagnostics[TEMPORAL_FILL] = (
            len(self.history) * self.hop / self.sample_rate
        )
        diagnostics["1k-2k_std"] = float(np.std(self.history))

        return Features(
            time=self.samples_seen / self.sample_rate,
            rms_db=rms_db,
            bands=bands,
            diagnostics=diagnostics,
        )


# ---------------------------------------------------------------------
# Detector rules over the firmware feature set
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Observed:
    fan_detected: bool
    compressor_detected: bool


@dataclass(frozen=True)
class FirmwareRules:
    """The firmware's stateless detectors, written over median features."""

    compressor_threshold: float
    sidebands_enable: bool = False
    sideband_lower_margin: float = 0.0
    sideband_upper_margin: float = 0.0
    fan: FanConfig = FanConfig()
    beep: BeepConfig = BeepConfig()

    def compressor(self, values) -> bool:
        primary = values["comp_primary"]
        seen = primary >= self.compressor_threshold

        if self.sidebands_enable:
            seen = (
                seen
                and primary - values["comp_lower"]
                >= self.sideband_lower_margin
                and primary - values["comp_upper"]
                >= self.sideband_upper_margin
            )

        return bool(seen)

    def observe(self, values) -> Observed:
        return Observed(
            fan_detected=bool(
                fan_energy(values, self.fan)
                and fan_stationary(values, self.fan)
            ),
            compressor_detected=self.compressor(values),
        )


@dataclass
class Replay:
    times: list[float] = field(default_factory=list)
    fan_candidate: list[bool] = field(default_factory=list)
    fan: list[bool | None] = field(default_factory=list)
    compressor_candidate: list[bool] = field(default_factory=list)
    compressor: list[bool | None] = field(default_factory=list)
    beeps: list = field(default_factory=list)
    rejected: dict = field(default_factory=dict)


def load_wav(path: Path) -> np.ndarray:
    """A recording as the int32 24-bit-domain samples the device sends."""
    rate, raw = wavfile.read(path)

    if rate != SAMPLE_RATE:
        raise ValueError(f"{path.name}: {rate} Hz, expected {SAMPLE_RATE}")

    if raw.ndim > 1:
        raw = raw[:, 0]

    # Same normalisation as the PC tools (analyze.to_float, then
    # * FULL_SCALE): WAVs are written as full-scale int32, so the
    # device's right-aligned 24-bit sample is the file value / 256.
    if raw.dtype == np.int32:
        return raw.astype(np.float64) / 256.0

    if raw.dtype == np.int16:
        return raw.astype(np.float64) * 256.0

    return raw.astype(np.float64) * FULL_SCALE


def replay(
    samples: np.ndarray,
    rules: FirmwareRules,
    median_seconds: float = 0.5,
    hold_seconds: float = 2.0,
) -> Replay:
    """Drive PC ObservationSmoother + BeepDetector with firmware features.

    Fed in 512-sample frames, as the device sends them.
    """
    extractor = RefExtractor()
    smoother = ObservationSmoother(
        rules,
        window_rate=SAMPLE_RATE / HOP,
        median_seconds=median_seconds,
        hold_seconds=hold_seconds,
        history_seconds=HISTORY_SECONDS,
    )
    beep = BeepDetector(
        rules.beep, hop_seconds=HOP_SECONDS,
        window_seconds=BEEP_FOOTPRINT_SECONDS,
    )
    out = Replay()

    for start in range(0, samples.size - HOP + 1, HOP):
        for window in extractor.push(samples[start: start + HOP]):
            decision = smoother.update(window)
            out.times.append(window.time)
            out.fan_candidate.append(decision.fan_candidate)
            out.fan.append(decision.fan_detected)
            out.compressor_candidate.append(decision.compressor_candidate)
            out.compressor.append(decision.compressor_detected)
            out.beeps.extend(beep.update(window))

    out.beeps.extend(beep.flush())
    out.rejected = dict(beep.rejected)
    return out
