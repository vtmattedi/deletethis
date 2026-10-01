"""Is the compressor running?

A running compressor is the only thing in the room that puts sustained
energy this low, in 30-80 Hz. That is the whole test: one band at or
above a threshold.

It is deliberately independent of the fan. A compressor running implies
the fan is turning, but this detector does not say so and does not
suppress anything: "compressor on" and "fan on" are separate
observations, and what they mean together is the controller's business.

The threshold is a property of the installation, not of the rule. The
OFF-state level in this band depends on where the microphone sits and
what else is in the building, and it has moved by about 10 dB between
recording sessions here. Re-run the evaluation when the microphone moves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .arrays import scalarise

# Chosen from the reviewed events: at -48 dB, 34% of OFF windows read as
# compressor in the newer sessions; at -38 dB the compressor is still
# found in 95.5% of its windows. The first recordings sat nearer -48.
DEFAULT_COMPRESSOR_THRESHOLD = -38.0


@dataclass(frozen=True)
class CompressorConfig:
    threshold: float = DEFAULT_COMPRESSOR_THRESHOLD


def detect_compressor(
    features: Mapping[str, object],
    config: CompressorConfig,
):
    """True where the 30-80 Hz level is at or above the threshold."""
    return scalarise(np.asarray(features["30-80"]) >= config.threshold)
