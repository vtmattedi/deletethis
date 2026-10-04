"""The labelled audio the firmware is validated against (read-only).

Three sources, all from the PC tool tree:

* recordings/*.wav       steady labelled takes. The label is in the file
                         name: off / fan / compressor, plus an
                         interference tag (talking, 3d_printer).
* results/events         v1 transition events, reviewed per event
                         (actualFrom / actualTo, with interference).
* results/v2/events      v2 events: per-observation reviews, and beeps.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import watson_ref as ref  # noqa: F401  (puts tools/audio on sys.path)

import evaluation  # noqa: E402  (tools/audio)

RECORDINGS = ref.TOOLS / "recordings"
EVENTS_V1 = ref.TOOLS / "results" / "events"
EVENTS_V2 = ref.TOOLS / "results" / "v2" / "events"

WARMUP_SECONDS = 5.0   # median (0.5 s) + temporal history (2 s) fill


@dataclass
class Segment:
    """One labelled stretch of audio."""

    name: str
    group: str
    path: Path
    fan: np.ndarray | None          # per-window truth: 1 / 0 / -1 unscored
    compressor: np.ndarray | None
    interference: str               # clean / talking / printer / ...
    kind: str                       # recording | v1 | v2
    meta: dict | None = None


def _recording_label(stem: str):
    name = stem.lower()
    if "compressor" in name or "comp" in name.split("_")[-1]:
        state = "COMPRESSOR"
    elif "fan" in name:
        state = "FAN"
    else:
        state = "OFF"

    if "talking" in name:
        interference = "talking"
    elif "3d_printer" in name:
        interference = "printer"
    else:
        interference = "clean"

    return state, interference


def truth_columns(times: np.ndarray, state: str):
    scored = times >= WARMUP_SECONDS
    fan = np.where(scored, 1 if state in ("FAN", "COMPRESSOR") else 0, -1)
    comp = np.where(scored, 1 if state == "COMPRESSOR" else 0, -1)
    return fan.astype(np.int8), comp.astype(np.int8)


def recording_segments() -> list[tuple[str, str, str, Path]]:
    items = []
    for path in sorted(RECORDINGS.glob("*.wav")):
        state, interference = _recording_label(path.stem)
        items.append((path.stem, state, interference, path))
    return items


def v1_records() -> list:
    records = evaluation.load_events([EVENTS_V1])
    return [r for r in records if r.labelled]


def v2_records() -> list:
    return evaluation.load_events([EVENTS_V2])
