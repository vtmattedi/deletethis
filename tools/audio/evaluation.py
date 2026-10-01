"""Replay reviewed events through the live classifier and score them.

The question this answers is "would a different rule have called these
real situations correctly?", asked of audio a person has already
labelled. The inputs are the event WAVs the backend recorded and the
``review`` block a human filled in; nothing else is ground truth.

    event WAV --> FeatureExtractor --> rule --> Smoother (median, hold)
                                                    |
                          review.actualFrom/To  ----+--> score

Two ways of running the rule, one rule
--------------------------------------
``replay_exact`` feeds the real Smoother window by window. It is what
the live backend does, and every number reported as a result comes from
it.

``FastEvaluator`` exists only so a threshold *search* is affordable: it
has to try thousands of settings, and the Smoother is a Python loop.
It does not reimplement the decision. The v2 rule is a single function
(``classify_v2_codes``) written to accept scalars or arrays, so the
search executes the same lines the live path does. What it does
duplicate is the rolling median and the hold -- plumbing, not rules --
and ``verify_fast_path`` checks it against the Smoother window for
window, so a disagreement is a failing check rather than a quiet error.

Ground truth, and its limits
----------------------------
An event is 30 s of audio around a moment the classifier changed its
mind. The reviewer says what the state really was before (``actualFrom``)
and after (``actualTo``). Windows are scored only where that is
trustworthy:

* the first ``warmup`` seconds are skipped, because replay starts cold
  and neither the median nor the 2 s temporal history has filled;
* when the state really changed, a ``guard`` either side of the
  transition is skipped, since the audio changed somewhere inside it
  and no label says exactly when;
* ``UNKNOWN`` and unreviewed events carry no label and are counted but
  never scored.

Windows inside one event are near-copies of each other, and adjacent
events often overlap in time, so anything that fits or chooses a
parameter must split by *group* (events within ``group_gap`` seconds of
one another), never by window.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.io import wavfile

import features as features_module
from acstream import FULL_SCALE
from analyze import to_float
from classifier.common import STATES, Smoother
from classifier.v2 import ThresholdsV2, classify_v2_codes
from features import (
    BAND_NAMES,
    FEATURE_NAMES,
    TEMPORAL_FILL,
    FeatureExtractor,
    Features,
)

# Live frames are this size; windows are cut from them the same way.
FRAME_SAMPLES = 512

# Seconds of replay before scores count: the 0.5 s median plus the 2 s
# temporal history, with a little to spare.
DEFAULT_WARMUP_SECONDS = 5.0

# Seconds either side of a real transition that are not scored. A
# transition is *published* hold + median seconds after the audio
# actually changed (about 2.5 s at the defaults), so the guard has to
# clear that.
DEFAULT_GUARD_SECONDS = 4.0

# Events whose audio lies within this many seconds of each other share
# a group. Neighbours in time share a room, a noise source and often
# literal samples.
DEFAULT_GROUP_GAP_SECONDS = 120.0

LABELS = ("OFF", "FAN", "COMPRESSOR")
LABEL_CODE = {name: index for index, name in enumerate(LABELS)}
UNSCORED = -1
NO_STATE = 3                      # confusion column for "nothing published"

CACHE_VERSION = 1


# ---------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------


@dataclass
class EventRecord:
    """One saved event and what a human said about it."""

    id: str
    json_path: Path
    wav_path: Path
    metadata: dict

    classifier_version: str
    reviewed: bool
    actual_from: str | None
    actual_to: str | None
    interference: tuple[str, ...]

    pre_seconds: float
    post_seconds: float
    moment: datetime                 # the transition moment in the WAV

    @property
    def labelled(self) -> bool:
        """Has a usable ground truth."""
        return (
            self.reviewed
            and self.actual_from in LABEL_CODE
            and self.actual_to in LABEL_CODE
        )

    @property
    def interference_label(self) -> str:
        return "+".join(self.interference) if self.interference else "clean"

    @property
    def start(self) -> datetime:
        return self.moment - timedelta(seconds=self.pre_seconds)

    @property
    def end(self) -> datetime:
        return self.moment + timedelta(seconds=self.post_seconds)


def parse_event(json_path: Path) -> EventRecord | None:
    """Read one event's JSON; None if it is not a usable event."""
    try:
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    if not isinstance(metadata, dict) or "id" not in metadata:
        return None

    audio = metadata.get("audio") or {}
    wav = json_path.with_name(audio.get("file", f"{metadata['id']}.wav"))

    if not wav.is_file():
        return None

    review = metadata.get("review") or {}
    reviewed = review.get("status") == "reviewed"

    try:
        moment = datetime.strptime(metadata["id"][:17], "%Y-%m-%d_%H%M%S")
    except ValueError:
        return None

    return EventRecord(
        id=metadata["id"],
        json_path=json_path,
        wav_path=wav,
        metadata=metadata,
        # Events written before versions existed were classified by v1.
        classifier_version=metadata.get("classifierVersion", "v1"),
        reviewed=reviewed,
        actual_from=review.get("actualFrom") if reviewed else None,
        actual_to=review.get("actualTo") if reviewed else None,
        interference=tuple(sorted(review.get("interference") or [])),
        pre_seconds=float(audio.get("preSeconds", 0.0)),
        post_seconds=float(audio.get("postSeconds", 0.0)),
        moment=moment,
    )


def load_events(directories: Iterable[Path]) -> list[EventRecord]:
    """Every readable event under the given directories, oldest first."""
    records: dict[str, EventRecord] = {}

    for directory in directories:
        for path in sorted(Path(directory).glob("*.json")):
            record = parse_event(path)

            if record is not None:
                records.setdefault(record.id, record)

    return sorted(records.values(), key=lambda item: item.moment)


def dataset_counts(records: list[EventRecord]) -> dict:
    """What is in the dataset, including everything that is not scored."""
    labelled = [r for r in records if r.labelled]
    unknown = [
        r for r in records
        if r.reviewed and not r.labelled
    ]

    return {
        "events": len(records),
        "labelled": len(labelled),
        "reviewedUnknown": len(unknown),
        "unreviewed": sum(1 for r in records if not r.reviewed),
        "steady": sum(
            1 for r in labelled if r.actual_from == r.actual_to
        ),
        "transitions": sum(
            1 for r in labelled if r.actual_from != r.actual_to
        ),
        "byFinalState": {
            label: sum(1 for r in labelled if r.actual_to == label)
            for label in LABELS
        },
        "byInterference": _count(r.interference_label for r in labelled),
    }


def _count(items: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}

    for item in items:
        counts[item] = counts.get(item, 0) + 1

    return dict(sorted(counts.items()))


def group_events(
    records: list[EventRecord],
    gap_seconds: float = DEFAULT_GROUP_GAP_SECONDS,
) -> dict[str, int]:
    """Group events whose audio is within ``gap_seconds`` of each other.

    Returns event id -> group number. A chain of events each close to
    the next becomes one group, however long: if the noise source ran
    for ten minutes, every event in those ten minutes shares it.
    """
    groups: dict[str, int] = {}
    group = -1
    horizon: datetime | None = None

    for record in sorted(records, key=lambda item: item.start):
        if horizon is None or record.start > horizon + timedelta(
            seconds=gap_seconds
        ):
            group += 1
            horizon = record.end
        else:
            horizon = max(horizon, record.end)

        groups[record.id] = group

    return groups


def truth_labels(
    record: EventRecord,
    times: np.ndarray,
    warmup: float = DEFAULT_WARMUP_SECONDS,
    guard: float = DEFAULT_GUARD_SECONDS,
) -> np.ndarray:
    """Per-window label code, or UNSCORED where there is none to trust."""
    labels = np.full(times.shape, UNSCORED, dtype=np.int8)

    if not record.labelled:
        return labels

    start = LABEL_CODE[record.actual_from]
    end = LABEL_CODE[record.actual_to]
    moment = record.pre_seconds

    if start == end:
        labels[:] = end
    else:
        labels[times < moment - guard] = start
        labels[times > moment + guard] = end

    labels[times < warmup] = UNSCORED

    return labels


# ---------------------------------------------------------------------
# Windows of features, cached
# ---------------------------------------------------------------------


@dataclass
class WindowSet:
    """Every analysis window of one WAV."""

    times: np.ndarray                  # (N,) seconds from the WAV start
    names: list[str]                   # feature names, column order
    matrix: np.ndarray                 # (N, F)
    sample_rate: int
    hop: int

    @property
    def window_rate(self) -> float:
        return self.sample_rate / self.hop

    def column(self, name: str) -> np.ndarray:
        return self.matrix[:, self.names.index(name)]

    def features(self) -> Iterator[Features]:
        """Rebuild Features objects exactly as the extractor made them."""
        bands = [n for n in self.names if n in BAND_NAMES]
        diagnostics = [
            n for n in self.names if n != "rms" and n not in BAND_NAMES
        ]
        index = {name: i for i, name in enumerate(self.names)}

        for time, row in zip(self.times, self.matrix):
            yield Features(
                time=float(time),
                rms_db=float(row[index["rms"]]),
                bands={n: float(row[index[n]]) for n in bands},
                diagnostics={n: float(row[index[n]]) for n in diagnostics},
            )


def extract_windows(wav_path: Path) -> WindowSet:
    """Run a WAV through the shared extractor, as live would."""
    sample_rate, raw = wavfile.read(wav_path)
    signal = to_float(raw) * FULL_SCALE

    extractor = FeatureExtractor(sample_rate)

    names: list[str] | None = None
    times: list[float] = []
    rows: list[list[float]] = []

    # In frames the size the ESP32 sends, so window boundaries land
    # exactly where they would live.
    for start in range(0, signal.size - FRAME_SAMPLES + 1, FRAME_SAMPLES):
        chunk = signal[start : start + FRAME_SAMPLES]

        for window in extractor.push(chunk):
            values = window.values

            if names is None:
                names = list(values)

            times.append(window.time)
            rows.append([values[name] for name in names])

    if names is None:
        raise ValueError(f"{wav_path.name}: too short for one window")

    return WindowSet(
        times=np.array(times),
        names=names,
        matrix=np.array(rows, dtype=np.float64),
        sample_rate=sample_rate,
        hop=extractor.hop,
    )


class FeatureCache:
    """Disk cache of extractor output, one file per event.

    The extractor is by far the slow part of a replay (about 40 s for a
    few hundred events) and it has no tunable settings: what it produces
    does not depend on any threshold. So its output is cached, and the
    cache is keyed on the WAV *and* on the source of features.py, so
    editing the DSP invalidates it without anyone having to remember to.
    """

    def __init__(self, directory: Path | None) -> None:
        self.directory = directory
        self.hits = 0
        self.misses = 0

        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)

        self._dsp = hashlib.sha1(
            Path(features_module.__file__).read_bytes()
        ).hexdigest()

    def _key(self, wav_path: Path) -> str:
        stat = wav_path.stat()

        return json.dumps(
            [CACHE_VERSION, self._dsp, stat.st_size, stat.st_mtime_ns]
        )

    def get(self, record: EventRecord) -> WindowSet:
        if self.directory is None:
            self.misses += 1
            return extract_windows(record.wav_path)

        path = self.directory / f"{record.id}.npz"
        key = self._key(record.wav_path)

        if path.is_file():
            try:
                with np.load(path, allow_pickle=False) as data:
                    if str(data["key"]) == key:
                        self.hits += 1

                        return WindowSet(
                            times=data["times"],
                            names=[str(n) for n in data["names"]],
                            matrix=data["matrix"],
                            sample_rate=int(data["sample_rate"]),
                            hop=int(data["hop"]),
                        )
            except (OSError, ValueError, KeyError):
                pass            # unreadable: just recompute it

        self.misses += 1
        windows = extract_windows(record.wav_path)

        np.savez_compressed(
            path,
            key=np.array(key),
            times=windows.times,
            names=np.array(windows.names),
            matrix=windows.matrix,
            sample_rate=np.array(windows.sample_rate),
            hop=np.array(windows.hop),
        )

        return windows


@dataclass
class ScoredEvent:
    """An event ready to be scored: its windows, truth and group."""

    record: EventRecord
    windows: WindowSet
    truth: np.ndarray
    group: int

    @property
    def scored(self) -> int:
        return int((self.truth != UNSCORED).sum())


def prepare(
    records: list[EventRecord],
    cache: FeatureCache,
    warmup: float = DEFAULT_WARMUP_SECONDS,
    guard: float = DEFAULT_GUARD_SECONDS,
    group_gap: float = DEFAULT_GROUP_GAP_SECONDS,
) -> list[ScoredEvent]:
    """Windows and truth for every labelled event with something to score."""
    labelled = [r for r in records if r.labelled]
    groups = group_events(labelled, group_gap)

    events: list[ScoredEvent] = []

    for record in labelled:
        windows = cache.get(record)
        truth = truth_labels(record, windows.times, warmup, guard)

        if (truth != UNSCORED).any():
            events.append(
                ScoredEvent(record, windows, truth, groups[record.id])
            )

    return events


# ---------------------------------------------------------------------
# Exact replay: the real Smoother
# ---------------------------------------------------------------------


def replay_exact(
    windows: WindowSet,
    rule,
    median_seconds: float,
    hold_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Candidate and published state per window, via the real Smoother.

    Returns two int arrays of state codes. Published state is -1 until
    the first candidate has held long enough to be published.
    """
    smoother = Smoother(
        rule,
        window_rate=windows.window_rate,
        median_seconds=median_seconds,
        hold_seconds=hold_seconds,
    )

    candidates = np.empty(len(windows.times), dtype=np.int8)
    states = np.empty(len(windows.times), dtype=np.int8)

    for index, window in enumerate(windows.features()):
        decision = smoother.update(window)

        candidates[index] = LABEL_CODE[decision.candidate]
        states[index] = (
            -1 if decision.state is None else LABEL_CODE[decision.state]
        )

    return candidates, states


# ---------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------


def confusion(truth: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """3x4 counts: truth (OFF, FAN, COMPRESSOR) x predicted + 'none'."""
    mask = truth != UNSCORED

    if not mask.any():
        return np.zeros((3, 4), dtype=np.int64)

    predicted = np.where(predicted < 0, NO_STATE, predicted)
    flat = truth[mask].astype(np.int64) * 4 + predicted[mask]

    return np.bincount(flat, minlength=12).reshape(3, 4)


def metrics(matrix: np.ndarray) -> dict:
    """Headline numbers from a 3x4 confusion matrix."""
    matrix = np.asarray(matrix, dtype=np.int64)
    totals = matrix.sum(axis=1)

    recall = np.array([
        matrix[i, i] / totals[i] if totals[i] else np.nan
        for i in range(3)
    ])

    predicted_totals = matrix[:, :3].sum(axis=0)
    precision = np.array([
        matrix[i, i] / predicted_totals[i] if predicted_totals[i] else np.nan
        for i in range(3)
    ])

    present = ~np.isnan(recall)

    return {
        "windows": int(matrix.sum()),
        "balancedAccuracy": (
            float(np.nanmean(recall)) if present.any() else None
        ),
        "recall": {
            LABELS[i]: _num(recall[i]) for i in range(3)
        },
        "precision": {
            LABELS[i]: _num(precision[i]) for i in range(3)
        },
        "offToFan": int(matrix[0, 1]),
        "offToCompressor": int(matrix[0, 2]),
        "fanToOff": int(matrix[1, 0]),
        "fanToCompressor": int(matrix[1, 2]),
        "compressorMisses": int(matrix[2].sum() - matrix[2, 2]),
        "unpublished": int(matrix[:, 3].sum()),
    }


def _num(value: float) -> float | None:
    return None if np.isnan(value) else float(value)


# ---------------------------------------------------------------------
# Fast path, for the threshold search only
# ---------------------------------------------------------------------


def rolling_median(matrix: np.ndarray, width: int) -> np.ndarray:
    """Median over the last ``width`` rows, fewer at the start.

    Matches the Smoother, which takes the median of however many
    windows it has up to its span.
    """
    rows = matrix.shape[0]
    out = np.empty_like(matrix)

    head = min(width - 1, rows)

    for index in range(head):
        out[index] = np.median(matrix[: index + 1], axis=0)

    if rows >= width:
        out[width - 1 :] = np.median(
            sliding_window_view(matrix, width, axis=0), axis=-1
        )

    return out


def hold_published(
    candidates: np.ndarray, window_rate: float, hold_seconds: float
) -> np.ndarray:
    """Published state per window from a candidate sequence.

    A candidate is published once it has been unchanged for
    ``hold_seconds``; until then the previous state stands. -1 means
    nothing has been published yet.
    """
    count = len(candidates)

    if count == 0:
        return candidates.astype(np.int8)

    need = int(np.ceil(hold_seconds * window_rate)) + 1

    change = np.empty(count, dtype=bool)
    change[0] = True
    change[1:] = candidates[1:] != candidates[:-1]

    index = np.arange(count)
    run_start = np.maximum.accumulate(np.where(change, index, 0))
    held = (index - run_start + 1) >= need

    last_held = np.maximum.accumulate(np.where(held, index, -1))

    return np.where(
        last_held >= 0, candidates[np.maximum(last_held, 0)], -1
    ).astype(np.int8)


class FastEvaluator:
    """Score many v2 settings quickly. See the module docstring.

    The smoothed features are computed once, since the median does not
    depend on any threshold; each setting then costs one vectorised call
    of the real rule plus a cheap hold per event.
    """

    def __init__(
        self,
        events: list[ScoredEvent],
        median_seconds: float,
        hold_seconds: float,
    ) -> None:
        self.events = events
        self.hold_seconds = hold_seconds

        self.names = events[0].windows.names
        self._index = {name: i for i, name in enumerate(self.names)}

        smoothed = []
        self.bounds: list[tuple[int, int]] = []
        self.rates: list[float] = []

        offset = 0

        for event in events:
            width = max(
                1,
                round(median_seconds * event.windows.window_rate),
            )

            matrix = rolling_median(event.windows.matrix, width)
            smoothed.append(matrix)

            self.bounds.append((offset, offset + len(matrix)))
            self.rates.append(event.windows.window_rate)

            offset += len(matrix)

        self.matrix = np.concatenate(smoothed)
        self.truth = [event.truth for event in events]

    def column(self, name: str) -> np.ndarray:
        return self.matrix[:, self._index[name]]

    def codes(self, rule: ThresholdsV2) -> np.ndarray:
        """Candidate code per window, for every window of every event."""
        view = {name: self.column(name) for name in self.names}

        return classify_v2_codes(view, rule).astype(np.int8)

    def published(self, rule: ThresholdsV2) -> list[np.ndarray]:
        codes = self.codes(rule)

        return [
            hold_published(codes[a:b], rate, self.hold_seconds)
            for (a, b), rate in zip(self.bounds, self.rates)
        ]

    def confusion(self, rule: ThresholdsV2) -> np.ndarray:
        """Per-event 3x4 confusion, shape (events, 3, 4)."""
        published = self.published(rule)

        return np.stack([
            confusion(truth, state)
            for truth, state in zip(self.truth, published)
        ])


def verify_fast_path(
    events: list[ScoredEvent],
    rule: ThresholdsV2,
    median_seconds: float,
    hold_seconds: float,
    limit: int | None = None,
) -> int:
    """Check FastEvaluator against the real Smoother, window for window.

    Raises AssertionError on the first disagreement. Returns the number
    of events compared.
    """
    chosen = events if limit is None else events[:limit]

    fast = FastEvaluator(chosen, median_seconds, hold_seconds)
    codes = fast.codes(rule)
    published = fast.published(rule)

    for event, (a, b), state in zip(chosen, fast.bounds, published):
        candidate_exact, state_exact = replay_exact(
            event.windows, rule, median_seconds, hold_seconds
        )

        assert np.array_equal(codes[a:b], candidate_exact), (
            event.record.id, "candidate",
        )
        assert np.array_equal(state, state_exact), (
            event.record.id, "published state",
        )

    return len(chosen)


# ---------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------


@dataclass
class Evaluation:
    """A classifier's replay over a prepared dataset."""

    label: str
    rule: object
    median_seconds: float
    hold_seconds: float
    candidate: list[np.ndarray] = field(default_factory=list)
    published: list[np.ndarray] = field(default_factory=list)

    def matrices(self, events: list[ScoredEvent], level: str = "published"):
        source = self.published if level == "published" else self.candidate

        return np.stack([
            confusion(event.truth, predicted)
            for event, predicted in zip(events, source)
        ])


def evaluate(
    label: str,
    events: list[ScoredEvent],
    rule,
    median_seconds: float,
    hold_seconds: float,
) -> Evaluation:
    """Exact replay of every event under one rule."""
    result = Evaluation(label, rule, median_seconds, hold_seconds)

    for event in events:
        candidate, state = replay_exact(
            event.windows, rule, median_seconds, hold_seconds
        )

        result.candidate.append(candidate)
        result.published.append(state)

    return result


__all__ = [
    "DEFAULT_GROUP_GAP_SECONDS",
    "DEFAULT_GUARD_SECONDS",
    "DEFAULT_WARMUP_SECONDS",
    "LABELS",
    "STATES",
    "TEMPORAL_FILL",
    "FEATURE_NAMES",
    "Evaluation",
    "EventRecord",
    "FastEvaluator",
    "FeatureCache",
    "ScoredEvent",
    "WindowSet",
    "confusion",
    "dataset_counts",
    "evaluate",
    "extract_windows",
    "group_events",
    "hold_published",
    "load_events",
    "metrics",
    "parse_event",
    "prepare",
    "replay_exact",
    "rolling_median",
    "truth_labels",
    "verify_fast_path",
]
