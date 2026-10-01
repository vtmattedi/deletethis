"""Replay reviewed events through the live classifiers and score them.

The question this answers is "would a different rule have called these
real situations correctly?", asked of audio a person has already
labelled. The inputs are the event WAVs the backend recorded and the
``review`` block a human filled in; nothing else is ground truth.

    event WAV --> FeatureExtractor --> detectors --> smoothing + hold
                                                         |
                          review.actualFrom/To  ---------+--> score

Observations, not a state
-------------------------
Watson v2 reports independent observations -- *is the fan running* and
*is the compressor running* -- and not one combined state. So each is
scored on its own, as a yes/no question with its own errors: a fan
called running when it was not, a compressor missed when it was.

The historical labels, though, are the old three-way ``OFF / FAN /
COMPRESSOR``. To score the new observations against them, this module
derives **compatibility truth**:

    OFF         fan = no    compressor = no
    FAN         fan = yes   compressor = no
    COMPRESSOR  fan = yes   compressor = yes

The assumption is that the compressor only runs while the fan is
turning, which holds for this air conditioner. It is a stand-in so the
old reviews remain usable evidence; it is *not* the review schema for
v2, which labels each observation on its own. Where it matters: a
"COMPRESSOR" window now counts against the fan detector if the fan is
not detected, which the old scoring (where the compressor shadowed the
fan) never exposed.

Two ways of running the rules, one set of rules
-----------------------------------------------
``replay_v2`` feeds the real ObservationSmoother window by window. It is
what the live backend does, and every reported result comes from it.

``FastEvaluator`` exists only so a threshold *search* is affordable: it
tries thousands of settings and the Smoother is a Python loop. It does
not reimplement the detectors -- they are written once, on numpy
booleans, and run on scalars (live) or arrays (search) alike. What it
does duplicate is the rolling median and the hold, which are plumbing,
and ``verify_fast_path`` checks them against the real thing window for
window so a disagreement is a failing check, not a quiet error.

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
from classifier.common import ObservationSmoother, Smoother
from classifier.detectors import (
    BeepConfig,
    BeepDetector,
    CompressorConfig,
    FanConfig,
    detect_compressor,
    detect_fan,
)
from classifier.v2 import ObservationRules
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

# The historical three-way labels, and the independent observations
# they are mapped onto.
LABELS = ("OFF", "FAN", "COMPRESSOR")
LABEL_CODE = {name: index for index, name in enumerate(LABELS)}
OBSERVATIONS = ("fan", "compressor")

UNSCORED = -1          # no trustworthy label for this window
UNKNOWN = -1           # a classifier had nothing published yet
NONE_COLUMN = 2        # confusion column for "nothing published"

CACHE_VERSION = 2      # bumped when the cached columns changed


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
    event_type: str                  # "transition" for v1-style events
    reviewed: bool
    actual_from: str | None
    actual_to: str | None
    interference: tuple[str, ...]

    pre_seconds: float
    post_seconds: float
    moment: datetime                 # the transition moment in the WAV

    @property
    def labelled(self) -> bool:
        """Has a usable ground truth in the historical three-way scheme.

        v2 events review each observation separately and carry no
        ``actualFrom`` / ``actualTo``, so they are not scored here.
        """
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
        event_type=metadata.get("eventType", "transition"),
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
    observation_events = [
        r for r in records if r.classifier_version == "v2"
        and r.event_type != "transition"
    ]
    unknown = [
        r for r in records
        if r.reviewed and not r.labelled and r not in observation_events
    ]

    return {
        "events": len(records),
        "labelled": len(labelled),
        "reviewedUnknown": len(unknown),
        "unreviewed": sum(
            1 for r in records
            if not r.reviewed and r not in observation_events
        ),
        # v2 observation events carry per-observation reviews that this
        # scoring does not read yet.
        "observationEvents": len(observation_events),
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
    """Per-window historical label code, or UNSCORED where untrusted."""
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


def observation_truth(truth: np.ndarray) -> dict[str, np.ndarray]:
    """Per-window yes/no truth for each observation, from the old labels.

    COMPATIBILITY ASSUMPTION: the compressor runs only while the fan is
    turning, so a COMPRESSOR label means both are on. See the module
    docstring; this is a bridge for historical reviews, not a schema.

    Returns int8 arrays: 1 = yes, 0 = no, UNSCORED where untrusted.
    """
    scored = truth != UNSCORED

    fan = np.where(scored, (truth >= LABEL_CODE["FAN"]).astype(np.int8),
                   UNSCORED).astype(np.int8)
    compressor = np.where(
        scored, (truth == LABEL_CODE["COMPRESSOR"]).astype(np.int8),
        UNSCORED,
    ).astype(np.int8)

    return {"fan": fan, "compressor": compressor}


def legacy_observations(codes: np.ndarray) -> dict[str, np.ndarray]:
    """Fold v1's single state onto the two observations.

    Same compatibility mapping as the truth, so v1 is scored on the
    same terms: its COMPRESSOR claims the fan is running too. -1 (nothing
    published yet) stays -1.
    """
    known = codes >= 0

    return {
        "fan": np.where(known, (codes >= 1).astype(np.int8), UNKNOWN)
        .astype(np.int8),
        "compressor": np.where(known, (codes == 2).astype(np.int8), UNKNOWN)
        .astype(np.int8),
    }


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
    nfft: int = 1024

    @property
    def window_rate(self) -> float:
        return self.sample_rate / self.hop

    @property
    def hop_seconds(self) -> float:
        return self.hop / self.sample_rate

    @property
    def window_seconds(self) -> float:
        return self.nfft / self.sample_rate

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
        nfft=extractor.nfft,
    )


class FeatureCache:
    """Disk cache of extractor output, one file per WAV.

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

    def get(self, record) -> WindowSet:
        """Windows for anything with ``.id`` and ``.wav_path``."""
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
                            nfft=int(data["nfft"]),
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
            nfft=np.array(windows.nfft),
        )

        return windows


@dataclass
class ScoredEvent:
    """An event ready to be scored: its windows, truth and group."""

    record: EventRecord
    windows: WindowSet
    truth: np.ndarray                  # historical three-way label codes
    group: int
    observed: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.observed:
            self.observed = observation_truth(self.truth)

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
# Exact replay: the real smoothers
# ---------------------------------------------------------------------


@dataclass
class ObservationRun:
    """One classifier's replay, as fan and compressor observations.

    ``candidate`` is the detector's verdict on the smoothed features;
    ``published`` is after the hold, which is what a consumer sees.
    Each maps observation -> one int8 array per event: 1 yes, 0 no,
    UNKNOWN (-1) before anything was published.
    """

    label: str
    candidate: dict[str, list[np.ndarray]] = field(
        default_factory=lambda: {o: [] for o in OBSERVATIONS}
    )
    published: dict[str, list[np.ndarray]] = field(
        default_factory=lambda: {o: [] for o in OBSERVATIONS}
    )


def replay_v1(
    windows: WindowSet,
    rule,
    median_seconds: float,
    hold_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """v1's candidate and published state codes, via the real Smoother.

    State codes are 0 OFF, 1 FAN, 2 COMPRESSOR; -1 means nothing has
    been published yet.
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


def replay_v2(
    windows: WindowSet,
    rules: ObservationRules,
    median_seconds: float,
    hold_seconds: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """v2's candidate and published observations, via the real smoother."""
    smoother = ObservationSmoother(
        rules,
        window_rate=windows.window_rate,
        median_seconds=median_seconds,
        hold_seconds=hold_seconds,
    )

    count = len(windows.times)
    candidate = {o: np.empty(count, dtype=np.int8) for o in OBSERVATIONS}
    published = {o: np.empty(count, dtype=np.int8) for o in OBSERVATIONS}

    def code(value):
        return UNKNOWN if value is None else int(value)

    for index, window in enumerate(windows.features()):
        decision = smoother.update(window)

        candidate["fan"][index] = int(decision.fan_candidate)
        candidate["compressor"][index] = int(decision.compressor_candidate)
        published["fan"][index] = code(decision.fan_detected)
        published["compressor"][index] = code(decision.compressor_detected)

    return candidate, published


def run_v1(
    label: str,
    events: list["ScoredEvent"],
    rule,
    median_seconds: float,
    hold_seconds: float,
) -> ObservationRun:
    """Replay every event under v1 and fold it onto the observations."""
    run = ObservationRun(label)

    for event in events:
        candidate, state = replay_v1(
            event.windows, rule, median_seconds, hold_seconds
        )

        for observation, values in legacy_observations(candidate).items():
            run.candidate[observation].append(values)

        for observation, values in legacy_observations(state).items():
            run.published[observation].append(values)

    return run


def run_v2(
    label: str,
    events: list["ScoredEvent"],
    rules: ObservationRules,
    median_seconds: float,
    hold_seconds: float,
) -> ObservationRun:
    """Replay every event under v2's independent observations."""
    run = ObservationRun(label)

    for event in events:
        candidate, published = replay_v2(
            event.windows, rules, median_seconds, hold_seconds
        )

        for observation in OBSERVATIONS:
            run.candidate[observation].append(candidate[observation])
            run.published[observation].append(published[observation])

    return run


def replay_beeps(
    windows: WindowSet, config: BeepConfig
) -> tuple[list, BeepDetector]:
    """Beep events in a recording, from the same raw windows live sees.

    Returns the events and the detector, whose counters say what it
    considered and turned down.
    """
    detector = BeepDetector(
        config,
        hop_seconds=windows.hop_seconds,
        window_seconds=windows.window_seconds,
    )
    events: list = []

    for window in windows.features():
        events.extend(detector.update(window))

    events.extend(detector.flush())

    return events, detector


# ---------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------


def confusion(truth: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """2x3 counts: truth (no, yes) x predicted (no, yes, nothing yet).

    Only windows with a trustworthy label count.
    """
    mask = truth != UNSCORED

    if not mask.any():
        return np.zeros((2, 3), dtype=np.int64)

    predicted = np.where(predicted < 0, NONE_COLUMN, predicted)
    flat = truth[mask].astype(np.int64) * 3 + predicted[mask]

    return np.bincount(flat, minlength=6).reshape(2, 3)


def metrics(matrix: np.ndarray) -> dict:
    """Headline numbers from a 2x3 confusion matrix.

    ``truePositive`` and friends are windows. Nothing-published-yet
    windows count as misses on whichever side they fall: not knowing is
    not the same as knowing it is off.
    """
    matrix = np.asarray(matrix, dtype=np.int64)

    absent, present = matrix[0], matrix[1]

    true_negative, false_positive = int(absent[0]), int(absent[1])
    false_negative, true_positive = int(present[0]), int(present[1])

    absent_total = int(absent.sum())
    present_total = int(present.sum())

    recall = true_positive / present_total if present_total else None
    specificity = true_negative / absent_total if absent_total else None
    called = true_positive + false_positive
    precision = true_positive / called if called else None

    rates = [r for r in (recall, specificity) if r is not None]

    return {
        "windows": int(matrix.sum()),
        "present": present_total,
        "absent": absent_total,
        "truePositive": true_positive,
        "falsePositive": false_positive,
        "falseNegative": false_negative,
        "trueNegative": true_negative,
        "unpublished": int(matrix[:, NONE_COLUMN].sum()),
        # fraction of windows where it really was running and we said so
        "recall": recall,
        # ... and where it was not running and we said so
        "specificity": specificity,
        "precision": precision,
        "balancedAccuracy": (
            float(np.mean(rates)) if rates else None
        ),
    }


def run_confusions(
    events: list[ScoredEvent],
    run: ObservationRun,
    level: str = "published",
) -> dict[str, np.ndarray]:
    """Per-event confusion for each observation: (events, 2, 3)."""
    source = run.published if level == "published" else run.candidate

    return {
        observation: np.stack([
            confusion(event.observed[observation], predicted)
            for event, predicted in zip(events, source[observation])
        ])
        for observation in OBSERVATIONS
    }


# ---------------------------------------------------------------------
# Fast path, for the threshold search only
# ---------------------------------------------------------------------


def rolling_median(matrix: np.ndarray, width: int) -> np.ndarray:
    """Median over the last ``width`` rows, fewer at the start.

    Matches the smoothers, which take the median of however many
    windows they have up to their span.
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
    """Published value per window from a candidate sequence.

    A candidate is published once it has been unchanged for
    ``hold_seconds``; until then the previous value stands. -1 means
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
    """Score many detector settings quickly. See the module docstring.

    The smoothed features are computed once, since the median does not
    depend on any threshold. The two detectors are independent, so each
    setting of one costs one vectorised call of that detector plus a
    cheap hold per event, and nothing of the other.
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
        self.truth = {
            observation: [e.observed[observation] for e in events]
            for observation in OBSERVATIONS
        }
        self._view = {
            name: self.matrix[:, i] for name, i in self._index.items()
        }

    def column(self, name: str) -> np.ndarray:
        return self._view[name]

    # -- one detector at a time -----------------------------------------

    def fan_candidates(self, config: FanConfig) -> np.ndarray:
        return np.asarray(detect_fan(self._view, config)).astype(np.int8)

    def compressor_candidates(self, config: CompressorConfig) -> np.ndarray:
        return np.asarray(
            detect_compressor(self._view, config)
        ).astype(np.int8)

    def _published(self, candidates: np.ndarray) -> list[np.ndarray]:
        return [
            hold_published(candidates[a:b], rate, self.hold_seconds)
            for (a, b), rate in zip(self.bounds, self.rates)
        ]

    def _confusion(
        self, observation: str, candidates: np.ndarray
    ) -> np.ndarray:
        return np.stack([
            confusion(truth, state)
            for truth, state in zip(
                self.truth[observation], self._published(candidates)
            )
        ])

    def fan_confusion(self, config: FanConfig) -> np.ndarray:
        """Per-event fan confusion, shape (events, 2, 3)."""
        return self._confusion("fan", self.fan_candidates(config))

    def compressor_confusion(self, config: CompressorConfig) -> np.ndarray:
        return self._confusion(
            "compressor", self.compressor_candidates(config)
        )

    # -- both, as v2 runs them -------------------------------------------

    def candidates(self, rules: ObservationRules) -> dict[str, np.ndarray]:
        return {
            "fan": self.fan_candidates(rules.fan),
            "compressor": self.compressor_candidates(rules.compressor),
        }

    def published(
        self, rules: ObservationRules
    ) -> dict[str, list[np.ndarray]]:
        return {
            observation: self._published(codes)
            for observation, codes in self.candidates(rules).items()
        }


def verify_fast_path(
    events: list[ScoredEvent],
    rules: ObservationRules,
    median_seconds: float,
    hold_seconds: float,
    limit: int | None = None,
) -> int:
    """Check FastEvaluator against the real smoother, window for window.

    Raises AssertionError on the first disagreement. Returns the number
    of events compared.
    """
    chosen = events if limit is None else events[:limit]

    fast = FastEvaluator(chosen, median_seconds, hold_seconds)
    candidates = fast.candidates(rules)
    published = fast.published(rules)

    for index, (event, (a, b)) in enumerate(zip(chosen, fast.bounds)):
        exact_candidate, exact_published = replay_v2(
            event.windows, rules, median_seconds, hold_seconds
        )

        for observation in OBSERVATIONS:
            assert np.array_equal(
                candidates[observation][a:b], exact_candidate[observation]
            ), (event.record.id, observation, "candidate")
            assert np.array_equal(
                published[observation][index], exact_published[observation]
            ), (event.record.id, observation, "published")

    return len(chosen)


__all__ = [
    "DEFAULT_GROUP_GAP_SECONDS",
    "DEFAULT_GUARD_SECONDS",
    "DEFAULT_WARMUP_SECONDS",
    "FEATURE_NAMES",
    "LABELS",
    "LABEL_CODE",
    "OBSERVATIONS",
    "TEMPORAL_FILL",
    "UNKNOWN",
    "UNSCORED",
    "EventRecord",
    "FastEvaluator",
    "FeatureCache",
    "ObservationRun",
    "ScoredEvent",
    "WindowSet",
    "confusion",
    "dataset_counts",
    "extract_windows",
    "group_events",
    "hold_published",
    "legacy_observations",
    "load_events",
    "metrics",
    "observation_truth",
    "parse_event",
    "prepare",
    "replay_beeps",
    "replay_v1",
    "replay_v2",
    "rolling_median",
    "run_confusions",
    "run_v1",
    "run_v2",
    "truth_labels",
    "verify_fast_path",
]
