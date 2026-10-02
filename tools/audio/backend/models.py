"""Validation models for mutating API requests."""

from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel, ConfigDict, Field, field_validator, model_validator,
)

from acstream import is_network_target, split_target
from classifier.v2 import STABILITY_FEATURES

# Spelled out for the type checker and for OpenAPI; a test pins it to
# STABILITY_FEATURES so the two cannot drift.
StabilityFeature = Literal[
    "1k-2k_std",
    "500-1k_std",
    "2k-4k_std",
    "rms_std",
    "spectral_flux_median",
    "spectral_flux_std",
]
assert set(StabilityFeature.__args__) == set(STABILITY_FEATURES)


class ConfigPatch(BaseModel):
    """Every field optional: a PATCH changes only what it names."""

    model_config = ConfigDict(extra="forbid")

    compressorThreshold: float | None = Field(
        default=None, ge=-120.0, le=0.0
    )
    fanMidThreshold: float | None = Field(
        default=None, ge=-120.0, le=0.0
    )
    fanHighThreshold: float | None = Field(
        default=None, ge=-120.0, le=0.0
    )
    fanRequire: Literal["either", "both"] | None = None

    medianSeconds: float | None = Field(
        default=None, gt=0.0, le=30.0
    )
    holdSeconds: float | None = Field(
        default=None, ge=0.0, le=120.0
    )

    eventPreSeconds: float | None = Field(
        default=None, ge=0.0, le=60.0
    )
    eventPostSeconds: float | None = Field(
        default=None, ge=0.0, le=60.0
    )

    # Classifier v2 only. The API layer rejects these on a v1 run.
    # `classifierVersion` is intentionally absent: switching version
    # changes where results are stored, so it is a restart, and
    # extra="forbid" turns an attempt to patch it into a 422.
    fanStabilityFeature: StabilityFeature | None = None
    fanStabilityThreshold: float | None = Field(
        default=None, gt=0.0, le=100.0
    )
    fanStabilityMinSeconds: float | None = Field(
        default=None, ge=0.0, le=30.0
    )

    # The beep detector, also v2 only.
    beepMinContrastDb: float | None = Field(
        default=None, ge=0.0, le=80.0
    )
    beepEdgeContrastDb: float | None = Field(
        default=None, ge=0.0, le=80.0
    )
    beepMinLevelDb: float | None = Field(
        default=None, ge=-120.0, le=0.0
    )
    beepMinMs: float | None = Field(default=None, gt=0.0, le=2000.0)
    beepMaxMs: float | None = Field(default=None, gt=0.0, le=5000.0)
    beepMinHz: float | None = Field(default=None, ge=0.0, le=8000.0)
    beepMaxHz: float | None = Field(default=None, ge=0.0, le=8000.0)
    beepMaxPeakSpreadHz: float | None = Field(
        default=None, ge=0.0, le=2000.0
    )

    def changes(self) -> dict:
        return self.model_dump(exclude_none=True)


class TargetPatch(BaseModel):
    """Where the backend reads the audio stream from."""

    model_config = ConfigDict(extra="forbid")

    target: str = Field(min_length=1, max_length=255)

    @field_validator("target")
    @classmethod
    def _tcp_address(cls, value: str) -> str:
        value = value.strip()
        # Network only: a serial port is not something to switch to
        # from a browser, and the firmware streams over TCP anyway.
        if not is_network_target(value):
            raise ValueError(
                "expected host:port or an IP/domain name, e.g. "
                "192.168.1.50:3333"
            )
        _host, port = split_target(value)
        if not 0 < port < 65536:
            raise ValueError("port must be between 1 and 65535")
        return value


StateLabel = Literal["OFF", "FAN", "COMPRESSOR", "UNKNOWN"]


class ReviewPatch(BaseModel):
    """A completed human review of the classifier result."""

    model_config = ConfigDict(extra="forbid")

    classificationCorrect: bool
    actualFrom: StateLabel
    actualTo: StateLabel
    interference: list[str] = Field(default_factory=list)
    notes: str = Field(default="", max_length=10000)


class CommandContext(BaseModel):
    """A command that was sent near an event.

    Watson records this and attaches it; it never reads it back to decide
    anything. Whether the command explains the event is for whoever
    analyses it later, or for the controller, which owns that logic.
    """

    model_config = ConfigDict(extra="forbid")

    command: str = Field(min_length=1, max_length=60)
    sentAt: float | None = None
    expectedBeep: bool = True
    note: str = Field(default="", max_length=1000)


# The three things that can be true of a reported boolean transition
# X -> Y: it happened, or the observation stayed X, or it stayed Y.
Outcome = Literal["correct", "stayed_true", "stayed_false"]


class ReviewV2Patch(BaseModel):
    """A human review of one v2 observation event.

    v1 reviews said what the combined state really was before and after
    (``actualFrom`` / ``actualTo``). A v2 event is about one observation,
    so the review is about that observation:

    * a fan or compressor change: was it right, and if not, what was the
      observation really afterwards (``actualValue``);
    * a beep: was it really a beep (``actualBeep``);
    * a manual capture: nothing was claimed, so the review says what was
      really happening: ``actualFan``, ``actualCompressor``, ``actualBeep``.
    """

    model_config = ConfigDict(extra="forbid")

    correct: bool | None = None
    # Fan / compressor: what really happened around the transition the
    # detector reported (X -> Y). See ``Outcome``.
    outcome: Outcome | None = None
    actualValue: bool | None = None
    actualBeep: bool | None = None
    actualFan: bool | None = None
    actualCompressor: bool | None = None
    interference: list[str] = Field(default_factory=list)
    notes: str = Field(default="", max_length=10000)
    commandContext: CommandContext | None = None


class CommandMarker(BaseModel):
    """The operator saying a command was just sent."""

    model_config = ConfigDict(extra="forbid")

    command: str = Field(min_length=1, max_length=60)
    expectedBeep: bool = True
    note: str = Field(default="", max_length=1000)


def _observation_review(metadata: dict, review: ReviewV2Patch, fields: dict):
    """A fan / compressor review, reduced to what reality did.

    The detector said X -> Y. With booleans there are three possibilities:

    * ``correct``      reality did X -> Y
    * ``stayed_true``  reality was True -> True
    * ``stayed_false`` reality was False -> False

    which is enough to rebuild the real timeline without asking for
    before and after separately. The older binary form (``correct`` with
    an optional ``actualValue``) is still accepted: wrong then means the
    observation was the opposite of ``to``, i.e. it stayed there.
    """
    origin, target = metadata.get("from"), metadata.get("to")
    outcome = review.outcome

    if outcome is None:
        if review.correct is None:
            raise ValueError("an outcome (or correct) is required")

        if review.correct:
            outcome = "correct"
        else:
            after = (
                review.actualValue
                if review.actualValue is not None else not target
            )
            outcome = "stayed_true" if after else "stayed_false"

    if outcome == "correct":
        actual_from, actual_to = origin, target
    else:
        actual_from = actual_to = outcome == "stayed_true"

    if review.correct is not None and review.correct != (outcome == "correct"):
        raise ValueError("correct contradicts the outcome")

    if review.actualValue is not None and review.actualValue != actual_to:
        raise ValueError("actualValue contradicts the outcome")

    fields.update(
        outcome=outcome,
        correct=outcome == "correct",
        actualFrom=actual_from,
        actualTo=actual_to,
        actualValue=actual_to,
    )

    return fields


def review_for_event(metadata: dict, body: dict) -> dict:
    """Validate a review against the kind of event it is about.

    Returns the review fields to store. v1-style events take
    ``actualFrom`` / ``actualTo``; v2 observation events take
    ``correct`` plus what is right for their type. Raises ValueError
    (and pydantic's ValidationError, a subclass) for a review that does
    not fit its event.
    """
    event_type = metadata.get("eventType")

    if metadata.get("classifierVersion") != "v2" or not event_type:
        return ReviewPatch.model_validate(body).model_dump()

    review = ReviewV2Patch.model_validate(body)
    fields = review.model_dump(exclude_none=True)

    if event_type == "manual":
        missing = [
            name for name in ("actualFan", "actualCompressor", "actualBeep")
            if getattr(review, name) is None
        ]
        if missing:
            raise ValueError(
                "a manual event needs " + ", ".join(missing)
                + ": what was really happening?"
            )
        if review.actualValue is not None:
            raise ValueError("a manual event has no actualValue")

        # "Correct" here means the live detectors agreed with the
        # reviewer about the fan and the compressor at that moment.
        seen = metadata.get("observations") or {}
        fields["correct"] = (
            seen.get("fan") == review.actualFan
            and seen.get("compressor") == review.actualCompressor
        )
        return fields

    for name in ("actualFan", "actualCompressor"):
        if getattr(review, name) is not None:
            raise ValueError(f"a {event_type} event has no {name}")

    if event_type in ("fan", "compressor"):
        if review.actualBeep is not None:
            raise ValueError(f"a {event_type} event has no actualBeep")

        return _observation_review(metadata, review, fields)

    if review.correct is None:
        raise ValueError("correct is required")

    if review.outcome is not None:
        raise ValueError(f"a {event_type} event has no outcome")

    if event_type == "beep":
        if review.actualValue is not None:
            raise ValueError("a beep event has no actualValue")

        expected = review.correct

        if "actualBeep" in fields and fields["actualBeep"] != expected:
            raise ValueError(
                "actualBeep contradicts the correct / wrong verdict"
            )

        fields["actualBeep"] = expected

    return fields


class EventDeleteRequest(BaseModel):
    """A bounded bundle of saved event identifiers to remove."""

    model_config = ConfigDict(extra="forbid")

    ids: list[str] = Field(min_length=1, max_length=100)


class EventBulkReviewRequest(BaseModel):
    """Apply one review decision to a bounded event selection."""

    model_config = ConfigDict(extra="forbid")

    ids: list[str] = Field(min_length=1, max_length=100)

    # v1-style: the combined state before and after.
    classificationCorrect: bool | None = None
    actualFrom: StateLabel | None = None
    actualTo: StateLabel | None = None

    # v2-style: one observation, or a beep.
    correct: bool | None = None
    outcome: Outcome | None = None
    actualValue: bool | None = None
    actualBeep: bool | None = None

    interference: list[str] = Field(default_factory=list)
    notes: str = Field(default="", max_length=10000)

    @model_validator(mode="after")
    def require_labels_for_incorrect_review(self):
        v2 = self.correct is not None or self.outcome is not None

        if (self.classificationCorrect is None) == (not v2) or (
            self.correct is not None and self.outcome is not None
        ):
            raise ValueError(
                "give exactly one of classificationCorrect (v1 events), "
                "or correct / outcome (v2 events)"
            )

        if self.classificationCorrect is False and (
            self.actualFrom is None or self.actualTo is None
        ):
            raise ValueError(
                "actualFrom and actualTo are required when marking "
                "events incorrect"
            )
        return self

    @property
    def verdict(self) -> bool:
        if self.outcome is not None:
            return self.outcome == "correct"

        return (
            self.correct if self.correct is not None
            else bool(self.classificationCorrect)
        )
