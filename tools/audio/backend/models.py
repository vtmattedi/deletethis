"""Validation models for mutating API requests."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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

    def changes(self) -> dict:
        return self.model_dump(exclude_none=True)


StateLabel = Literal["OFF", "FAN", "COMPRESSOR", "UNKNOWN"]


class ReviewPatch(BaseModel):
    """A completed human review of the classifier result."""

    model_config = ConfigDict(extra="forbid")

    classificationCorrect: bool
    actualFrom: StateLabel
    actualTo: StateLabel
    interference: list[str] = Field(default_factory=list)
    notes: str = Field(default="", max_length=10000)


class EventDeleteRequest(BaseModel):
    """A bounded bundle of saved event identifiers to remove."""

    model_config = ConfigDict(extra="forbid")

    ids: list[str] = Field(min_length=1, max_length=100)


class EventBulkReviewRequest(BaseModel):
    """Apply one review decision to a bounded event selection."""

    model_config = ConfigDict(extra="forbid")

    ids: list[str] = Field(min_length=1, max_length=100)
    classificationCorrect: bool
    actualFrom: StateLabel | None = None
    actualTo: StateLabel | None = None
    interference: list[str] = Field(default_factory=list)
    notes: str = Field(default="", max_length=10000)

    @model_validator(mode="after")
    def require_labels_for_incorrect_review(self):
        if not self.classificationCorrect and (
            self.actualFrom is None or self.actualTo is None
        ):
            raise ValueError(
                "actualFrom and actualTo are required when marking "
                "events incorrect"
            )
        return self
