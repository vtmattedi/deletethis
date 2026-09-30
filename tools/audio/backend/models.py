"""Validation models for mutating API requests."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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
