"""Classifier versions.

    v1   threshold classifier: 30-80 Hz -> COMPRESSOR, else fan bands
    v2   v1 plus a stationarity gate on FAN

Everything version-independent -- the states, the smoother and the
publication hold -- lives in ``common``. A version is a *rule*: an
object with ``decide(values) -> state``. Live classification, event
timelines and offline replay all drive the same rule through the same
Smoother, so there is exactly one implementation of each decision.
"""

from .common import (
    COMPRESSOR,
    DEFAULT_HOLD_SECONDS,
    DEFAULT_MEDIAN_SECONDS,
    FAN,
    OFF,
    STATES,
    Decision,
    Rule,
    Smoother,
)

VERSIONS = ("v1", "v2")

__all__ = [
    "COMPRESSOR",
    "DEFAULT_HOLD_SECONDS",
    "DEFAULT_MEDIAN_SECONDS",
    "FAN",
    "OFF",
    "STATES",
    "VERSIONS",
    "Decision",
    "Rule",
    "Smoother",
]
