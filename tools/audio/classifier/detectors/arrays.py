"""Detectors that work on one window or on a million.

A stateless detector is a comparison. Written with numpy booleans it is
the same comparison whether it is given one window's features (live, one
at a time) or every window of every recorded event at once (the
threshold search), so the evaluation tools run the very lines the live
backend runs rather than a copy that could drift.
"""

from __future__ import annotations

import numpy as np


def scalarise(result):
    """A plain ``bool`` for scalar input, the array itself otherwise."""
    result = np.asarray(result)

    return bool(result) if result.ndim == 0 else result
