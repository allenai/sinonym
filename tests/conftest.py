"""
Pytest configuration and fixtures for sinonym test suite.

This module provides shared fixtures to optimize test performance by avoiding
repeated expensive initialization of ChineseNameDetector instances.

Also supports injecting candidate weight vectors via the optional
environment variable `SINONYM_WEIGHTS` (JSON-encoded list of 8 or 9
floats; shorter vectors are padded with the default trailing coefficients by
`NameParsingService`), to enable automated optimization scripts to evaluate
weight configurations. A malformed or wrong-shaped value raises immediately
rather than silently falling back to the default weights, so a broken
optimization run fails loudly instead of quietly scoring the wrong vector.
"""

import json
import os

import pytest

from sinonym import ChineseNameDetector


@pytest.fixture(scope="session")
def detector():
    """
    Session-scoped ChineseNameDetector instance shared across all tests.

    This fixture creates a single detector instance that is reused for the entire
    test suite, eliminating the ~4-7 second initialization cost that would otherwise
    be repeated for each test file.

    The detector is stateless after initialization, so sharing it across tests
    does not affect test isolation.

    Returns:
        ChineseNameDetector: Fully initialized detector instance
    """
    # Optionally override weight vector via environment for optimization runs.
    # Production (NameParsingService) accepts 8/9-element vectors and pads
    # the trailing defaults, so we mirror that here; anything else is a
    # configuration error and must fail the session rather than silently
    # falling back to default weights.
    weights = None
    raw = os.getenv("SINONYM_WEIGHTS")
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            message = f"SINONYM_WEIGHTS is not valid JSON: {raw!r}"
            raise ValueError(message) from e
        if not isinstance(parsed, list) or len(parsed) not in (8, 9):
            message = f"SINONYM_WEIGHTS must be a JSON list of 8 or 9 floats; got {raw!r}"
            raise ValueError(message)
        weights = [float(x) for x in parsed]

    return ChineseNameDetector(weights=weights)


@pytest.fixture(scope="session")
def fast_detector(detector):
    """
    Alias for detector fixture for backwards compatibility.

    Some tests may prefer this name to emphasize performance optimization.
    """
    return detector
