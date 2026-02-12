"""
Pytest configuration and fixtures for sinonym test suite.

This module provides shared fixtures to optimize test performance by avoiding
repeated expensive initialization of ChineseNameDetector instances.

Also supports injecting candidate weight vectors via the optional
environment variable `SINONYM_WEIGHTS` (JSON-encoded list of 8 floats), to
enable automated optimization scripts to evaluate weight configurations.
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
    # Optionally override weight vector via environment for optimization runs
    weights = None
    raw = os.getenv("SINONYM_WEIGHTS")
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list) and len(parsed) == 8:
                weights = [float(x) for x in parsed]
        except Exception:
            weights = None

    return ChineseNameDetector(weights=weights)


@pytest.fixture(scope="session") 
def fast_detector(detector):
    """
    Alias for detector fixture for backwards compatibility.
    
    Some tests may prefer this name to emphasize performance optimization.
    """
    return detector
