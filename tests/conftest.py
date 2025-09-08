"""
Pytest configuration and fixtures for sinonym test suite.

This module provides shared fixtures to optimize test performance by avoiding
repeated expensive initialization of ChineseNameDetector instances.
"""

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
    return ChineseNameDetector()


@pytest.fixture(scope="session") 
def fast_detector():
    """
    Alias for detector fixture for backwards compatibility.
    
    Some tests may prefer this name to emphasize performance optimization.
    """
    return ChineseNameDetector()