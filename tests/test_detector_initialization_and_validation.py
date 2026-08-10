"""Regression tests for detector initialization and public batch validation."""

import math
import threading

import pytest

from sinonym import ChineseNameDetector
from sinonym.services.normalization import NormalizationService


def test_lazy_initialization_retries_after_post_data_failure(monkeypatch):
    """A transient dependent-service failure must not publish partial readiness."""
    original_initialize_services = ChineseNameDetector._initialize_services  # noqa: SLF001
    attempts = 0

    def fail_once(self, data):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            message = "transient service initialization failure"
            raise RuntimeError(message)
        return original_initialize_services(self, data)

    monkeypatch.setattr(ChineseNameDetector, "_initialize_services", fail_once)

    detector = ChineseNameDetector()

    assert detector._data is None  # noqa: SLF001
    assert detector._parsing_service is None  # noqa: SLF001

    result = detector.normalize_name("Zhang Wei")

    assert attempts == 2
    assert detector._data is not None  # noqa: SLF001
    assert detector._parsing_service is not None  # noqa: SLF001
    assert result.success


def test_lazy_initialization_retries_after_context_injection_failure(monkeypatch):
    """The original post-load failure trigger must remain retryable."""
    original_set_data_context = NormalizationService.set_data_context
    attempts = 0

    def fail_once(self, data):
        nonlocal attempts
        if data is None:
            return original_set_data_context(self, data)
        attempts += 1
        if attempts == 1:
            message = "transient context injection failure"
            raise RuntimeError(message)
        return original_set_data_context(self, data)

    monkeypatch.setattr(NormalizationService, "set_data_context", fail_once)

    detector = ChineseNameDetector()
    result = detector.normalize_name("Zhang Wei")

    assert attempts == 2
    assert result.success


def test_concurrent_lazy_initialization_never_exposes_partial_readiness(monkeypatch):
    """A second caller waits until the first caller publishes every service."""
    original_initialize_services = ChineseNameDetector._initialize_services  # noqa: SLF001
    initializing = threading.Event()
    release_initialization = threading.Event()
    second_finished = threading.Event()
    attempts = 0

    def fail_then_pause(self, data):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            message = "force lazy initialization"
            raise RuntimeError(message)
        initializing.set()
        assert release_initialization.wait(timeout=2)
        return original_initialize_services(self, data)

    monkeypatch.setattr(ChineseNameDetector, "_initialize_services", fail_then_pause)
    detector = ChineseNameDetector()
    results = []

    first = threading.Thread(target=lambda: results.append(detector.normalize_name("Zhang Wei")))

    def run_second():
        results.append(detector.normalize_name("Li Ming"))
        second_finished.set()

    second = threading.Thread(target=run_second)
    first.start()
    assert initializing.wait(timeout=2)
    second.start()
    try:
        assert not second_finished.wait(timeout=0.05)
    finally:
        release_initialization.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert len(results) == 2
    assert all(result.success for result in results)
    assert detector._data is not None  # noqa: SLF001
    assert detector._batch_analysis_service is not None  # noqa: SLF001


@pytest.mark.parametrize("format_threshold", [-0.01, 1.01, math.nan, math.inf, -math.inf])
def test_batch_apis_reject_invalid_format_threshold(detector, format_threshold):
    """Invalid thresholds surface instead of degrading to per-name fallback."""
    with pytest.raises(ValueError, match="format_threshold"):
        detector.analyze_name_batch(["Zhang Wei"], format_threshold=format_threshold)
    with pytest.raises(ValueError, match="format_threshold"):
        detector.detect_batch_format(["Zhang Wei"], format_threshold=format_threshold)
    with pytest.raises(ValueError, match="format_threshold"):
        detector.analyze_name_batches([], format_threshold=format_threshold)
    with pytest.raises(ValueError, match="format_threshold"):
        detector.analyze_name_batches_strict([], format_threshold=format_threshold)


@pytest.mark.parametrize("minimum_batch_size", [0, -1])
def test_batch_apis_reject_invalid_minimum_batch_size(detector, minimum_batch_size):
    """Non-positive minimum sizes fail even when there are no batches to iterate."""
    with pytest.raises(ValueError, match="minimum_batch_size"):
        detector.analyze_name_batch(["Zhang Wei"], minimum_batch_size=minimum_batch_size)
    with pytest.raises(ValueError, match="minimum_batch_size"):
        detector.analyze_name_batches([], minimum_batch_size=minimum_batch_size)
    with pytest.raises(ValueError, match="minimum_batch_size"):
        detector.analyze_name_batches_strict([], minimum_batch_size=minimum_batch_size)
