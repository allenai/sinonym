"""
Tests for persistent multi-process normalization.
"""

import pytest


TEST_NAMES = [
    "Li Wei",
    "Wang Weiming",
    "巩俐",
    "John Smith",
    "Kim Min-jun",
    "Ou-yang Ming",
    "Duanmu Wenjie",
    "Zhou(Mary)Li",
    "DAN CHEN",
    "Li(Peter)Chen",
]


def _decision_signature(result):
    normalized = result.result if result.success else ""
    return result.success, normalized


def test_process_name_batch_multiprocess_matches_single_process(detector):
    """Multi-process convenience path should match single-process output exactly."""
    expected = [detector.normalize_name(name) for name in TEST_NAMES]
    actual = detector.process_name_batch_multiprocess(
        TEST_NAMES,
        max_workers=2,
        chunk_size=4,
    )

    assert [_decision_signature(result) for result in actual] == [_decision_signature(result) for result in expected]


def test_persistent_multiprocess_pool_can_be_reused(detector):
    """Persistent pool should support repeated batch calls without changing outputs."""
    names_a = TEST_NAMES[:5]
    names_b = TEST_NAMES[5:]

    expected_a = [detector.normalize_name(name) for name in names_a]
    expected_b = [detector.normalize_name(name) for name in names_b]

    with detector.create_persistent_multiprocess_pool(max_workers=2, chunk_size=3) as pool:
        actual_a = pool.normalize_names(names_a)
        actual_b = pool.normalize_names(names_b)

    assert [_decision_signature(result) for result in actual_a] == [_decision_signature(result) for result in expected_a]
    assert [_decision_signature(result) for result in actual_b] == [_decision_signature(result) for result in expected_b]


def test_persistent_multiprocess_pool_rejects_calls_after_close(detector):
    """Closed pool should raise a clear error on subsequent use."""
    pool = detector.create_persistent_multiprocess_pool(max_workers=2, chunk_size=2)
    pool.close()

    with pytest.raises(RuntimeError, match="closed"):
        pool.normalize_names(["Li Wei"])
