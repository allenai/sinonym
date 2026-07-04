"""
Tests for persistent multi-process normalization.
"""

from concurrent.futures.process import BrokenProcessPool

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
TEMP_POOL_WORKERS = 3
TEMP_POOL_CHUNK_SIZE = 7
CUSTOM_FORMAT_THRESHOLD = 0.7
CUSTOM_MINIMUM_BATCH_SIZE = 3


def _decision_signature(result):
    normalized = result.result if result.success else ""
    return result.success, normalized


def _decision_signatures(results):
    return [_decision_signature(result) for result in results]


def _batch_decision_signatures(batch_results):
    return [_decision_signatures(results) for results in batch_results]


def test_process_name_batch_multiprocess_matches_per_name_normalization(detector):
    """One-shot multi-process normalization should match per-name normalize_name semantics."""
    expected = [detector.normalize_name(name) for name in TEST_NAMES]
    actual = detector.process_name_batch_multiprocess(
        TEST_NAMES,
        max_workers=2,
        chunk_size=4,
    )

    assert _decision_signatures(actual) == _decision_signatures(expected)


def test_process_name_batch_multiprocess_does_not_apply_batch_overrides(detector):
    """One-shot multi-process normalization is parallel per-name parsing, not batch correction."""
    names = ["Wang An", "Yan Li", "Wu Gang", "Li Bao"]
    expected = [detector.normalize_name(name) for name in names]
    batch = detector.process_name_batch(names)
    actual = detector.process_name_batch_multiprocess(names, max_workers=2, chunk_size=2)

    assert [result.result for result in expected] != [result.result for result in batch]
    assert _decision_signatures(actual) == _decision_signatures(expected)


def test_process_name_batch_multiprocess_uses_temporary_process_pool(detector, monkeypatch):
    """Wrapper should delegate to the real temporary process-pool helper with detector state."""
    captured = {}
    expected = [detector.normalize_name("Li Wei")]

    def fake_normalize_names_multiprocess(names, **kwargs):
        captured["names"] = names
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(
        "sinonym.detector.normalize_names_multiprocess",
        fake_normalize_names_multiprocess,
    )

    actual = detector.process_name_batch_multiprocess(
        ["Li Wei"],
        max_workers=TEMP_POOL_WORKERS,
        chunk_size=TEMP_POOL_CHUNK_SIZE,
        mp_start_method="spawn",
    )

    assert actual is expected
    assert captured["names"] == ["Li Wei"]
    assert captured["max_workers"] == TEMP_POOL_WORKERS
    assert captured["chunk_size"] == TEMP_POOL_CHUNK_SIZE
    assert captured["mp_start_method"] == "spawn"
    assert captured["detector_config"] is detector._config  # noqa: SLF001
    assert captured["detector_weights"] == detector._weights  # noqa: SLF001


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_workers": 0}, "max_workers"),
        ({"chunk_size": 0}, "chunk_size"),
        ({"mp_start_method": "not_a_method"}, "start method"),
    ],
)
def test_process_name_batch_multiprocess_rejects_invalid_pool_options(detector, kwargs, message):
    """One-shot multi-process normalization should surface invalid pool options."""
    with pytest.raises(ValueError, match=message):
        detector.process_name_batch_multiprocess(["Li Wei"], **kwargs)


def test_persistent_multiprocess_pool_can_be_reused(detector):
    """Persistent pool should support repeated normalization calls without changing outputs."""
    names_a = TEST_NAMES[:5]
    names_b = TEST_NAMES[5:]

    expected_a = [detector.normalize_name(name) for name in names_a]
    expected_b = [detector.normalize_name(name) for name in names_b]

    with detector.create_persistent_multiprocess_pool(max_workers=2, chunk_size=3) as pool:
        actual_a = pool.normalize_names(names_a)
        actual_b = pool.normalize_names(names_b)

    assert _decision_signatures(actual_a) == _decision_signatures(expected_a)
    assert _decision_signatures(actual_b) == _decision_signatures(expected_b)


def test_persistent_multiprocess_pool_processes_batches_with_batch_context(detector):
    """Persistent pool should process many independent batches without losing batch correction."""
    batches = [
        ["Wang An", "Yan Li", "Wu Gang", "Li Bao"],
        TEST_NAMES[:5],
        TEST_NAMES[5:],
    ]
    individual_first_batch = [detector.normalize_name(name) for name in batches[0]]
    expected = [detector.process_name_batch(batch) for batch in batches]

    with detector.create_persistent_multiprocess_pool(max_workers=2, chunk_size=1) as pool:
        actual = pool.process_name_batches(batches)

    assert [result.result for result in individual_first_batch] != [result.result for result in expected[0]]
    assert _batch_decision_signatures(actual) == _batch_decision_signatures(expected)


def test_persistent_multiprocess_pool_analyzes_batches_with_batch_evidence(detector):
    """Persistent pool should preserve full analyze_name_batch results for independent batches."""
    batches = [
        ["Wang An", "Yan Li", "Wu Gang", "Li Bao"],
        TEST_NAMES[:5],
    ]
    expected = [detector.analyze_name_batch(batch) for batch in batches]

    with detector.create_persistent_multiprocess_pool(max_workers=2, chunk_size=1) as pool:
        actual = pool.analyze_name_batches(batches)

    assert [result.names for result in actual] == [result.names for result in expected]
    assert [result.improvements for result in actual] == [result.improvements for result in expected]
    assert _batch_decision_signatures([result.results for result in actual]) == _batch_decision_signatures(
        [result.results for result in expected],
    )


def test_persistent_multiprocess_pool_process_name_batches_handles_empty_outer_list(detector):
    """Persistent batch processing should return no results for no submitted batches."""
    with detector.create_persistent_multiprocess_pool(max_workers=2, chunk_size=1) as pool:
        assert pool.process_name_batches([]) == []


def test_detector_normalize_names_auto_uses_local_path_when_worker_count_is_one(detector, monkeypatch):
    """Auto per-name normalization should avoid MP when max_workers=1."""

    def fail_create_pool(**_kwargs):
        message = "pool should not be created"
        raise AssertionError(message)

    monkeypatch.setattr(detector, "create_persistent_multiprocess_pool", fail_create_pool)

    actual = detector.normalize_names(TEST_NAMES, parallel="auto", min_parallel_names=1, max_workers=1)
    expected = [detector.normalize_name(name) for name in TEST_NAMES]

    assert _decision_signatures(actual) == _decision_signatures(expected)


def test_detector_process_name_batches_auto_uses_pool_above_threshold(detector, monkeypatch):
    """Auto batch processing should delegate to the persistent pool once the threshold is met."""
    captured = {}
    expected_batch = detector.analyze_name_batch(
        ["Li Wei"],
        format_threshold=CUSTOM_FORMAT_THRESHOLD,
        minimum_batch_size=CUSTOM_MINIMUM_BATCH_SIZE,
    )

    class FakePool:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            captured["closed"] = True

        def analyze_name_batches(self, batches, **kwargs):
            captured["batches"] = batches
            captured.update(kwargs)
            return [expected_batch]

    def fake_create_pool(**kwargs):
        captured["pool_kwargs"] = kwargs
        return FakePool()

    monkeypatch.setattr(detector, "create_persistent_multiprocess_pool", fake_create_pool)

    actual = detector.process_name_batches(
        [["Li Wei"]],
        parallel="auto",
        min_parallel_batches=1,
        format_threshold=CUSTOM_FORMAT_THRESHOLD,
        minimum_batch_size=CUSTOM_MINIMUM_BATCH_SIZE,
        max_workers=TEMP_POOL_WORKERS,
        chunk_size=TEMP_POOL_CHUNK_SIZE,
    )

    assert actual == [expected_batch.results]
    assert captured["batches"] == [["Li Wei"]]
    assert captured["format_threshold"] == CUSTOM_FORMAT_THRESHOLD
    assert captured["minimum_batch_size"] == CUSTOM_MINIMUM_BATCH_SIZE
    assert captured["pool_kwargs"]["max_workers"] == TEMP_POOL_WORKERS
    assert captured["pool_kwargs"]["chunk_size"] == TEMP_POOL_CHUNK_SIZE
    assert captured["closed"]


def test_detector_parallel_wrappers_reject_invalid_parallel_mode(detector):
    """High-level wrappers should reject unknown parallel modes explicitly."""
    with pytest.raises(ValueError, match="parallel"):
        detector.normalize_names(["Li Wei"], parallel="sometimes")

    with pytest.raises(ValueError, match="parallel"):
        detector.process_name_batches([["Li Wei"]], parallel="sometimes")


def test_persistent_multiprocess_pool_rejects_calls_after_close(detector):
    """Closed pool should raise a clear error on subsequent use."""
    pool = detector.create_persistent_multiprocess_pool(max_workers=2, chunk_size=2)
    pool.close()

    with pytest.raises(RuntimeError, match="closed"):
        pool.normalize_names(["Li Wei"])

    with pytest.raises(RuntimeError, match="closed"):
        pool.process_name_batches([["Li Wei"]])


def test_persistent_multiprocess_pool_surfaces_broken_pool_causes(detector):
    """A broken worker pool surfaces both init failure and mid-batch worker death."""

    class _BrokenExecutor:
        """Stand-in executor whose map fails as a broken pool would."""

        def map(self, *_args, **_kwargs):
            raise BrokenProcessPool

    pool = detector.create_persistent_multiprocess_pool(max_workers=2, chunk_size=2)
    pool.close()  # Shut the real workers down; we only exercise the error handler.
    pool._closed = False  # noqa: SLF001 - reopen so normalize_names reaches executor.map.
    pool._executor = _BrokenExecutor()  # noqa: SLF001 - inject the broken executor.

    with pytest.raises(RuntimeError) as excinfo:
        pool.normalize_names(["Li Wei"])

    message = str(excinfo.value)
    assert "initialize" in message  # initialization-failure cause
    assert "mid-batch" in message  # worker-death-mid-run cause
    assert isinstance(excinfo.value.__cause__, BrokenProcessPool)  # original chained
