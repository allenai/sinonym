"""
Tests for persistent multi-process normalization.
"""

import typing
from concurrent.futures.process import BrokenProcessPool

import pytest

from scripts.verify_multiprocess import _rate_per_second
from sinonym import detector as detector_module
from sinonym.detector import _should_use_multiprocessing
from sinonym.services import process_pool

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


def test_verify_multiprocess_rate_per_second_handles_zero_elapsed():
    assert _rate_per_second(unit_count=1, elapsed=0.0) > 0


def _batch_decision_signatures(batch_results):
    return [_decision_signatures(results) for results in batch_results]


def test_process_pool_public_annotations_resolve_at_runtime():
    """Public process-pool APIs should support runtime type introspection."""
    hints = typing.get_type_hints(process_pool.normalize_names_multiprocess)

    assert "detector_config" in hints
    assert "return" in hints


def test_process_name_batch_multiprocess_matches_batch_context(detector):
    """One-shot multi-process batch processing should match process_name_batch semantics."""
    expected = detector.process_name_batch(TEST_NAMES)
    actual = detector.process_name_batch_multiprocess(
        TEST_NAMES,
        max_workers=2,
        chunk_size=4,
    )

    assert _decision_signatures(actual) == _decision_signatures(expected)


def test_process_name_batch_multiprocess_preserves_batch_overrides(detector):
    """One-shot multi-process batch processing should preserve batch correction."""
    names = ["Wang An", "Yan Li", "Wu Gang", "Li Bao"]
    individual_results = [detector.normalize_name(name) for name in names]
    expected = detector.process_name_batch(names)
    actual = detector.process_name_batch_multiprocess(names, max_workers=2, chunk_size=2)

    assert [result.result for result in individual_results] != [result.result for result in expected]
    assert _decision_signatures(actual) == _decision_signatures(expected)


def test_process_name_batch_multiprocess_uses_temporary_process_pool(detector, monkeypatch):
    """Wrapper should delegate to process_name_batches through a temporary pool."""
    captured = {}
    expected = [detector.process_name_batch(["Li Wei"])[0]]

    def fake_process_name_batches(batches, **kwargs):
        captured["batches"] = batches
        captured.update(kwargs)
        return [expected]

    monkeypatch.setattr(detector, "process_name_batches", fake_process_name_batches)

    actual = detector.process_name_batch_multiprocess(
        ["Li Wei"],
        max_workers=TEMP_POOL_WORKERS,
        chunk_size=TEMP_POOL_CHUNK_SIZE,
        mp_start_method="spawn",
    )

    assert actual is expected
    assert captured["batches"] == [["Li Wei"]]
    assert captured["parallel"] == "always"
    assert captured["min_parallel_batches"] == 1
    assert captured["max_workers"] == TEMP_POOL_WORKERS
    assert captured["chunk_size"] == TEMP_POOL_CHUNK_SIZE
    assert captured["mp_start_method"] == "spawn"


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


def test_multiprocessing_threshold_is_validated_only_for_auto_mode():
    """Manual parallel modes should not validate an unused auto threshold."""
    common = {
        "item_count": 1,
        "auto_threshold": 0,
        "default_auto_threshold": 2,
        "linux_auto_threshold": 2,
        "max_workers": None,
    }

    assert not _should_use_multiprocessing(parallel="never", **common)
    assert _should_use_multiprocessing(parallel="always", **common)
    with pytest.raises(ValueError, match="threshold"):
        _should_use_multiprocessing(parallel="auto", **common)


def test_high_level_auto_start_method_prefers_spawn_on_linux(detector, monkeypatch):
    """High-level auto should not fork by default inside serving processes."""
    captured = {}

    class FakePool:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            captured["closed"] = True

        @staticmethod
        def normalize_names(names):
            return []

    def fake_create_pool(**kwargs):
        captured["pool_kwargs"] = kwargs
        return FakePool()

    monkeypatch.setattr(detector_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(detector, "create_persistent_multiprocess_pool", fake_create_pool)

    detector.normalize_names(["Li Wei"], parallel="auto", min_parallel_names=1, max_workers=2)

    assert captured["pool_kwargs"]["mp_start_method"] == "spawn"
    assert captured["closed"]


def test_persistent_multiprocess_pool_rejects_calls_after_close(detector):
    """Closed pool should raise a clear error on subsequent use."""
    pool = detector.create_persistent_multiprocess_pool(max_workers=2, chunk_size=2)
    pool.close()

    with pytest.raises(RuntimeError, match="closed"):
        pool.normalize_names(["Li Wei"])


def test_persistent_multiprocess_pool_wraps_worker_task_errors(detector):
    """Ordinary worker exceptions should include pool context and close the pool."""

    class _FailingExecutor:
        def __init__(self):
            self.shutdown_calls = []

        def map(self, *_args, **_kwargs):
            message = "worker exploded"
            raise ValueError(message)

        def shutdown(self, **kwargs):
            self.shutdown_calls.append(kwargs)

    pool = detector.create_persistent_multiprocess_pool(max_workers=2, chunk_size=2)
    pool.close()
    pool._closed = False  # noqa: SLF001 - force the injected executor path.
    failing_executor = _FailingExecutor()
    pool._executor = failing_executor  # noqa: SLF001 - inject failing executor.

    with pytest.raises(RuntimeError, match="multiprocessing worker task failed") as excinfo:
        pool.normalize_names(["Li Wei", "Wang Wei", "Chen Ming"])

    assert isinstance(excinfo.value.__cause__, ValueError)
    assert "3 item" in str(excinfo.value)
    assert "2 chunk" in str(excinfo.value)
    assert pool.closed
    assert failing_executor.shutdown_calls == [{"wait": False, "cancel_futures": True}]

    with pytest.raises(RuntimeError, match="closed"):
        pool.process_name_batches([["Li Wei"]])


def test_persistent_multiprocess_pool_surfaces_broken_pool_causes(detector):
    """A broken worker pool surfaces both init failure and mid-batch worker death."""

    class _BrokenExecutor:
        """Stand-in executor whose map fails as a broken pool would."""

        def __init__(self):
            self.shutdown_calls = []

        def map(self, *_args, **_kwargs):
            raise BrokenProcessPool

        def shutdown(self, **kwargs):
            self.shutdown_calls.append(kwargs)

    pool = detector.create_persistent_multiprocess_pool(max_workers=2, chunk_size=2)
    pool.close()  # Shut the real workers down; we only exercise the error handler.
    pool._closed = False  # noqa: SLF001 - reopen so normalize_names reaches executor.map.
    broken_executor = _BrokenExecutor()
    pool._executor = broken_executor  # noqa: SLF001 - inject the broken executor.

    with pytest.raises(RuntimeError) as excinfo:
        pool.normalize_names(["Li Wei"])

    message = str(excinfo.value)
    assert "initialize" in message  # initialization-failure cause
    assert "mid-batch" in message  # worker-death-mid-run cause
    assert isinstance(excinfo.value.__cause__, BrokenProcessPool)  # original chained
    assert pool.closed
    assert broken_executor.shutdown_calls == [{"wait": False, "cancel_futures": True}]
    with pytest.raises(RuntimeError, match="closed"):
        pool.normalize_names(["Li Wei"])
