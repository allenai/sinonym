"""
Persistent multi-process helpers for Chinese name normalization.

This module provides a cross-platform process pool that keeps worker processes
alive across multiple calls. Each worker initializes one `ChineseNameDetector`
instance and reuses it for all subsequent chunks.
"""

from __future__ import annotations

import pickle
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import get_all_start_methods, get_context
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

    from sinonym.coretypes import BatchParseResult, ChineseNameConfig, ParseResult
    from sinonym.detector import ChineseNameDetector


_T = TypeVar("_T")
_R = TypeVar("_R")
_BatchRequest = tuple[list[str], float, int]
_WORKER_DETECTOR: ChineseNameDetector | None = None
POOL_CLOSED_MESSAGE = "process pool is closed"
WORKER_NOT_INITIALIZED_MESSAGE = "process pool worker is not initialized"


def validate_multiprocess_options(
    *,
    max_workers: int | None = None,
    chunk_size: int = 64,
    mp_start_method: str = "spawn",
) -> None:
    """Validate process-pool knobs without starting worker processes."""
    if max_workers is not None and max_workers < 1:
        message = "max_workers must be >= 1"
        raise ValueError(message)
    if chunk_size < 1:
        message = "chunk_size must be >= 1"
        raise ValueError(message)

    try:
        get_context(mp_start_method)
    except ValueError as exc:
        available = ", ".join(get_all_start_methods())
        message = f"unsupported multiprocessing start method '{mp_start_method}'. Available methods: {available}"
        raise ValueError(message) from exc


def _init_worker(config: ChineseNameConfig | None, weights: tuple[float, ...] | None) -> None:
    """Initialize one detector per worker process."""
    from sinonym.detector import ChineseNameDetector  # noqa: PLC0415

    global _WORKER_DETECTOR  # noqa: PLW0603
    worker_weights = list(weights) if weights is not None else None
    _WORKER_DETECTOR = ChineseNameDetector(config=config, weights=worker_weights)


def _normalize_chunk(names: list[str]) -> list[ParseResult]:
    """Normalize one chunk inside a worker process."""
    if _WORKER_DETECTOR is None:
        raise RuntimeError(WORKER_NOT_INITIALIZED_MESSAGE)
    return [_WORKER_DETECTOR.normalize_name(name) for name in names]


def _process_name_batch_chunk(batch_requests: list[_BatchRequest]) -> list[list[ParseResult]]:
    """Process one chunk of independent name batches inside a worker process."""
    if _WORKER_DETECTOR is None:
        raise RuntimeError(WORKER_NOT_INITIALIZED_MESSAGE)
    return [
        _WORKER_DETECTOR.process_name_batch(
            names,
            format_threshold=format_threshold,
            minimum_batch_size=minimum_batch_size,
        )
        for names, format_threshold, minimum_batch_size in batch_requests
    ]


def _analyze_name_batch_chunk(batch_requests: list[_BatchRequest]) -> list[BatchParseResult]:
    """Analyze one chunk of independent name batches inside a worker process."""
    if _WORKER_DETECTOR is None:
        raise RuntimeError(WORKER_NOT_INITIALIZED_MESSAGE)
    return [
        _WORKER_DETECTOR.analyze_name_batch(
            names,
            format_threshold=format_threshold,
            minimum_batch_size=minimum_batch_size,
        )
        for names, format_threshold, minimum_batch_size in batch_requests
    ]


def _chunk_items(items: list[_T], chunk_size: int) -> list[list[_T]]:
    """Split a list into contiguous chunks."""
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


def _batch_requests(
    batches: list[list[str]],
    *,
    format_threshold: float,
    minimum_batch_size: int,
) -> list[_BatchRequest]:
    """Attach shared batch options to each submitted batch."""
    return [(batch, format_threshold, minimum_batch_size) for batch in batches]


class PersistentMultiprocessNormalizer:
    """
    Persistent cross-platform process pool for name normalization.

    Notes:
    - Uses `spawn` by default for consistent behavior across Windows/macOS/Linux.
    - Keep this pool alive across multiple calls to avoid repeated process start-up.
    """

    def __init__(
        self,
        *,
        max_workers: int | None = None,
        chunk_size: int = 64,
        mp_start_method: str = "spawn",
        detector_config: ChineseNameConfig | None = None,
        detector_weights: list[float] | None = None,
    ) -> None:
        validate_multiprocess_options(
            max_workers=max_workers,
            chunk_size=chunk_size,
            mp_start_method=mp_start_method,
        )

        self._chunk_size = chunk_size
        self._closed = False
        self._config = detector_config
        self._weights = tuple(detector_weights) if detector_weights is not None else None

        self._validate_picklable_state()

        mp_context = get_context(mp_start_method)

        self._executor = ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp_context,
            initializer=_init_worker,
            initargs=(self._config, self._weights),
        )

    def _validate_picklable_state(self) -> None:
        """Validate config and weights can be sent to worker processes."""
        try:
            pickle.dumps((self._config, self._weights))
        except (pickle.PicklingError, AttributeError, TypeError) as exc:
            message = "detector config/weights are not picklable; cannot initialize multiprocessing workers"
            raise ValueError(message) from exc

    @property
    def chunk_size(self) -> int:
        """Chunk size used for worker IPC batching."""
        return self._chunk_size

    @property
    def closed(self) -> bool:
        """Whether the underlying executor has been shut down."""
        return self._closed

    def _map_chunks(self, items: list[_T], worker: Callable[[list[_T]], list[_R]]) -> list[_R]:
        """Map chunked work through the executor and preserve item order."""
        if self._closed:
            raise RuntimeError(POOL_CLOSED_MESSAGE)
        if not items:
            return []

        chunks = _chunk_items(items, self._chunk_size)
        results: list[_R] = []
        try:
            for chunk_result in self._executor.map(worker, chunks, chunksize=1):
                results.extend(chunk_result)
        except BrokenProcessPool as exc:
            message = (
                "the multiprocessing worker pool broke. Either the workers failed to "
                "initialize (on Windows/macOS, call this API from a module guarded by "
                "`if __name__ == '__main__':`) or a worker died mid-batch (for example, "
                "out-of-memory or a native crash)."
            )
            raise RuntimeError(message) from exc
        return results

    def normalize_names(self, names: list[str]) -> list[ParseResult]:
        """
        Normalize names in parallel using persistent worker processes.

        The output list preserves the exact input order.
        """
        return self._map_chunks(names, _normalize_chunk)

    def process_name_batches(
        self,
        batches: list[list[str]],
        *,
        format_threshold: float = 0.55,
        minimum_batch_size: int = 2,
    ) -> list[list[ParseResult]]:
        """
        Process independent name batches in parallel using persistent workers.

        Each inner list is processed with `ChineseNameDetector.process_name_batch()`,
        so batch-format correction is isolated to that list. The outer and inner
        output order both preserve the input order. For this method, `chunk_size`
        controls how many submitted batches are sent to each worker task.
        """
        requests = _batch_requests(
            batches,
            format_threshold=format_threshold,
            minimum_batch_size=minimum_batch_size,
        )
        return self._map_chunks(requests, _process_name_batch_chunk)

    def analyze_name_batches(
        self,
        batches: list[list[str]],
        *,
        format_threshold: float = 0.55,
        minimum_batch_size: int = 2,
    ) -> list[BatchParseResult]:
        """
        Analyze independent name batches in parallel using persistent workers.

        Each inner list is processed with `ChineseNameDetector.analyze_name_batch()`,
        so batch-format correction and evidence are isolated to that list.
        """
        requests = _batch_requests(
            batches,
            format_threshold=format_threshold,
            minimum_batch_size=minimum_batch_size,
        )
        return self._map_chunks(requests, _analyze_name_batch_chunk)

    def close(self) -> None:
        """Shutdown worker processes and release resources."""
        if self._closed:
            return
        self._executor.shutdown(wait=True, cancel_futures=False)
        self._closed = True

    def __enter__(self) -> PersistentMultiprocessNormalizer:  # noqa: PYI034
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def normalize_names_multiprocess(  # noqa: PLR0913
    names: list[str],
    *,
    max_workers: int | None = None,
    chunk_size: int = 64,
    mp_start_method: str = "spawn",
    detector_config: ChineseNameConfig | None = None,
    detector_weights: list[float] | None = None,
) -> list[ParseResult]:
    """Normalize one batch in a temporary multi-process pool."""
    with PersistentMultiprocessNormalizer(
        max_workers=max_workers,
        chunk_size=chunk_size,
        mp_start_method=mp_start_method,
        detector_config=detector_config,
        detector_weights=detector_weights,
    ) as pool:
        return pool.normalize_names(names)
