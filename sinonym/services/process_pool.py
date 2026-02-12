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
from typing import TYPE_CHECKING

from sinonym.coretypes import ChineseNameConfig, ParseResult

if TYPE_CHECKING:
    from sinonym.detector import ChineseNameDetector


_WORKER_DETECTOR: ChineseNameDetector | None = None


def _init_worker(config: ChineseNameConfig | None, weights: tuple[float, ...] | None) -> None:
    """Initialize one detector per worker process."""
    from sinonym.detector import ChineseNameDetector

    global _WORKER_DETECTOR
    worker_weights = list(weights) if weights is not None else None
    _WORKER_DETECTOR = ChineseNameDetector(config=config, weights=worker_weights)


def _normalize_chunk(names: list[str]) -> list[ParseResult]:
    """Normalize one chunk inside a worker process."""
    if _WORKER_DETECTOR is None:
        raise RuntimeError("process pool worker is not initialized")
    return [_WORKER_DETECTOR.normalize_name(name) for name in names]


def _chunk_names(names: list[str], chunk_size: int) -> list[list[str]]:
    """Split a name list into contiguous chunks."""
    return [names[index : index + chunk_size] for index in range(0, len(names), chunk_size)]


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
        if max_workers is not None and max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")

        self._chunk_size = chunk_size
        self._closed = False
        self._config = detector_config
        self._weights = tuple(detector_weights) if detector_weights is not None else None

        self._validate_picklable_state()

        try:
            mp_context = get_context(mp_start_method)
        except ValueError as exc:
            available = ", ".join(get_all_start_methods())
            raise ValueError(
                f"unsupported multiprocessing start method '{mp_start_method}'. Available methods: {available}",
            ) from exc

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
            raise ValueError(
                "detector config/weights are not picklable; cannot initialize multiprocessing workers",
            ) from exc

    @property
    def chunk_size(self) -> int:
        """Chunk size used for worker IPC batching."""
        return self._chunk_size

    @property
    def closed(self) -> bool:
        """Whether the underlying executor has been shut down."""
        return self._closed

    def normalize_names(self, names: list[str]) -> list[ParseResult]:
        """
        Normalize names in parallel using persistent worker processes.

        The output list preserves the exact input order.
        """
        if self._closed:
            raise RuntimeError("process pool is closed")
        if not names:
            return []

        chunks = _chunk_names(names, self._chunk_size)
        results: list[ParseResult] = []
        try:
            for chunk_result in self._executor.map(_normalize_chunk, chunks, chunksize=1):
                results.extend(chunk_result)
        except BrokenProcessPool as exc:
            message = (
                "failed to initialize process workers. On Windows/macOS, call this API "
                "from a module guarded by `if __name__ == '__main__':`."
            )
            raise RuntimeError(message) from exc
        return results

    def close(self) -> None:
        """Shutdown worker processes and release resources."""
        if self._closed:
            return
        self._executor.shutdown(wait=True, cancel_futures=False)
        self._closed = True

    def __enter__(self) -> PersistentMultiprocessNormalizer:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def normalize_names_multiprocess(
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
