#!/usr/bin/env python3
"""
Persistent multi-process benchmark and parity check for sinonym.

This script compares single-process throughput to a persistent process pool and
verifies that multi-process outputs match single-process outputs.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import statistics
import time

from sinonym import ChineseNameDetector
from tests.test_performance import TestChineseNameDetectorPerformance


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile persistent multi-process normalization throughput.")
    parser.add_argument("--names", type=int, default=12000, help="Number of deterministic test names.")
    parser.add_argument("--warmup", type=int, default=3000, help="Warm-up names per worker/process mode.")
    parser.add_argument("--runs", type=int, default=3, help="Number of timed runs per mode.")
    parser.add_argument("--workers", type=int, default=6, help="Number of worker processes for pool mode.")
    parser.add_argument("--chunk-size", type=int, default=64, help="Chunk size for worker IPC batching.")
    return parser.parse_args()


def _accuracy_signature(result) -> tuple[bool, str | tuple[list[str], list[str]]]:
    normalized_output = result.result if result.success else ""
    return result.success, normalized_output


def _time_single_process(detector: ChineseNameDetector, names: list[str], runs: int) -> tuple[list[float], list]:
    run_times: list[float] = []
    baseline_results = []

    for run_idx in range(runs):
        start = time.perf_counter()
        results = [detector.normalize_name(name) for name in names]
        elapsed = time.perf_counter() - start

        if run_idx == 0:
            baseline_results = results
        run_times.append(elapsed)

    return run_times, baseline_results


def _time_persistent_pool(
    detector: ChineseNameDetector,
    names: list[str],
    warmup_names: list[str],
    runs: int,
    workers: int,
    chunk_size: int,
) -> tuple[list[float], list]:
    run_times: list[float] = []
    first_results = []

    with detector.create_persistent_multiprocess_pool(max_workers=workers, chunk_size=chunk_size) as pool:
        if warmup_names:
            pool.normalize_names(warmup_names)

        for run_idx in range(runs):
            start = time.perf_counter()
            results = pool.normalize_names(names)
            elapsed = time.perf_counter() - start

            if run_idx == 0:
                first_results = results
            run_times.append(elapsed)

    return run_times, first_results


def _print_stats(label: str, times: list[float], name_count: int) -> float:
    rates = [name_count / elapsed for elapsed in times]
    mean_rate = statistics.mean(rates)
    median_rate = statistics.median(rates)
    cv = (statistics.stdev(rates) / mean_rate * 100.0) if len(rates) > 1 and mean_rate > 0 else 0.0
    print(f"{label}:")
    print(f"  runs_sec={','.join(f'{t:.4f}' for t in times)}")
    print(f"  rates_names_per_sec={','.join(f'{r:.0f}' for r in rates)}")
    print(f"  mean_rate={mean_rate:.2f} median_rate={median_rate:.2f} cv_percent={cv:.2f}")
    return median_rate


def main() -> int:
    args = _parse_args()

    if args.names < 1:
        raise ValueError("--names must be >= 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.runs < 1:
        raise ValueError("--runs must be >= 1")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be >= 1")

    os.environ.setdefault("PYTHONHASHSEED", "42")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    detector = ChineseNameDetector()
    helper = TestChineseNameDetectorPerformance()
    names = helper.generate_test_names(detector, args.names)
    warmup = min(args.warmup, len(names))
    warmup_names = names[:warmup]

    for name in warmup_names:
        detector.normalize_name(name)

    print("=" * 88)
    print("SINONYM PERSISTENT MULTI-PROCESS PROFILE")
    print("=" * 88)
    print(
        f"names={len(names)} warmup={warmup} runs={args.runs} workers={args.workers} chunk_size={args.chunk_size}",
    )
    print()

    single_times, baseline_results = _time_single_process(detector, names, args.runs)
    pool_times, pool_results = _time_persistent_pool(
        detector,
        names,
        warmup_names,
        args.runs,
        args.workers,
        args.chunk_size,
    )

    baseline_signature = [_accuracy_signature(result) for result in baseline_results]
    pool_signature = [_accuracy_signature(result) for result in pool_results]
    outputs_match = baseline_signature == pool_signature

    single_median_rate = _print_stats("single_process", single_times, len(names))
    print()
    pool_median_rate = _print_stats("persistent_pool", pool_times, len(names))
    print()
    print(f"decision_outputs_match={outputs_match}")
    print(f"speedup_median={pool_median_rate / single_median_rate:.2f}x")

    return 0 if outputs_match else 1


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
