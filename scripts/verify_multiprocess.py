#!/usr/bin/env python3
"""Cross-platform multiprocessing correctness and throughput verifier."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sinonym import ChineseNameDetector
from tests.test_performance import TestChineseNameDetectorPerformance

BASE_BATCHES = [
    ["Wang An", "Yan Li", "Wu Gang", "Li Bao"],
    ["Zhang Wei", "Li Ming", "Bei Yu", "Wang Xiaoli"],
    ["Li Wei", "Wang Weiming", "Zhang Ming", "Chen Huang"],
    ["Duanmu Wenjie", "Ou-yang Ming", "DAN CHEN", "Li(Peter)Chen"],
]


@dataclass(frozen=True)
class TimingStats:
    label: str
    unit: str
    unit_count: int
    runs_sec: list[float]
    rates_per_sec: list[float]
    mean_rate: float
    median_rate: float
    cv_percent: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Sinonym multiprocessing correctness and throughput.")
    parser.add_argument("--names", type=int, default=6000, help="Number of deterministic flat names.")
    parser.add_argument("--batches", type=int, default=1000, help="Number of deterministic author-list batches.")
    parser.add_argument("--warmup-names", type=int, default=1000, help="Flat names used to warm caches/workers.")
    parser.add_argument("--warmup-batches", type=int, default=100, help="Batches used to warm caches/workers.")
    parser.add_argument("--runs", type=int, default=3, help="Timed runs per measured path.")
    parser.add_argument("--workers", type=int, default=6, help="Process workers for MP paths.")
    parser.add_argument("--name-chunk-size", type=int, default=64, help="Names per worker task.")
    parser.add_argument("--batch-chunk-size", type=int, default=32, help="Author-list batches per worker task.")
    parser.add_argument("--start-method", default="auto", help="multiprocessing start method, or 'auto'.")
    parser.add_argument("--min-persistent-name-speedup", type=float, default=1.25)
    parser.add_argument("--min-persistent-batch-speedup", type=float, default=1.10)
    parser.add_argument("--min-auto-name-speedup", type=float, default=0.0)
    parser.add_argument("--min-auto-batch-speedup", type=float, default=0.0)
    parser.add_argument("--json-output", type=Path, default=None, help="Optional path for machine-readable metrics.")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    positive_fields = (
        "names",
        "batches",
        "runs",
        "workers",
        "name_chunk_size",
        "batch_chunk_size",
    )
    for field in positive_fields:
        if getattr(args, field) < 1:
            message = f"--{field.replace('_', '-')} must be >= 1"
            raise ValueError(message)
    if args.warmup_names < 0:
        message = "--warmup-names must be >= 0"
        raise ValueError(message)
    if args.warmup_batches < 0:
        message = "--warmup-batches must be >= 0"
        raise ValueError(message)


def _result_signature(result) -> tuple[bool, str]:
    return result.success, result.result if result.success else ""


def _batch_result_signature(batch_result) -> tuple[Any, ...]:
    pattern = batch_result.format_pattern
    return (
        tuple(batch_result.names),
        tuple(_result_signature(result) for result in batch_result.results),
        tuple(batch_result.improvements),
        pattern.dominant_format.value,
        pattern.threshold_met,
        pattern.surname_first_count,
        pattern.given_first_count,
        pattern.total_count,
    )


def _result_list_signature(results) -> tuple[tuple[bool, str], ...]:
    return tuple(_result_signature(result) for result in results)


def _batch_list_signature(batch_results) -> tuple[tuple[Any, ...], ...]:
    return tuple(_batch_result_signature(batch_result) for batch_result in batch_results)


def _make_batches(count: int) -> list[list[str]]:
    return [list(BASE_BATCHES[index % len(BASE_BATCHES)]) for index in range(count)]


def _rate_per_second(unit_count: int, elapsed: float) -> float:
    """Return throughput while guarding timer-resolution zero elapsed values."""
    measured_elapsed = max(elapsed, time.get_clock_info("perf_counter").resolution)
    return unit_count / measured_elapsed


def _time_runs(label: str, unit: str, unit_count: int, runs: int, fn) -> tuple[TimingStats, Any]:
    run_times: list[float] = []
    first_result = None
    for run_index in range(runs):
        start = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - start
        if run_index == 0:
            first_result = result
        run_times.append(elapsed)

    rates = [_rate_per_second(unit_count, elapsed) for elapsed in run_times]
    mean_rate = statistics.mean(rates)
    median_rate = statistics.median(rates)
    cv_percent = (statistics.stdev(rates) / mean_rate * 100.0) if len(rates) > 1 and mean_rate > 0 else 0.0
    return (
        TimingStats(
            label=label,
            unit=unit,
            unit_count=unit_count,
            runs_sec=run_times,
            rates_per_sec=rates,
            mean_rate=mean_rate,
            median_rate=median_rate,
            cv_percent=cv_percent,
        ),
        first_result,
    )


def _print_stats(stats: TimingStats) -> None:
    print(f"{stats.label}:")
    print(f"  runs_sec={','.join(f'{value:.4f}' for value in stats.runs_sec)}")
    print(f"  rates_{stats.unit}_per_sec={','.join(f'{value:.0f}' for value in stats.rates_per_sec)}")
    print(f"  mean_rate={stats.mean_rate:.2f} median_rate={stats.median_rate:.2f} cv_percent={stats.cv_percent:.2f}")


def _record_check(checks: dict[str, bool], label: str, *, passed: bool) -> None:
    checks[label] = passed
    print(f"{label}={'PASS' if passed else 'FAIL'}")


def _speedup(label: str, numerator: TimingStats, denominator: TimingStats) -> float:
    value = numerator.median_rate / denominator.median_rate if denominator.median_rate > 0 else 0.0
    print(f"{label}={value:.2f}x")
    return value


def _resolve_start_method(start_method: str) -> str:
    """Resolve CLI-level auto start method for low-level pool checks."""
    if start_method != "auto":
        return start_method
    return "fork" if platform.system() == "Linux" else "spawn"


def _jsonable_config(args: argparse.Namespace) -> dict[str, Any]:
    """Return argparse config with paths converted for JSON output."""
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def main() -> int:  # noqa: PLR0915
    args = _parse_args()
    _validate_args(args)
    resolved_start_method = _resolve_start_method(args.start_method)

    os.environ.setdefault("PYTHONHASHSEED", "42")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    detector = ChineseNameDetector()
    helper = TestChineseNameDetectorPerformance()
    names = helper.generate_test_names(detector, args.names)
    batches = _make_batches(args.batches)
    warmup_names = names[: min(args.warmup_names, len(names))]
    warmup_batches = batches[: min(args.warmup_batches, len(batches))]

    print("=" * 88)
    print("SINONYM MULTIPROCESS VERIFIER")
    print("=" * 88)
    print(
        f"platform={platform.platform()} python={sys.version.split()[0]} "
        f"start_method={args.start_method} resolved_start_method={resolved_start_method} "
        f"names={len(names)} batches={len(batches)} runs={args.runs} workers={args.workers}",
    )
    print()

    for name in warmup_names:
        detector.normalize_name(name)
    for batch in warmup_batches:
        detector.process_name_batch(batch)

    checks: dict[str, bool] = {}
    timings: dict[str, TimingStats] = {}

    local_names_stats, local_name_results = _time_runs(
        "local_normalize_names",
        "names",
        len(names),
        args.runs,
        lambda: detector.normalize_names(names, parallel="never"),
    )
    timings[local_names_stats.label] = local_names_stats
    _print_stats(local_names_stats)

    auto_names_stats, auto_name_results = _time_runs(
        "auto_normalize_names_total_call",
        "names",
        len(names),
        args.runs,
        lambda: detector.normalize_names(
            names,
            parallel="auto",
            min_parallel_names=1,
            max_workers=args.workers,
            chunk_size=args.name_chunk_size,
            mp_start_method=args.start_method,
        ),
    )
    timings[auto_names_stats.label] = auto_names_stats
    _print_stats(auto_names_stats)

    with detector.create_persistent_multiprocess_pool(
        max_workers=args.workers,
        chunk_size=args.name_chunk_size,
        mp_start_method=resolved_start_method,
    ) as pool:
        if warmup_names:
            pool.normalize_names(warmup_names)
        persistent_names_stats, persistent_name_results = _time_runs(
            "persistent_pool_normalize_names_reused",
            "names",
            len(names),
            args.runs,
            lambda: pool.normalize_names(names),
        )
    timings[persistent_names_stats.label] = persistent_names_stats
    _print_stats(persistent_names_stats)

    one_shot_batch = BASE_BATCHES[0]
    one_shot_results = detector.process_name_batch_multiprocess(
        one_shot_batch,
        max_workers=args.workers,
        chunk_size=args.name_chunk_size,
        mp_start_method=resolved_start_method,
    )

    local_name_sig = _result_list_signature(local_name_results)
    _record_check(checks, "auto_names_match_local", passed=_result_list_signature(auto_name_results) == local_name_sig)
    _record_check(
        checks,
        "persistent_names_match_local",
        passed=_result_list_signature(persistent_name_results) == local_name_sig,
    )
    _record_check(
        checks,
        "one_shot_batch_matches_local",
        passed=_result_list_signature(one_shot_results)
        == _result_list_signature(detector.process_name_batch(one_shot_batch)),
    )
    print()

    local_batch_stats, local_batch_results = _time_runs(
        "local_analyze_name_batches",
        "batches",
        len(batches),
        args.runs,
        lambda: detector.analyze_name_batches(batches, parallel="never"),
    )
    timings[local_batch_stats.label] = local_batch_stats
    _print_stats(local_batch_stats)

    auto_batch_stats, auto_batch_results = _time_runs(
        "auto_analyze_name_batches_total_call",
        "batches",
        len(batches),
        args.runs,
        lambda: detector.analyze_name_batches(
            batches,
            parallel="auto",
            min_parallel_batches=1,
            max_workers=args.workers,
            chunk_size=args.batch_chunk_size,
            mp_start_method=args.start_method,
        ),
    )
    timings[auto_batch_stats.label] = auto_batch_stats
    _print_stats(auto_batch_stats)

    with detector.create_persistent_multiprocess_pool(
        max_workers=args.workers,
        chunk_size=args.batch_chunk_size,
        mp_start_method=resolved_start_method,
    ) as pool:
        if warmup_batches:
            pool.analyze_name_batches(warmup_batches)
        persistent_batch_stats, persistent_batch_results = _time_runs(
            "persistent_pool_analyze_name_batches_reused",
            "batches",
            len(batches),
            args.runs,
            lambda: pool.analyze_name_batches(batches),
        )
    timings[persistent_batch_stats.label] = persistent_batch_stats
    _print_stats(persistent_batch_stats)

    local_batch_sig = _batch_list_signature(local_batch_results)
    _record_check(checks, "auto_batches_match_local", passed=_batch_list_signature(auto_batch_results) == local_batch_sig)
    _record_check(
        checks,
        "persistent_batches_match_local",
        passed=_batch_list_signature(persistent_batch_results) == local_batch_sig,
    )

    persistent_name_speedup = _speedup("persistent_name_speedup", persistent_names_stats, local_names_stats)
    persistent_batch_speedup = _speedup("persistent_batch_speedup", persistent_batch_stats, local_batch_stats)
    auto_name_speedup = _speedup("auto_name_total_call_speedup", auto_names_stats, local_names_stats)
    auto_batch_speedup = _speedup("auto_batch_total_call_speedup", auto_batch_stats, local_batch_stats)

    _record_check(
        checks,
        "persistent_name_speedup_gate",
        passed=persistent_name_speedup >= args.min_persistent_name_speedup,
    )
    _record_check(
        checks,
        "persistent_batch_speedup_gate",
        passed=persistent_batch_speedup >= args.min_persistent_batch_speedup,
    )
    _record_check(checks, "auto_name_speedup_gate", passed=auto_name_speedup >= args.min_auto_name_speedup)
    _record_check(checks, "auto_batch_speedup_gate", passed=auto_batch_speedup >= args.min_auto_batch_speedup)

    summary = {
        "checks": checks,
        "speedups": {
            "persistent_name": persistent_name_speedup,
            "persistent_batch": persistent_batch_speedup,
            "auto_name_total_call": auto_name_speedup,
            "auto_batch_total_call": auto_batch_speedup,
        },
        "timings": {label: asdict(stats) for label, stats in timings.items()},
        "config": _jsonable_config(args),
        "platform": {
            "platform": platform.platform(),
            "python": sys.version,
            "start_method": args.start_method,
            "resolved_start_method": resolved_start_method,
        },
    }

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    all_passed = all(checks.values())
    print()
    print(f"overall={'PASS' if all_passed else 'FAIL'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
