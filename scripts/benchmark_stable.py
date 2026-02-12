#!/usr/bin/env python3
"""
Stable benchmark runner for sinonym with median-based gating.

This script runs multiple isolated worker measurements (fresh subprocess
per run) so hash seed and process-level state are controlled, then reports
mean/median/CV and supports a minimum median throughput gate.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import time

from sinonym.detector import ChineseNameDetector
from tests.test_performance import TestChineseNameDetectorPerformance


def _run_worker(names_count: int, warmup_count: int) -> dict[str, float | int]:
    """Run one isolated benchmark measurement in-process."""
    detector = ChineseNameDetector()
    helper = TestChineseNameDetectorPerformance()
    names = helper.generate_test_names(detector, names_count)

    warmup = min(warmup_count, len(names))
    for name in names[:warmup]:
        detector.normalize_name(name)

    gc.collect()
    gc_enabled = gc.isenabled()
    if gc_enabled:
        gc.disable()

    try:
        start = time.perf_counter()
        for name in names:
            detector.normalize_name(name)
        end = time.perf_counter()
    finally:
        if gc_enabled:
            gc.enable()

    elapsed = end - start
    names_per_sec = len(names) / elapsed if elapsed > 0 else 0.0
    return {
        "elapsed_seconds": elapsed,
        "names_per_second": names_per_sec,
        "name_count": len(names),
    }


def _run_subprocess_worker(args: argparse.Namespace, run_idx: int) -> dict[str, float | int]:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(args.hash_seed)
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")

    cmd = [
        sys.executable,
        __file__,
        "--worker",
        "--names",
        str(args.names),
        "--warmup",
        str(args.warmup),
    ]
    result = subprocess.run(  # noqa: S603
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        env=env,
    )
    if result.returncode != 0:
        message = (
            f"Worker {run_idx} failed with code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        raise RuntimeError(message)

    for raw_line in reversed(result.stdout.splitlines()):
        candidate = raw_line.strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            return json.loads(candidate)

    message = f"Worker {run_idx} did not emit JSON output.\nSTDOUT:\n{result.stdout}"
    raise RuntimeError(message)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stable benchmark runner with median gate.")
    parser.add_argument("--runs", type=int, default=5, help="Number of isolated runs.")
    parser.add_argument("--names", type=int, default=3000, help="Number of deterministic test names.")
    parser.add_argument(
        "--warmup",
        type=int,
        default=3000,
        help="Number of names to pre-warm before timing in each run.",
    )
    parser.add_argument(
        "--hash-seed",
        type=int,
        default=42,
        help="PYTHONHASHSEED used for each worker subprocess.",
    )
    parser.add_argument(
        "--min-median-names-per-sec",
        type=float,
        default=0.0,
        help="Optional gate: fail (exit 1) when median names/sec is below this value.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()

    if args.worker:
        payload = _run_worker(args.names, args.warmup)
        print(json.dumps(payload))
        return 0

    if args.runs < 1:
        message = "--runs must be >= 1"
        raise ValueError(message)

    print("=" * 72)
    print("SINONYM STABLE BENCHMARK")
    print("=" * 72)
    print(f"runs={args.runs} names={args.names} warmup={args.warmup} hash_seed={args.hash_seed}")
    print()

    run_payloads: list[dict[str, float | int]] = []
    for run_idx in range(1, args.runs + 1):
        payload = _run_subprocess_worker(args, run_idx)
        run_payloads.append(payload)
        print(
            f"run {run_idx}: "
            f"{payload['elapsed_seconds']:.6f}s | "
            f"{payload['names_per_second']:.0f} names/sec",
        )

    rates = [float(payload["names_per_second"]) for payload in run_payloads]
    elapsed = [float(payload["elapsed_seconds"]) for payload in run_payloads]

    median_rate = statistics.median(rates)
    mean_rate = statistics.mean(rates)
    stdev_rate = statistics.stdev(rates) if len(rates) > 1 else 0.0
    cv_rate = (stdev_rate / mean_rate * 100.0) if mean_rate else 0.0

    mean_elapsed = statistics.mean(elapsed)
    median_elapsed = statistics.median(elapsed)

    print()
    print("Summary")
    print("-" * 72)
    print(f"rate_mean_names_per_second={mean_rate:.2f}")
    print(f"rate_median_names_per_second={median_rate:.2f}")
    print(f"rate_stddev_names_per_second={stdev_rate:.2f}")
    print(f"rate_cv_percent={cv_rate:.2f}")
    print(f"elapsed_mean_seconds={mean_elapsed:.6f}")
    print(f"elapsed_median_seconds={median_elapsed:.6f}")
    print(f"runs_names_per_second={','.join(f'{r:.0f}' for r in rates)}")
    print()
    print(f"MEDIAN_NAMES_PER_SECOND={median_rate:.2f}")

    if args.min_median_names_per_sec > 0 and median_rate < args.min_median_names_per_sec:
        print(
            "GATE=FAIL "
            f"(median {median_rate:.2f} < required {args.min_median_names_per_sec:.2f})",
        )
        return 1

    print("GATE=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
