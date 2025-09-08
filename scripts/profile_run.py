#!/usr/bin/env python3
"""
Low-variance performance profiling script for sinonym.

This script minimizes measurement variance by:
- Setting deterministic hash seed
- Using reproducible test data generation
- Pre-warming caches to eliminate cold start effects
- Taking multiple measurements and reporting statistics
- Forcing garbage collection between runs
"""

import cProfile
import gc
import os
import pstats
import statistics
import time

# Set deterministic hash seed for consistent dict iteration
os.environ["PYTHONHASHSEED"] = "42"

from sinonym.detector import ChineseNameDetector
from tests.test_performance import TestChineseNameDetectorPerformance


def run_single_measurement(detector, names, enable_profiling=True):
    """Run a single performance measurement with optional profiling."""
    gc.collect()  # Clean state between runs

    if enable_profiling:
        pr = cProfile.Profile()
        pr.enable()

    start = time.perf_counter()
    for name in names:
        detector.is_chinese_name(name)
    end = time.perf_counter()

    if enable_profiling:
        pr.disable()
        return end - start, pr
    return end - start, None


def main():
    print("Sinonym Performance Profiler (Low Variance)")
    print("=" * 45)

    # Create detector and generate deterministic test data
    detector = ChineseNameDetector()
    helper = TestChineseNameDetectorPerformance()
    names = helper.generate_test_names(detector, 3000)

    print(f"Generated {len(names)} deterministic test names")

    # Pre-warm all caches to eliminate cold start variance
    print("Pre-warming caches...")
    # Full warm-up to ensure all code paths are exercised
    for name in names:  # Warm up with ALL names to fully load caches
        detector.is_chinese_name(name)

    print("Caches fully warmed. Now measuring warm performance...")

    # First, run pure measurements for accurate performance assessment
    print("Running 5 pure measurements (no profiling overhead)...")
    pure_times = []

    for i in range(5):
        runtime, _ = run_single_measurement(detector, names, enable_profiling=False)
        pure_times.append(runtime)
        print(f"Pure run {i+1}: {runtime:.4f}s")

    # Statistical summary for pure measurements
    mean_time = statistics.mean(pure_times)
    std_time = statistics.stdev(pure_times) if len(pure_times) > 1 else 0
    cv = (std_time / mean_time * 100) if mean_time > 0 else 0

    print("\nPure Performance Statistics:")
    print(f"Mean:     {mean_time:.4f}s")
    print(f"Std Dev:  {std_time:.4f}s")
    print(f"CV:       {cv:.2f}%")
    print(f"Range:    {min(pure_times):.4f}s - {max(pure_times):.4f}s")

    # Performance metrics
    names_per_sec = len(names) / mean_time
    us_per_name = (mean_time / len(names)) * 1_000_000

    print("\nPerformance Metrics:")
    print(f"Rate:     {names_per_sec:.0f} names/second")
    print(f"Per name: {us_per_name:.1f} microseconds")

    # Variance assessment
    if cv < 3.0:
        print(f"✅ EXCELLENT: Very low variance ({cv:.2f}%)")
    elif cv < 5.0:
        print(f"✅ GOOD: Low variance ({cv:.2f}%)")
    elif cv < 10.0:
        print(f"⚠️  FAIR: Moderate variance ({cv:.2f}%)")
    else:
        print(f"❌ HIGH: High variance ({cv:.2f}%) - check system load")

    # Now run one profiled measurement for detailed analysis
    print("\nDetailed Profiling (single run with cProfile overhead):")
    print("=" * 58)
    _, profiler = run_single_measurement(detector, names, enable_profiling=True)
    ps = pstats.Stats(profiler).strip_dirs().sort_stats("tottime")
    ps.print_stats(25)


if __name__ == "__main__":
    main()

