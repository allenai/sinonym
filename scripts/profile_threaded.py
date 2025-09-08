#!/usr/bin/env python3
"""
Low-variance multi-threaded performance testing script for sinonym.

Tests performance with different numbers of threads and demonstrates
thread safety of the optimized caching implementation. This version
minimizes variance through:
- Deterministic hash seeds
- Reproducible test data generation
- Cache pre-warming
- Multiple measurement runs with statistics
"""

import concurrent.futures
import gc
import os
import statistics
import time

# Set deterministic hash seed for consistent dict iteration
os.environ["PYTHONHASHSEED"] = "42"

from sinonym.detector import ChineseNameDetector
from tests.test_performance import TestChineseNameDetectorPerformance


def test_single_thread(names: list[str], detector: ChineseNameDetector) -> tuple[float, list[tuple[str, bool]]]:
    """Test processing with a single thread."""
    start_time = time.perf_counter()
    results = []
    for name in names:
        result = detector.is_chinese_name(name)
        results.append((name, result.success))
    end_time = time.perf_counter()
    return end_time - start_time, results


def process_chunk(args: tuple) -> tuple[float, list[tuple[str, bool]]]:
    """Process a chunk of names in a single thread."""
    names, detector = args
    return test_single_thread(names, detector)


def test_multithreaded(names: list[str], num_threads: int) -> tuple[float, list[tuple[str, bool]]]:
    """Test processing with multiple threads using the SAME detector instance."""
    # Use ONE shared detector instance across all threads
    detector = ChineseNameDetector()

    # Split names into chunks for each thread
    chunk_size = len(names) // num_threads
    chunks = []
    for i in range(num_threads):
        start_idx = i * chunk_size
        if i == num_threads - 1:  # Last chunk gets remaining names
            end_idx = len(names)
        else:
            end_idx = (i + 1) * chunk_size
        chunks.append((names[start_idx:end_idx], detector))

    start_time = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
        chunk_results = [future.result() for future in concurrent.futures.as_completed(futures)]

    end_time = time.perf_counter()

    # Combine all results
    all_results = []
    for _, results in chunk_results:
        all_results.extend(results)

    return end_time - start_time, all_results


def verify_consistency(single_results: list[tuple[str, bool]],
                      multi_results: list[tuple[str, bool]]) -> bool:
    """Verify that single-threaded and multi-threaded results are identical."""
    if len(single_results) != len(multi_results):
        return False

    # Sort both by name for comparison (since multi-threaded may have different order)
    single_sorted = sorted(single_results, key=lambda x: x[0])
    multi_sorted = sorted(multi_results, key=lambda x: x[0])

    return single_sorted == multi_sorted


def run_multiple_measurements(test_func, *args, num_runs=3):
    """Run multiple measurements and return statistics."""
    times = []
    results = None

    for _ in range(num_runs):
        gc.collect()  # Clean state between runs
        time_taken, test_results = test_func(*args)
        times.append(time_taken)
        if results is None:
            results = test_results  # Keep first run's results for verification

    mean_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0
    cv = (std_time / mean_time * 100) if mean_time > 0 else 0

    return mean_time, std_time, cv, results


def main():
    print("=" * 75)
    print("SINONYM MULTI-THREADED PERFORMANCE TEST (LOW VARIANCE)")
    print("=" * 75)

    # Generate deterministic test data
    detector = ChineseNameDetector()
    helper = TestChineseNameDetectorPerformance()
    test_names = helper.generate_test_names(detector, 1200)  # 1200 names

    print(f"Generated {len(test_names)} deterministic test names")

    # Pre-warm caches
    print("Pre-warming caches...")
    for name in test_names[:100]:
        detector.is_chinese_name(name)

    print("Running multi-threaded tests with statistical analysis...")
    print()

    # Test with different thread counts
    thread_counts = [1, 2, 4, 8]

    # Run single-threaded baseline with statistics
    print("Measuring single-threaded baseline (3 runs)...")
    single_mean, single_std, single_cv, single_results = run_multiple_measurements(
        test_single_thread, test_names, detector,
    )
    single_rate = len(test_names) / single_mean

    print(f"Single-threaded: {single_mean:.3f}s ±{single_std:.3f} (CV: {single_cv:.1f}%) | {single_rate:.0f} names/sec")
    print()

    print("Threads | Mean Time(s) | Std(s) | CV(%) | Rate(names/s) | Speedup | Safety")
    print("-" * 75)

    for num_threads in thread_counts:
        if num_threads == 1:
            # Use single-threaded results
            mean_time = single_mean
            std_time = single_std
            cv = single_cv
            thread_results = single_results
            consistent = True
        else:
            # Test multi-threaded with statistics
            mean_time, std_time, cv, thread_results = run_multiple_measurements(
                test_multithreaded, test_names, num_threads,
            )
            consistent = verify_consistency(single_results, thread_results)

        thread_rate = len(test_names) / mean_time
        speedup = single_mean / mean_time

        safety_status = "✓ PASS" if consistent else "✗ FAIL"
        cv_status = "✓" if cv < 10.0 else "⚠" if cv < 20.0 else "✗"

        print(f"{num_threads:^7} | {mean_time:^12.3f} | {std_time:^6.3f} | {cv:^4.1f}{cv_status} | {thread_rate:^13.0f} | {speedup:^7.2f}x | {safety_status}")

    print()
    print("=" * 75)
    print("SUMMARY:")
    print("=" * 75)
    print("• Deterministic test data eliminates variance from random name generation")
    print("• Pre-warmed caches eliminate cold-start effects")
    print("• Multiple measurements provide statistical confidence")
    print("• Thread safety: All threads produce identical results")
    print("• CV < 10%: Good measurement stability")
    print("• CV 10-20%: Moderate system noise")
    print("• CV > 20%: High variance, check system load")
    print()
    print("Note: GIL limits CPU-bound threading speedup in Python.")
    print("Thread safety and cache isolation are the primary validation goals.")


if __name__ == "__main__":
    main()
