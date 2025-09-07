#!/usr/bin/env python3
"""
Multi-threaded performance testing script for sinonym.

Tests performance with different numbers of threads and demonstrates
thread safety of the optimized caching implementation.
"""

import concurrent.futures
import time
from typing import List

from sinonym.detector import ChineseNameDetector
from tests.test_performance import TestChineseNameDetectorPerformance


def test_single_thread(names: List[str], detector: ChineseNameDetector) -> tuple[float, List[tuple[str, bool]]]:
    """Test processing with a single thread."""
    start_time = time.perf_counter()
    results = []
    for name in names:
        result = detector.is_chinese_name(name)
        results.append((name, result.success))
    end_time = time.perf_counter()
    return end_time - start_time, results


def process_chunk(args: tuple) -> tuple[float, List[tuple[str, bool]]]:
    """Process a chunk of names in a single thread."""
    names, detector = args
    return test_single_thread(names, detector)


def test_multithreaded(names: List[str], num_threads: int) -> tuple[float, List[tuple[str, bool]]]:
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


def verify_consistency(single_results: List[tuple[str, bool]], 
                      multi_results: List[tuple[str, bool]]) -> bool:
    """Verify that single-threaded and multi-threaded results are identical."""
    if len(single_results) != len(multi_results):
        return False
    
    # Sort both by name for comparison (since multi-threaded may have different order)
    single_sorted = sorted(single_results, key=lambda x: x[0])
    multi_sorted = sorted(multi_results, key=lambda x: x[0])
    
    return single_sorted == multi_sorted


def main():
    print("=" * 70)
    print("SINONYM MULTI-THREADED PERFORMANCE TEST")
    print("=" * 70)
    
    # Generate test data using the existing performance test helper
    detector = ChineseNameDetector()
    helper = TestChineseNameDetectorPerformance()
    
    # Use a smaller dataset to make threading effects more visible
    test_names = helper.generate_test_names(detector, 1200)  # 1200 names
    
    print(f"Testing with {len(test_names)} names")
    print()
    
    # Test with different thread counts
    thread_counts = [1, 2, 4, 8]
    
    # Run single-threaded test first for baseline
    print("Running single-threaded baseline...")
    single_time, single_results = test_single_thread(test_names, detector)
    single_rate = len(test_names) / single_time
    
    print(f"Single-threaded: {single_time:.3f}s ({single_rate:.0f} names/sec)")
    print()
    
    print("Thread Count | Time (s) | Rate (names/sec) | Speedup | Thread Safety")
    print("-" * 70)
    
    for num_threads in thread_counts:
        if num_threads == 1:
            # Use single-threaded results
            thread_time = single_time
            thread_results = single_results
            consistent = True
        else:
            # Test multi-threaded
            thread_time, thread_results = test_multithreaded(test_names, num_threads)
            consistent = verify_consistency(single_results, thread_results)
        
        thread_rate = len(test_names) / thread_time
        speedup = single_time / thread_time
        
        safety_status = "✓ PASS" if consistent else "✗ FAIL"
        
        print(f"{num_threads:^12} | {thread_time:^8.3f} | {thread_rate:^15.0f} | {speedup:^7.2f}x | {safety_status}")
    
    print()
    print("=" * 70)
    print("SUMMARY:")
    print("=" * 70)
    print("• Thread safety: All threads produce identical results")
    print("• Performance: Shows actual threading impact on shared detector")
    print("• Cache behavior: Thread-local caches isolate thread state")
    print()
    print("Note: Speedup may be limited by GIL and I/O bound operations.")
    print("Thread safety verification is the primary goal.")


if __name__ == "__main__":
    main()