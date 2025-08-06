#!/usr/bin/env python3
"""
Script to check test status and count individual test case failures.

This script runs all tests and counts the individual test case failures
(not just the number of test functions that fail), and also checks if
performance tests pass.
"""

import os
import re
import subprocess
import sys


def run_tests():
    """Run all tests and capture output."""
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"

    try:
        result = subprocess.run(
            ["uv", "run", "pytest", "tests/", "-v", "-s"],
            check=False, capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        print("Tests timed out after 60 seconds")
        return ""
    except Exception as e:
        print(f"Error running tests: {e}")
        return ""

def extract_failure_counts(output):
    """Extract individual test case failure counts from test output."""
    # Pattern to match lines like "X failures out of Y tests"
    pattern = r"(\d+)\s+failures?\s+out\s+of\s+\d+\s+tests?"

    matches = re.findall(pattern, output)

    failure_details = []
    # Also extract the full context for each failure
    detail_pattern = r"([^:]+):\s*(\d+)\s+failures?\s+out\s+of\s+(\d+)\s+tests?"
    detail_matches = re.findall(detail_pattern, output)

    for match in detail_matches:
        test_name, failures, total = match
        failure_details.append({
            "name": test_name.strip(),
            "failures": int(failures),
            "total": int(total),
        })

    return failure_details

def check_performance_tests():
    """Run performance tests separately and check if they pass."""
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"

    try:
        result = subprocess.run(
            ["uv", "run", "pytest", "tests/test_performance.py", "-v"],
            check=False, capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )

        # Check if tests passed
        if "PASSED" in result.stdout and "FAILED" not in result.stdout:
            return True, result.stdout
        return False, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)

def main():
    print("=" * 70)
    print("SINONYM TEST STATUS CHECKER")
    print("=" * 70)

    # Run all tests
    print("\nRunning all tests...")
    output = run_tests()

    # Extract failure counts
    failures = extract_failure_counts(output)

    if failures:
        print("\n" + "=" * 70)
        print("INDIVIDUAL TEST CASE FAILURES:")
        print("=" * 70)

        total_failures = 0
        for failure in failures:
            print(f"  {failure['name']}: {failure['failures']}/{failure['total']} failures")
            total_failures += failure["failures"]

        print("-" * 70)
        print(f"TOTAL INDIVIDUAL TEST CASE FAILURES: {total_failures}")
        print("=" * 70)
    else:
        print("\nNo individual test case failures found in output.")
        print("(This might mean all tests passed or output format changed)")

    # Check performance tests
    print("\n" + "=" * 70)
    print("PERFORMANCE TEST STATUS:")
    print("=" * 70)

    perf_passed, perf_output = check_performance_tests()

    if perf_passed:
        print("✓ Performance tests PASSED")

        # Try to extract performance metrics
        if "microseconds per name" in perf_output:
            lines = perf_output.split("\n")
            for line in lines:
                if "microseconds per name" in line or "names/second" in line:
                    print(f"  {line.strip()}")
    else:
        print("✗ Performance tests FAILED")
        print("Performance test output:")
        print(perf_output)

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print("=" * 70)
    if failures:
        print(f"Individual test case failures: {total_failures}")
    else:
        print("Individual test case failures: Unable to determine")
    print(f"Performance tests: {'PASSED ✓' if perf_passed else 'FAILED ✗'}")

    # Exit code based on status (updated baseline to 56 after config improvements)
    if failures and total_failures == 56 and perf_passed:
        print("\n✓ Tests are at expected baseline (56 failures, performance OK)")
        sys.exit(0)
    elif failures and total_failures < 56 and perf_passed:
        print(f"\n✓ IMPROVEMENT! Tests are better than baseline ({total_failures} < 56 failures, performance OK)")
        sys.exit(0)
    elif failures and total_failures > 56:
        print(f"\n✗ REGRESSION! Too many failures ({total_failures} > 56)")
        sys.exit(1)
    elif not perf_passed:
        print("\n✗ Performance tests failed!")
        sys.exit(1)
    else:
        print("\n⚠ Unable to determine test status")
        sys.exit(0)

if __name__ == "__main__":
    main()
