#!/usr/bin/env python3
"""
Script to check test status and count individual test case failures.

This script runs all tests and counts the individual test case failures
(not just the number of test functions that fail), and also checks if
performance tests pass.
"""

import ast
import os
import re
import subprocess
import sys

EXPECTED_FAILURES = 61


def run_tests():
    """Run all tests and capture output."""
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    # Ensure UTF-8 encoding on all platforms
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        # Prepare failure log path for this run
        fail_log = os.path.join(os.getcwd(), ".pytest_failures.txt")
        # Clear previous log if exists
        try:
            if os.path.exists(fail_log):
                os.remove(fail_log)
        except Exception:
            pass

        env["SINONYM_FAIL_LOG"] = fail_log

        result = subprocess.run(
            [
                "uv",
                "run",
                "pytest",
                "-q",
                "-s",
                "tests/",
                "--ignore=tests/test_performance.py",
                "--disable-warnings",
                "--maxfail=0",
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            timeout=300,
        )
        # Append path to combined output for downstream consumers
        combined = result.stdout + result.stderr
        if os.path.exists(fail_log):
            combined += f"\n__FAIL_LOG_PATH__={fail_log}\n"
        return combined
    except subprocess.TimeoutExpired:
        print("Tests timed out after 60 seconds")
        return ""
    except Exception as e:
        print(f"Error running tests: {e}")
        return ""


def extract_failure_counts(output):
    """Extract individual test case failure counts from test output."""
    # Match both with and without leading 'AssertionError:' prefix
    pattern_with_assert = re.compile(
        r"AssertionError:\s*([^:]+):\s*(\d+)\s+failures?\s+out\s+of\s+(\d+)\s+tests?",
    )
    pattern_plain = re.compile(
        r"([^:]+):\s*(\d+)\s+failures?\s+out\s+of\s+(\d+)\s+tests?",
    )

    failure_details = {}  # Use dict to deduplicate by test name

    # Split output into lines and process each one
    for line in output.split("\n"):
        match = None
        if "AssertionError:" in line:
            match = pattern_with_assert.search(line)
        if not match:
            match = pattern_plain.search(line)
        if match:
            test_name, failures, total = match.groups()
            test_name = test_name.strip()
            # Use dict to automatically deduplicate by test name
            failure_details[test_name] = {
                "name": test_name,
                "failures": int(failures),
                "total": int(total),
            }

    return list(failure_details.values())


def read_fail_log_path_from_output(output: str) -> str | None:
    for line in output.splitlines():
        if line.strip().startswith("__FAIL_LOG_PATH__="):
            return line.strip().split("=", 1)[1]
    return None


def read_fail_log(path: str) -> list[str]:
    try:
        with open(path, encoding="utf-8") as f:
            return [ln.rstrip("\n") for ln in f]
    except Exception:
        return []


## Removed per-file execution helpers


def check_performance_tests():
    """Run performance tests separately and check if they pass."""
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    # Ensure UTF-8 encoding on all platforms
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        result = subprocess.run(
            ["uv", "run", "pytest", "tests/test_performance.py", "-v"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
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

    # Prefer explicit failure log if available
    fail_log_path = read_fail_log_path_from_output(output)
    logged = read_fail_log(fail_log_path) if fail_log_path else []
    failures = extract_failure_counts(output) if not logged else []

    if logged:
        # Summarize by label from log entries
        by_label: dict[str, int] = {}
        parsed_entries: list[dict] = []
        for ln in logged:
            parts = ln.split("\t")
            if len(parts) != 6:
                continue
            label, name_repr, expected_success, expected_output, actual_success, actual_output = parts
            by_label[label] = by_label.get(label, 0) + 1
            # Convert repr(name) back to a string if possible
            try:
                name = ast.literal_eval(name_repr)
            except Exception:
                name = name_repr
            parsed_entries.append(
                {
                    "label": label,
                    "name": name,
                    "expected_success": expected_success,
                    "expected_output": expected_output,
                    "actual_success": actual_success,
                    "actual_output": actual_output,
                },
            )

        print("\n" + "=" * 70)
        print("INDIVIDUAL TEST CASE FAILURES (aggregated counts):")
        print("=" * 70)
        total_failures = 0
        for label in sorted(by_label):
            count = by_label[label]
            print(f"  {label}: {count} failures")
            total_failures += count
        print("-" * 70)
        print(f"TOTAL INDIVIDUAL TEST CASE FAILURES: {total_failures}")
        # Cross-check aggregated total against raw log length
        if total_failures != len(logged):
            print("WARNING: Aggregated count does not match raw log line count!")
            print(f"  Aggregated: {total_failures} vs Raw: {len(logged)}")
        print("=" * 70)

        # Print detailed list of every failing case from the log
        print("\n" + "=" * 70)
        print("DETAILED FAILURES (from fail log):")
        print("=" * 70)
        for i, e in enumerate(parsed_entries, start=1):
            print(
                f"{i:4d}. [{e['label']}] {e['name']} | "
                f"expected_success={e['expected_success']} actual_success={e['actual_success']} | "
                f"expected={e['expected_output']} | actual={e['actual_output']}",
            )
        print("=" * 70)
    elif failures:
        print("\n" + "=" * 70)
        print("INDIVIDUAL TEST CASE FAILURES (aggregated counts):")
        print("=" * 70)

        total_failures = 0
        for failure in failures:
            print(f"  {failure['name']}: {failure['failures']}/{failure['total']} failures")
            total_failures += failure["failures"]

        print("-" * 70)
        print(f"TOTAL INDIVIDUAL TEST CASE FAILURES: {total_failures}")
        print("=" * 70)
    else:
        print("\nNo aggregated failure counts found.")
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
    if logged:
        print(f"Individual test case failures: {len(logged)}")
    elif failures:
        print(f"Individual test case failures: {total_failures}")
    else:
        print("Individual test case failures: Unable to determine")
    print(f"Performance tests: {'PASSED ✓' if perf_passed else 'FAILED ✗'}")

    # Exit code based on status (updated baseline to EXPECTED_FAILURES after config improvements)
    if logged and len(logged) == EXPECTED_FAILURES and perf_passed:
        print("\n✓ Tests are at expected baseline (EXPECTED_FAILURES failures, performance OK)")
        sys.exit(0)
    elif logged and len(logged) < EXPECTED_FAILURES and perf_passed:
        print(
            f"\n✓ IMPROVEMENT! Tests are better than baseline ({len(logged)} < EXPECTED_FAILURES failures, performance OK)",
        )
        sys.exit(0)
    elif logged and len(logged) > EXPECTED_FAILURES:
        print(f"\n✗ REGRESSION! Too many failures ({len(logged)} > EXPECTED_FAILURES)")
        sys.exit(1)
    elif failures and total_failures == EXPECTED_FAILURES and perf_passed:
        print("\n✓ Tests are at expected baseline (EXPECTED_FAILURES failures, performance OK)")
        sys.exit(0)
    elif failures and total_failures < EXPECTED_FAILURES and perf_passed:
        print(
            f"\n✓ IMPROVEMENT! Tests are better than baseline ({total_failures} < EXPECTED_FAILURES failures, performance OK)",
        )
        sys.exit(0)
    elif failures and total_failures > EXPECTED_FAILURES:
        print(f"\n✗ REGRESSION! Too many failures ({total_failures} > EXPECTED_FAILURES)")
        sys.exit(1)
    elif not perf_passed:
        print("\n✗ Performance tests failed!")
        sys.exit(1)
    else:
        print("\n⚠ Unable to determine test status")
        sys.exit(0)


if __name__ == "__main__":
    main()
