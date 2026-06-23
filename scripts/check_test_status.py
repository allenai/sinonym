#!/usr/bin/env python3
"""
Script to check test status and count individual test case failures.

This script runs all tests and counts the individual test case failures
(not just the number of test functions that fail), and also checks if
performance tests pass.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

EXPECTED_FAILURES = 44
FAIL_LOG_FIELDS = (
    "nodeid",
    "label",
    "name",
    "expected_success",
    "expected_output",
    "actual_success",
    "actual_output",
)
PYTEST_BASELINE_RETURN_CODES = {0, 1}
INVALID_FAIL_LOG_ENTRY_MESSAGE = "failure log entry has invalid fields"

FAILURE_COUNT_PATTERNS = (
    re.compile(
        r"AssertionError:\s*([^:]+):\s*(\d+)\s+failures?\s+out\s+of\s+(\d+)\s+tests?",
    ),
    re.compile(
        r"([^:]+):\s*(\d+)\s+failures?\s+out\s+of\s+(\d+)\s+tests?",
    ),
)


class UvNotFoundError(OSError):
    def __init__(self) -> None:
        super().__init__("uv executable not found on PATH")


def _uv_executable() -> str:
    uv_path = shutil.which("uv")
    if not uv_path:
        raise UvNotFoundError
    return uv_path


def _safe_for_console(value: object) -> str:
    """Return text that is safe to print on the current console encoding."""
    text = str(value)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(encoding, errors="backslashreplace").decode(encoding, errors="replace")


def run_tests():
    """Run all tests and capture output."""
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    # Ensure UTF-8 encoding on all platforms
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        # Prepare failure log path for this run
        fail_log = Path(tempfile.mkdtemp(prefix="sinonym-pytest-failures-")) / "failures.jsonl"

        env["SINONYM_FAIL_LOG"] = str(fail_log)

        result = subprocess.run(  # noqa: S603
            [
                _uv_executable(),
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
    except subprocess.TimeoutExpired:
        print("Tests timed out after 300 seconds")
        return "", None
    except OSError as e:
        print(f"Error running tests: {e}")
        return "", None
    else:
        # Append path to combined output for downstream consumers
        combined = result.stdout + result.stderr
        if fail_log.exists():
            combined += f"\n__FAIL_LOG_PATH__={fail_log}\n"
        return combined, result.returncode


def _failure_count_match(line: str) -> re.Match[str] | None:
    for pattern in FAILURE_COUNT_PATTERNS:
        match = pattern.search(line)
        if match:
            return match
    return None


def extract_failure_counts(output):
    """Extract individual test case failure counts from test output."""
    failure_details = {}  # Use dict to deduplicate by test name

    # Split output into lines and process each one
    for line in output.split("\n"):
        match = _failure_count_match(line)
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


def _matches_known_failure_label(message: str | None, known_labels: list[str]) -> bool:
    if not message:
        return False

    label_fragment = re.split(r":|\.\.\.|\u2026", message, maxsplit=1)[0]
    normalized_fragment = _normalized_label(label_fragment)
    if not normalized_fragment:
        return False

    return any(_normalized_label(label).startswith(normalized_fragment) for label in known_labels)


def _is_truncated_assertion_message(message: str | None) -> bool:
    """Return whether pytest reported only a duplicate/truncated assertion summary."""
    if message is None:
        return True

    stripped = message.strip()
    return stripped in {"", "...", "\u2026"}


def _is_truncated_assertion_summary(message: str) -> bool:
    """Return whether pytest truncated the AssertionError class name itself."""
    stripped = message.strip()
    if not stripped.endswith(("...", "\u2026")):
        return False

    prefix = re.split(r"\.\.\.|\u2026", stripped, maxsplit=1)[0].strip(": ")
    return "assertionerror".startswith(prefix.lower())


def _is_duplicate_logged_assertion_summary(
    nodeid: str,
    message: str,
    known_labels: list[str],
    logged_nodeids: set[str],
) -> bool:
    """Return whether a pytest summary duplicates exact-nodeid fail-log rows."""
    if nodeid not in logged_nodeids:
        return False
    if _is_truncated_assertion_summary(message):
        return True
    if not message.startswith("AssertionError"):
        return False

    assertion_message = message.removeprefix("AssertionError").removeprefix(":").strip()
    return _is_truncated_assertion_message(assertion_message) or _matches_known_failure_label(
        assertion_message,
        known_labels,
    )


def extract_unaggregated_pytest_failures(
    output: str,
    known_labels: list[str] | None = None,
    logged_nodeids: set[str] | None = None,
) -> list[dict[str, str]]:
    """Extract pytest failures/errors not represented by aggregate counts."""
    known_labels = known_labels or []
    logged_nodeids = logged_nodeids or set()
    failures: dict[str, dict[str, str]] = {}
    summary_pattern = re.compile(r"^(FAILED|ERROR)\s+(.+?)\s+-\s+(.+?)\s*$")

    for line in output.splitlines():
        match = summary_pattern.match(line.strip())
        if not match:
            continue
        if _failure_count_match(line):
            continue
        status, nodeid, message = match.groups()
        if _is_duplicate_logged_assertion_summary(nodeid, message, known_labels, logged_nodeids):
            continue
        failures[nodeid] = {
            "name": nodeid,
            "message": message,
            "status": status,
        }

    return list(failures.values())


def extract_unaggregated_assertion_failures(output: str, known_labels: list[str] | None = None) -> list[dict[str, str]]:
    """Backward-compatible wrapper for unaggregated pytest failure extraction."""
    return extract_unaggregated_pytest_failures(output, known_labels)


def read_fail_log_path_from_output(output: str) -> str | None:
    for line in output.splitlines():
        if line.strip().startswith("__FAIL_LOG_PATH__="):
            return line.strip().split("=", 1)[1]
    return None


def read_fail_log(path: str) -> list[str]:
    try:
        with Path(path).open(encoding="utf-8") as f:
            return [ln.rstrip("\n") for ln in f]
    except OSError:
        return []


def format_fail_log_line(entry: dict) -> str:
    """Serialize one failure-log entry as JSONL."""
    if set(entry) != set(FAIL_LOG_FIELDS):
        raise ValueError(INVALID_FAIL_LOG_ENTRY_MESSAGE)
    return json.dumps({field: entry[field] for field in FAIL_LOG_FIELDS}, ensure_ascii=False)


def parse_fail_log_line(line: str) -> dict | None:
    """Parse one failure-log line, returning None when the line is malformed."""
    try:
        entry = json.loads(line)
    except json.JSONDecodeError:
        return None

    if not isinstance(entry, dict) or set(entry) != set(FAIL_LOG_FIELDS):
        return None
    if not isinstance(entry["nodeid"], str):
        return None
    if not isinstance(entry["label"], str):
        return None
    return entry


def parse_fail_log(logged: list[str]) -> tuple[dict[str, int], list[dict], list[str]]:
    """Parse shared failure log lines into aggregate and detailed views."""
    by_label: dict[str, int] = {}
    parsed_entries: list[dict] = []
    malformed_entries: list[str] = []

    for ln in logged:
        parsed_entry = parse_fail_log_line(ln)
        if parsed_entry is None:
            malformed_entries.append(ln)
            continue
        label = parsed_entry["label"]
        by_label[label] = by_label.get(label, 0) + 1
        parsed_entries.append(parsed_entry)

    return by_label, parsed_entries, malformed_entries


def _normalized_label(label: str) -> str:
    return re.sub(r"\s+", " ", label.strip().lower())


def _summary_label_base(label: str) -> str:
    normalized = _normalized_label(label)
    return re.sub(r"\s+tests?$", "", normalized)


def _matching_logged_labels(summary_label: str, logged_counts: dict[str, int]) -> list[str]:
    """Find fail-log labels that represent the same aggregate pytest summary."""
    summary_normalized = _normalized_label(summary_label)
    summary_base = _summary_label_base(summary_label)
    matches: list[str] = []

    for logged_label in logged_counts:
        logged_normalized = _normalized_label(logged_label)
        if logged_normalized == summary_normalized:
            return [logged_label]
        if logged_normalized.startswith((f"{summary_base} ", f"{summary_base}(")):
            matches.append(logged_label)

    return matches


def combine_failure_sources(logged: list[str], output: str) -> dict:
    """Combine fail-log entries and pytest assertion summaries without double-counting."""
    logged_counts, parsed_entries, malformed_entries = parse_fail_log(logged)
    aggregate_counts = extract_failure_counts(output)
    aggregate_only: list[dict] = []
    aggregate_deficits: list[dict] = []
    aggregate_covered: list[dict] = []

    for failure in aggregate_counts:
        matching_labels = _matching_logged_labels(failure["name"], logged_counts)
        if not matching_labels:
            aggregate_only.append(failure)
            continue

        logged_count = sum(logged_counts[label] for label in matching_labels)
        if failure["failures"] > logged_count:
            aggregate_deficits.append(
                {
                    **failure,
                    "failures": failure["failures"] - logged_count,
                    "reported_failures": failure["failures"],
                    "logged_failures": logged_count,
                    "logged_labels": matching_labels,
                },
            )
        else:
            aggregate_covered.append(
                {
                    **failure,
                    "logged_failures": logged_count,
                    "logged_labels": matching_labels,
                },
            )

    known_labels = [failure["name"] for failure in aggregate_counts]
    known_labels.extend(logged_counts)
    logged_nodeids = {entry["nodeid"] for entry in parsed_entries if entry["nodeid"]}
    unaggregated_pytest_failures = extract_unaggregated_pytest_failures(output, known_labels, logged_nodeids)
    aggregate_only_total = sum(failure["failures"] for failure in aggregate_only)
    aggregate_deficit_total = sum(failure["failures"] for failure in aggregate_deficits)
    total_failures = len(logged) + aggregate_only_total + aggregate_deficit_total + len(unaggregated_pytest_failures)

    return {
        "logged_counts": logged_counts,
        "parsed_entries": parsed_entries,
        "malformed_entries": malformed_entries,
        "aggregate_only": aggregate_only,
        "aggregate_deficits": aggregate_deficits,
        "aggregate_covered": aggregate_covered,
        "unaggregated_assertions": unaggregated_pytest_failures,
        "unaggregated_pytest_failures": unaggregated_pytest_failures,
        "total_failures": total_failures,
    }


## Removed per-file execution helpers


def check_performance_tests():
    """Run performance tests separately and check if they pass."""
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    # Ensure UTF-8 encoding on all platforms
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        result = subprocess.run(  # noqa: S603
            [_uv_executable(), "run", "pytest", "tests/test_performance.py", "-v"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            timeout=30,
        )

        # Check if tests passed
        if result.returncode == 0:
            return True, result.stdout
        return False, result.stdout + result.stderr
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, str(e)


def status_exit_decision(
    total_failures: int,
    *,
    perf_passed: bool,
    pytest_returncode: int | None,
    malformed_entries: list[str],
) -> tuple[int, str]:
    """Return process exit code and final status message."""
    if pytest_returncode is None:
        exit_code = 1
        message = "Test execution failed before pytest returned a status."
    elif pytest_returncode not in PYTEST_BASELINE_RETURN_CODES:
        exit_code = 1
        message = f"Pytest execution failed with exit code {pytest_returncode}."
    elif pytest_returncode == 1 and total_failures == 0:
        exit_code = 1
        message = "Pytest failed without parseable expected failure counts."
    elif pytest_returncode == 0 and total_failures > 0:
        exit_code = 1
        message = "Parsed failure output even though pytest exited successfully."
    elif malformed_entries:
        exit_code = 1
        message = f"Fail log has {len(malformed_entries)} malformed entries."
    elif not perf_passed:
        exit_code = 1
        message = "Performance tests failed!"
    elif total_failures > EXPECTED_FAILURES:
        exit_code = 1
        message = f"REGRESSION! Too many failures ({total_failures} > EXPECTED_FAILURES)"
    elif total_failures == EXPECTED_FAILURES:
        exit_code = 0
        message = "Tests are at expected baseline (EXPECTED_FAILURES failures, performance OK)"
    else:
        exit_code = 0
        message = f"IMPROVEMENT! Tests are better than baseline ({total_failures} < EXPECTED_FAILURES failures, performance OK)"

    return exit_code, message


def main():  # noqa: C901, PLR0912, PLR0915
    print("=" * 70)
    print("SINONYM TEST STATUS CHECKER")
    print("=" * 70)

    # Run all tests
    print("\nRunning all tests...")
    output, pytest_returncode = run_tests()

    # Combine explicit failure log entries with assertion summaries from pytest
    # output. Some aggregate tests do not write to SINONYM_FAIL_LOG.
    fail_log_path = read_fail_log_path_from_output(output)
    logged = read_fail_log(fail_log_path) if fail_log_path else []
    failure_report = combine_failure_sources(logged, output)
    logged_counts = failure_report["logged_counts"]
    parsed_entries = failure_report["parsed_entries"]
    malformed_entries = failure_report["malformed_entries"]
    aggregate_only = failure_report["aggregate_only"]
    aggregate_deficits = failure_report["aggregate_deficits"]
    unaggregated_pytest_failures = failure_report["unaggregated_pytest_failures"]
    total_failures = failure_report["total_failures"]

    if total_failures:
        print("\n" + "=" * 70)
        print("INDIVIDUAL TEST CASE FAILURES (aggregated counts):")
        print("=" * 70)

        if logged_counts:
            print("  From fail log:")
            for label in sorted(logged_counts):
                count = logged_counts[label]
                print(f"    {label}: {count} failures")
        if aggregate_only or aggregate_deficits:
            print("  From pytest assertion summaries:")
            for failure in aggregate_only:
                print(f"    {failure['name']}: {failure['failures']}/{failure['total']} failures")
            for failure in aggregate_deficits:
                print(
                    f"    {failure['name']}: {failure['failures']} additional failures "
                    f"({failure['reported_failures']} reported, {failure['logged_failures']} in fail log)",
                )
        if unaggregated_pytest_failures:
            print("  From unaggregated pytest failures/errors:")
            for failure in unaggregated_pytest_failures:
                print(f"    {failure['name']}: 1 failure")
        if malformed_entries:
            print("  Malformed fail-log entries:")
            print(f"    {len(malformed_entries)} failures")
        print("-" * 70)
        print(f"TOTAL INDIVIDUAL TEST CASE FAILURES: {total_failures}")
        # Cross-check aggregated total against raw log length
        if logged and total_failures != len(logged):
            print("NOTE: Total includes failures not present in the raw fail log.")
            print(f"  Combined: {total_failures} vs Raw fail log: {len(logged)}")
        print("=" * 70)

        # Print detailed list of every failing case from the log
        if parsed_entries:
            print("\n" + "=" * 70)
            print("DETAILED FAILURES (from fail log):")
            print("=" * 70)
            for i, e in enumerate(parsed_entries, start=1):
                label = _safe_for_console(e["label"])
                name = _safe_for_console(e["name"])
                expected_output = _safe_for_console(e["expected_output"])
                actual_output = _safe_for_console(e["actual_output"])
                print(
                    f"{i:4d}. [{label}] {name} | "
                    f"expected_success={e['expected_success']} actual_success={e['actual_success']} | "
                    f"expected={expected_output} | actual={actual_output}",
                )
            print("=" * 70)

        if aggregate_only or aggregate_deficits or unaggregated_pytest_failures:
            print("\n" + "=" * 70)
            print("DETAILED FAILURES (not in fail log):")
            print("=" * 70)
            for failure in aggregate_only:
                print(f"  [{failure['name']}] {failure['failures']}/{failure['total']} aggregate failures")
            for failure in aggregate_deficits:
                labels = ", ".join(failure["logged_labels"])
                print(
                    f"  [{failure['name']}] {failure['failures']} additional aggregate failures "
                    f"beyond fail-log labels: {labels}",
                )
            for failure in unaggregated_pytest_failures:
                message = _safe_for_console(failure["message"])
                print(f"  [{failure['name']}] {message}")
            print("=" * 70)
    else:
        print("\nNo assertion failures found.")
        if pytest_returncode not in (0, None):
            print("(Pytest exited non-zero without parseable expected failure counts.)")

    # Check performance tests
    print("\n" + "=" * 70)
    print("PERFORMANCE TEST STATUS:")
    print("=" * 70)

    perf_passed, perf_output = check_performance_tests()

    if perf_passed:
        print("Performance tests PASSED")

        # Try to extract performance metrics
        if "microseconds per name" in perf_output:
            lines = perf_output.split("\n")
            for line in lines:
                if "microseconds per name" in line or "names/second" in line:
                    print(_safe_for_console(f"  {line.strip()}"))
    else:
        print("Performance tests FAILED")
        print("Performance test output:")
        print(_safe_for_console(perf_output))

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print("=" * 70)
    print(f"Individual test case failures: {total_failures}")
    print(f"Performance tests: {'PASSED' if perf_passed else 'FAILED'}")

    # Exit code based on status (updated baseline to EXPECTED_FAILURES after config improvements)
    exit_code, status_message = status_exit_decision(
        total_failures,
        perf_passed=perf_passed,
        pytest_returncode=pytest_returncode,
        malformed_entries=malformed_entries,
    )
    print(f"\n{status_message}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
