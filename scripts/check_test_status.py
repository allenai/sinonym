#!/usr/bin/env python3
"""Run the test suite and count real pytest failures from JUnit XML."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

PYTEST_BASELINE_RETURN_CODES = {0, 1}
UNEXPECTED_FAILURE_SAMPLE_SIZE = 5
PERFORMANCE_METRIC_MARKERS = ("Time per name:", "names/second")
EXPECTED_NORMALIZED_NAME_FAILURES = (
    (
        "tests.test_acl::test_acl_chinese_names[Fei Yu-Fei Yu]",
        "Fei Yu",
        "Fei Yu",
        "Yu Fei",
    ),
    (
        "tests.test_acl::test_acl_order_preservation[Hao Fei-Hao Fei]",
        "Hao Fei",
        "Hao Fei",
        "Fei Hao",
    ),
    (
        "tests.test_acl::test_acl_order_preservation[Li Kuang-Li Kuang]",
        "Li Kuang",
        "Li Kuang",
        "Kuang Li",
    ),
    (
        "tests.test_acl::test_acl_order_preservation[Yao Shu-Yao Shu]",
        "Yao Shu",
        "Yao Shu",
        "Shu Yao",
    ),
    (
        "tests.test_basic_chinese_names::test_basic_chinese_names[Li Gong-expected13]",
        "Li Gong",
        "Gong Li",
        "Li Gong",
    ),
    (
        "tests.test_basic_chinese_names::test_basic_chinese_names[Lu Xun-expected75]",
        "Lu Xun",
        "Xun Lu",
        "Lu Xun",
    ),
    (
        "tests.test_basic_chinese_names::test_basic_chinese_names[Qin Shi-expected80]",
        "Qin Shi",
        "Shi Qin",
        "Qin Shi",
    ),
    (
        "tests.test_misc::test_misc_chinese_names[Jin Hua-expected13]",
        "Jin Hua",
        "Hua Jin",
        "Jin Hua",
    ),
    (
        "tests.test_misc::test_misc_chinese_names[Miao Yu-expected21]",
        "Miao Yu",
        "Miao Yu",
        "Yu Miao",
    ),
    (
        "tests.test_misc::test_misc_chinese_names[Yu Miao-expected22]",
        "Yu Miao",
        "Miao Yu",
        "Yu Miao",
    ),
    (
        "tests.test_misc::test_misc_chinese_names[Wen Jing-expected33]",
        "Wen Jing",
        "Jing Wen",
        "Wen Jing",
    ),
    (
        "tests.test_misc::test_misc_chinese_names[Jing Wen-expected34]",
        "Jing Wen",
        "Jing Wen",
        "Wen Jing",
    ),
    (
        "tests.test_mixed_production_cases::test_mixed_cases[Shu Yao-expected_result13]",
        "Shu Yao",
        "Yao Shu",
        "Shu Yao",
    ),
    (
        "tests.test_mixed_production_cases::test_mixed_cases[Jia Jian Feng-expected_result24]",
        "Jia Jian Feng",
        "Jian-Feng Jia",
        "Jia-Jian Feng",
    ),
    (
        "tests.test_mixed_production_cases::test_mixed_cases[Wei Wen Xing-expected_result48]",
        "Wei Wen Xing",
        "Wen-Xing Wei",
        "Wei-Wen Xing",
    ),
    (
        "tests.test_mixed_production_cases::test_mixed_cases[Xi Zhao-expected_result79]",
        "Xi Zhao",
        "Zhao Xi",
        "Xi Zhao",
    ),
    (
        "tests.test_mixed_production_cases::test_mixed_cases[xu feng-expected_result120]",
        "xu feng",
        "Feng Xu",
        "Xu Feng",
    ),
)
EXPECTED_FAILURES = len(EXPECTED_NORMALIZED_NAME_FAILURES)
EXPECTED_FAILURE_SIGNATURES = tuple(
    (
        nodeid,
        "failure",
        f"AssertionError: {raw_name!r}: expected normalized name {expected!r}, got {actual!r}",
    )
    for nodeid, raw_name, expected, actual in EXPECTED_NORMALIZED_NAME_FAILURES
)
EXPECTED_FAILURE_NODEIDS = tuple(nodeid for nodeid, *_ in EXPECTED_NORMALIZED_NAME_FAILURES)

if TYPE_CHECKING:
    from collections.abc import Iterable


class UvNotFoundError(OSError):
    """Raised when uv is not available on PATH."""

    def __init__(self) -> None:
        super().__init__("uv executable not found on PATH")


class JunitReadError(RuntimeError):
    """Raised when pytest did not produce readable JUnit XML."""

    def __init__(self, junit_path: Path, error: Exception) -> None:
        message = f"could not read JUnit XML at {junit_path}: {error}"
        super().__init__(message)


@dataclass(frozen=True)
class FailureDetail:
    """One failed or errored pytest testcase."""

    nodeid: str
    kind: str
    message: str


@dataclass(frozen=True)
class PytestRunResult:
    """Captured pytest execution result."""

    returncode: int | None


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


def run_tests(junit_path: Path) -> PytestRunResult:
    """Run non-performance tests and write a JUnit XML report."""
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        result = subprocess.run(  # noqa: S603
            [
                _uv_executable(),
                "run",
                "pytest",
                "-q",
                "tests/",
                "--ignore=tests/test_performance.py",
                "--disable-warnings",
                "--maxfail=0",
                "--junitxml",
                str(junit_path),
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
        return PytestRunResult(returncode=None)
    except OSError as e:
        print(f"Error running tests: {e}")
        return PytestRunResult(returncode=None)

    return PytestRunResult(returncode=result.returncode)


def _tag_name(element: ET.Element) -> str:
    """Return an XML tag name without a namespace."""
    return element.tag.rsplit("}", maxsplit=1)[-1]


def _testcase_nodeid(testcase: ET.Element) -> str:
    """Return a readable pytest-ish node id for a JUnit testcase."""
    classname = testcase.attrib.get("classname", "")
    name = testcase.attrib.get("name", "")
    if classname and name:
        return f"{classname}::{name}"
    return name or classname or "<unknown testcase>"


def _failure_children(testcase: ET.Element) -> list[ET.Element]:
    """Return failure/error children for one testcase."""
    return [child for child in testcase if _tag_name(child) in {"failure", "error"}]


def read_junit_failures(junit_path: Path) -> list[FailureDetail]:
    """Read failed or errored testcases from a pytest JUnit XML report."""
    try:
        root = ET.parse(junit_path).getroot()  # noqa: S314 - local XML emitted by this pytest run.
    except (OSError, ET.ParseError) as e:
        raise JunitReadError(junit_path, e) from e

    failures: list[FailureDetail] = []
    for testcase in root.iter():
        if _tag_name(testcase) != "testcase":
            continue

        children = _failure_children(testcase)
        if not children:
            continue

        first = children[0]
        failures.append(
            FailureDetail(
                nodeid=_testcase_nodeid(testcase),
                kind=_tag_name(first),
                message=first.attrib.get("message", "").strip(),
            ),
        )

    return failures


def check_performance_tests() -> tuple[bool, str]:
    """Run performance tests separately and return whether they pass."""
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        result = subprocess.run(  # noqa: S603
            [_uv_executable(), "run", "pytest", "tests/test_performance.py", "-q", "--capture=no"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, str(e)

    if result.returncode == 0:
        return True, result.stdout
    return False, result.stdout + result.stderr


def is_performance_metric_line(line: str) -> bool:
    """Return whether a performance-test output line contains a reported metric."""
    return any(marker in line for marker in PERFORMANCE_METRIC_MARKERS)


def _failure_signature(failure: FailureDetail) -> tuple[str, str, str]:
    """Return the stable fields used to compare a failure with the known baseline."""
    return failure.nodeid, failure.kind, failure.message


def unexpected_failure_signatures(failures: Iterable[FailureDetail]) -> list[FailureDetail]:
    """Return current failures whose node id, kind, or message is not part of the known baseline."""
    expected_counts = Counter(EXPECTED_FAILURE_SIGNATURES)
    unexpected = []
    for failure in failures:
        signature = _failure_signature(failure)
        if expected_counts[signature] > 0:
            expected_counts[signature] -= 1
        else:
            unexpected.append(failure)

    return sorted(unexpected, key=_failure_signature)


def _unexpected_failure_sample(failures: list[FailureDetail]) -> str:
    """Return a concise sample of unexpected failure signatures for the status line."""
    samples = [f"{failure.nodeid} ({failure.kind}: {failure.message})" for failure in failures[:UNEXPECTED_FAILURE_SAMPLE_SIZE]]
    suffix = (
        ""
        if len(failures) <= UNEXPECTED_FAILURE_SAMPLE_SIZE
        else f", ... (+{len(failures) - UNEXPECTED_FAILURE_SAMPLE_SIZE} more)"
    )
    return f"{', '.join(samples)}{suffix}"


def status_exit_decision(
    total_failures: int,
    *,
    perf_passed: bool,
    pytest_returncode: int | None,
    junit_error: str | None = None,
    failures: Iterable[FailureDetail] | None = None,
) -> tuple[int, str]:
    """Return process exit code and final status message."""
    unexpected_signatures = unexpected_failure_signatures(failures or ())
    if pytest_returncode is None:
        exit_code = 1
        status_message = "Test execution failed before pytest returned a status."
    elif junit_error is not None:
        exit_code = 1
        status_message = junit_error
    elif pytest_returncode not in PYTEST_BASELINE_RETURN_CODES:
        exit_code = 1
        status_message = f"Pytest execution failed with exit code {pytest_returncode}."
    elif pytest_returncode == 1 and total_failures == 0:
        exit_code = 1
        status_message = "Pytest failed without JUnit-reported failures."
    elif pytest_returncode == 0 and total_failures > 0:
        exit_code = 1
        status_message = "JUnit reported failures even though pytest exited successfully."
    elif unexpected_signatures:
        exit_code = 1
        status_message = f"Unexpected pytest failure signatures: {_unexpected_failure_sample(unexpected_signatures)}"
    elif not perf_passed:
        exit_code = 1
        status_message = "Performance tests failed!"
    elif total_failures > EXPECTED_FAILURES:
        exit_code = 1
        status_message = f"REGRESSION! Too many failures ({total_failures} > EXPECTED_FAILURES)"
    elif total_failures == EXPECTED_FAILURES:
        exit_code = 0
        status_message = "Tests are at expected baseline (EXPECTED_FAILURES failures, performance OK)"
    else:
        exit_code = 0
        status_message = (
            f"IMPROVEMENT! Tests are better than baseline ({total_failures} < EXPECTED_FAILURES failures, performance OK)"
        )

    return exit_code, status_message


def _print_failures(failures: list[FailureDetail]) -> None:
    """Print a compact failure report."""
    if not failures:
        print("\nNo test failures found.")
        return

    print("\n" + "=" * 70)
    print("PYTEST FAILURES / ERRORS")
    print("=" * 70)
    for index, failure in enumerate(failures, start=1):
        message = f": {failure.message}" if failure.message else ""
        print(f"{index:4d}. [{failure.kind}] {_safe_for_console(failure.nodeid)}{_safe_for_console(message)}")
    print("-" * 70)
    print(f"TOTAL PYTEST FAILURES / ERRORS: {len(failures)}")
    print("=" * 70)


def main() -> None:
    """Run status checks and exit according to the configured baseline."""
    print("=" * 70)
    print("SINONYM TEST STATUS CHECKER")
    print("=" * 70)

    with tempfile.TemporaryDirectory(prefix="sinonym-pytest-status-") as tmpdir:
        junit_path = Path(tmpdir) / "pytest.xml"

        print("\nRunning non-performance tests...")
        test_run = run_tests(junit_path)

        junit_error = None
        failures: list[FailureDetail] = []
        if test_run.returncode is not None:
            try:
                failures = read_junit_failures(junit_path)
            except JunitReadError as e:
                junit_error = str(e)

        _print_failures(failures)

    print("\n" + "=" * 70)
    print("PERFORMANCE TEST STATUS:")
    print("=" * 70)
    perf_passed, perf_output = check_performance_tests()
    if perf_passed:
        print("Performance tests PASSED")
        for line in perf_output.splitlines():
            if is_performance_metric_line(line):
                print(_safe_for_console(f"  {line.strip()}"))
    else:
        print("Performance tests FAILED")
        print("Performance test output:")
        print(_safe_for_console(perf_output))

    total_failures = len(failures)
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print("=" * 70)
    print(f"Pytest failures/errors: {total_failures}")
    print(f"Performance tests: {'PASSED' if perf_passed else 'FAILED'}")

    exit_code, status_message = status_exit_decision(
        total_failures,
        perf_passed=perf_passed,
        pytest_returncode=test_run.returncode,
        junit_error=junit_error,
        failures=failures,
    )
    print(f"\n{status_message}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
