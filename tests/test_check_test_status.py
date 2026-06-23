# ruff: noqa: INP001

import pytest

from scripts import check_test_status

COMBINED_FAILURE_TOTAL = 6
DEFICIT_FAILURE_TOTAL = 3
DEFICIT_ONLY_TOTAL = 2
EXPECTED_WITH_EXTRA_ASSERTION_TOTAL = 2


def _log_line(
    label: str,
    name: str = "Example Name",
    nodeid: str = "tests/test_example.py::test_example",
) -> str:
    return check_test_status.format_fail_log_line(
        {
            "nodeid": nodeid,
            "label": label,
            "name": name,
            "expected_success": True,
            "expected_output": "expected",
            "actual_success": True,
            "actual_output": "actual",
        },
    )


def _baseline_log_lines() -> list[str]:
    return [_log_line("Expected bucket", str(index)) for index in range(check_test_status.EXPECTED_FAILURES)]


def test_combines_logged_and_unlogged_aggregate_failures_without_double_counting():
    logged = [
        _log_line("Basic Chinese name tests", "A", "tests/test_basic_chinese_names.py::test_basic"),
        _log_line("Basic Chinese name tests", "B", "tests/test_basic_chinese_names.py::test_basic"),
        _log_line("Middle name individual (formatted)", "C", "tests/test_middle_names.py::test_middle"),
    ]
    output = (
        "E AssertionError: Basic Chinese name tests: 2 failures out of 10 tests\n"
        "FAILED tests/test_basic_chinese_names.py::test_basic "
        "- AssertionError: Basic Chinese name tests: 2 failures out of 10 tests\n"
        "FAILED tests/test_basic_chinese_names.py::test_basic - AssertionError: Basic Chines...\n"
        "E AssertionError: Middle name individual tests: 1 failures out of 7 tests\n"
        "E AssertionError: Non-Chinese rejection tests: 3 failures out of 308 tests\n"
        "FAILED tests/test_non_chinese_rejection.py::test_non_chinese "
        "- AssertionError: Non-Chinese rejection tests: 3 failures out of 308 tests"
    )

    report = check_test_status.combine_failure_sources(logged, output)

    assert report["total_failures"] == COMBINED_FAILURE_TOTAL
    assert [failure["name"] for failure in report["aggregate_only"]] == ["Non-Chinese rejection tests"]
    assert report["aggregate_deficits"] == []
    assert report["unaggregated_assertions"] == []


def test_counts_aggregate_deficit_when_fail_log_is_incomplete():
    logged = [_log_line("Basic Chinese name tests", "A")]
    output = "E AssertionError: Basic Chinese name tests: 3 failures out of 10 tests"

    report = check_test_status.combine_failure_sources(logged, output)

    assert report["total_failures"] == DEFICIT_FAILURE_TOTAL
    assert report["aggregate_deficits"][0]["failures"] == DEFICIT_ONLY_TOTAL


def test_skips_ellipsis_summary_when_nodeid_matches_logged_bucket():
    logged = [_log_line("Name formatting tests", "A", "tests/test_name_formatting.py::test_name_formatting")]
    output = "FAILED tests/test_name_formatting.py::test_name_formatting - AssertionError: ..."

    report = check_test_status.combine_failure_sources(logged, output)

    assert report["total_failures"] == 1
    assert report["unaggregated_assertions"] == []


def test_skips_truncated_assertion_class_when_nodeid_matches_logged_bucket():
    logged = [_log_line("Basic Chinese name tests", "A", "tests/test_basic_chinese_names.py::test_basic_chinese_names")]
    output = "FAILED tests/test_basic_chinese_names.py::test_basic_chinese_names - AssertionErro..."

    report = check_test_status.combine_failure_sources(logged, output)

    assert report["total_failures"] == 1
    assert report["unaggregated_pytest_failures"] == []


def test_counts_truncated_assertion_class_when_nodeid_is_not_logged():
    logged = [_log_line("Basic Chinese name tests", "A", "tests/test_basic_chinese_names.py::test_basic_chinese_names")]
    output = "FAILED tests/test_other.py::test_other - AssertionErro..."

    report = check_test_status.combine_failure_sources(logged, output)

    assert report["total_failures"] == EXPECTED_WITH_EXTRA_ASSERTION_TOTAL
    assert report["unaggregated_pytest_failures"][0]["name"] == "tests/test_other.py::test_other"


def test_counts_unaggregated_assertion_failure():
    output = "FAILED tests/test_plain.py::test_plain - AssertionError: assert 1 == 2"

    report = check_test_status.combine_failure_sources([], output)

    assert report["total_failures"] == 1
    assert report["unaggregated_assertions"][0]["name"] == "tests/test_plain.py::test_plain"


def test_counts_real_assertion_in_same_node_as_logged_bucket():
    logged = [_log_line("Name formatting tests", "A", "tests/test_name_formatting.py::test_name_formatting")]
    output = "FAILED tests/test_name_formatting.py::test_name_formatting - AssertionError: assert 1 == 2"

    report = check_test_status.combine_failure_sources(logged, output)

    assert report["total_failures"] == EXPECTED_WITH_EXTRA_ASSERTION_TOTAL
    assert report["unaggregated_assertions"][0]["name"] == "tests/test_name_formatting.py::test_name_formatting"


@pytest.mark.parametrize(
    "output",
    [
        "ERROR tests/test_new.py::test_new - ValueError: boom",
        "FAILED tests/test_new.py::test_new - ValueError: boom",
    ],
)
def test_counts_unaggregated_pytest_errors_beyond_expected_baseline(output):
    report = check_test_status.combine_failure_sources(_baseline_log_lines(), output)

    assert report["total_failures"] == check_test_status.EXPECTED_FAILURES + 1
    assert report["unaggregated_pytest_failures"][0]["name"] == "tests/test_new.py::test_new"

    exit_code, message = check_test_status.status_exit_decision(
        report["total_failures"],
        perf_passed=True,
        pytest_returncode=1,
        malformed_entries=report["malformed_entries"],
    )
    assert exit_code == 1
    assert "Too many failures" in message


def test_counts_non_assertion_failure_even_when_node_matches_logged_bucket():
    logged = [_log_line("Name formatting tests", "A", "tests/test_name_formatting.py::test_name_formatting")]
    output = "FAILED tests/test_name_formatting.py::test_name_formatting - ValueError: boom"

    report = check_test_status.combine_failure_sources(logged, output)

    assert report["total_failures"] == EXPECTED_WITH_EXTRA_ASSERTION_TOTAL
    assert report["unaggregated_pytest_failures"][0]["message"] == "ValueError: boom"


@pytest.mark.parametrize("pytest_returncode", [None, 2])
def test_status_fails_when_pytest_execution_did_not_complete(pytest_returncode):
    exit_code, message = check_test_status.status_exit_decision(
        total_failures=0,
        perf_passed=True,
        pytest_returncode=pytest_returncode,
        malformed_entries=[],
    )

    assert exit_code == 1
    assert "pytest" in message.lower() or "test execution" in message.lower()


def test_status_fails_when_pytest_failed_without_parseable_failures():
    exit_code, message = check_test_status.status_exit_decision(
        total_failures=0,
        perf_passed=True,
        pytest_returncode=1,
        malformed_entries=[],
    )

    assert exit_code == 1
    assert "parseable" in message


def test_status_fails_on_malformed_fail_log_entries():
    exit_code, message = check_test_status.status_exit_decision(
        total_failures=check_test_status.EXPECTED_FAILURES,
        perf_passed=True,
        pytest_returncode=1,
        malformed_entries=["bad\tline"],
    )

    assert exit_code == 1
    assert "malformed" in message


def test_performance_status_requires_successful_pytest_return_code(monkeypatch):
    class CompletedProcess:
        stdout = "tests/test_performance.py::test_speed PASSED\nERROR collecting later"
        stderr = ""
        returncode = 1

    monkeypatch.setattr(check_test_status, "_uv_executable", lambda: "uv")
    monkeypatch.setattr(check_test_status.subprocess, "run", lambda *args, **kwargs: CompletedProcess())

    passed, output = check_test_status.check_performance_tests()

    assert not passed
    assert "ERROR collecting later" in output
