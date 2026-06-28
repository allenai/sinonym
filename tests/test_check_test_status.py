from pathlib import Path

import pytest

from scripts import check_test_status


def _expected_failure_details() -> list[check_test_status.FailureDetail]:
    return [
        check_test_status.FailureDetail(nodeid, kind, message)
        for nodeid, kind, message in check_test_status.EXPECTED_FAILURE_SIGNATURES
    ]


def _write_junit(path: Path, body: str) -> None:
    path.write_text(f'<?xml version="1.0" encoding="utf-8"?><testsuite>{body}</testsuite>', encoding="utf-8")


def test_sdist_includes_check_test_status_dependency_for_packaged_tests():
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    sdist_section = pyproject.split("[tool.hatch.build.targets.sdist]", maxsplit=1)[1]
    sdist_section = sdist_section.split("\n[", maxsplit=1)[0]

    assert '"tests"' in sdist_section
    assert '"scripts/check_test_status.py"' in sdist_section


def test_read_junit_failures_counts_failed_and_errored_testcases(tmp_path):
    junit_path = tmp_path / "pytest.xml"
    _write_junit(
        junit_path,
        """
        <testcase classname="tests.test_example" name="test_passes" />
        <testcase classname="tests.test_example" name="test_fails[param]">
            <failure message="assert 1 == 2">traceback</failure>
        </testcase>
        <testcase classname="tests.test_example" name="test_errors">
            <error message="ValueError: boom">traceback</error>
        </testcase>
        """,
    )

    failures = check_test_status.read_junit_failures(junit_path)

    assert [failure.kind for failure in failures] == ["failure", "error"]
    assert [failure.nodeid for failure in failures] == [
        "tests.test_example::test_fails[param]",
        "tests.test_example::test_errors",
    ]
    assert [failure.message for failure in failures] == ["assert 1 == 2", "ValueError: boom"]


def test_read_junit_failures_counts_one_real_failure_per_testcase(tmp_path):
    junit_path = tmp_path / "pytest.xml"
    _write_junit(
        junit_path,
        """
        <testcase classname="tests.test_example" name="test_one_case">
            <failure message="first field failed">traceback</failure>
            <failure message="second field failed">traceback</failure>
        </testcase>
        """,
    )

    failures = check_test_status.read_junit_failures(junit_path)

    assert len(failures) == 1
    assert failures[0].nodeid == "tests.test_example::test_one_case"


def test_read_junit_failures_fails_on_missing_or_malformed_xml(tmp_path):
    missing_path = tmp_path / "missing.xml"
    malformed_path = tmp_path / "malformed.xml"
    malformed_path.write_text("<testsuite>", encoding="utf-8")

    with pytest.raises(check_test_status.JunitReadError):
        check_test_status.read_junit_failures(missing_path)

    with pytest.raises(check_test_status.JunitReadError):
        check_test_status.read_junit_failures(malformed_path)


@pytest.mark.parametrize("pytest_returncode", [None, 2])
def test_status_fails_when_pytest_execution_did_not_complete_cleanly(pytest_returncode):
    exit_code, message = check_test_status.status_exit_decision(
        total_failures=0,
        perf_passed=True,
        pytest_returncode=pytest_returncode,
    )

    assert exit_code == 1
    assert "pytest" in message.lower() or "test execution" in message.lower()


def test_status_fails_when_pytest_failed_without_junit_failures():
    exit_code, message = check_test_status.status_exit_decision(
        total_failures=0,
        perf_passed=True,
        pytest_returncode=1,
    )

    assert exit_code == 1
    assert "junit" in message.lower()


def test_status_fails_on_junit_error():
    exit_code, message = check_test_status.status_exit_decision(
        total_failures=0,
        perf_passed=True,
        pytest_returncode=1,
        junit_error="could not read JUnit XML",
    )

    assert exit_code == 1
    assert "JUnit XML" in message


def test_status_detects_regressions_beyond_expected_baseline():
    exit_code, message = check_test_status.status_exit_decision(
        total_failures=check_test_status.EXPECTED_FAILURES + 1,
        perf_passed=True,
        pytest_returncode=1,
    )

    assert exit_code == 1
    assert "Too many failures" in message


def test_status_allows_expected_baseline():
    expected_failures = _expected_failure_details()

    exit_code, message = check_test_status.status_exit_decision(
        total_failures=check_test_status.EXPECTED_FAILURES,
        perf_passed=True,
        pytest_returncode=1,
        failures=expected_failures,
    )

    assert exit_code == 0
    assert "expected baseline" in message


def test_status_fails_on_unexpected_failure_with_same_total_count():
    expected_failures = _expected_failure_details()
    swapped_failures = [
        *expected_failures[:-1],
        check_test_status.FailureDetail(
            "tests.test_new_regression::test_wrong_output",
            "failure",
            "AssertionError: wrong output",
        ),
    ]

    exit_code, message = check_test_status.status_exit_decision(
        total_failures=check_test_status.EXPECTED_FAILURES,
        perf_passed=True,
        pytest_returncode=1,
        failures=swapped_failures,
    )

    assert exit_code == 1
    assert "Unexpected pytest failure signatures" in message
    assert "tests.test_new_regression::test_wrong_output" in message


def test_status_fails_on_changed_failure_message_with_same_nodeids():
    expected_failures = _expected_failure_details()
    changed_failures = [
        check_test_status.FailureDetail(
            expected_failures[0].nodeid,
            expected_failures[0].kind,
            "AssertionError: different wrong output",
        ),
        *expected_failures[1:],
    ]

    exit_code, message = check_test_status.status_exit_decision(
        total_failures=check_test_status.EXPECTED_FAILURES,
        perf_passed=True,
        pytest_returncode=1,
        failures=changed_failures,
    )

    assert exit_code == 1
    assert "Unexpected pytest failure signatures" in message
    assert expected_failures[0].nodeid in message


def test_status_allows_improvement():
    exit_code, message = check_test_status.status_exit_decision(
        total_failures=check_test_status.EXPECTED_FAILURES - 1,
        perf_passed=True,
        pytest_returncode=1,
    )

    assert exit_code == 0
    assert "IMPROVEMENT" in message


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
