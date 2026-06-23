import os
from pathlib import Path

from scripts.check_test_status import format_fail_log_line


def log_failure(  # noqa: PLR0913
    label: str,
    name: str,
    expected_success: bool,
    expected_output: str,
    actual_success: bool,
    actual_output: str,
) -> None:
    """Append a normalized failure line to the shared failure log if enabled.

    The status script sets SINONYM_FAIL_LOG to a writable filepath. When present,
    tests append one line per failing case in a uniform, parse-friendly format.
    """
    path = os.getenv("SINONYM_FAIL_LOG")
    if not path:
        return
    try:
        # Ensure parent dir exists when a nested path is provided
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("a", encoding="utf-8") as f:
            line = format_fail_log_line(
                {
                    "label": label,
                    "name": name,
                    "expected_success": expected_success,
                    "expected_output": expected_output,
                    "actual_success": actual_success,
                    "actual_output": actual_output,
                },
            )
            f.write(f"{line}\n")
    except OSError:
        # Best-effort logging; never break tests due to logging
        return

