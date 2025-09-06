import os
from pathlib import Path


def log_failure(label: str, name: str, expected_success: bool, expected_output: str, actual_success: bool, actual_output: str) -> None:
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
        with open(path, "a", encoding="utf-8") as f:
            f.write(
                f"{label}\t{name!r}\t{expected_success}\t{expected_output}\t{actual_success}\t{actual_output}\n",
            )
    except Exception:
        # Best-effort logging; never break tests due to logging
        return

