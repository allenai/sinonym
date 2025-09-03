"""
Mixed Scripts Test Suite

This module contains tests for names with mixed scripts and special characters:
- Han characters mixed with romanization
- Diacritical marks and accented characters
- Full-width characters (common in PDFs)
- OCR artifacts and scanning errors
- Unicode normalization issues
"""

import sys
from pathlib import Path

# Add the parent directory to path to import sinonym
sys.path.insert(0, str(Path(__file__).parent.parent))

from sinonym import ChineseNameDetector
from tests._fail_log import log_failure

# Test cases for mixed scripts, diacritics, and special characters
CHINESE_NAME_TEST_CASES = [
    ("Chao（冯超） Feng", (True, "Chao Feng")),
    ("Chü Chen", (True, "Chu Chen")),
    ("Li Yü", (True, "Yu Li")),
    ("Lü Buwei", (True, "Bu-Wei Lu")),
    ("Lü Wei", (True, "Wei Lu")),
    ("Wei-min Zhang 张为民", (True, "Wei-Min Zhang")),
    ("Xiaohong Li 张小红", (True, "Xiao-Hong Li")),
    ("Yü Li", (True, "Yu Li")),
    ("Yü Ying-shih", (True, "Ying-Shih Yu")),
    ("Zhou（Mary）Li", (True, "Li Zhou")),
    ("刘（Thomas）Wang", (True, "Liu Wang")),
    ("张为民 Wei-min Zhang", (True, "Wei-Min Zhang")),
    ("李（Peter）Chen", (True, "Li Chen")),
    ("贺娟 He Juan", (True, "Juan He")),
    ("陈丹 Chen Dan", (True, "Dan Chen")),
    ("陈（David）Liu", (True, "Chen Liu")),
    ("Chao（冯超） Feng", (True, "Chao Feng")),
    ("Wei-min Zhang 张为民", (True, "Wei-Min Zhang")),
    ("Xiaohong Li 张小红", (True, "Xiao-Hong Li")),
    ("张为民 Wei-min Zhang", (True, "Wei-Min Zhang")),
]


def test_mixed_scripts():
    """Test names with mixed scripts, diacritics, and special characters."""
    detector = ChineseNameDetector()

    passed = 0
    failed = 0

    for input_name, expected in CHINESE_NAME_TEST_CASES:
        result = detector.is_chinese_name(input_name)
        # Convert ParseResult to tuple format for comparison
        result_tuple = (result.success, result.result if result.success else result.error_message)

        if result_tuple == expected:
            passed += 1
        else:
            failed += 1
            expected_success, expected_name = expected
            actual = result.result if result.success else result.error_message
            print(
                f"FAILED: '{input_name}': expected ({expected_success}, '{expected_name}'), got ({result.success}, '{actual}')",
            )
            log_failure("Mixed scripts tests", input_name, expected_success, expected_name, result.success, actual)

    if failed:
        print(f"Mixed scripts tests: {failed} failures out of {len(CHINESE_NAME_TEST_CASES)} tests")
    assert failed == 0, f"Mixed scripts tests: {failed} failures out of {len(CHINESE_NAME_TEST_CASES)} tests"
    print(f"Mixed scripts tests: {passed} passed, {failed} failed")


if __name__ == "__main__":
    test_mixed_scripts()
