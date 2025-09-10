"""
Compound Names Test Suite

This module contains tests for compound and multi-syllable Chinese names including:
- Compound given names
- Multi-part surnames (Ouyang, Sima, etc.)
- Three-token given names
- Complex name splitting patterns
"""

import sys
from pathlib import Path

# Add the parent directory to path to import sinonym
sys.path.insert(0, str(Path(__file__).parent.parent))

from sinonym import ChineseNameDetector
from tests._fail_log import log_failure

# Test cases for compound and multi-syllable names
CHINESE_NAME_TEST_CASES = [
    ("Au-Yeung Chun", (True, "Chun Au-Yeung")),
    ("Cai Yun-hui", (True, "Yun-Hui Cai")),
    ("Chen Niran", (True, "Ni-Ran Chen")),
    ("Jiangzhou Wang", (True, "Jiang-Zhou Wang")),
    ("Jianping Fan", (True, "Jian-Ping Fan")),
    ("Jianwei Zhang", (True, "Jian-Wei Zhang")),
    ("Jianying Zhou", (True, "Jian-Ying Zhou")),
    ("Leung Ka Fai", (True, "Ka-Fai Leung")),
    ("Li Siran", (True, "Si-Ran Li")),
    ("Li Zeze", (True, "Ze-Ze Li")),
    ("Murong Xue", (True, "Xue Murong")),
    ("Ouyang Xiaoming", (True, "Xiao-Ming Ouyang")),
    ("Ouyang Xiu", (True, "Xiu Ouyang")),
    ("Sa Beining", (True, "Bei-Ning Sa")),
    ("Shangguan Wen", (True, "Wen Shangguan")),
    ("Sun Xiao-long", (True, "Xiao-Long Sun")),
    ("Szeto Wah", (True, "Wah Szeto")),
]


def test_compound_names(detector):
    """Test compound and multi-syllable Chinese names."""

    passed = 0
    failed = 0

    for input_name, expected in CHINESE_NAME_TEST_CASES:
        result = detector.normalize_name(input_name)
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
            log_failure("Compound name tests", input_name, expected_success, expected_name, result.success, actual)

    if failed:
        print(f"Compound name tests: {failed} failures out of {len(CHINESE_NAME_TEST_CASES)} tests")
    assert failed == 0, f"Compound name tests: {failed} failures out of {len(CHINESE_NAME_TEST_CASES)} tests"
    print(f"Compound name tests: {passed} passed, {failed} failed")


if __name__ == "__main__":
    test_compound_names()
