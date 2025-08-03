"""
Regional Variants Test Suite

This module contains tests for different regional Chinese name romanization systems:
- Cantonese romanizations (Hong Kong style)
- Wade-Giles forms (Traditional/Taiwanese)
- Historical and alternative romanization systems
"""

import sys
from pathlib import Path

# Add the parent directory to path to import sinonym
sys.path.insert(0, str(Path(__file__).parent.parent))

from sinonym import ChineseNameDetector

# Test cases for regional variants and romanization systems
CHINESE_NAME_TEST_CASES = [
    ("Chan Tai Man", (True, "Tai-Man Chan")),
    ("Cheung Hok Yau", (True, "Hok-Yau Cheung")),
    ("Chow Yun Fat", (True, "Yun-Fat Chow")),
    ("Fung Hiu Man", (True, "Hiu-Man Fung")),
    ("Goh Chok Tong", (True, "Chok-Tong Goh")),
    ("Kwok Fu Shing", (True, "Fu-Shing Kwok")),
    ("Lam Ching Ying", (True, "Ching-Ying Lam")),
    ("Lau Suk Yan", (True, "Suk-Yan Lau")),
    ("Lau Tak Wah", (True, "Tak-Wah Lau")),
    ("Leung Chiu Wai", (True, "Chiu-Wai Leung")),
    ("Ng Man Tat", (True, "Man-Tat Ng")),
    ("Siu Ming Wong", (True, "Siu-Ming Wong")),
    ("Szeto Wai Kin", (True, "Wai-Kin Szeto")),
    ("Teo Chee Hean", (True, "Chee-Hean Teo")),
    ("Tsang Chi Wai", (True, "Chi-Wai Tsang")),
    ("Tse Ting Fung", (True, "Ting-Fung Tse")),
    ("Wong Siu Ming", (True, "Siu-Ming Wong")),
    ("Yeung Chin Wah", (True, "Chin-Wah Yeung")),
    ("Chan Tai Man", (True, "Tai-Man Chan")),
    ("Cheung Hok Yau", (True, "Hok-Yau Cheung")),
    ("Chow Yun Fat", (True, "Yun-Fat Chow")),
    ("Fung Hiu Man", (True, "Hiu-Man Fung")),
    ("Goh Chok Tong", (True, "Chok-Tong Goh")),
    ("Kwok Fu Shing", (True, "Fu-Shing Kwok")),
    ("Lam Ching Ying", (True, "Ching-Ying Lam")),
    ("Lau Tak Wah", (True, "Tak-Wah Lau")),
    ("Leung Chiu Wai", (True, "Chiu-Wai Leung")),
    ("Ng Man Tat", (True, "Man-Tat Ng")),
    ("Siu Ming Wong", (True, "Siu-Ming Wong")),
    ("Teo Chee Hean", (True, "Chee-Hean Teo")),
    ("Tsang Chi Wai", (True, "Chi-Wai Tsang")),
    ("Tse Ting Fung", (True, "Ting-Fung Tse")),
    ("Wong Siu Ming", (True, "Siu-Ming Wong")),
    ("Yeung Chin Wah", (True, "Chin-Wah Yeung")),
]


def test_regional_variants():
    """Test regional variants including Cantonese, Wade-Giles, and Taiwanese forms."""
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
            print(f"FAILED: '{input_name}': expected {expected}, got {result_tuple}")

    assert failed == 0, f"Regional variant tests: {failed} failures out of {len(CHINESE_NAME_TEST_CASES)} tests"
    print(f"Regional variant tests: {passed} passed, {failed} failed")


if __name__ == "__main__":
    test_regional_variants()
