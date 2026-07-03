"""
Regional Variants Test Suite

This module contains tests for different regional Chinese name romanization systems:
- Cantonese romanizations (Hong Kong style)
- Wade-Giles forms (Traditional/Taiwanese)
- Historical and alternative romanization systems
"""

import pytest

from tests._case_assertions import assert_normalized_name

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


@pytest.mark.parametrize(("input_name", "expected"), CHINESE_NAME_TEST_CASES)
def test_regional_variants(detector, input_name, expected):
    """Test regional variants including Cantonese, Wade-Giles, and Taiwanese forms."""
    assert_normalized_name(detector, input_name, expected)
