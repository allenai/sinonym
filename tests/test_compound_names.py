"""
Compound Names Test Suite

This module contains tests for compound and multi-syllable Chinese names including:
- Compound given names
- Multi-part surnames (Ouyang, Sima, etc.)
- Three-token given names
- Complex name splitting patterns
"""

import pytest

from tests._case_assertions import assert_normalized_name

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


@pytest.mark.parametrize(("input_name", "expected"), CHINESE_NAME_TEST_CASES)
def test_compound_names(detector, input_name, expected):
    """Test compound and multi-syllable Chinese names."""
    assert_normalized_name(detector, input_name, expected)
