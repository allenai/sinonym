"""
Mixed Scripts Test Suite

This module contains tests for names with mixed scripts and special characters:
- Han characters mixed with romanization
- Diacritical marks and accented characters
- Full-width characters (common in PDFs)
- OCR artifacts and scanning errors
- Unicode normalization issues
"""

import pytest

from tests._case_assertions import assert_normalized_name

# Test cases for mixed scripts, diacritics, and special characters
CHINESE_NAME_TEST_CASES = [
    ("Chao（冯超） Feng", (True, "Chao Feng")),
    ("Chü Chen", (True, "Chu Chen")),
    ("Li Yü", (True, "Yu Li")),
    ("Lü Buwei", (True, "Bu-Wei Lu")),
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


@pytest.mark.parametrize(("input_name", "expected"), CHINESE_NAME_TEST_CASES)
def test_mixed_scripts(detector, input_name, expected):
    """Test names with mixed scripts, diacritics, and special characters."""
    assert_normalized_name(detector, input_name, expected)
