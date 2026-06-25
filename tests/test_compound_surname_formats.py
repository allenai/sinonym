"""
Test compound surname format preservation.

This module tests that compound surnames preserve their input structure format
while allowing proper capitalization normalization.

Format preservation rules:
- Compact: "duanmu" → "Duanmu" (compact stays compact)
- Spaced: "au yeung" → "Au Yeung" (spaced stays spaced)
- Hyphenated: "au-yeung" → "Au-Yeung" (hyphenated stays hyphenated)
- CamelCase: "AuYeung" → "AuYeung" (camelCase stays camelCase)
"""

import pytest

from tests._case_assertions import assert_normalized_name

# Format preservation test cases
COMPOUND_SURNAME_TEST_CASES = [
    # Compact format (no separator)
    ("Duanmu Wenjie", (True, "Wen-Jie Duanmu")),
    ("duanmu wenjie", (True, "Wen-Jie Duanmu")),  # Fix capitalization
    ("DUANMU wenjie", (True, "Wen-Jie Duanmu")),  # Fix capitalization
    ("Sima Xiangru", (True, "Xiang-Ru Sima")),
    ("sima xiangru", (True, "Xiang-Ru Sima")),
    # Spaced format
    ("Au Yeung Chun", (True, "Chun Au Yeung")),
    ("au yeung chun", (True, "Chun Au Yeung")),  # Fix capitalization
    ("AU YEUNG chun", (True, "Chun Au Yeung")),  # Fix capitalization
    ("Ou Yang Wei Ming", (True, "Wei-Ming Ou Yang")),
    ("ou yang wei ming", (True, "Wei-Ming Ou Yang")),
    ("Si Ma Qian Feng", (True, "Qian-Feng Si Ma")),
    # Hyphenated format
    ("Au-Yeung Chun", (True, "Chun Au-Yeung")),
    ("au-yeung chun", (True, "Chun Au-Yeung")),  # Fix capitalization
    ("AU-YEUNG chun", (True, "Chun Au-Yeung")),  # Fix capitalization
    ("Ou-Yang Wei Ming", (True, "Wei-Ming Ou-Yang")),
    ("Si-Ma Qian Feng", (True, "Qian-Feng Si-Ma")),
    # CamelCase format
    ("AuYeung Ka Ming", (True, "Ka-Ming AuYeung")),
    ("OuYang Wei Ming", (True, "Wei-Ming OuYang")),
    ("SiMa Qian Feng", (True, "Qian-Feng SiMa")),
    # Mixed format edge cases
    ("auyeung ka ming", (True, "Ka-Ming Auyeung")),  # All lowercase -> treated as compact
    ("AUYEUNG ka ming", (True, "Ka-Ming Auyeung")),  # All uppercase -> treated as compact
    # Single vs multiple given names
    ("AuYeung Li", (True, "Li AuYeung")),
    ("Au Yeung Li", (True, "Li Au Yeung")),
    ("Au-Yeung Li", (True, "Li Au-Yeung")),
    ("Duanmu Li", (True, "Li Duanmu")),
    # Capitalization normalization cases
    ("AuYeung Ka Ming", (True, "Ka-Ming AuYeung")),
    ("aUyEUNG ka ming", (True, "Ka-Ming AuYeung")),  # Malformed camelCase -> normalized to proper camelCase
    ("Au Yeung Ka Ming", (True, "Ka-Ming Au Yeung")),
    ("au yeung ka ming", (True, "Ka-Ming Au Yeung")),
    ("AU YEUNG ka ming", (True, "Ka-Ming Au Yeung")),
    ("Au-Yeung Ka Ming", (True, "Ka-Ming Au-Yeung")),
    ("au-yeung ka ming", (True, "Ka-Ming Au-Yeung")),
    ("AU-YEUNG ka ming", (True, "Ka-Ming Au-Yeung")),
    ("Duanmu Ka Ming", (True, "Ka-Ming Duanmu")),
    ("duanmu ka ming", (True, "Ka-Ming Duanmu")),
    ("DUANMU ka ming", (True, "Ka-Ming Duanmu")),
]


@pytest.mark.parametrize(("input_name", "expected_result"), COMPOUND_SURNAME_TEST_CASES)
def test_compound_surname_format_preservation(detector, input_name, expected_result):
    """Test that compound surname formats are preserved while fixing capitalization."""
    assert_normalized_name(detector, input_name, expected_result)
