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

from sinonym.detector import ChineseNameDetector
from tests._fail_log import log_failure

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


def test_compound_surname_format_preservation(detector):
    """Test that compound surname formats are preserved while fixing capitalization."""

    passed = 0
    failed = 0

    for input_name, expected_result in COMPOUND_SURNAME_TEST_CASES:
        result = detector.normalize_name(input_name)

        # Extract expected success status and name from tuple
        expected_success, expected_name = expected_result

        if result.success == expected_success and (not expected_success or result.result == expected_name):
            passed += 1
        else:
            failed += 1
            actual = result.result if result.success else f"ERROR: {result.error_message}"
            actual_text = result.result if result.success else result.error_message
            print(
                f"FAILED: '{input_name}': expected ({expected_success}, '{expected_name}'), got ({result.success}, '{actual_text}')",
            )
            log_failure(
                "Compound surname format tests",
                input_name,
                expected_success,
                expected_name,
                result.success,
                actual_text,
            )

    if failed:
        print(f"Compound surname format tests: {failed} failures out of {len(COMPOUND_SURNAME_TEST_CASES)} tests")
    assert failed == 0, f"Compound surname format tests: {failed} failures out of {len(COMPOUND_SURNAME_TEST_CASES)} tests"
    print(f"Compound surname format tests: {passed} passed, {failed} failed")


if __name__ == "__main__":
    test_compound_surname_format_preservation()
    print("All compound surname format tests passed!")
