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


def test_compound_surname_format_preservation():
    """Test that compound surname formats are preserved while fixing capitalization."""
    detector = ChineseNameDetector()

    # Format preservation test cases
    format_preservation_cases = [
        # Compact format (no separator)
        ("Duanmu Wenjie", "Wen-Jie Duanmu"),
        ("duanmu wenjie", "Wen-Jie Duanmu"),  # Fix capitalization
        ("DUANMU wenjie", "Wen-Jie Duanmu"),  # Fix capitalization
        ("Sima Xiangru", "Xiang-Ru Sima"),
        ("sima xiangru", "Xiang-Ru Sima"),

        # Spaced format
        ("Au Yeung Chun", "Chun Au Yeung"),
        ("au yeung chun", "Chun Au Yeung"),  # Fix capitalization
        ("AU YEUNG chun", "Chun Au Yeung"),  # Fix capitalization
        ("Ou Yang Wei Ming", "Wei-Ming Ou Yang"),
        ("ou yang wei ming", "Wei-Ming Ou Yang"),
        ("Si Ma Qian Feng", "Qian-Feng Si Ma"),

        # Hyphenated format
        ("Au-Yeung Chun", "Chun Au-Yeung"),
        ("au-yeung chun", "Chun Au-Yeung"),  # Fix capitalization
        ("AU-YEUNG chun", "Chun Au-Yeung"),  # Fix capitalization
        ("Ou-Yang Wei Ming", "Wei-Ming Ou-Yang"),
        ("Si-Ma Qian Feng", "Qian-Feng Si-Ma"),

        # CamelCase format
        ("AuYeung Ka Ming", "Ka-Ming AuYeung"),
        ("OuYang Wei Ming", "Wei-Ming OuYang"),
        ("SiMa Qian Feng", "Qian-Feng SiMa"),
    ]

    passed = 0
    failed = 0

    for input_name, expected_result in format_preservation_cases:
        result = detector.is_chinese_name(input_name)

        if result.success and result.result == expected_result:
            passed += 1
        else:
            failed += 1
            actual = result.result if result.success else f"ERROR: {result.error_message}"
            print(f"FAILED: '{input_name}': expected '{expected_result}', got '{actual}'")

    assert failed == 0, f"Format preservation tests: {failed} failures out of {len(format_preservation_cases)} tests"


def test_mixed_format_edge_cases():
    """Test edge cases and mixed formats."""
    detector = ChineseNameDetector()

    edge_cases = [
        # Capitalization variations - should be handled robustly
        ("auyeung ka ming", "Ka-Ming Auyeung"),  # All lowercase -> treated as compact
        ("AUYEUNG ka ming", "Ka-Ming Auyeung"),  # All uppercase -> treated as compact

        # Single vs multiple given names
        ("AuYeung Li", "Li AuYeung"),
        ("Au Yeung Li", "Li Au Yeung"),
        ("Au-Yeung Li", "Li Au-Yeung"),
        ("Duanmu Li", "Li Duanmu"),
    ]

    passed = 0
    failed = 0

    for input_name, expected_result in edge_cases:
        result = detector.is_chinese_name(input_name)

        if result.success and result.result == expected_result:
            passed += 1
        else:
            failed += 1
            actual = result.result if result.success else f"ERROR: {result.error_message}"
            print(f"FAILED: '{input_name}': expected '{expected_result}', got '{actual}'")

    assert failed == 0, f"Edge case tests: {failed} failures out of {len(edge_cases)} tests"


def test_format_detection_logic():
    """Test that format detection logic works correctly."""
    detector = ChineseNameDetector()

    # Test cases to verify format detection is working
    format_detection_cases = [
        # These should be detected as different formats and formatted accordingly
        ("AuYeung", "camelCase"),  # Should stay camelCase
        ("au yeung", "spaced"),   # Should stay spaced
        ("au-yeung", "hyphenated"), # Should stay hyphenated
        ("auyang", "compact"),    # Should stay compact (though this might not be a real compound)
    ]

    for compound, format_type in format_detection_cases:
        # Test with a simple given name to isolate compound surname formatting
        test_name = f"{compound} Li"
        result = detector.is_chinese_name(test_name)

        if result.success:
            # Extract the surname part - for spaced compounds, we need multiple words
            parts = result.result.split()
            if format_type == "spaced":
                # For spaced compounds, surname is typically the last 2 words
                surname_part = " ".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
            else:
                # For other formats, surname is the last word
                surname_part = parts[-1]
            print(f"Format test: '{compound}' ({format_type}) -> '{surname_part}'")

            # Verify format is preserved
            if format_type == "camelCase":
                assert "AuYeung" in surname_part, f"CamelCase not preserved: {surname_part}"
            elif format_type == "spaced":
                assert " " in surname_part, f"Spaced format not preserved: {surname_part}"
            elif format_type == "hyphenated":
                assert "-" in surname_part, f"Hyphenated format not preserved: {surname_part}"
            elif format_type == "compact":
                assert " " not in surname_part and "-" not in surname_part, f"Compact format not preserved: {surname_part}"


def test_capitalization_normalization():
    """Test that capitalization is properly normalized while preserving format."""
    detector = ChineseNameDetector()

    capitalization_cases = [
        # Various capitalization inputs should all normalize to same output
        ("AuYeung Ka Ming", "Ka-Ming AuYeung"),
        ("aUyEUNG ka ming", "Ka-Ming AuYeung"),  # Malformed camelCase -> normalized to proper camelCase

        ("Au Yeung Ka Ming", "Ka-Ming Au Yeung"),
        ("au yeung ka ming", "Ka-Ming Au Yeung"),
        ("AU YEUNG ka ming", "Ka-Ming Au Yeung"),

        ("Au-Yeung Ka Ming", "Ka-Ming Au-Yeung"),
        ("au-yeung ka ming", "Ka-Ming Au-Yeung"),
        ("AU-YEUNG ka ming", "Ka-Ming Au-Yeung"),

        ("Duanmu Ka Ming", "Ka-Ming Duanmu"),
        ("duanmu ka ming", "Ka-Ming Duanmu"),
        ("DUANMU ka ming", "Ka-Ming Duanmu"),
    ]

    passed = 0
    failed = 0

    for input_name, expected_result in capitalization_cases:
        result = detector.is_chinese_name(input_name)

        if result.success and result.result == expected_result:
            passed += 1
        else:
            failed += 1
            actual = result.result if result.success else f"ERROR: {result.error_message}"
            print(f"FAILED: '{input_name}': expected '{expected_result}', got '{actual}'")

    assert failed == 0, f"Capitalization normalization tests: {failed} failures out of {len(capitalization_cases)} tests"


if __name__ == "__main__":
    test_compound_surname_format_preservation()
    test_mixed_format_edge_cases()
    test_format_detection_logic()
    test_capitalization_normalization()
    print("All compound surname format tests passed!")
