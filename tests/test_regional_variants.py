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
    # Cantonese names (Hong Kong/Guangdong romanization)
    ("Chan Tai Man", (True, "Tai-Man Chan")),
    ("Wong Siu Ming", (True, "Siu-Ming Wong")),
    ("Lee Ka Fai", (True, "Ka-Fai Lee")),
    ("Lau Tak Wah", (True, "Tak-Wah Lau")),
    ("Cheung Hok Yau", (True, "Hok-Yau Cheung")),
    ("Chow Yun Fat", (True, "Yun-Fat Chow")),
    ("Ng Man Tat", (True, "Man-Tat Ng")),
    ("Leung Chiu Wai", (True, "Chiu-Wai Leung")),
    ("Kwok Fu Shing", (True, "Fu-Shing Kwok")),
    ("Lam Ching Ying", (True, "Ching-Ying Lam")),
    ("Yeung Chin Wah", (True, "Chin-Wah Yeung")),
    ("Tsang Chi Wai", (True, "Chi-Wai Tsang")),
    ("Fung Hiu Man", (True, "Hiu-Man Fung")),
    ("Tse Ting Fung", (True, "Ting-Fung Tse")),
    ("Yip Man", (True, "Man Yip")),
    ("Siu Ming Wong", (True, "Siu-Ming Wong")),
    ("Ka Fai Lee", (True, "Ka-Fai Lee")),
    ("Chan Tai-Man", (True, "Tai-Man Chan")),
    ("Wong Kit", (True, "Kit Wong")),
    # Cantonese romanization cases (preserved as-is, not mapped through pypinyin alias)
    ("Tsang Wei", (True, "Wei Tsang")),  # Cantonese tsang preserved as surname
    ("Yuen Li", (True, "Yuen Li")),  # Cantonese yuen preserved as surname
    ("Au Ming", (True, "Ming Au")),  # Cantonese au preserved as surname
    ("Kam Hua", (True, "Hua Kam")),  # Cantonese kam preserved as surname
    # Wade-Giles forms (Traditional Chinese romanization)
    ("Li Tsu", (True, "Tsu Li")),  # Tests tsu→cu syllable precedence over ts→z prefix
    ("Wang Tseng", (True, "Tseng Wang")),  # Tests ts→z prefix when no syllable match
    ("Chen Tsi", (True, "Tsi Chen")),  # Tests tsi→ci syllable precedence
    ("Wu Tzu", (True, "Tzu Wu")),  # Tests tzu→zi syllable precedence
    ("Zhang Hsien", (True, "Hsien Zhang")),  # Tests hs→x prefix conversion
    ("Huang Hsia", (True, "Hsia Huang")),  # Tests hsia→xia syllable conversion
    ("Zhou Chuang", (True, "Chuang Zhou")),  # Tests chuang→zhuang syllable conversion
    ("Gao Chuai", (True, "Chuai Gao")),  # Tests chuai→zhuai syllable conversion
    ("Sun Chueh", (True, "Chueh Sun")),  # Tests chueh→jue syllable conversion
    ("Ma Chui", (True, "Chui Ma")),  # Tests chui→zhui syllable conversion
    ("Xu Erh", (True, "Erh Xu")),  # Tests erh→er syllable conversion
    ("Fan Chien", (True, "Chien Fan")),  # Tests ien→ian suffix conversion potential
    # Wade-Giles forms that convert to moved surnames
    ("Wei Kung", (True, "Wei Kung")),  # kung→gong conversion test (name order preserved)
    # Wade-Giles syllables with ü (now work correctly with comprehensive diacritical support)
    ("Yü Li", (True, "Yu Li")),  # Wade-Giles yü -> yu conversion works correctly
    ("Li Yü", (True, "Yu Li")),  # Wade-Giles yü -> yu conversion works correctly
    ("Lü Wei", (True, "Wei Lu")),  # Wade-Giles lü -> lu conversion now works correctly
    ("Chü Chen", (True, "Chu Chen")),  # Wade-Giles chü -> ju conversion works correctly
    ("Lü Buwei", (True, "Bu-Wei Lu")),  # Historical Chinese name now works with ü support
    # Taiwanese / Wade-Giles / Older Forms
    ("Chiang Kai Shek", (True, "Kai-Shek Chiang")),  # Keep as-is if not recognized
    ("Hsu Wen Hsiung", (True, "Wen-Hsiung Hsu")),  # Keep as-is if not recognized
    # Mixed Cantonese patterns with overlapping surnames    # Singapore/Malaysian Chinese variants
    ("Teo Chee Hean", (True, "Chee-Hean Teo")),
    ("Goh Chok Tong", (True, "Chok-Tong Goh")),
    # Cantonese names with compound patterns - preserved romanization
    ("Au Yeung Chun", (True, "Chun Au Yeung")),  # Preserves spaced compound format
    ("Szeto Wai Kin", (True, "Wai-Kin Szeto")),  # Keep Szeto, not Si Tu
    ("Lau Suk Yan", (True, "Suk-Yan Lau")),  # Keep Lau, not Liu
    # Traditional forms with apostrophes
    ("Ts'ao Ming", (True, "Ming Ts'ao")),  # Full-width apostrophe: preserves Wade-Giles form
    ("Ch'en Wei", (True, "Wei Ch'en")),  # Ch'en → qen → chen via Wade-Giles + SYLLABLE_RULES
    ("K'ung Fu", (True, "Fu K'ung")),  # Full-width apostrophe: preserves Wade-Giles form
    ("T'ang Li", (True, "T'ang Li")),  # Full-width apostrophe: preserves Wade-Giles form
    ("P'eng Yu", (True, "Yu P'eng")),  # Full-width apostrophe: preserves Wade-Giles form
    ("Ts'ao Ts'ai", (True, "Ts'ai Ts'ao")),  # Mixed ASCII and full-width apostrophes
    ("Yeh Ming-hsun", (True, "Ming-Hsun Yeh")),  # FAILS
    ("Chou En-lai", (True, "En-Lai Chou")),  # FAILS
    ("Tsiang Tingfu", (True, "Ting-Fu Tsiang")),  # FAILS
    ("Yü Ying-shih", (True, "Ying-Shih Yu")),  # FAILS
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
