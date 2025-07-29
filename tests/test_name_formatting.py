"""
Name Formatting Test Suite

This module contains tests for various name formatting patterns including:
- Hyphenated names
- Comma-separated format ("Last, First")
- Names with periods/dots
- Whitespace handling
- Different capitalization patterns
"""

import sys
from pathlib import Path

# Add the parent directory to path to import sinonym
sys.path.insert(0, str(Path(__file__).parent.parent))

from sinonym import ChineseNameDetector

# Test cases for name formatting and separators
CHINESE_NAME_TEST_CASES = [
    # Hyphenated names
    ("Yu-Zhong Wei", (True, "Yu-Zhong Wei")),
    ("Yu-zhong Wei", (True, "Yu-Zhong Wei")),
    ("Yuzhong Wei", (True, "Yu-Zhong Wei")),
    ("YuZhong Wei", (True, "Yu-Zhong Wei")),
    ("Yu Zhong Wei", (True, "Yu-Zhong Wei")),
    ("Xiao-Hong Li", (True, "Xiao-Hong Li")),
    ("Xiaohong Li", (True, "Xiao-Hong Li")),
    ("Liu Zhi-guo", (True, "Zhi-Guo Liu")),
    ("Yu Jian-guo", (True, "Jian-Guo Yu")),
    ("He Jian-guo", (True, "Jian-Guo He")),
    ("Zhang Hong-xin", (True, "Hong-Xin Zhang")),
    ("Chen-Hung Huang", (True, "Chen-Hung Huang")),
    ("Cheng-Hung Huang", (True, "Cheng-Hung Huang")),
    ("Chia-Ming Chang", (True, "Chia-Ming Chang")),
    ("Chine-Feng Wu", (True, "Chine-Feng Wu")),
    ("Shu-Juan Li", (True, "Shu-Juan Li")),
    ("Dan-Dan Zhang", (True, "Dan-Dan Zhang")),
    ("Dan Dan Zhang", (True, "Dan-Dan Zhang")),
    ("Dan-dan Zhang", (True, "Dan-Dan Zhang")),
    ("Shi-Juan Li", (True, "Shi-Juan Li")),
    ("LI Xiao-juan", (True, "Xiao-Juan Li")),
    ("XIAO-JUAN LI", (True, "Xiao-Juan Li")),
    ("Xiao Juan Li", (True, "Xiao-Juan Li")),
    ("Xiao-juan Li", (True, "Xiao-Juan Li")),
    ("Li Xiao-juan", (True, "Xiao-Juan Li")),
    ("Li Xiao-Juan", (True, "Xiao-Juan Li")),
    ("Xiao-Juan Li", (True, "Xiao-Juan Li")),
    ("Min-Hung Lee", (True, "Min-Hung Lee")),  # Korean-style hyphenation but Chinese content
    ("Au-Yeung Ka-Ming", (True, "Ka-Ming Au-Yeung")),
    ("Chan Tai-Man", (True, "Tai-Man Chan")),

    # Names with periods and initials
    ("Y. Z. Wei", (True, "Y-Z Wei")),
    ("X.-H. Li", (True, "X-H Li")),
    ("A. I. Lee", (True, "A-I Lee")),
    ("P.Y. Huang", (True, "P-Y Huang")),
    ("D. W. Wang", (True, "D-W Wang")),
    ("Li.Wei.Zhang", (True, "Li-Wei Zhang")),
    ("X. Han", (True, "X Han")),
    ("L Han", (True, "L Han")),
    ("X. F. Han", (True, "X-F Han")),
    ("R. Han", (True, "R Han")),
    ("L. Han", (True, "L Han")),
    ("X F Han", (True, "X-F Han")),
    ("X. -F. Han", (True, "X-F Han")),
    (". X.F.Han", (True, "X-F Han")),
    ("Zhang W.", (True, "W Zhang")),  # Keep initial as-is
    ("Liu X.Y.", (True, "X-Y Liu")),  # Keep initials as-is
    ("Chen J.-M.", (True, "J-M Chen")),  # Keep initials as-is
    ("Wu M.J.", (True, "M-J Wu")),  # Keep initials as-is
    ("Wang B.", (True, "B Wang")),

    # Comma-separated "LAST, First" format tests (academic/professional contexts)
    ("Wei, Yu-Zhong", (True, "Yu-Zhong Wei")),
    ("Liu, Dehua", (True, "De-Hua Liu")),
    ("Zhang, Wei", (True, "Wei Zhang")),
    ("Chen, Yu", (True, "Yu Chen")),
    ("Wang, Li Ming", (True, "Li-Ming Wang")),
    ("Ouyang, Xiaoming", (True, "Xiao-Ming Ouyang")),
    ("Wong, Siu Ming", (True, "Siu-Ming Wong")),
    ("Chan, Tai Man", (True, "Tai-Man Chan")),
    ("Au-Yeung, Ka-Ming", (True, "Ka-Ming Au-Yeung")),
    ("Choi, Suk-Zan", (True, "Suk-Zan Choi")),
    ("Chen, Mei Ling", (True, "Mei-Ling Chen")),
    ("Wu, Yufei", (True, "Yu-Fei Wu")),
    ("Yuan, Li-Ming", (True, "Li-Ming Yuan")),  # Comma with compound given name
    ("Zeng, Wei", (True, "Wei Zeng")),  # Comma-separated format

    # Test with extra whitespace (should be handled gracefully)
    ("Wei,   Yu-Zhong", (True, "Yu-Zhong Wei")),
    ("Liu,Dehua", (True, "De-Hua Liu")),  # No space after comma
    ("  Zhang  ,  Wei  ", (True, "Wei Zhang")),  # Extra whitespace
    ("Chen,Mei Ling", (True, "Mei-Ling Chen")),  # Missing space after comma
    ("Wu,Yu Fei", (True, "Yu-Fei Wu")),  # Missing space after comma
    ("Liu, Xiao-ming", (True, "Xiao-Ming Liu")),  # Already hyphenated

    # Extended hyphenated patterns
    ("Xiao Ming-hui Li", (True, "Xiao-Ming-Hui Li")),

    # Mixed script with parenthetical names
    ("Zhang（Wei）Ming", (True, "Ming Zhang")),  # Han surname + roman given name in parentheses
    ("张（Wei）Ming", (True, "Ming Zhang")),  # Han surname + roman given name in parentheses
    ("李（Peter）Chen", (True, "Li Chen")),  # Han surname, Western given name stripped
    ("Wang（小明）Zhang", (True, "Zhang Wang")),  # Roman surname + Han given name in parentheses
    ("陈（David）Liu", (True, "Chen Liu")),  # Han surname, Western given name stripped
    ("Zhou（Mary）Li", (True, "Li Zhou")),  # Mixed Han/Roman with Western name stripped
    ("刘（Thomas）Wang", (True, "Liu Wang")),  # Han surname, Western given name stripped
    ("Chao（冯超） Feng", (True, "Chao Feng")),

    # Concatenated name handling - single token cases (fixed in this session)
    ("LinShu", (True, "Shu Lin")),  # Simple CamelCase → Lin Shu
    ("XIAOChen", (True, "Xiao Chen")),  # Mixed caps → XIAO Chen
    ("LuWANG", (True, "Lu Wang")),  # Mixed caps → Lu WANG
    ("XiaoMing", (True, "Ming Xiao")),  # Chinese CamelCase (surname first format)
    ("ZhangWei", (True, "Wei Zhang")),  # Chinese CamelCase (surname first format)
    ("ZengWei", (True, "Wei Zeng")),  # CamelCase concatenation
    ("YuanLi", (True, "Yuan Li")),  # CamelCase concatenation - Yuan detected as surname
    ("OuMing", (True, "Ming Ou")),  # CamelCase concatenation
    ("JinHua", (True, "Hua Jin")),  # CamelCase concatenation

    # Multi-token cases with concatenated elements
    ("XiaoMing Li", (True, "Xiao-Ming Li")),  # Multi-token with concatenated given name

    # Special apostrophe handling (Asian keyboard input)
    ("Ts'ao Ming", (True, "Ming Ts'ao")),  # Full-width apostrophe: preserves Wade-Giles form
    ("Ch'en Wei", (True, "Wei Ch'en")),  # Ch'en → qen → chen via Wade-Giles + SYLLABLE_RULES
    ("K'ung Fu", (True, "Fu K'ung")),  # Full-width apostrophe: preserves Wade-Giles form
    ("T'ang Li", (True, "T'ang Li")),  # Full-width apostrophe: preserves Wade-Giles form
    ("P'eng Yu", (True, "Yu P'eng")),  # Full-width apostrophe: preserves Wade-Giles form
    ("Ts'ao Ts'ai", (True, "Ts'ai Ts'ao")),  # Mixed ASCII and full-width apostrophes

    # Complex formatting patterns
    ("Ou Yang Wei Ming", (True, "Wei-Ming Ou Yang")),  # Preserves spaced compound format
    ("Ou-Yang Wei Ming", (True, "Wei-Ming Ou-Yang")),  # Tests that hyphenated compounds expand correctly
    ("Si-Ma Qian Feng", (True, "Qian-Feng Si-Ma")),  # Tests section 3 vs section 2 parse generation
    ("AuYeung Ka Ming", (True, "Ka-Ming AuYeung")),  # Preserves camelCase format

    # Mixed Han characters with romanization formatting
    ("张为民 Wei-min Zhang", (True, "Wei-Min Zhang")),
    ("Wei-min Zhang 张为民", (True, "Wei-Min Zhang")),
    ("Wei Min Zhang", (True, "Wei-Min Zhang")),
    ("Xiaohong Li 张小红", (True, "Xiao-Hong Li")),
    ("張Wei Ming", (True, "Wei-Ming Zhang")),
    ("贺娟 He Juan", (True, "Juan He")),  # Mixed Han characters
    ("陈丹 Chen Dan", (True, "Dan Chen")),  # Mixed Han characters

    # Special character edge cases
    ("Ying-Nan P. Chen", (True, "Ying Nan P Chen")),  # TODO
]


def test_name_formatting():
    """Test various name formatting patterns including hyphens, commas, periods."""
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

    assert failed == 0, f"Name formatting tests: {failed} failures out of {len(CHINESE_NAME_TEST_CASES)} tests"
    print(f"Name formatting tests: {passed} passed, {failed} failed")


if __name__ == "__main__":
    test_name_formatting()
