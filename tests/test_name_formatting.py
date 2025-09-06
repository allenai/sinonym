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
from tests._fail_log import log_failure

# Test cases for name formatting and separators
CHINESE_NAME_TEST_CASES = [
    ("  Zhang  ,  Wei  ", (True, "Wei Zhang")),
    (". X.F.Han", (True, "X-F Han")),
    ("A. I. Lee", (True, "A-I Lee")),
    ("Ch'en Wei", (True, "Wei Ch'en")),
    ("Chan Tai-Man", (True, "Tai-Man Chan")),
    ("Chen J.-M.", (True, "J-M Chen")),
    ("Chen,Mei Ling", (True, "Mei-Ling Chen")),
    ("D. W. Wang", (True, "D-W Wang")),
    ("Dan-dan Zhang", (True, "Dan-Dan Zhang")),
    ("JinHua", (True, "Hua Jin")),
    ("L. Han", (True, "L Han")),
    ("LI Xiao-juan", (True, "Xiao-Juan Li")),
    ("Li.Wei.Zhang", (True, "Li-Wei Zhang")),
    ("Liu X.Y.", (True, "X-Y Liu")),
    ("Liu, Xiao-ming", (True, "Xiao-Ming Liu")),
    ("LuWANG", (True, "Lu Wang")),
    ("Min-Hung Lee", (True, "Min-Hung Lee")),
    ("OuMing", (True, "Ming Ou")),
    ("P.Y. Huang", (True, "P-Y Huang")),
    ("R. Han", (True, "R Han")),
    ("Ts'ao Ming", (True, "Ming Ts'ao")),
    ("Wang B.", (True, "B Wang")),
    ("Wei,   Yu-Zhong", (True, "Yu-Zhong Wei")),
    ("Wei, Yu-Zhong", (True, "Yu-Zhong Wei")),
    ("Wu M.J.", (True, "M-J Wu")),
    ("Wu,Yu Fei", (True, "Yu-Fei Wu")),
    ("X. -F. Han", (True, "X-F Han")),
    ("X. F. Han", (True, "X-F Han")),
    ("X. Han", (True, "X Han")),
    ("X.-H. Li", (True, "X-H Li")),
    ("XIAO-JUAN LI", (True, "Xiao-Juan Li")),
    ("XIAOChen", (True, "Xiao Chen")),
    ("Xiao Ming-hui Li", (True, "Xiao-Ming-Hui Li")),
    ("Y. Z. Wei", (True, "Y-Z Wei")),
    ("Yuan, Li-Ming", (True, "Li-Ming Yuan")),
    ("YuanLi", (True, "Yuan Li")),
    ("Zeng, Wei", (True, "Wei Zeng")),
    ("ZengWei", (True, "Wei Zeng")),
    ("Zhang W.", (True, "W Zhang")),
    (". X.F.Han", (True, "X-F Han")),
    ("A. I. Lee", (True, "A-I Lee")),
    ("Au-Yeung Ka-Ming", (True, "Ka-Ming Au-Yeung")),
    ("Au-Yeung, Ka-Ming", (True, "Ka-Ming Au-Yeung")),
    ("Chan Tai-Man", (True, "Tai-Man Chan")),
    ("Chan, Tai Man", (True, "Tai-Man Chan")),
    ("Chen, Mei Ling", (True, "Mei-Ling Chen")),
    ("Chen, Yu", (True, "Yu Chen")),
    ("Chen-Hung Huang", (True, "Chen-Hung Huang")),
    ("Cheng-Hung Huang", (True, "Cheng-Hung Huang")),
    ("Chia-Ming Chang", (True, "Chia-Ming Chang")),
    ("Chine-Feng Wu", (True, "Chine-Feng Wu")),
    ("Choi, Suk-Zan", (True, "Suk-Zan Choi")),
    ("D. W. Wang", (True, "D-W Wang")),
    ("Dan-Dan Zhang", (True, "Dan-Dan Zhang")),
    ("Dan-dan Zhang", (True, "Dan-Dan Zhang")),
    ("He Jian-guo", (True, "Jian-Guo He")),
    ("L. Han", (True, "L Han")),
    ("LI Xiao-juan", (True, "Xiao-Juan Li")),
    ("Li Xiao-Juan", (True, "Xiao-Juan Li")),
    ("Li Xiao-juan", (True, "Xiao-Juan Li")),
    ("Li.Wei.Zhang", (True, "Li-Wei Zhang")),
    ("Liu Zhi-guo", (True, "Zhi-Guo Liu")),
    ("Liu, Dehua", (True, "De-Hua Liu")),
    ("Ouyang, Xiaoming", (True, "Xiao-Ming Ouyang")),
    ("P.Y. Huang", (True, "P-Y Huang")),
    ("R. Han", (True, "R Han")),
    ("Shi-Juan Li", (True, "Shi-Juan Li")),
    ("Shu-Juan Li", (True, "Shu-Juan Li")),
    ("Wang B.", (True, "B Wang")),
    ("Wang, Li Ming", (True, "Li-Ming Wang")),
    ("Wei Min Zhang", (True, "Wei-Min Zhang")),
    ("Wei,   Yu-Zhong", (True, "Yu-Zhong Wei")),
    ("Wei, Yu-Zhong", (True, "Yu-Zhong Wei")),
    ("Wong, Siu Ming", (True, "Siu-Ming Wong")),
    ("Wu, Yufei", (True, "Yu-Fei Wu")),
    ("X. -F. Han", (True, "X-F Han")),
    ("X. F. Han", (True, "X-F Han")),
    ("X. Han", (True, "X Han")),
    ("X.-H. Li", (True, "X-H Li")),
    ("XIAO-JUAN LI", (True, "Xiao-Juan Li")),
    ("Xiao Juan Li", (True, "Xiao-Juan Li")),
    ("Xiao Ming-hui Li", (True, "Xiao-Ming-Hui Li")),
    ("Xiao-Hong Li", (True, "Xiao-Hong Li")),
    ("Xiao-Juan Li", (True, "Xiao-Juan Li")),
    ("Xiao-juan Li", (True, "Xiao-Juan Li")),
    ("Xiaohong Li", (True, "Xiao-Hong Li")),
    ("Y. Z. Wei", (True, "Y-Z Wei")),
    ("Yu Jian-guo", (True, "Jian-Guo Yu")),
    ("Yu Zhong Wei", (True, "Yu-Zhong Wei")),
    ("Yu-Zhong Wei", (True, "Yu-Zhong Wei")),
    ("Yu-zhong Wei", (True, "Yu-Zhong Wei")),
    ("YuZhong Wei", (True, "Yu-Zhong Wei")),
    ("Yuzhong Wei", (True, "Yu-Zhong Wei")),
    ("Zhang Hong-xin", (True, "Hong-Xin Zhang")),
    ("Zhang, Wei", (True, "Wei Zhang")),
    ("LinShu", (True, "Shu Lin")),
    ("Chen C", (True, "C Chen")),
    ("Li A", (True, "A Li")),
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
            expected_success, expected_name = expected
            actual = result.result if result.success else result.error_message
            print(
                f"FAILED: '{input_name}': expected ({expected_success}, '{expected_name}'), got ({result.success}, '{actual}')",
            )
            log_failure("Name formatting tests", input_name, expected_success, expected_name, result.success, actual)

    if failed:
        print(f"Name formatting tests: {failed} failures out of {len(CHINESE_NAME_TEST_CASES)} tests")
    assert failed == 0, f"Name formatting tests: {failed} failures out of {len(CHINESE_NAME_TEST_CASES)} tests"
    print(f"Name formatting tests: {passed} passed, {failed} failed")


if __name__ == "__main__":
    test_name_formatting()
