"""
Edge Cases Test Suite

This module contains tests for edge cases and ambiguous names:
- Very short names
- Names with initials
- Reduplicated names
- Overlapping surnames
- Concatenated names
- Ambiguous patterns
"""

import sys
from pathlib import Path

# Add the parent directory to path to import sinonym
sys.path.insert(0, str(Path(__file__).parent.parent))

from sinonym import ChineseNameDetector

# Test cases for edge cases and ambiguous names
CHINESE_NAME_TEST_CASES = [
    # Very Short Names (Could Be Ambiguous)
    ("Li A", (True, "A Li")),
    ("Wang B.", (True, "B Wang")),
    ("Chen C", (True, "C Chen")),
    # Names with initials
    ("Y. Z. Wei", (True, "Y-Z Wei")),
    ("X.-H. Li", (True, "X-H Li")),
    ("A. I. Lee", (True, "A-I Lee")),
    ("P.Y. Huang", (True, "P-Y Huang")),
    ("D. W. Wang", (True, "D-W Wang")),
    ("X. Han", (True, "X Han")),
    ("L Han", (True, "L Han")),
    ("X. F. Han", (True, "X-F Han")),
    ("R. Han", (True, "R Han")),
    ("L. Han", (True, "L Han")),
    ("X F Han", (True, "X-F Han")),
    ("X. -F. Han", (True, "X-F Han")),
    (". X.F.Han", (True, "X-F Han")),
    # Initials & Abbreviated Given Names
    ("Zhang W.", (True, "W Zhang")),  # Keep initial as-is
    ("Liu X.Y.", (True, "X-Y Liu")),  # Keep initials as-is
    ("Chen J.-M.", (True, "J-M Chen")),  # Keep initials as-is
    ("Wu M.J.", (True, "M-J Wu")),  # Keep initials as-is
    # Reduplicated names    ("Dan-Dan Zhang", (True, "Dan-Dan Zhang")),
    ("Dan Dan Zhang", (True, "Dan-Dan Zhang")),
    ("Dan-dan Zhang", (True, "Dan-Dan Zhang")),
    # Reduplicated Given Names (Common in Southern China)
    ("Chen Linlin", (True, "Lin-Lin Chen")),  # System uses hyphens
    ("Wang Nini", (True, "Ni-Ni Wang")),  # System uses hyphens
    ("Li Lili", (True, "Li-Li Li")),  # System uses hyphens
    ("Tu Youyou", (True, "You-You Tu")),  # System uses hyphens for reduplicated names
    # Concatenated name handling - single token cases
    ("LinShu", (True, "Shu Lin")),  # Simple CamelCase → Lin Shu
    ("XIAOChen", (True, "Xiao Chen")),  # Mixed caps → XIAO Chen
    ("LuWANG", (True, "Lu Wang")),  # Mixed caps → Lu WANG
    ("XiaoMing", (True, "Ming Xiao")),  # Chinese CamelCase (surname first format)
    ("ZhangWei", (True, "Wei Zhang")),  # Chinese CamelCase (surname first format)
    # Multi-token cases with concatenated elements
    ("XiaoMing Li", (True, "Xiao-Ming Li")),  # Multi-token with concatenated given name
    # Overlapping surnames with Korean patterns
    ("Han Jun", (True, "Jun Han")),  # Previously rejected due to Korean overlap
    ("Han jun", (True, "Jun Han")),  # Case variant
    ("Jun Han", (True, "Jun Han")),  # Different order
    ("JUN HAN", (True, "Jun Han")),  # All caps
    ("Min-Hung Lee", (True, "Min-Hung Lee")),  # Korean-style hyphenation but Chinese content
    ("Xuefeng Han", (True, "Xue-Feng Han")),  # Chinese compound + overlapping surname
    ("Xiaojuan Han", (True, "Xiao-Juan Han")),  # Chinese compound + overlapping surname
    # Names that were being accepted as Chinese (proper behavior with new algorithm)
    ("Ho Yung Lee", (True, "Ho-Yung Lee")),  # Strong Chinese surname evidence overrides Korean patterns
    ("Ho Yun Lee", (True, "Ho-Yun Lee")),  # Strong Chinese surname evidence overrides Korean patterns
    ("Jin Ho Lee", (True, "Jin-Ho Lee")),  # Strong Chinese surname evidence overrides Korean patterns
    ("Min Soo Lee", (True, "Min-Soo Lee")),  # Strong Chinese surname evidence overrides Korean patterns
    # Additional tests for moved surnames (Korean overlap fixes)
    ("Li Gong", (True, "Gong Li")),  # gong as given name
    ("Koo Ming", (True, "Ming Koo")),  # koo surname test
    ("Zhang Koo", (True, "Zhang Koo")),  # koo as given name (name order preserved)
    ("Wang Kang", (True, "Kang Wang")),  # kang as given name
    ("An Li", (True, "An Li")),  # an surname test (name order preserved)
    ("Chen An", (True, "An Chen")),
    ("Hu Cha", (True, "Cha Hu")),  # cha moved from Korean-only to overlapping
    ("Feng Cha", (True, "Cha Feng")),  # cha moved from Korean-only to overlapping
    ("He Cha", (True, "Cha He")),  # cha moved from Korean-only to overlapping
    # Edge case pattern with dots
    ("Li.Wei.Zhang", (True, "Li-Wei Zhang")),
    # Special character edge cases
    # Typos & Misspellings that should still work
    ("Wang Xueyin", (True, "Xue-Yin Wang")),  # System uses hyphens
    ("Zou Shaoqi", (True, "Shao-Qi Zou")),  # System uses hyphens
    ("Huang Yixuan", (True, "Yi-Xuan Huang")),  # System uses hyphens
    ("Wei, Yu-Zhong", (True, "Yu-Zhong Wei")),
    ("Wei,   Yu-Zhong", (True, "Yu-Zhong Wei")),  # Extra whitespace
    ("  Zhang  ,  Wei  ", (True, "Wei Zhang")),  # Extra whitespace around comma
    ("Xiao Ming-hui Li", (True, "Xiao-Ming-Hui Li")),
    # Missing Space After Comma (OCR Glitch)
    ("Chen,Mei Ling", (True, "Mei-Ling Chen")),  # Missing space after comma
    ("Wu,Yu Fei", (True, "Yu-Fei Wu")),  # Missing space after comma
    ("Liu, Xiao-ming", (True, "Xiao-Ming Liu")),  # Already hyphenated
    # Title Case vs ALL CAPS (from metadata export)
    ("JUN HAN", (True, "Jun Han")),  # All caps
    ("XIAO-JUAN LI", (True, "Xiao-Juan Li")),
    ("LI Xiao-juan", (True, "Xiao-Juan Li")),
]


def test_edge_cases():
    """Test edge cases and ambiguous names."""
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

    assert failed == 0, f"Edge case tests: {failed} failures out of {len(CHINESE_NAME_TEST_CASES)} tests"
    print(f"Edge case tests: {passed} passed, {failed} failed")


if __name__ == "__main__":
    test_edge_cases()
