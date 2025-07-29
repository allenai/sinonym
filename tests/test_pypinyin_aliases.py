"""
Pypinyin Frequency Aliases Test Suite

This module contains tests for pypinyin frequency-based surname mappings.
These test cases validate that surnames work correctly even when pypinyin output
differs from romanization system expectations. Each group tests a specific alias.
"""

import sys
from pathlib import Path

# Add the parent directory to path to import sinonym
sys.path.insert(0, str(Path(__file__).parent.parent))

from sinonym import ChineseNameDetector

# Test cases for pypinyin frequency aliases
CHINESE_NAME_TEST_CASES = [
    # ═══════════════════════════════════════════════════════════════════════════════
    # PYPINYIN FREQUENCY ALIAS TEST CASES
    # ═══════════════════════════════════════════════════════════════════════════════
    # These test cases validate that surnames work correctly even when pypinyin output
    # differs from romanization system expectations. Each group tests a specific alias.

    # 曾 (Han) → "ceng" (pypinyin) → "zeng" (expected romanization)
    ("Zeng Wei", (True, "Wei Zeng")),  # Basic alias: zeng surname should work
    ("Wei Zeng", (True, "Wei Zeng")),  # Name order variation
    ("Zeng Ming-Li", (True, "Ming-Li Zeng")),  # Compound given name
    ("Hao-Jun Zeng", (True, "Hao-Jun Zeng")),  # Given name first format
    ("Zeng Xiao-Hong", (True, "Xiao-Hong Zeng")),  # Common given name pattern

    # 阮 (Han) → "ruan" (pypinyin) → "yuan" (expected romanization)
    ("Yuan Li", (True, "Yuan Li")),  # Basic alias: yuan surname should work
    ("Li Yuan", (True, "Yuan Li")),  # Name order variation - Yuan detected as surname
    ("Yuan Jian-Guo", (True, "Jian-Guo Yuan")),  # Compound given name
    ("Wei-Ming Yuan", (True, "Wei-Ming Yuan")),  # Given name first format

    # 区 (Han) → "qu" (pypinyin) → "ou" (expected romanization)
    ("Ou Ming", (True, "Ming Ou")),  # Basic alias: ou surname should work
    ("Ming Ou", (True, "Ming Ou")),  # Name order variation
    ("Ou Xiao-Li", (True, "Xiao-Li Ou")),  # Compound given name
    ("Yu-Bin Ou", (True, "Yu-Bin Ou")),  # Given name first format

    # 甘 (Han) → "gan" (pypinyin) → "jin" (expected romanization)
    ("Jin Hua", (True, "Hua Jin")),  # Basic alias: jin surname should work
    ("Hua Jin", (True, "Hua Jin")),  # Name order variation
    ("Jin Li-Ming", (True, "Li-Ming Jin")),  # Compound given name
    ("Xiao-Yu Jin", (True, "Xiao-Yu Jin")),  # Given name first format

    # 黎 (Han) → "li" (pypinyin) → "lai" (expected romanization)
    ("Lai Bin", (True, "Bin Lai")),  # Basic alias: lai surname should work
    ("Bin Lai", (True, "Bin Lai")),  # Name order variation
    ("Lai Wei-Jun", (True, "Wei-Jun Lai")),  # Compound given name
    ("Ming-Hua Lai", (True, "Ming-Hua Lai")),  # Given name first format

    # 缪 (Han) → "mou" (pypinyin) → "miao" (expected romanization)
    ("Miao Yu", (True, "Miao Yu")),  # Basic alias: miao surname should work
    ("Yu Miao", (True, "Miao Yu")),  # Name order variation - Miao detected as surname
    ("Miao Jian-Wei", (True, "Jian-Wei Miao")),  # Compound given name
    ("Li-Jun Miao", (True, "Li-Jun Miao")),  # Given name first format

    # 翟 (Han) → "di" (pypinyin) → "zhai" (expected romanization)
    ("Zhai Jun", (True, "Jun Zhai")),  # Basic alias: zhai surname should work
    ("Jun Zhai", (True, "Jun Zhai")),  # Name order variation
    ("Zhai Yu-Ming", (True, "Yu-Ming Zhai")),  # Compound given name
    ("Xiao-Wei Zhai", (True, "Xiao-Wei Zhai")),  # Given name first format

    # 毛 (Han) → "mao" (pypinyin) → "mo" (expected romanization)
    ("Mo Wei", (True, "Wei Mo")),  # Basic alias: mo surname should work
    ("Wei Mo", (True, "Wei Mo")),  # Name order variation
    ("Mo Li-Hua", (True, "Li-Hua Mo")),  # Compound given name
    ("Jun-Ming Mo", (True, "Jun-Ming Mo")),  # Given name first format

    # 尹 (Han) → "yin" (pypinyin) → "wen" (expected romanization)
    ("Wen Jing", (True, "Jing Wen")),  # Basic alias: wen surname should work
    ("Jing Wen", (True, "Jing Wen")),  # Name order variation
    ("Wen Xiao-Jun", (True, "Xiao-Jun Wen")),  # Compound given name
    ("Yu-Li Wen", (True, "Yu-Li Wen")),  # Given name first format

    # Mixed format cases with pypinyin aliases
    ("Zeng, Wei", (True, "Wei Zeng")),  # Comma-separated format
    ("Yuan, Li-Ming", (True, "Li-Ming Yuan")),  # Comma with compound given name
    ("Ou-Ming Li", (True, "Ou-Ming Li")),  # Compound with alias as given name
    ("Jin Wei-Hua", (True, "Wei-Hua Jin")),  # Standard compound pattern

    # Edge cases with pypinyin aliases
    ("ZengWei", (True, "Wei Zeng")),  # CamelCase concatenation
    ("YuanLi", (True, "Yuan Li")),  # CamelCase concatenation - Yuan detected as surname
    ("OuMing", (True, "Ming Ou")),  # CamelCase concatenation
    ("JinHua", (True, "Hua Jin")),  # CamelCase concatenation

    # Realistic full names using pypinyin alias surnames
    ("Zeng Xiao-Ming", (True, "Xiao-Ming Zeng")),  # Very common Chinese given name
    ("Yuan Jing-Wei", (True, "Jing-Wei Yuan")),  # Traditional compound given name
    ("Ou Li-Hua", (True, "Li-Hua Ou")),  # Classic Chinese female name
    ("Jin Peng-Fei", (True, "Peng-Fei Jin")),  # Aspirational Chinese male name
    ("Lai Yu-Qing", (True, "Yu-Qing Lai")),  # Literary Chinese name
    ("Miao Zi-Han", (True, "Zi-Han Miao")),  # Modern Chinese name
    ("Zhai Hong-Yu", (True, "Hong-Yu Zhai")),  # Traditional meaningful name
    ("Mo Rui-Xin", (True, "Rui-Xin Mo")),  # Contemporary Chinese name
    ("Wen Mei-Li", (True, "Mei-Li Wen")),  # Beautiful Chinese female name
]


def test_pypinyin_aliases():
    """Test pypinyin frequency-based surname mappings."""
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

    assert failed == 0, f"Pypinyin alias tests: {failed} failures out of {len(CHINESE_NAME_TEST_CASES)} tests"
    print(f"Pypinyin alias tests: {passed} passed, {failed} failed")


if __name__ == "__main__":
    test_pypinyin_aliases()
