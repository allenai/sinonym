"""
Miscellaneous test cases for Chinese name detection.

These test cases focus on pypinyin aliases, CamelCase variations, and other edge cases
that were not covered in the existing test suite.
"""

import pytest

from sinonym import ChineseNameDetector
from tests._fail_log import log_failure

# Test cases for miscellaneous Chinese name detection scenarios
MISC_TEST_CASES = [
    # Pypinyin aliases - Zeng (曾)
    ("Zeng Wei", (True, "Wei Zeng")),
    ("Wei Zeng", (True, "Wei Zeng")),
    ("Zeng Ming-Li", (True, "Ming-Li Zeng")),
    ("Hao-Jun Zeng", (True, "Hao-Jun Zeng")),
    ("Zeng Xiao-Hong", (True, "Xiao-Hong Zeng")),

    # Pypinyin aliases - Yuan (阮)
    ("Yuan Li", (True, "Yuan Li")),
    ("Li Yuan", (True, "Yuan Li")),
    ("Yuan Jian-Guo", (True, "Jian-Guo Yuan")),
    ("Wei-Ming Yuan", (True, "Wei-Ming Yuan")),

    # Pypinyin aliases - Ou (区)
    ("Ou Ming", (True, "Ming Ou")),
    ("Ming Ou", (True, "Ming Ou")),
    ("Ou Xiao-Li", (True, "Xiao-Li Ou")),
    ("Yu-Bin Ou", (True, "Yu-Bin Ou")),

    # Pypinyin aliases - Jin (甘)
    ("Jin Hua", (True, "Hua Jin")),
    ("Hua Jin", (True, "Hua Jin")),
    ("Jin Li-Ming", (True, "Li-Ming Jin")),
    ("Xiao-Yu Jin", (True, "Xiao-Yu Jin")),

    # Pypinyin aliases - Lai (黎)
    ("Lai Bin", (True, "Bin Lai")),
    ("Bin Lai", (True, "Bin Lai")),
    ("Lai Wei-Jun", (True, "Wei-Jun Lai")),
    ("Ming-Hua Lai", (True, "Ming-Hua Lai")),

    # Pypinyin aliases - Miao (缪)
    ("Miao Yu", (True, "Miao Yu")),
    ("Yu Miao", (True, "Miao Yu")),
    ("Miao Jian-Wei", (True, "Jian-Wei Miao")),
    ("Li-Jun Miao", (True, "Li-Jun Miao")),

    # Pypinyin aliases - Zhai (翟)
    ("Zhai Jun", (True, "Jun Zhai")),
    ("Jun Zhai", (True, "Jun Zhai")),
    ("Zhai Yu-Ming", (True, "Yu-Ming Zhai")),
    ("Xiao-Wei Zhai", (True, "Xiao-Wei Zhai")),

    # Pypinyin aliases - Mo (毛)
    ("Mo Wei", (True, "Wei Mo")),
    ("Wei Mo", (True, "Wei Mo")),
    ("Mo Li-Hua", (True, "Li-Hua Mo")),
    ("Jun-Ming Mo", (True, "Jun-Ming Mo")),

    # Pypinyin aliases - Wen (尹)
    ("Wen Jing", (True, "Jing Wen")),
    ("Jing Wen", (True, "Jing Wen")),
    ("Wen Xiao-Jun", (True, "Xiao-Jun Wen")),
    ("Yu-Li Wen", (True, "Yu-Li Wen")),

    # Mixed cases with pypinyin aliases
    ("Ou-Ming Li", (True, "Ou-Ming Li")),
    ("Jin Wei-Hua", (True, "Wei-Hua Jin")),

    # Realistic full names using pypinyin alias surnames
    ("Zeng Xiao-Ming", (True, "Xiao-Ming Zeng")),
    ("Yuan Jing-Wei", (True, "Jing-Wei Yuan")),
    ("Ou Li-Hua", (True, "Li-Hua Ou")),
    ("Jin Peng-Fei", (True, "Peng-Fei Jin")),
    ("Lai Yu-Qing", (True, "Yu-Qing Lai")),
    ("Miao Zi-Han", (True, "Zi-Han Miao")),
    ("Zhai Hong-Yu", (True, "Hong-Yu Zhai")),
    ("Mo Rui-Xin", (True, "Rui-Xin Mo")),
    ("Wen Mei-Li", (True, "Mei-Li Wen")),

    # Edge cases with initials
    ("L Han", (True, "L Han")),
    ("X F Han", (True, "X-F Han")),

    # Spacing variations
    ("Dan Dan Zhang", (True, "Dan-Dan Zhang")),

    # CamelCase variations
    ("XiaoMing", (True, "Ming Xiao")),
    ("ZhangWei", (True, "Wei Zhang")),
    ("XiaoMing Li", (True, "Xiao-Ming Li")),

    # Korean overlap edge cases
    ("Ho Yung Lee", (True, "Ho-Yung Lee")),

    # Additional compound name patterns
    ("Wang Xueyin", (True, "Xue-Yin Wang")),
    ("Zou Shaoqi", (True, "Shao-Qi Zou")),
    ("Huang Yixuan", (True, "Yi-Xuan Huang")),
]


@pytest.fixture
def detector():
    """Create a ChineseNameDetector instance for testing."""
    return ChineseNameDetector()


def test_misc_chinese_names(detector):
    """Test miscellaneous Chinese name detection scenarios."""
    passed = 0
    failed = 0

    for test_input, expected in MISC_TEST_CASES:
        expected_success, expected_output = expected
        result = detector.is_chinese_name(test_input)

        if result.success == expected_success:
            if expected_success:
                # For successful cases, check the output format
                if result.result == expected_output:
                    passed += 1
                else:
                    failed += 1
                    print(
                        f"FAILED: '{test_input}': expected (True, '{expected_output}'), got (True, '{result.result}')",
                    )
                    log_failure(
                        "Miscellaneous tests",
                        test_input,
                        True,
                        expected_output,
                        True,
                        result.result,
                    )
            else:
                # For failed cases, just check that it failed
                passed += 1
        else:
            failed += 1
            if expected_success:
                print(
                    f"FAILED: '{test_input}': expected (True, '{expected_output}'), got (False, '{result.error_message}')",
                )
                log_failure(
                    "Miscellaneous tests",
                    test_input,
                    True,
                    expected_output,
                    False,
                    result.error_message,
                )
            else:
                print(
                    f"FAILED: '{test_input}': expected (False, '{expected[1]}'), got (True, '{result.result}')",
                )
                log_failure(
                    "Miscellaneous tests",
                    test_input,
                    False,
                    expected[1],
                    True,
                    result.result,
                )

    # Print detailed results
    print(f"\nMiscellaneous Chinese name tests: {passed}/{len(MISC_TEST_CASES)} passed")

    # Assert that all tests pass
    if failed:
        print(f"Miscellaneous tests: {failed} failures out of {len(MISC_TEST_CASES)} tests")
    assert failed == 0, f"Miscellaneous tests: {failed} failures out of {len(MISC_TEST_CASES)} tests"
