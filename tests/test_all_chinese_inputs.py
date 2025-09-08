"""
All-Chinese Input Test Suite

This module contains tests for names that are entirely composed of Chinese characters,
validating the surname-first ordering convention and frequency-based disambiguation.
"""

import sys
from pathlib import Path

# Add the parent directory to path to import sinonym
sys.path.insert(0, str(Path(__file__).parent.parent))

from sinonym import ChineseNameDetector
from tests._fail_log import log_failure

# Test cases for all-Chinese inputs
ALL_CHINESE_INPUT_TEST_CASES = [
    # Basic 2-character names with clear frequency differences
    ("巩俐", (True, "Li Gong")),  # 巩(surname,freq:293) 俐(given) -> Li (surname,freq:63038) Gong (given,freq:159)
    ("李伟", (True, "Wei Li")),  # Li(surname,freq:63038) Wei(given) - surname first (correct Chinese order)
    ("王明", (True, "Ming Wang")),  # Wang(surname,freq:68479) Ming(given) - most common surname
    ("张华", (True, "Hua Zhang")),  # Zhang(surname,freq:62799) Hua(given) - very common surname
    ("刘强", (True, "Qiang Liu")),  # Liu(surname,freq:45525) Qiang(given)
    ("陈静", (True, "Jing Chen")),  # Chen(surname,freq:35080) Jing(given)
    ("杨红", (True, "Hong Yang")),  # Yang(surname,freq:26347) Hong(given)
    ("赵敏", (True, "Min Zhao")),  # Zhao(surname,freq:19288) Min(given)

    # Cases where both characters can be surnames, surname-first convention applies
    ("李王", (True, "Wang Li")),  # Li(surname,first) Wang(given,second) - surname-first convention
    ("张李", (True, "Li Zhang")),  # Zhang(surname,first) Li(given,second) - surname-first convention
    ("陈杨", (True, "Yang Chen")),  # Chen(surname,first) Yang(given,second) - surname-first convention

    # Historical/famous Chinese names
    ("孔子", (True, "Zi Kong")),  # Kong(surname) Zi(given) - Confucius
    ("孟子", (True, "Zi Meng")),  # Meng(surname) Zi(given) - Mencius
    ("老子", (True, "Zi Lao")),  # Lao(surname) Zi(given) - Laozi

    # Compound surnames with given names
    ("欧阳明", (True, "Ming Ou Yang")),  # Ou Yang(compound surname) Ming(given name)
    ("司马华", (True, "Hua Si Ma")),   # Si Ma(compound surname) Hua(given name)
    ("诸葛", (True, "Ge Zhu")),  # Need to check this case

    # Edge cases - single character repeated
    ("林林", (True, "Lin Lin")),  # Lin(surname) Lin(given) - reduplicated name
    ("华华", (True, "Hua Hua")),  # Hua(surname) Hua(given) - reduplicated name

    # Names where the second character has higher surname frequency (but gets surname-first bonus)
    ("明李", (True, "Ming Li")),  # Ming(not surname) + Li(very common surname) -> Ming wins with surname-first bonus
    ("华王伟", (True, "Wang-Wei Hua")),  # Hua(surname) + Wang Wei(compound given name) -> Western order

    # Names with traditional characters (if supported)
    ("張偉", (True, "Wei Zhang")),  # Traditional Zhang Wei
    ("劉強", (True, "Qiang Liu")),  # Traditional Liu Qiang

    # Three character names should fall back to original logic (not handled by special all-Chinese logic)
    ("李明华", (True, "Ming-Hua Li")),  # 3 characters should use existing logic
    ("王小明", (True, "Xiao-Ming Wang")),  # 3 characters should use existing logic
]

def test_all_chinese_inputs(detector):
    """Test all-Chinese input processing with surname-first preference."""

    passed = 0
    failed = 0

    for input_name, expected in ALL_CHINESE_INPUT_TEST_CASES:
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
            log_failure("All-Chinese input tests", input_name, expected_success, expected_name, result.success, actual)

    if failed:
        print(f"All-Chinese input tests: {failed} failures out of {len(ALL_CHINESE_INPUT_TEST_CASES)} tests")
    assert failed == 0, f"All-Chinese input tests: {failed} failures out of {len(ALL_CHINESE_INPUT_TEST_CASES)} tests"
    print(f"All-Chinese input tests: {passed} passed, {failed} failed")


if __name__ == "__main__":
    test_all_chinese_inputs()
