"""
Non-Chinese Rejection Test Suite

This module contains tests for names that should be properly rejected as non-Chinese:
- Western names
- Korean names
- Vietnamese names
- Japanese names
- Mixed Western/Chinese names
- Names with forbidden patterns
"""

import sys
from pathlib import Path

# Add the parent directory to path to import sinonym
sys.path.insert(0, str(Path(__file__).parent.parent))

from sinonym import ChineseNameDetector

# Non-Chinese names that should return False (failure reason varies)
NON_CHINESE_TEST_CASES = [
    "Bruce Lee",
    "John Smith",
    "Maria Garcia",
    "Kim Min Soo",
    "Nguyen Van Anh",
    "Le Mai Anh",
    "Tran Thi Lan",
    "Pham Minh Tuan",
    "Sunil Gupta",
    "Sergey Feldman",

    # Korean false positive tests
    "Park Min Jung",
    "Lee Bo-ram",
    "Kim Min-jun",  # Hyphenated Korean name
    "Park Hye-jin",
    "Choi Seung-hyun",
    "Jung Hoon-ki",
    "Lee Seul-gi",
    "Yoon Soo-bin",
    "Han Ji-min",
    "Lim Young-woong",

    # Non-hyphenated Korean names (should be caught by enhanced Korean detection)
    "Kim Minjun",  # Should be caught by multi-syllable Korean pattern detection
    "Park Hyejin",  # Should be caught by multi-syllable Korean pattern detection
    "Lim Soo Jin",  # Should be caught by multiple Korean given name patterns
    "Yoon Soojin",  # Should be caught by multi-syllable Korean pattern detection
    "Choi Seunghyun",  # Should be caught by multi-syllable Korean pattern detection

    # Vietnamese false positive tests
    "Nguyen An He",  # Should be caught by Vietnamese-only surname "nguyen"
    "Hoang Thu Mai",  # Should be caught by Vietnamese structural patterns
    "Le Thi Lan",  # Should be caught by Vietnamese "Thi" middle name pattern
    "Pham Van Duc",  # Should be caught by Vietnamese structural patterns
    "Tran Minh Tuan",  # Should be caught by Vietnamese structural patterns
    "Vo Thanh Son",  # Should be caught by Vietnamese structural patterns
    "Truong Minh Duc",  # Should be caught by Vietnamese-only surname "truong"
    "Trinh Thi Lan",  # Should be caught by Vietnamese-only surname "trinh" + "Thi" pattern
    "Dinh Van Duc",  # Should be caught by Vietnamese-only surname "dinh"
    "Nguyen Thi Mai",  # Should be caught by Vietnamese-only surname + "Thi" pattern

    # Overlapping surname differentiation tests
    "Lim Hye-jin",

    # Western names with initials
    "De Pace A",
    "A. Rubin",
    "E. Moulin",

    # Session fixes - Western names with forbidden phonetic patterns
    "Julian Lee",  # Contains "ian" pattern - should be blocked by cultural validation
    "Christian Wong",  # Contains "ian" pattern
    "Adrian Liu",  # Contains "ian" pattern
    "Adrian Chen",  # Contains "ian" pattern - should be blocked by cultural validation
    "Brian Chen",  # Contains "br" + "ian" patterns

    # Additional Western names ending in "-ian" that should be rejected
    "Julian Smith",
    "Adrian Brown",
    "Christian Jones",
    "Vivian White",
    "Fabian Garcia",
    "Damian Miller",

    # Western names with forbidden patterns that should remain blocked
    "Gloria Martinez",  # Contains "gl" pattern - should be blocked
    "Glenn Johnson",  # Contains "gl" pattern - should be blocked
    "Gloria Chen",  # Western name with Chinese surname - should be blocked
    "Clara Wong",  # Contains "cl" pattern - should be blocked
    "Frank Liu",  # Contains "fr" pattern - should be blocked

    # Session fixes - Korean names (overlapping surnames + Korean given names)
    "Ho-Young Lee",  # Contains "young" Korean pattern

    # Comprehensive Western name pattern fixes - names ending in -ian
    "Sebastian Davis",  # sebastian + -ian pattern
    "Damian Wilson",  # damian + -ian pattern
    "Brian Johnson",  # brian + -ian pattern
    "Ryan Thompson",  # ryan + -ian pattern

    # Western names ending in -an
    "Alan Wilson",  # alan + -an pattern with specific prefix rule
    "Susan Davis",  # susan + -an pattern with specific prefix rule
    "Urban Miller",  # urban + -an pattern
    "Logan Brown",  # logan + -an pattern
    "Jordan Smith",  # jordan + -an pattern
    "Morgan Jones",  # morgan + -an pattern
    "Megan Anderson",  # megan + -an pattern

    # Western names ending in -ana
    "Ana Martinez",  # ana + -ana pattern
    "Dana Wilson",  # dana + -ana pattern
    "Diana Johnson",  # diana + -ana pattern
    "Lana Thompson",  # lana + -ana pattern

    # Western names ending in -na
    "Tina Anderson",  # tina + -na pattern
    "Nina Davis",  # nina + -na pattern
    "Anna Thompson",  # anna + -na pattern
    "Gina Wilson",  # gina + -na pattern
    "Vera Martinez",  # vera + -na pattern
    "Sara Johnson",  # sara + -na pattern
    "Mira Brown",  # mira + -na pattern
    "Nora Smith",  # nora + -na pattern
    "Hanna Jones",  # hanna + -na pattern
    "Sina Miller",  # sina + -na pattern
    "Kina Davis",  # kina + -na pattern

    # Western names ending in -ta
    "Rita Wilson",  # rita + -ta pattern
    "Beta Johnson",  # beta + -ta pattern (technical name)
    "Meta Thompson",  # meta + -ta pattern (technical name)
    "Delta Brown",  # delta + -ta pattern (technical name)

    # Western names ending in -ena
    "Dena Smith",  # dena + -ena pattern
    "Lena Jones",  # lena + -ena pattern
    "Rena Martinez",  # rena + -ena pattern
    "Sena Anderson",  # sena + -ena pattern

    # Western names ending in -ne
    "Anne Wilson",  # anne + -ne pattern
    "Diane Davis",  # diane + -ne pattern
    "June Johnson",  # june + -ne pattern
    "Wayne Thompson",  # wayne + -ne pattern

    # Western names ending in -ina
    "Zina Brown",  # zina + -ina pattern

    # Western names ending in -nna
    "Channa Smith",  # channa + -nna pattern
    "Jenna Jones",  # jenna + -nna pattern

    # Western names ending in -ie
    "Genie Martinez",  # genie + -ie pattern
    "Julie Anderson",  # julie + -ie pattern

    # Individual Western names that don't fit suffix patterns
    "Milan Rodriguez",  # milan individual pattern
    "Liam Garcia",  # liam individual pattern
    "Adam Wilson",  # adam individual pattern
    "Noah Davis",  # noah individual pattern
    "Dean Johnson",  # dean individual pattern
    "Sean Thompson",  # sean individual pattern
    "Juan Brown",  # juan individual pattern
    "Ivan Smith",  # ivan individual pattern
    "Ethan Jones",  # ethan individual pattern
    "Duncan Martinez",  # duncan individual pattern
    "Leon Anderson",  # leon individual pattern
    "Sage Wilson",  # sage individual pattern
    "Karen Davis",  # karen individual pattern
    "Lisa Johnson",  # lisa individual pattern
    "Linda Thompson",  # linda individual pattern
    "Kate Brown",  # kate individual pattern
    "Mike Smith",  # mike individual pattern
    "Eli Jones",  # eli individual pattern
    "Wade Martinez",  # wade individual pattern
    "Heidi Anderson",  # heidi individual pattern

    # Comma-separated non-Chinese names (should still be rejected)
    "Smith, John",
    "Garcia, Maria",
    "Johnson, Brian",
    "Brown, Adrian",
    "Soo, Kim Min",  # Korean name in comma format
    "Anh, Nguyen Van",  # Vietnamese name in comma format
    "Martinez, Gloria",  # Western name with forbidden "gl" pattern

    # Korean names with overlapping surnames (should still be rejected due to Korean given names)
    "Gong Min-soo",  # Overlapping surname + Korean given name patterns
    "Kang Young-ho",  # Overlapping surname + Korean given name patterns
    "An Bo-ram",  # Overlapping surname + Korean given name patterns
    "Koo Hye-jin",  # Overlapping surname + Korean given name patterns
    "Ha Min-jun",  # Overlapping surname + Korean given name patterns

    # Western names with specific "ew" patterns (should still be blocked after pattern refinement)
    "Andrew Smith",  # Contains "drew" pattern
    "Matthew Johnson",  # Contains "thew" pattern
    "Drew Wilson",  # Contains "drew" pattern
    "Stewart Jones",  # Contains "stew" pattern
    "Newton Miller",  # Contains "newt" pattern
    "Hewitt Davis",  # Contains "witt" pattern
    "Newell Garcia",  # Contains "well" pattern
    "Powell Martinez",  # Contains "owell" pattern
    "Andrew Chen",  # Western first name + Chinese surname (should be blocked)
    "Matthew Li",  # Western first name + Chinese surname (should be blocked)

    # Concatenated Western names that should be rejected (fixed in this session)
    "BrownPaul",  # Western CamelCase should be rejected
    "FurukawaKoichi",  # Japanese CamelCase should be rejected
    "SmithJohn",  # Western CamelCase should be rejected
    "JohnsonMike",  # Western CamelCase should be rejected

    # Mixed parenthetical cases that should be rejected (Western names in parentheses)
    "Zhang（Andrew）Smith",  # Mixed with Western name
    "李（Peter）Johnson",  # Mixed with Western surname

    # ═══════════════════════════════════════════════════════════════════════════════
    # ADDITIONAL NON-CHINESE TEST CASES
    # ═══════════════════════════════════════════════════════════════════════════════
    # Extended coverage for various non-Chinese names that should be properly rejected

    # --- Ambiguous Ordering (Should Be Rejected) ---
    "Alexander Wang",  # Western given name → skip
    "Michelle Zhang",  # Western given name
    "Bruce Lee Jun Fan",  # Mixed Western/Chinese
    "Leslie Cheung Kwok Wing",  # Mixed Western/Chinese

    # --- Additional Vietnamese Names ---
    "Nguyen Van Hai",
    "Tran Thi Bich Hang",
    "Le Duy Anh",
    "Pham Tuan Dat",

    # --- Additional Korean Names ---
    "Kim Min Jung",
    "Lee Joon Ho",
    "Park Ji Hoon",
    "Choi Soo Ahn",
    "Jeong Yuna",
    "Hwang Byung Chul",
    "Kang Daniel",

    # --- Japanese Names ---
    "Sato Taro",
    "Tanaka Hanako",
    "Yamamoto Ken",
    "Watanabe Aiko",

    # --- Other Western Names ---
    "Mohammed Ali",

    # --- Korean-style Given Name But Chinese Author (Borderline) ---
    "Kim Jong Il",  # North Korean leader → not Chinese
    "Ryu Seung Hee",  # Korean name pattern
    "Woo Suk Hwan",

    # --- Japanese On'yomi Readings That Look Chinese ---
    "Kato Koichi",  # Japanese academic
    "Honda Masaru",
    "Fujiwara Tetsuya",
]


def test_non_chinese_rejection():
    """Test that non-Chinese names are correctly rejected."""
    detector = ChineseNameDetector()

    passed = 0
    failed = 0

    for input_name in NON_CHINESE_TEST_CASES:
        result = detector.is_chinese_name(input_name)
        if result.success is False:
            passed += 1
        else:
            failed += 1
            print(f"FAILED: '{input_name}': expected False, got {result.success}")

    assert failed == 0, f"Non-Chinese rejection tests: {failed} failures out of {len(NON_CHINESE_TEST_CASES)} tests"
    print(f"Non-Chinese rejection tests: {passed} passed, {failed} failed")


if __name__ == "__main__":
    test_non_chinese_rejection()
