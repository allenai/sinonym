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
    ("Bruce Lee", (False, "should_be_rejected")),
    ("John Smith", (False, "should_be_rejected")),
    ("Maria Garcia", (False, "should_be_rejected")),
    ("Kim Min Soo", (False, "should_be_rejected")),
    ("Nguyen Van Anh", (False, "should_be_rejected")),
    ("Le Mai Anh", (False, "should_be_rejected")),
    ("Tran Thi Lan", (False, "should_be_rejected")),
    ("Pham Minh Tuan", (False, "should_be_rejected")),
    ("Sunil Gupta", (False, "should_be_rejected")),
    ("Sergey Feldman", (False, "should_be_rejected")),
    # Korean false positive tests
    ("Kwang-Hwi Cho", (False, "should_be_rejected")),
    ("Park Min Jung", (False, "should_be_rejected")),
    ("Lee Bo-ram", (False, "should_be_rejected")),
    ("Kim Min-jun", (False, "should_be_rejected")),  # Hyphenated Korean name
    ("Park Hye-jin", (False, "should_be_rejected")),
    ("Choi Seung-hyun", (False, "should_be_rejected")),
    ("Jung Hoon-ki", (False, "should_be_rejected")),
    ("Lee Seul-gi", (False, "should_be_rejected")),
    ("Yoon Soo-bin", (False, "should_be_rejected")),
    ("Han Ji-min", (False, "should_be_rejected")),
    ("Lim Young-woong", (False, "should_be_rejected")),
    ("Cho Kwang-Hwi", (False, "should_be_rejected")),
    ("Lee Young-ho", (False, "should_be_rejected")),
    ("Ha-Young Lee", (False, "should_be_rejected")),
    ("Kim Hye-ji", (False, "should_be_rejected")),
    ("Kim Eun-ji", (False, "should_be_rejected")),
    ("Kim Yeon-woo", (False, "should_be_rejected")),
    ("Park Hyeon-woo", (False, "should_be_rejected")),
    ("Park Ji-hye", (False, "should_be_rejected")),
    ("Choi Woo-jin", (False, "should_be_rejected")),
    ("Jung Seok-ho", (False, "should_be_rejected")),
    ("Yoon Seung-hyun", (False, "should_be_rejected")),
    ("Kwon Tae-hyun", (False, "should_be_rejected")),
    ("Ryu Seung-yeon", (False, "should_be_rejected")),
    ("Hwang Kyu-tae", (False, "should_be_rejected")),
    ("Ahn Jae-yong", (False, "should_be_rejected")),
    ("Seo Yeon-woo", (False, "should_be_rejected")),
    ("Oh Hye-rin", (False, "should_be_rejected")),
    ("Bae Hyun-woo", (False, "should_be_rejected")),
    ("Roh Myung-hoon", (False, "should_be_rejected")),
    ("Jeon Hye-won", (False, "should_be_rejected")),
    ("Yoo Ji-hoon", (False, "should_be_rejected")),
    ("Nam Bo-ram", (False, "should_be_rejected")),
    ("Shim Young-jin", (False, "should_be_rejected")),
    ("Noh Kyung-ho", (False, "should_be_rejected")),
    ("Joo Min-jae", (False, "should_be_rejected")),
    ("Bae Eun-jeong", (False, "should_be_rejected")),
    # More diverse real Korean names with varied dash/no-dash forms
    ("Son Heung-min", (False, "should_be_rejected")),
    ("Kim Yuna", (False, "should_be_rejected")),
    ("Lee Byung-hun", (False, "should_be_rejected")),
    ("Ha Jung-woo", (False, "should_be_rejected")),
    ("Gong Yoo", (False, "should_be_rejected")),
    ("Hyun Bin", (False, "should_be_rejected")),
    ("Park Seo-joon", (False, "should_be_rejected")),
    ("Ji Chang-wook", (False, "should_be_rejected")),
    ("Song Joong-ki", (False, "should_be_rejected")),
    ("Lee Seung-gi", (False, "should_be_rejected")),
    ("Cha Eun-woo", (False, "should_be_rejected")),
    ("Kim Woo-bin", (False, "should_be_rejected")),
    ("Yoo Ah-in", (False, "should_be_rejected")),
    ("Youn Yuh-jung", (False, "should_be_rejected")),
    ("Jeon Do-yeon", (False, "should_be_rejected")),
    ("Park Shin-hye", (False, "should_be_rejected")),
    ("Ryu Jun-yeol", (False, "should_be_rejected")),
    ("Choi Woo-shik", (False, "should_be_rejected")),
    ("Kim Go-eun", (False, "should_be_rejected")),
    ("Son Ye-jin", (False, "should_be_rejected")),
    ("Ma Dong-seok", (False, "should_be_rejected")),
    ("Ok Taec-yeon", (False, "should_be_rejected")),
    ("Bae Su-ji", (False, "should_be_rejected")),
    ("Namgoong Min", (False, "should_be_rejected")),
    ("Syngman Rhee", (False, "should_be_rejected")),
    # Additional diverse real Korean names (public figures)
    ("Moon Jae-in", (False, "should_be_rejected")),
    ("Park Geun-hye", (False, "should_be_rejected")),
    ("Roh Moo-hyun", (False, "should_be_rejected")),
    ("Park Bo-gum", (False, "should_be_rejected")),
    ("Kim Soo-hyun", (False, "should_be_rejected")),
    ("Lee Min-ho", (False, "should_be_rejected")),
    ("Song Hye-kyo", (False, "should_be_rejected")),
    ("Jun Ji-hyun", (False, "should_be_rejected")),
    ("Kim Hee-ae", (False, "should_be_rejected")),
    ("Kim Hye-soo", (False, "should_be_rejected")),
    ("Lee Sung-kyung", (False, "should_be_rejected")),
    ("Kang Seung-yoon", (False, "should_be_rejected")),
    ("Choi Min-ho", (False, "should_be_rejected")),
    ("Kim Dong-hyun", (False, "should_be_rejected")),
    ("Park Ji-sung", (False, "should_be_rejected")),
    ("Hwang Hee-chan", (False, "should_be_rejected")),
    ("Kim Seok-jin", (False, "should_be_rejected")),
    ("Min Yoon-gi", (False, "should_be_rejected")),
    ("Jung Ho-seok", (False, "should_be_rejected")),
    ("Park Ji-min", (False, "should_be_rejected")),
    ("Kim Nam-joon", (False, "should_be_rejected")),
    ("Yoo Seung-ho", (False, "should_be_rejected")),
    ("Kim Myung-soo", (False, "should_be_rejected")),
    ("Lee Joon-gi", (False, "should_be_rejected")),
    ("Bong Joon-ho", (False, "should_be_rejected")),
    # Non-hyphenated Korean names (should be caught by enhanced Korean detection)
    ("Kim Minjun", (False, "should_be_rejected")),  # Should be caught by multi-syllable Korean pattern detection
    ("Park Hyejin", (False, "should_be_rejected")),  # Should be caught by multi-syllable Korean pattern detection
    ("Lim Soo Jin", (False, "should_be_rejected")),  # Should be caught by multiple Korean given name patterns
    ("Yoon Soojin", (False, "should_be_rejected")),  # Should be caught by multi-syllable Korean pattern detection
    ("Choi Seunghyun", (False, "should_be_rejected")),  # Should be caught by multi-syllable Korean pattern detection
    # Vietnamese false positive tests
    ("Nguyen An He", (False, "should_be_rejected")),  # Should be caught by Vietnamese-only surname "nguyen"
    ("Hoang Thu Mai", (False, "should_be_rejected")),  # Should be caught by Vietnamese structural patterns
    ("Le Thi Lan", (False, "should_be_rejected")),  # Should be caught by Vietnamese "Thi" middle name pattern
    ("Pham Van Duc", (False, "should_be_rejected")),  # Should be caught by Vietnamese structural patterns
    ("Tran Minh Tuan", (False, "should_be_rejected")),  # Should be caught by Vietnamese structural patterns
    ("Vo Thanh Son", (False, "should_be_rejected")),  # Should be caught by Vietnamese structural patterns
    ("Truong Minh Duc", (False, "should_be_rejected")),  # Should be caught by Vietnamese-only surname "truong"
    (
        "Trinh Thi Lan",
        (False, "should_be_rejected"),
    ),  # Should be caught by Vietnamese-only surname "trinh" + "Thi" pattern
    ("Dinh Van Duc", (False, "should_be_rejected")),  # Should be caught by Vietnamese-only surname "dinh"
    ("Nguyen Thi Mai", (False, "should_be_rejected")),  # Should be caught by Vietnamese-only surname + "Thi" pattern
    # Overlapping surname differentiation tests
    ("Lim Hye-jin", (False, "should_be_rejected")),
    # Western names with initials
    ("De Pace A", (False, "should_be_rejected")),
    ("A. Rubin", (False, "should_be_rejected")),
    ("E. Moulin", (False, "should_be_rejected")),
    # Session fixes - Western names with forbidden phonetic patterns
    ("Julian Lee", (False, "should_be_rejected")),  # Contains "ian" pattern - should be blocked by cultural validation
    ("Christian Wong", (False, "should_be_rejected")),  # Contains "ian" pattern
    ("Adrian Liu", (False, "should_be_rejected")),  # Contains "ian" pattern
    ("Adrian Chen", (False, "should_be_rejected")),  # Contains "ian" pattern - should be blocked by cultural validation
    ("Brian Chen", (False, "should_be_rejected")),  # Contains "br" + "ian" patterns
    # Additional Western names ending in "-ian" that should be rejected
    ("Julian Smith", (False, "should_be_rejected")),
    ("Adrian Brown", (False, "should_be_rejected")),
    ("Christian Jones", (False, "should_be_rejected")),
    ("Vivian White", (False, "should_be_rejected")),
    ("Fabian Garcia", (False, "should_be_rejected")),
    ("Damian Miller", (False, "should_be_rejected")),
    # Western names with forbidden patterns that should remain blocked
    ("Gloria Martinez", (False, "should_be_rejected")),  # Contains "gl" pattern - should be blocked
    ("Glenn Johnson", (False, "should_be_rejected")),  # Contains "gl" pattern - should be blocked
    ("Gloria Chen", (False, "should_be_rejected")),  # Western name with Chinese surname - should be blocked
    ("Clara Wong", (False, "should_be_rejected")),  # Contains "cl" pattern - should be blocked
    ("Frank Liu", (False, "should_be_rejected")),  # Contains "fr" pattern - should be blocked
    # Session fixes - Korean names (overlapping surnames + Korean given names)
    ("Ho-Young Lee", (False, "should_be_rejected")),  # Contains "young" Korean pattern
    ("Ha Young Lee", (False, "should_be_rejected")),  # Contains "young" Korean pattern
    # Additional Korean hyphenated given names with overlapping surname 'Lee'
    ("In-Ho Lee", (False, "should_be_rejected")),
    ("Byung-Ho Lee", (False, "should_be_rejected")),
    ("In-Soo Lee", (False, "should_be_rejected")),
    ("Dong-Hyun Lee", (False, "should_be_rejected")),
    ("Sang-Hoon Lee", (False, "should_be_rejected")),
    ("Jae-Ho Lee", (False, "should_be_rejected")),
    ("Joon-Ho Lee", (False, "should_be_rejected")),
    ("Min-Jae Lee", (False, "should_be_rejected")),
    ("Seung-Hyun Lee", (False, "should_be_rejected")),
    ("Hyun-Woo Lee", (False, "should_be_rejected")),
    ("Hye-Jin Lee", (False, "should_be_rejected")),
    ("Eun-Ji Lee", (False, "should_be_rejected")),
    ("Yeon-Woo Lee", (False, "should_be_rejected")),
    ("Ji-Hoon Lee", (False, "should_be_rejected")),
    # Same style as above but with varied Korean surnames (both overlapping and Korean-only)
    ("In-Ho Kim", (False, "should_be_rejected")),
    ("Byung-Ho Park", (False, "should_be_rejected")),
    ("In-Soo Choi", (False, "should_be_rejected")),
    ("Dong-Hyun Jung", (False, "should_be_rejected")),
    ("Sang-Hoon Yoon", (False, "should_be_rejected")),
    ("Jae-Ho Kwon", (False, "should_be_rejected")),
    ("Joon-Ho Ryu", (False, "should_be_rejected")),
    ("Min-Jae Hwang", (False, "should_be_rejected")),
    ("Seung-Hyun Ahn", (False, "should_be_rejected")),
    ("Hyun-Woo Seo", (False, "should_be_rejected")),
    ("Hye-Jin Oh", (False, "should_be_rejected")),
    ("Eun-Ji Bae", (False, "should_be_rejected")),
    ("Yeon-Woo Roh", (False, "should_be_rejected")),
    ("Ji-Hoon Jeon", (False, "should_be_rejected")),
    ("Tae-Hyun Kang", (False, "should_be_rejected")),
    ("Kyu-Tae Kwon", (False, "should_be_rejected")),
    # Surname-first Korean formatting variants
    ("Kim In-Ho", (False, "should_be_rejected")),
    ("Park Byung-Ho", (False, "should_be_rejected")),
    ("Choi In-Soo", (False, "should_be_rejected")),
    ("Jung Dong-Hyun", (False, "should_be_rejected")),
    ("Yoon Sang-Hoon", (False, "should_be_rejected")),
    ("Kwon Jae-Ho", (False, "should_be_rejected")),
    ("Ryu Joon-Ho", (False, "should_be_rejected")),
    ("Hwang Min-Jae", (False, "should_be_rejected")),
    ("Ahn Seung-Hyun", (False, "should_be_rejected")),
    ("Seo Hyun-Woo", (False, "should_be_rejected")),
    ("Oh Hye-Jin", (False, "should_be_rejected")),
    ("Bae Eun-Ji", (False, "should_be_rejected")),
    ("Roh Yeon-Woo", (False, "should_be_rejected")),
    ("Jeon Ji-Hoon", (False, "should_be_rejected")),
    ("Kang Tae-Hyun", (False, "should_be_rejected")),
    # Comprehensive Western name pattern fixes - names ending in -ian
    ("Sebastian Davis", (False, "should_be_rejected")),  # sebastian + -ian pattern
    ("Damian Wilson", (False, "should_be_rejected")),  # damian + -ian pattern
    ("Brian Johnson", (False, "should_be_rejected")),  # brian + -ian pattern
    ("Ryan Thompson", (False, "should_be_rejected")),  # ryan + -ian pattern
    # Western names ending in -an
    ("Alan Wilson", (False, "should_be_rejected")),  # alan + -an pattern with specific prefix rule
    ("Susan Davis", (False, "should_be_rejected")),  # susan + -an pattern with specific prefix rule
    ("Urban Miller", (False, "should_be_rejected")),  # urban + -an pattern
    ("Logan Brown", (False, "should_be_rejected")),  # logan + -an pattern
    ("Jordan Smith", (False, "should_be_rejected")),  # jordan + -an pattern
    ("Morgan Jones", (False, "should_be_rejected")),  # morgan + -an pattern
    ("Megan Anderson", (False, "should_be_rejected")),  # megan + -an pattern
    # Western names ending in -ana
    ("Ana Martinez", (False, "should_be_rejected")),
    ("Dana Wilson", (False, "should_be_rejected")),
    ("Diana Johnson", (False, "should_be_rejected")),
    ("Lana Thompson", (False, "should_be_rejected")),
    # Western names ending in -na
    ("Tina Anderson", (False, "should_be_rejected")),
    ("Nina Davis", (False, "should_be_rejected")),
    ("Anna Thompson", (False, "should_be_rejected")),
    ("Gina Wilson", (False, "should_be_rejected")),
    ("Vera Martinez", (False, "should_be_rejected")),
    ("Sara Johnson", (False, "should_be_rejected")),
    ("Mira Brown", (False, "should_be_rejected")),
    ("Nora Smith", (False, "should_be_rejected")),
    ("Hanna Jones", (False, "should_be_rejected")),
    ("Sina Miller", (False, "should_be_rejected")),
    ("Kina Davis", (False, "should_be_rejected")),
    # Western names ending in -ta
    ("Rita Wilson", (False, "should_be_rejected")),
    ("Beta Johnson", (False, "should_be_rejected")),
    ("Meta Thompson", (False, "should_be_rejected")),
    ("Delta Brown", (False, "should_be_rejected")),
    # Western names ending in -ena
    ("Dena Smith", (False, "should_be_rejected")),
    ("Lena Jones", (False, "should_be_rejected")),
    ("Rena Martinez", (False, "should_be_rejected")),
    ("Sena Anderson", (False, "should_be_rejected")),
    # Western names ending in -ne
    ("Anne Wilson", (False, "should_be_rejected")),
    ("Diane Davis", (False, "should_be_rejected")),
    ("June Johnson", (False, "should_be_rejected")),
    ("Wayne Thompson", (False, "should_be_rejected")),
    # Western names ending in -ina
    ("Zina Brown", (False, "should_be_rejected")),
    # Western names ending in -nna
    ("Channa Smith", (False, "should_be_rejected")),
    ("Jenna Jones", (False, "should_be_rejected")),
    # Western names ending in -ie
    ("Genie Martinez", (False, "should_be_rejected")),
    ("Julie Anderson", (False, "should_be_rejected")),
    # Individual Western names that don't fit suffix patterns
    ("Milan Rodriguez", (False, "should_be_rejected")),
    ("Liam Garcia", (False, "should_be_rejected")),
    ("Adam Wilson", (False, "should_be_rejected")),
    ("Noah Davis", (False, "should_be_rejected")),
    ("Dean Johnson", (False, "should_be_rejected")),
    ("Sean Thompson", (False, "should_be_rejected")),
    ("Juan Brown", (False, "should_be_rejected")),
    ("Ivan Smith", (False, "should_be_rejected")),
    ("Ethan Jones", (False, "should_be_rejected")),
    ("Duncan Martinez", (False, "should_be_rejected")),
    ("Leon Anderson", (False, "should_be_rejected")),
    ("Sage Wilson", (False, "should_be_rejected")),
    ("Karen Davis", (False, "should_be_rejected")),
    ("Lisa Johnson", (False, "should_be_rejected")),
    ("Linda Thompson", (False, "should_be_rejected")),
    ("Kate Brown", (False, "should_be_rejected")),
    ("Mike Smith", (False, "should_be_rejected")),
    ("Eli Jones", (False, "should_be_rejected")),
    ("Wade Martinez", (False, "should_be_rejected")),
    ("Heidi Anderson", (False, "should_be_rejected")),
    # Comma-separated non-Chinese names (should still be rejected)
    ("Smith, John", (False, "should_be_rejected")),
    ("Garcia, Maria", (False, "should_be_rejected")),
    ("Johnson, Brian", (False, "should_be_rejected")),
    ("Brown, Adrian", (False, "should_be_rejected")),
    ("Soo, Kim Min", (False, "should_be_rejected")),  # Korean name in comma format
    ("Anh, Nguyen Van", (False, "should_be_rejected")),  # Vietnamese name in comma format
    ("Martinez, Gloria", (False, "should_be_rejected")),  # Western name with forbidden "gl" pattern
    # Korean names with overlapping surnames (should still be rejected due to Korean given names)
    ("Gong Min-soo", (False, "should_be_rejected")),
    ("Kang Young-ho", (False, "should_be_rejected")),
    ("An Bo-ram", (False, "should_be_rejected")),
    ("Koo Hye-jin", (False, "should_be_rejected")),
    ("Ha Min-jun", (False, "should_be_rejected")),
    # Western names with specific "ew" patterns (should still be blocked after pattern refinement)
    ("Andrew Smith", (False, "should_be_rejected")),
    ("Matthew Johnson", (False, "should_be_rejected")),
    ("Drew Wilson", (False, "should_be_rejected")),
    ("Stewart Jones", (False, "should_be_rejected")),
    ("Newton Miller", (False, "should_be_rejected")),
    ("Hewitt Davis", (False, "should_be_rejected")),
    ("Newell Garcia", (False, "should_be_rejected")),
    ("Powell Martinez", (False, "should_be_rejected")),
    ("Andrew Chen", (False, "should_be_rejected")),
    ("Matthew Li", (False, "should_be_rejected")),
    # Concatenated Western names that should be rejected
    ("BrownPaul", (False, "should_be_rejected")),
    ("FurukawaKoichi", (False, "should_be_rejected")),
    ("SmithJohn", (False, "should_be_rejected")),
    ("JohnsonMike", (False, "should_be_rejected")),
    # Mixed parenthetical cases that should be rejected
    ("Zhang（Andrew）Smith", (False, "should_be_rejected")),
    ("李（Peter）Johnson", (False, "should_be_rejected")),
    # Additional non-Chinese names
    ("Alexander Wang", (False, "should_be_rejected")),
    ("Michelle Zhang", (False, "should_be_rejected")),
    ("Bruce Lee Jun Fan", (False, "should_be_rejected")),
    ("Leslie Cheung Kwok Wing", (False, "should_be_rejected")),
    # Additional Vietnamese Names
    ("Nguyen Van Hai", (False, "should_be_rejected")),
    ("Tran Thi Bich Hang", (False, "should_be_rejected")),
    ("Le Duy Anh", (False, "should_be_rejected")),
    ("Pham Tuan Dat", (False, "should_be_rejected")),
    # Additional Korean Names
    ("Kim Min Jung", (False, "should_be_rejected")),
    ("Lee Joon Ho", (False, "should_be_rejected")),
    ("Park Ji Hoon", (False, "should_be_rejected")),
    ("Choi Soo Ahn", (False, "should_be_rejected")),
    ("Jeong Yuna", (False, "should_be_rejected")),
    ("Hwang Byung Chul", (False, "should_be_rejected")),
    ("Kang Daniel", (False, "should_be_rejected")),
    # Japanese Names
    ("Sato Taro", (False, "should_be_rejected")),
    ("Tanaka Hanako", (False, "should_be_rejected")),
    ("Yamamoto Ken", (False, "should_be_rejected")),
    ("Watanabe Aiko", (False, "should_be_rejected")),
    # Other Western Names
    ("Mohammed Ali", (False, "should_be_rejected")),
    # Korean-style Given Name But Chinese Author (Borderline)
    ("Kim Jong Il", (False, "should_be_rejected")),
    ("Ryu Seung Hee", (False, "should_be_rejected")),
    ("Woo Suk Hwan", (False, "should_be_rejected")),
    # Japanese On'yomi Readings That Look Chinese
    ("Kato Koichi", (False, "should_be_rejected")),
    ("Honda Masaru", (False, "should_be_rejected")),
    ("Fujiwara Tetsuya", (False, "should_be_rejected")),
    # All-Chinese Character Japanese Names (should be rejected)
    ("佐藤太郎", (False, "should_be_rejected")),  # Sato Taro in Kanji
    ("田中花子", (False, "should_be_rejected")),  # Tanaka Hanako in Kanji
    ("山本健", (False, "should_be_rejected")),  # Yamamoto Ken in Kanji
    ("渡邊愛子", (False, "should_be_rejected")),  # Watanabe Aiko in Kanji
    ("加藤浩一", (False, "should_be_rejected")),  # Kato Koichi in Kanji
    ("本田勝", (False, "should_be_rejected")),  # Honda Masaru in Kanji
    ("藤原哲也", (False, "should_be_rejected")),  # Fujiwara Tetsuya in Kanji
    ("鈴木一郎", (False, "should_be_rejected")),  # Suzuki Ichiro in Kanji
    ("高橋美咲", (False, "should_be_rejected")),  # Takahashi Misaki in Kanji
    ("伊藤博文", (False, "should_be_rejected")),  # Ito Hirobumi (historical figure) in Kanji
]


def test_non_chinese_rejection():
    """Test that non-Chinese names are correctly rejected."""
    detector = ChineseNameDetector()

    passed = 0
    failed = 0

    for input_name, expected_result in NON_CHINESE_TEST_CASES:
        result = detector.is_chinese_name(input_name)

        # Extract expected success status from tuple
        expected_success, _ = expected_result

        if result.success == expected_success:
            passed += 1
        else:
            failed += 1
            actual = result.result if result.success else f"ERROR: {result.error_message}"
            print(f"FAILED: '{input_name}': expected {expected_result}, got ({result.success}, '{actual}')")

    assert failed == 0, f"Non-Chinese rejection tests: {failed} failures out of {len(NON_CHINESE_TEST_CASES)} tests"
    print(f"Non-Chinese rejection tests: {passed} passed, {failed} failed")


if __name__ == "__main__":
    test_non_chinese_rejection()
