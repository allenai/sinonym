"""
Basic Chinese Names Test Suite

This module contains tests for simple, common Chinese name patterns.
Tests basic surname + given name combinations and common Chinese names.
"""

import sys
from pathlib import Path

# Add the parent directory to path to import sinonym
sys.path.insert(0, str(Path(__file__).parent.parent))

from sinonym import ChineseNameDetector

# Basic Chinese names with simple patterns
CHINESE_NAME_TEST_CASES = [
    # Simple surname + given name patterns
    ("Liu Dehua", (True, "De-Hua Liu")),
    ("Dehua Liu", (True, "De-Hua Liu")),
    ("Zhou Xun", (True, "Xun Zhou")),
    ("Xun Zhou", (True, "Xun Zhou")),
    ("Li Ming", (True, "Ming Li")),
    ("Li Na", (True, "Na Li")),
    ("Gao Wei", (True, "Wei Gao")),
    ("Zhang Wei", (True, "Wei Zhang")),
    ("Wang Jun", (True, "Jun Wang")),
    ("Jun Wang", (True, "Jun Wang")),
    ("Chen Yu", (True, "Yu Chen")),
    ("Yu Chen", (True, "Yu Chen")),
    ("Yu Murong", (True, "Yu Murong")),
    ("Tsai Yu", (True, "Yu Tsai")),
    ("Yu Tsai", (True, "Yu Tsai")),
    ("Ma Long", (True, "Long Ma")),
    ("Lee Min", (True, "Min Lee")),
    ("Lee Jun", (True, "Jun Lee")),
    ("Chen Chen Yu", (True, "Chen-Yu Chen")),
    ("Wang Li Ming", (True, "Li-Ming Wang")),
    ("Chung Ming Wang", (True, "Chung-Ming Wang")),
    ("Wei Wei", (True, "Wei Wei")),
    ("Xu Xu", (True, "Xu Xu")),

    # Names with overlapping surnames
    ("Choi Suk-Zan", (True, "Suk-Zan Choi")),
    ("Choi Ka-Fai", (True, "Ka-Fai Choi")),
    ("Choi Ming", (True, "Ming Choi")),
    ("Jung Chi-Wai", (True, "Chi-Wai Jung")),
    ("Lim Wai-Kit", (True, "Wai-Kit Lim")),
    ("Im Siu-Ming", (True, "Siu-Ming Im")),

    # Singapore/Malaysian Chinese names
    ("Teo Chee Hean", (True, "Chee-Hean Teo")),
    ("Goh Chok Tong", (True, "Chok-Tong Goh")),

    # Names with initials + Chinese surnames
    ("H Y Tiong", (True, "H-Y Tiong")),
    ("Z D Chen", (True, "Z-D Chen")),
    ("Y Z Wang", (True, "Y-Z Wang")),
    ("H M Zhang", (True, "H-M Zhang")),

    # Simple capitalization variants
    ("DAN CHEN", (True, "Dan Chen")),
    ("Juan He", (True, "Juan He")),
    ("Dan Sun", (True, "Dan Sun")),
    ("Nan Huang", (True, "Nan Huang")),
    ("Dan CHEN", (True, "Dan Chen")),
    ("Chen Dan", (True, "Dan Chen")),
    ("He Juan", (True, "Juan He")),
    ("Juan Yu", (True, "Juan Yu")),
    ("Dan Chen", (True, "Dan Chen")),
    ("Liu Nan", (True, "Nan Liu")),
    ("Sun Dan", (True, "Dan Sun")),
    ("Dan Cheng", (True, "Dan Cheng")),
    ("Juan Song", (True, "Juan Song")),
    ("Juan Liang", (True, "Juan Liang")),
    ("DAN SUN", (True, "Dan Sun")),
    ("Huang Nan", (True, "Nan Huang")),
    ("JUAN HE", (True, "Juan He")),

    # Names with overlapping surnames (Korean overlap fixes)
    ("Han Jun", (True, "Jun Han")),  # Previously rejected due to Korean overlap
    ("Han jun", (True, "Jun Han")),  # Case variant
    ("Jun Han", (True, "Jun Han")),  # Different order
    ("JUN HAN", (True, "Jun Han")),  # All caps
    ("Xuefeng Han", (True, "Xue-Feng Han")),  # Chinese compound + overlapping surname
    ("Xiaojuan Han", (True, "Xiao-Juan Han")),  # Chinese compound + overlapping surname

    # Additional Chinese names with compound syllables
    ("Lianhua Wang", (True, "Lian-Hua Wang")),
    ("Tianjian Li", (True, "Tian-Jian Li")),

    # Session fixes - legitimate Chinese names that should be preserved
    ("Zixuan Wang", (True, "Zi-Xuan Wang")),  # Should pass tiered confidence system
    ("Weiming Zhang", (True, "Wei-Ming Zhang")),  # Gold standard (anchor + anchor)

    # Additional real Chinese names to ensure comprehensive coverage
    ("Xiuxian Zhang", (True, "Xiu-Xian Zhang")),
    ("Chunfang Li", (True, "Chun-Fang Li")),
    ("Guangming Wang", (True, "Guang-Ming Wang")),
    ("Jianchun Liu", (True, "Jian-Chun Liu")),
    ("Wenxuan Chen", (True, "Wen-Xuan Chen")),
    ("Yongquan Zhou", (True, "Yong-Quan Zhou")),
    ("Xuefeng Gao", (True, "Xue-Feng Gao")),
    ("Zhenghua Yang", (True, "Zheng-Hua Yang")),
    ("Meiling Wu", (True, "Mei-Ling Wu")),
    ("Qiuying Zhang", (True, "Qiu-Ying Zhang")),
    ("Ruigang Li", (True, "Rui-Gang Li")),
    ("Shuangxi Wang", (True, "Shuang-Xi Wang")),
    ("Tianhua Liu", (True, "Tian-Hua Liu")),
    ("Xiaoqing Chen", (True, "Xiao-Qing Chen")),
    ("Yuanfang Zhou", (True, "Yuan-Fang Zhou")),
    ("Zhiyuan Yang", (True, "Zhi-Yuan Yang")),
    ("Lingfeng Wu", (True, "Ling-Feng Wu")),
    ("Baoguo Xu", (True, "Bao-Guo Xu")),

    # Dynamic system test cases - previously problematic syllables now working
    ("Li Qionghua", (True, "Qiong-Hua Li")),  # qiong syllable from givenname.csv
    ("Chen Siming", (True, "Si-Ming Chen")),  # si syllable from givenname.csv
    ("Liu Chuanyu", (True, "Chuan-Yu Liu")),  # chuan syllable from givenname.csv
    ("Wu Leping", (True, "Le-Ping Wu")),  # le syllable from givenname.csv
    ("Zhou Shuaibin", (True, "Shuai-Bin Zhou")),  # shuai syllable from givenname.csv
    ("Huang Bihong", (True, "Bi-Hong Huang")),  # bi syllable from givenname.csv
    ("Chen Cuanfen", (True, "Cuan-Fen Chen")),  # cuan syllable from manual supplement
    ("Wang Dongliang", (True, "Dong-Liang Wang")),  # compound name with fixed forbidden pattern
    ("Zhang Xiaoming", (True, "Xiao-Ming Zhang")),  # common name validation
    ("Liu Hunyu", (True, "Hun-Yu Liu")),  # hun syllable from manual supplement
    ("Wu Zabing", (True, "Za-Bing Wu")),  # za syllable from manual supplement

    # High-frequency syllables from givenname.csv now included
    ("Wang Zehua", (True, "Ze-Hua Wang")),  # ze syllable (4,513.6 ppm)
    ("Zhang Chuan", (True, "Chuan Zhang")),  # chuan syllable (2,741.0 ppm)
    ("Chen Leming", (True, "Le-Ming Chen")),  # le syllable (2,658.5 ppm)
    ("Liu Shuai", (True, "Shuai Liu")),  # shuai syllable (1,977.2 ppm)
    ("Wu Laiming", (True, "Lai-Ming Wu")),  # lai syllable (1,702.8 ppm)
    ("Zhou Rundong", (True, "Run-Dong Zhou")),  # run syllable (1,645.9 ppm)
    ("Huang Daoming", (True, "Dao-Ming Huang")),  # dao syllable (1,605.6 ppm)
    ("Wang Huaiyu", (True, "Huai-Yu Wang")),  # huai syllable (1,587.0 ppm)
    ("Zhang Hangfei", (True, "Hang-Fei Zhang")),  # hang syllable (1,552.8 ppm)
    ("Li Wangming", (True, "Wang-Ming Li")),  # wang syllable (1,531.8 ppm)
    ("Liu Zenghua", (True, "Zeng-Hua Liu")),  # zeng syllable (1,413.2 ppm)
    ("Wu Cunming", (True, "Cun-Ming Wu")),  # cun syllable (1,319.0 ppm)
    ("Zhou Kuihua", (True, "Kui-Hua Zhou")),  # kui syllable (1,293.5 ppm)
    ("Huang Dingyu", (True, "Ding-Yu Huang")),  # ding syllable (1,180.0 ppm)

    # Names that were being accepted as Chinese (proper behavior with new algorithm)
    ("Ho Yung Lee", (True, "Ho-Yung Lee")),  # Strong Chinese surname evidence overrides Korean patterns
    ("Ho Yun Lee", (True, "Ho-Yun Lee")),  # Strong Chinese surname evidence overrides Korean patterns
    ("Jin Ho Lee", (True, "Jin-Ho Lee")),  # Strong Chinese surname evidence overrides Korean patterns
    ("Min Soo Lee", (True, "Min-Soo Lee")),  # Strong Chinese surname evidence overrides Korean patterns

    # Test cases for compound splitting after plausible_components filtering (Concern 1)
    ("Zhang Xuefeng", (True, "Xue-Feng Zhang")),  # Tests 'xue' syllable preserved in plausible_components
    ("Liu Yuehua", (True, "Yue-Hua Liu")),  # Tests 'yue' syllable preserved
    ("Chen Jueming", (True, "Jue-Ming Chen")),  # Tests 'jue' syllable preserved
    ("Wu Kuaile", (True, "Kuai-Le Wu")),  # Tests 'kuai' syllable preserved
    ("Wang Shuaiming", (True, "Shuai-Ming Wang")),  # Tests 'shuai' syllable preserved
    ("Li Hualiang", (True, "Hua-Liang Li")),  # Tests compound splitting still works

    # Additional tests for moved surnames (Korean overlap fixes)
    ("Gong Wei", (True, "Wei Gong")),  # Direct gong test
    ("Li Gong", (True, "Gong Li")),  # gong as given name
    ("Koo Ming", (True, "Ming Koo")),  # koo surname test
    ("Zhang Koo", (True, "Zhang Koo")),  # koo as given name (name order preserved)
    ("Kang Wei", (True, "Wei Kang")),  # kang surname test
    ("Wang Kang", (True, "Kang Wang")),  # kang as given name
    ("An Li", (True, "An Li")),  # an surname test (name order preserved)
    ("Chen An", (True, "An Chen")),  # an as given name
    ("Ha Wei", (True, "Ha Wei")),
    ("Liu Ha", (True, "Ha Liu")),

    # Korean surname overlap fix verification
    ("Kong Kung", (True, "Kung Kong")),

    ("Li Weiwei", (True, "Wei-Wei Li")),
    ("Zhang Weiwei", (True, "Wei-Wei Zhang")),
    ("Wang Weiming", (True, "Wei-Ming Wang")),
    ("Chen Wenjun", (True, "Wen-Jun Chen")),

    # Common pattern tests
    ("Hu Cha", (True, "Cha Hu")),  # cha moved from Korean-only to overlapping
    ("Feng Cha", (True, "Cha Feng")),  # cha moved from Korean-only to overlapping
    ("He Cha", (True, "Cha He")),  # cha moved from Korean-only to overlapping

    # Ambiguous Surname + Common Word
    ("Gao Shan", (True, "Shan Gao")),
    ("Feng Yun", (True, "Yun Feng")),
    ("Jin Bo", (True, "Bo Jin")),
    ("Qin Shi", (True, "Shi Qin")),
    ("Han Han", (True, "Han Han")),

    # Reduplicated Given Names (Common in Southern China)
    ("Chen Linlin", (True, "Lin-Lin Chen")),  # System uses hyphens
    ("Wang Nini", (True, "Ni-Ni Wang")),  # System uses hyphens
    ("Li Lili", (True, "Li-Li Li")),  # System uses hyphens
    ("Zhou Zhou", (True, "Zhou Zhou")),

    # Title Case vs ALL CAPS (from metadata export)
    ("ZHANG WEI", (True, "Wei Zhang")),
    ("LI XIAOMING", (True, "Xiao-Ming Li")),  # System uses hyphens

    # Very common Chinese names
    ("Lu Xun", (True, "Xun Lu")),
    ("Mo Yan", (True, "Yan Mo")),
    ("Tu Youyou", (True, "You-You Tu")),  # System uses hyphens for reduplicated names
    ("Xi Jinping", (True, "Jin-Ping Xi")),  # System uses hyphens
]


def test_basic_chinese_names():
    """Test basic Chinese names with simple patterns."""
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

    assert failed == 0, f"Basic Chinese name tests: {failed} failures out of {len(CHINESE_NAME_TEST_CASES)} tests"
    print(f"Basic Chinese name tests: {passed} passed, {failed} failed")


if __name__ == "__main__":
    test_basic_chinese_names()
