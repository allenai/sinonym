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
from tests._fail_log import log_failure

# Basic Chinese names with simple patterns
CHINESE_NAME_TEST_CASES = [
    ("An Li", (True, "An Li")),
    ("Chen An", (True, "An Chen")),
    ("Chen Jueming", (True, "Jue-Ming Chen")),
    ("Chen Linlin", (True, "Lin-Lin Chen")),
    ("Feng Cha", (True, "Cha Feng")),
    ("Han Jun", (True, "Jun Han")),
    ("Han jun", (True, "Jun Han")),
    ("He Cha", (True, "Cha He")),
    ("Ho Yun Lee", (True, "Ho-Yun Lee")),
    ("Hu Cha", (True, "Cha Hu")),
    ("Jin Ho Lee", (True, "Jin-Ho Lee")),
    ("Jun Han", (True, "Jun Han")),
    ("Koo Ming", (True, "Ming Koo")),
    ("Li Gong", (True, "Gong Li")),
    ("Li Hualiang", (True, "Hua-Liang Li")),
    ("Li Lili", (True, "Li-Li Li")),
    ("Liu Yuehua", (True, "Yue-Hua Liu")),
    ("Min Soo Lee", (True, "Min-Soo Lee")),
    ("Tu Youyou", (True, "You-You Tu")),
    ("Wang Kang", (True, "Kang Wang")),
    ("Wang Nini", (True, "Ni-Ni Wang")),
    ("Wang Shuaiming", (True, "Shuai-Ming Wang")),
    ("Wu Kuaile", (True, "Kuai-Le Wu")),
    ("Xiaojuan Han", (True, "Xiao-Juan Han")),
    ("Xuefeng Han", (True, "Xue-Feng Han")),
    ("Zhang Koo", (True, "Zhang Koo")),
    ("Zhang Xuefeng", (True, "Xue-Feng Zhang")),
    ("Baoguo Xu", (True, "Bao-Guo Xu")),
    ("Chen Chen Yu", (True, "Chen-Yu Chen")),
    ("Chen Dan", (True, "Dan Chen")),
    ("Chen Wenjun", (True, "Wen-Jun Chen")),
    ("Chen Yu", (True, "Yu Chen")),
    ("Choi Ka-Fai", (True, "Ka-Fai Choi")),
    ("Choi Ming", (True, "Ming Choi")),
    ("Choi Suk-Zan", (True, "Suk-Zan Choi")),
    ("Chunfang Li", (True, "Chun-Fang Li")),
    ("Chung Ming Wang", (True, "Chung-Ming Wang")),
    ("DAN CHEN", (True, "Dan Chen")),
    ("DAN SUN", (True, "Dan Sun")),
    ("Dan CHEN", (True, "Dan Chen")),
    ("Dan Chen", (True, "Dan Chen")),
    ("Dan Cheng", (True, "Dan Cheng")),
    ("Dan Sun", (True, "Dan Sun")),
    ("Dehua Liu", (True, "De-Hua Liu")),
    ("Feng Yun", (True, "Yun Feng")),
    ("Gao Shan", (True, "Shan Gao")),
    ("Gao Wei", (True, "Wei Gao")),
    ("Guangming Wang", (True, "Guang-Ming Wang")),
    ("H M Zhang", (True, "H-M Zhang")),
    ("H Y Tiong", (True, "H-Y Tiong")),
    ("Ha Wei", (True, "Ha Wei")),
    ("Han Han", (True, "Han Han")),
    ("He Juan", (True, "Juan He")),
    ("Huang Nan", (True, "Nan Huang")),
    ("Im Siu-Ming", (True, "Siu-Ming Im")),
    ("JUAN HE", (True, "Juan He")),
    ("Jianchun Liu", (True, "Jian-Chun Liu")),
    ("Jin Bo", (True, "Bo Jin")),
    ("Juan He", (True, "Juan He")),
    ("Juan Liang", (True, "Juan Liang")),
    ("Juan Song", (True, "Juan Song")),
    ("Juan Yu", (True, "Juan Yu")),
    ("Jun Wang", (True, "Jun Wang")),
    ("Jung Chi-Wai", (True, "Chi-Wai Jung")),
    ("Kong Kung", (True, "Kung Kong")),
    ("Lee Jun", (True, "Jun Lee")),
    ("Lee Min", (True, "Min Lee")),
    ("Li Ming", (True, "Ming Li")),
    ("Li Na", (True, "Na Li")),
    ("Li Weiwei", (True, "Wei-Wei Li")),
    ("Lianhua Wang", (True, "Lian-Hua Wang")),
    ("Lim Wai-Kit", (True, "Wai-Kit Lim")),
    ("Lingfeng Wu", (True, "Ling-Feng Wu")),
    ("Liu Dehua", (True, "De-Hua Liu")),
    ("Liu Ha", (True, "Ha Liu")),
    ("Liu Nan", (True, "Nan Liu")),
    ("Lu Xun", (True, "Xun Lu")),
    ("Ma Long", (True, "Long Ma")),
    ("Meiling Wu", (True, "Mei-Ling Wu")),
    ("Mo Yan", (True, "Yan Mo")),
    ("Nan Huang", (True, "Nan Huang")),
    ("Qin Shi", (True, "Shi Qin")),
    ("Qiuying Zhang", (True, "Qiu-Ying Zhang")),
    ("Ruigang Li", (True, "Rui-Gang Li")),
    ("Shuangxi Wang", (True, "Shuang-Xi Wang")),
    ("Sun Dan", (True, "Dan Sun")),
    ("Tianhua Liu", (True, "Tian-Hua Liu")),
    ("Tianjian Li", (True, "Tian-Jian Li")),
    ("Tsai Yu", (True, "Yu Tsai")),
    ("Wang Jun", (True, "Jun Wang")),
    ("Wang Li Ming", (True, "Li-Ming Wang")),
    ("Wang Weiming", (True, "Wei-Ming Wang")),
    ("Wei Wei", (True, "Wei Wei")),
    ("Wenxuan Chen", (True, "Wen-Xuan Chen")),
    ("Xiaoqing Chen", (True, "Xiao-Qing Chen")),
    ("Xiuxian Zhang", (True, "Xiu-Xian Zhang")),
    ("Xu Xu", (True, "Xu Xu")),
    ("Xuefeng Gao", (True, "Xue-Feng Gao")),
    ("Xun Zhou", (True, "Xun Zhou")),
    ("Y Z Wang", (True, "Y-Z Wang")),
    ("Yongquan Zhou", (True, "Yong-Quan Zhou")),
    ("Yu Chen", (True, "Yu Chen")),
    ("Yu Murong", (True, "Yu Murong")),
    ("Yu Tsai", (True, "Yu Tsai")),
    ("Yuanfang Zhou", (True, "Yuan-Fang Zhou")),
    ("Z D Chen", (True, "Z-D Chen")),
    ("ZHANG WEI", (True, "Wei Zhang")),
    ("Zhang Wei", (True, "Wei Zhang")),
    ("Zhang Weiwei", (True, "Wei-Wei Zhang")),
    ("Zhenghua Yang", (True, "Zheng-Hua Yang")),
    ("Zhiyuan Yang", (True, "Zhi-Yuan Yang")),
    ("Zhou Xun", (True, "Xun Zhou")),
    ("Zhou Zhou", (True, "Zhou Zhou")),
]


def test_basic_chinese_names(detector):
    """Test basic Chinese names with simple patterns."""

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
            log_failure("Basic Chinese name tests", input_name, expected_success, expected_name, result.success, actual)

    if failed:
        print(f"Basic Chinese name tests: {failed} failures out of {len(CHINESE_NAME_TEST_CASES)} tests")
    assert failed == 0, f"Basic Chinese name tests: {failed} failures out of {len(CHINESE_NAME_TEST_CASES)} tests"
    print(f"Basic Chinese name tests: {passed} passed, {failed} failed")


if __name__ == "__main__":
    test_basic_chinese_names()
