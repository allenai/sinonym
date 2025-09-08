"""
Mixed Test Cases from Production Results

This test file contains a 50/50 mix of cases that:
- Failed in the current production system (error cases)
- Succeeded in the current production system (success cases)

This helps validate that improvements don't break existing functionality
while fixing known issues.
"""

import sys
from pathlib import Path

# Add the parent directory to path to import sinonym
sys.path.insert(0, str(Path(__file__).parent.parent))

from sinonym import ChineseNameDetector
from tests._fail_log import log_failure

# Mixed test cases: 50% production errors, 50% production successes
MIXED_TEST_CASES = [
    # production_success_case - 李冠龙 (3 tokens)
    ("Li Guan Long", (True, "Guan-Long Li")),
    # production_success_case - 张春帆 (3 tokens)
    ("Zhang Chun Fan", (True, "Chun-Fan Zhang")),
    # production_success_case - 韩爱红 (3 tokens)
    ("Han Ai Hong", (True, "Ai-Hong Han")),
    # production_error_case - 门好 (2 tokens)
    ("Men Hao", (True, "Hao Men")),
    # production_success_case - 吴德隆 (3 tokens)
    ("Wu De Long", (True, "De-Long Wu")),
    # production_success_case - 杨毓明 (3 tokens)
    ("Yang Yu Ming", (True, "Yu-Ming Yang")),
    # production_error_case - 桂瑞 (2 tokens)
    ("Gui Rui", (True, "Rui Gui")),
    # production_success_case - 曹昌祺 (3 tokens)
    ("Cao Chang Qi", (True, "Chang-Qi Cao")),
    # production_success_case - 杜伟刚 (3 tokens)
    ("Du Wei Gang", (True, "Wei-Gang Du")),
    # production_error_case - 金玲瑜 (3 tokens)
    ("Jin Ling Yu", (True, "Ling-Yu Jin")),
    # production_success_case - 叶青波 (3 tokens)
    ("Ye Qing Bo", (True, "Qing-Bo Ye")),
    # production_success_case - 张春先 (3 tokens)
    ("Zhang Chun Xian", (True, "Chun-Xian Zhang")),
    # production_success_case - 肖环环 (3 tokens)
    ("Xiao Huan Huan", (True, "Huan-Huan Xiao")),
    # production_error_case - 舒遥 (2 tokens)
    ("Shu Yao", (True, "Yao Shu")),
    # production_success_case - 袁长明 (3 tokens)
    ("Yuan Zhang Ming", (True, "Zhang-Ming Yuan")),
    # production_success_case - 沈衍庆 (3 tokens)
    ("Shen Yan Qing", (True, "Yan-Qing Shen")),
    # production_error_case - 郑朝辉 (3 tokens)
    ("Zheng Chao Hui", (True, "Chao-Hui Zheng")),
    # production_success_case - 张宝权 (3 tokens)
    ("Zhang Bao Quan", (True, "Bao-Quan Zhang")),
    # production_success_case - 黄红丹 (3 tokens)
    ("Huang Hong Dan", (True, "Hong-Dan Huang")),
    # production_error_case - 黄裕昌 (3 tokens)
    ("Huang Yu Chang", (True, "Yu-Chang Huang")),
    # production_error_case - 花夏 (2 tokens)
    ("Xi Cheng", (True, "Cheng Xi")),
    # production_success_case - 张日龙 (3 tokens)
    ("Zhang Ri Long", (True, "Ri-Long Zhang")),
    # production_success_case - 刘晓贵 (3 tokens)
    ("Liu Xiao Gui", (True, "Xiao-Gui Liu")),
    # production_success_case - 安易 (2 tokens)
    ("An Yi", (True, "An Yi")),
    # production_error_case - 贾建锋 (3 tokens)
    ("Jia Jian Feng", (True, "Jian-Feng Jia")),
    # production_success_case - 陈嘉成 (3 tokens)
    ("Chen Jia Cheng", (True, "Jia-Cheng Chen")),
    # production_error_case - 邱兆林 (3 tokens)
    ("Qiu Zhao Lin", (True, "Zhao-Lin Qiu")),
    # production_success_case - 王立航 (3 tokens)
    ("Wang Li Hang", (True, "Li-Hang Wang")),
    # production_success_case - 余和明 (3 tokens)
    ("Yu He Ming", (True, "He-Ming Yu")),
    # production_error_case - 王秋艳 (3 tokens) [prod: Wang-Qiu Yan]
    ("Wang Qiu Yan", (True, "Qiu-Yan Wang")),
    # production_success_case - 赵继民 (3 tokens)
    ("Zhao Ji Min", (True, "Ji-Min Zhao")),
    # production_error_case - 王修林 (3 tokens) [prod: Wang-Xiu Lin]
    ("Wang Xiu Lin", (True, "Xiu-Lin Wang")),
    # production_success_case - 许卿 (2 tokens)
    ("Xu Qing", (True, "Qing Xu")),
    # production_error_case - 青卓 (2 tokens) [prod: Qing Zhuo]
    ("Qing Zhuo", (True, "Qing Zhuo")),
    # production_success_case - 吴艺雯 (3 tokens)
    ("Wu Yi Wen", (True, "Yi-Wen Wu")),
    # production_success_case - 何学升 (3 tokens)
    ("He Xue Sheng", (True, "Xue-Sheng He")),
    # production_success_case - 郭海旭 (3 tokens) [prod: Guo-Hai Xu]
    ("Guo Hai Xu", (True, "Guo-Hai Xu")),
    # production_error_case - 杨国燕 (3 tokens) [prod: Yang-Guo Yan]
    ("Yang Guo Yan", (True, "Guo-Yan Yang")),
    # production_success_case - 许嘉宝 (3 tokens)
    ("Xu Jia Bao", (True, "Jia-Bao Xu")),
    # production_success_case - 高特 (2 tokens)
    ("Gao Te", (True, "Te Gao")),
    # production_success_case - 李星刚 (3 tokens)
    ("Li Xing Gang", (True, "Xing-Gang Li")),
    # production_success_case - 樊家良 (3 tokens)
    ("Fan Jia Liang", (True, "Jia-Liang Fan")),
    # production_success_case - 申建广 (3 tokens)
    ("Shen Jian Guang", (True, "Jian-Guang Shen")),
    # production_error_case - 魏亚红 (3 tokens) [prod: Wei-Ya Hong]
    ("Wei Ya Hong", (True, "Ya-Hong Wei")),
    # production_error_case - 叶皓然 (3 tokens) [prod: Ye-Hao Ran]
    ("Ye Hao Ran", (True, "Hao-Ran Ye")),
    # production_error_case - 王保石 (3 tokens) [prod: Wang-Bao Shi]
    ("Wang Bao Shi", (True, "Bao-Shi Wang")),
    # production_success_case - 赵江辉 (3 tokens)
    ("Zhao Jiang Hui", (True, "Jiang-Hui Zhao")),
    # production_success_case - 韩玉静 (3 tokens)
    ("Han Yu Jing", (True, "Yu-Jing Han")),
    # production_error_case - 魏文兴 (3 tokens) [prod: Wei-Wen Xing]
    ("Wei Wen Xing", (True, "Wen-Xing Wei")),
    # production_success_case - 金彭年 (3 tokens)
    ("Jin Peng Nian", (True, "Peng-Nian Jin")),
    # production_success_case - 吴铭双 (3 tokens)
    ("Wu Ming Shuang", (True, "Ming-Shuang Wu")),
    # production_error_case - 张绪仁 (3 tokens) [prod: Zhang-Xu Ren]
    ("Zhang Xu Ren", (True, "Xu-Ren Zhang")),
    # production_error_case - 王金忠 (3 tokens) [prod: Wang-Jin Zhong]
    ("Wang Jin Zhong", (True, "Jin-Zhong Wang")),
    # production_error_case - 严友祥 (3 tokens) [prod: Yan-You Xiang]
    ("Yan You Xiang", (True, "You-Xiang Yan")),
    # production_error_case - 王庆兰 (3 tokens) [prod: Wang-Qing Lan]
    ("Wang Qing Lan", (True, "Qing-Lan Wang")),
    # production_success_case - 刘玉荣 (3 tokens)
    ("Liu Yu Rong", (True, "Yu-Rong Liu")),
    # production_error_case - 李菊琴 (3 tokens) [prod: Li-Ju Qin]
    ("Li Ju Qin", (True, "Ju-Qin Li")),
    # production_success_case - 季星宇 (3 tokens)
    ("Ji Xing Yu", (True, "Xing-Yu Ji")),
    # production_error_case - 王克凤 (3 tokens) [prod: Wang-Ke Feng]
    ("Wang Ke Feng", (True, "Ke-Feng Wang")),
    # production_success_case - 杨倩骅 (3 tokens)
    ("Yang Qian Hua", (True, "Qian-Hua Yang")),
    # production_success_case - 张芷珊 (3 tokens)
    ("Zhang Zhi Shan", (True, "Zhi-Shan Zhang")),
    # production_error_case - 萧雨轩 (3 tokens) [prod: Xiao-Yu Xuan]
    ("Xiao Yu Xuan", (True, "Yu-Xuan Xiao")),
    # production_success_case - 丰志勇 (3 tokens)
    ("Feng Zhi Yong", (True, "Zhi-Yong Feng")),
    # production_success_case - 曹善华 (3 tokens)
    ("Cao Shan Hua", (True, "Shan-Hua Cao")),
    # production_error_case - 张一弓 (3 tokens) [prod: Zhang-Yi Gong]
    ("Zhang Yi Gong", (True, "Yi-Gong Zhang")),
    # production_error_case - 陈思琦 (3 tokens) [prod: Chen-Si Qi]
    ("Chen Si Qi", (True, "Si-Qi Chen")),
    # production_error_case - 周宇和 (3 tokens) [prod: Zhou-Yu He]
    ("Zhou Yu He", (True, "Yu-He Zhou")),
    # production_error_case - 黄海弟 (3 tokens) [prod: Huang-Hai Di]
    ("Huang Hai Di", (True, "Hai-Di Huang")),
    # production_success_case - 毛敏 (2 tokens)
    ("Mao Min", (True, "Min Mao")),
    # production_success_case - 杨得志 (3 tokens)
    ("Yang De Zhi", (True, "De-Zhi Yang")),
    # production_success_case - 潘汉潮 (3 tokens)
    ("Pan Han Chao", (True, "Han-Chao Pan")),
    # production_success_case - 高心 (2 tokens)
    ("Gao Xin", (True, "Xin Gao")),
    # production_success_case - 佟杰 (2 tokens)
    ("Tong Jie", (True, "Jie Tong")),
    # production_success_case - 牛一兵 (3 tokens)
    ("Niu Yi Bing", (True, "Yi-Bing Niu")),
    # production_success_case - 陈莹超 (3 tokens)
    ("Chen Ying Chao", (True, "Ying-Chao Chen")),
    # production_success_case - 何兴洋 (3 tokens)
    ("He Xing Yang", (True, "Xing-Yang He")),
    # production_success_case - 许旦 (2 tokens)
    ("Xu Dan", (True, "Dan Xu")),
    # production_success_case - 施博仁 (3 tokens) [prod: Shi-Bo Ren]
    ("Shi Bo Ren", (True, "Shi-Bo Ren")),
    # production_success_case - 韦少芬 (3 tokens)
    ("Wei Shao Fen", (True, "Shao-Fen Wei")),
    # production_error_case - 西钊 (2 tokens) [prod: Xi Zhao]
    ("Xi Zhao", (True, "Zhao Xi")),
    # production_error_case - 林森相 (3 tokens) [prod: Lin-Sen Xiang]
    ("Lin Sen Xiang", (True, "Sen-Xiang Lin")),
    # production_success_case - 白树华 (3 tokens)
    ("Bai Shu Hua", (True, "Shu-Hua Bai")),
    # production_success_case - 曹庆森 (3 tokens)
    ("Cao Qing Sen", (True, "Qing-Sen Cao")),
    # production_error_case - 张道成 (3 tokens) [prod: Zhang-Dao Cheng]
    ("Zhang Dao Cheng", (True, "Dao-Cheng Zhang")),
    # production_error_case - 盛春凤 (3 tokens) [prod: Sheng-Chun Feng]
    ("Chen Lan Qin", (True, "Lan-Qin Chen")),
    # production_success_case - 王远晴 (3 tokens)
    ("Wang Yuan Qing", (True, "Yuan-Qing Wang")),
    # production_error_case - 傅梦婷 (3 tokens) [prod: Fu-Meng Ting]
    ("Fu Meng Ting", (True, "Meng-Ting Fu")),
    # production_success_case - 曹光耀 (3 tokens)
    ("Cao Guang Yao", (True, "Guang-Yao Cao")),
    # production_success_case - 于爱珍 (3 tokens)
    ("Yu Ai Zhen", (True, "Ai-Zhen Yu")),
    # production_success_case - 戴爱莲 (3 tokens)
    ("Dai Ai Lian", (True, "Ai-Lian Dai")),
    # production_error_case - 陈晓湘 (3 tokens) [prod: Chen-Xiao Xiang]
    ("Chen Xiao Xiang", (True, "Xiao-Xiang Chen")),
    # production_success_case - 张芷云 (3 tokens)
    ("Zhang Zhi Yun", (True, "Zhi-Yun Zhang")),
    # production_error_case - 刘世松 (3 tokens) [prod: Liu-Shi Song]
    ("Liu Shi Song", (True, "Shi-Song Liu")),
    # production_error_case - 刘观林 (3 tokens) [prod: Liu-Guan Lin]
    ("Liu Guan Lin", (True, "Guan-Lin Liu")),
    # production_error_case - 李翰祥 (3 tokens) [prod: Li-Han Xiang]
    ("Li Han Xiang", (True, "Han-Xiang Li")),
    # production_error_case - 刘明珠 (3 tokens) [prod: Liu-Ming Zhu]
    ("Liu Ming Zhu", (True, "Ming-Zhu Liu")),
    # production_success_case - 夏庆庆 (3 tokens)
    ("Xia Qing Qing", (True, "Qing-Qing Xia")),
    # production_success_case - 黎振强 (3 tokens)
    ("Li Zhen Qiang", (True, "Zhen-Qiang Li")),
]


def test_mixed_cases(detector):
    """Test mixed cases from production results."""

    passed = 0
    failed = 0

    for input_name, expected_result in MIXED_TEST_CASES:
        result = detector.is_chinese_name(input_name)

        # Extract expected success status and name from tuple
        expected_success, expected_name = expected_result

        if result.success == expected_success and (not expected_success or result.result == expected_name):
            passed += 1
        else:
            failed += 1
            actual = result.result if result.success else f"ERROR: {result.error_message}"
            actual_text = result.result if result.success else result.error_message
            print(
                f"FAILED: '{input_name}': expected ({expected_success}, '{expected_name}'), got ({result.success}, '{actual_text}')",
            )
            log_failure(
                "ML ranker test data tests",
                input_name,
                expected_success,
                expected_name,
                result.success,
                actual_text,
            )

    if failed:
        print(f"ML ranker test data tests: {failed} failures out of {len(MIXED_TEST_CASES)} tests")
    assert failed == 0, f"ML ranker test data tests: {failed} failures out of {len(MIXED_TEST_CASES)} tests"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
