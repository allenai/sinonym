"""
ACL 2025 Authors Test Suite

This module contains tests for all author names from ACL 2025 accepted papers.
Tests the sinonym library against real-world academic author names.
"""

import sys
from pathlib import Path

# Add the parent directory to path to import sinonym
sys.path.insert(0, str(Path(__file__).parent.parent))

from sinonym import ChineseNameDetector

# Ground truth: manually verified Chinese names from ACL 2025
ACL_CHINESE_NAMES = [
    # Obviously Chinese names (common Chinese surnames + given names)
    ("Weiqi Wang", "Wei-Qi Wang"),
    ("Xin Liu", "Xin Liu"),
    ("Yang Li", "Yang Li"),
    ("Hui Liu", "Hui Liu"),
    ("Yifan Gao", "Yi-Fan Gao"),
    ("Zheng Zhang", "Zheng Zhang"),
    ("Yifei Zhang", "Yi-Fei Zhang"),
    ("Liang Zhao", "Liang Zhao"),
    ("Chen Huang", "Chen Huang"),
    ("Tong Zhang", "Tong Zhang"),
    ("Wenqiang Lei", "Wen-Qiang Lei"),
    ("Chenhao Tan", "Chen-Hao Tan"),
    ("Kai Tian", "Kai Tian"),
    ("Ning Ding", "Ning Ding"),
    ("Bowen Zhou", "Bo-Wen Zhou"),
    ("Xinyu Ma", "Xin-Yu Ma"),
    ("Yutao Zhu", "Yu-Tao Zhu"),
    ("Dawei Yin", "Da-Wei Yin"),
    ("Zhicheng Dou", "Zhi-Cheng Dou"),
    ("Yinghui Li", "Ying-Hui Li"),
    ("Qingyu Zhou", "Qing-Yu Zhou"),
    ("Ying Shen", "Ying Shen"),
    ("Wenhao Jiang", "Wen-Hao Jiang"),
    ("Baobao Chang", "Bao-Bao Chang"),
    ("Xiaoming Zhang", "Xiao-Ming Zhang"),
    ("Baosong Yang", "Bao-Song Yang"),
    ("Bei Yu", "Bei Yu"),
    ("Chengzhong LIU", "Cheng-Zhong Liu"),
    ("Chin-Jou Li", "Chin-Jou Li"),
    ("Chong Li", "Chong Li"),
    ("Chuxu Zhang", "Chu-Xu Zhang"),
    ("Fan Wang", "Fan Wang"),
    ("Fanxin Li", "Fan-Xin Li"),
    ("Fei Yu", "Fei Yu"),
    ("Guoliang Kang", "Guo-Liang Kang"),
    ("Haitao Li", "Hai-Tao Li"),
    ("Hao Li", "Hao Li"),
    ("Haonan Zhang", "Hao-Nan Zhang"),
    ("Haoyu Wang", "Hao-Yu Wang"),
    ("Hongling Xu", "Hong-Ling Xu"),
    ("Hongning Wang", "Hong-Ning Wang"),
    ("Huixuan Zhang", "Hui-Xuan Zhang"),
    ("Jiahan Ren", "Jia-Han Ren"),
    ("Jiajun Liu", "Jia-Jun Liu"),
    ("Jiajun Tan", "Jia-Jun Tan"),
    ("Jiajun Zhang", "Jia-Jun Zhang"),
    ("Jianwei Wang", "Jian-Wei Wang"),
    ("Jiarong Xu", "Jia-Rong Xu"),
    ("Jue Hong", "Jue Hong"),
    ("Jun Zhang", "Jun Zhang"),
    ("Junhao Shi", "Jun-Hao Shi"),
    ("Kaige Li", "Kai-Ge Li"),
    ("Ke Zeng", "Ke Zeng"),
    ("Kexin Fan", "Ke-Xin Fan"),
    ("Kunpeng Liu", "Kun-Peng Liu"),
    ("Maochuan Lu", "Mao-Chuan Lu"),
    ("Mingxuan Li", "Ming-Xuan Li"),
    ("Po-Kai Chen", "Po-Kai Chen"),
    ("Ruize Gao", "Rui-Ze Gao"),
    ("Shanshan Huang", "Shan-Shan Huang"),
    ("Sicheng Yu", "Si-Cheng Yu"),
    ("Siliang Tang", "Si-Liang Tang"),
    ("Tianhao Shen", "Tian-Hao Shen"),
    ("Tingyu Song", "Ting-Yu Song"),
    ("Wei Shen", "Wei Shen"),
    ("Weilun Zhao", "Wei-Lun Zhao"),
    ("Xiangqi Wang", "Xiang-Qi Wang"),
    ("Xiao Liang", "Xiao Liang"),
    ("Xiaowen Chu", "Xiao-Wen Chu"),
    ("Xintong Li", "Xin-Tong Li"),
    ("Xiwen Zhang", "Xi-Wen Zhang"),
    ("Xuchao Zhang", "Xu-Chao Zhang"),
    ("Xuefeng Bai", "Xue-Feng Bai"),
    ("Yanmin Qian", "Yan-Min Qian"),
    ("Yaowu Chen", "Yao-Wu Chen"),
    ("Yi Tay", "Yi Tay"),
    ("Yongmei Zhou", "Yong-Mei Zhou"),
    ("Yongxin Huang", "Yong-Xin Huang"),
    ("Yuhao QING", "Yu-Hao Qing"),
    ("Yuxi Xie", "Yu-Xi Xie"),
    ("Ze Liu", "Ze Liu"),
    ("Zhaoxin Fan", "Zhao-Xin Fan"),
    ("Zhehuai Chen", "Zhe-Huai Chen"),
    ("Zhihong Zhang", "Zhi-Hong Zhang"),
    ("Zhiyi Tian", "Zhi-Yi Tian"),
]

# Ground truth: obviously non-Chinese names from ACL 2025
ACL_NON_CHINESE_NAMES = [
    "Aaron Mueller",
    "Sebastian Ruder",
    "Nathan Lambert",
    "Sara Hooker",
    "Michael Ekstrand",
    "Aaron Nicolson",
    "Jason Dowling",
    "Bevan Koopman",
    "Kevin Robinson",
    "Chirag Nagpal",
    "Alexander Nicholas D'Amour",
    "Kristian Lum",
    "Avi Feller",
    "Adam Fourney",
    "Adam Golinski",
    "Adam Jardine",
    "Adam Nohejl",
    "Adam Stein",
    "David Wadden",
    "Jeremy Clifton",
    # Additional verified non-Chinese names from random sample (seed=42)
    "Adrian Weller",
    "Ankita Gupta",
    "Anneliese Brei",
    "Anukriti Bhatnagar",
    "Archiki Prasad",
    "Carlos Gómez-Rodríguez",
    "Chen Amiraz",
    "Christopher Clark",
    "Claude Humbel",
    "Clayton Marr",
    "Daniel Deutsch",
    "Derry Tanti Wijaya",
    "Drishti Sharma",
    "Harish Tayyar Madabushi",
    "Helena Gomez Adorno",
    "Hiroaki Yamagiwa",
    "James Gung",
    "Kangil Kim",
    "Katharine Henry",
    "LLMs Trust Humans More",
    "Lai Hou Tim",
    "Marcos Zampieri",
    "Maria José Bocorny Finatto",
    "Minjun Park",
    "Mohit Raghavendra",
    "Reut Tsarfaty",
    "Rosey Billington",
    "Roy Bar-Haim",
    "Sajjadur Rahman",
    "Shinji Watanabe",
    "Shivani Kumar",
    "Su Lin Blodgett",
    "Tanmay Rajpurohit",
    "Tharindu Ranasinghe",
    "Viktor Moskvoretskii",
    "YoungBin Kim",
    "Yulia Tsvetkov",
    "Zeynab Ashourinezhad",
]

# ALL 191 ACL names with order flips
# These should preserve their original 'Given Surname' format
ACL_ORDER_PRESERVATION_TEST_CASES = [
    ("Cao Xiao", "Cao Xiao"),  # Currently: Xiao Cao
    ("Chang Ao", "Chang Ao"),  # Currently: Ao Chang
    ("Chang Ma", "Chang Ma"),  # Currently: Ma Chang
    ("Chen Feiyang", "Chen Feiyang"),  # Currently: Fei-Yang Chen
    ("Chen Gong", "Chen Gong"),  # Currently: Gong Chen
    ("Chen Li", "Chen Li"),  # Currently: Li Chen
    ("Chen Li Yu", "Chen Li Yu"),  # Currently: Li-Yu Chen
    ("Chen Lin", "Chen Lin"),  # Currently: Lin Chen
    ("Chen Luo", "Chen Luo"),  # Currently: Luo Chen
    ("Chen Qian", "Chen Qian"),  # Currently: Qian Chen
    ("Chen Shen", "Chen Shen"),  # Currently: Shen Chen
    ("Chen Tang", "Chen Tang"),  # Currently: Tang Chen
    ("Chen Zhi", "Chen Zhi"),  # Currently: Zhi Chen
    ("Chen Zhu", "Chen Zhu"),  # Currently: Zhu Chen
    ("Cheng Qian", "Cheng Qian"),  # Currently: Qian Cheng
    ("Cheng Wan", "Cheng Wan"),  # Currently: Wan Cheng
    ("Cheng Wen", "Cheng Wen"),  # Currently: Wen Cheng
    ("Chong Feng", "Chong Feng"),  # Currently: Feng Chong
    ("Chong Ruan", "Chong Ruan"),  # Currently: Ruan Chong
    ("Deng Cai", "Deng Cai"),  # Currently: Cai Deng
    ("Deng Qiyuan", "Deng Qiyuan"),  # Currently: Qi-Yuan Deng
    ("Di Shang", "Di Shang"),  # Currently: Shang Di
    ("Dong Yu", "Dong Yu"),  # Currently: Yu Dong
    ("Fan Bu", "Fan Bu"),  # Currently: Bu Fan
    ("Fan Yin", "Fan Yin"),  # Currently: Yin Fan
    ("Feng Wei", "Feng Wei"),  # Currently: Wei Feng
    ("Feng Xia", "Feng Xia"),  # Currently: Xia Feng
    ("GAO Zuchen", "Gao Zuchen"),  # Currently: Zu-Chen Gao
    ("Ge Qu", "Ge Qu"),  # Currently: Qu Ge
    ("Ge Shi", "Ge Shi"),  # Currently: Shi Ge
    ("Ge Yu", "Ge Yu"),  # Currently: Yu Ge
    ("Guo Gan", "Guo Gan"),  # Currently: Gan Guo
    ("Han Fang", "Han Fang"),  # Currently: Fang Han
    ("Han Meng", "Han Meng"),  # Currently: Meng Han
    ("Han Qiu", "Han Qiu"),  # Currently: Qiu Han
    ("Han Shi", "Han Shi"),  # Currently: Shi Han
    ("Han Xiao", "Han Xiao"),  # Currently: Xiao Han
    ("Han Yuan", "Han Yuan"),  # Currently: Yuan Han
    ("Hao Fei", "Hao Fei"),  # Currently: Fei Hao
    ("Hao-Ran Wei", "Hao-Ran Wei"),  # Currently: Wei Hao-Ran
    ("Haoran Jin", "Hao-Ran Jin"),  # Currently: Jin Haoran
    ("Haoran Que", "Hao-Ran Que"),  # Currently: Que Haoran
    ("Haoran Ye", "Hao-Ran Ye"),  # Currently: Ye Haoran
    ("He Yan", "He Yan"),  # Currently: Yan He
    ("Hu Kai", "Hu Kai"),  # Currently: Kai Hu
    ("Hu Yiwen", "Hu Yiwen"),  # Currently: Yi-Wen Hu
    ("Huang Qing", "Huang Qing"),  # Currently: Qing Huang
    ("Hui Su", "Hui Su"),  # Currently: Su Hui
    ("Hui Wei", "Hui Wei"),  # Currently: Wei Hui
    ("Jiang Bian", "Jiang Bian"),  # Currently: Bian Jiang
    ("Jiang Gui", "Jiang Gui"),  # Currently: Gui Jiang
    ("Jiang Tian", "Jiang Tian"),  # Currently: Tian Jiang
    ("Jiang Zhong", "Jiang Zhong"),  # Currently: Zhong Jiang
    ("Jie Ying", "Jie Ying"),  # Currently: Ying Jie
    ("Junjie Fang", "Jun-Jie Fang"),  # Currently: Fang Junjie
    ("Junjie Peng", "Jun-Jie Peng"),  # Currently: Peng Junjie
    ("Junjie Ye", "Jun-Jie Ye"),  # Currently: Ye Junjie
    ("Ke Lei", "Ke Lei"),  # Currently: Lei Ke
    ("Ke Yi", "Ke Yi"),  # Currently: Yi Ke
    ("Kun Ji", "Kun Ji"),  # Currently: Ji Kun
    ("Kun Kuang", "Kun Kuang"),  # Currently: Kuang Kun
    ("Lecheng Zheng", "Lecheng Zheng"),  # Currently: Zheng Lecheng
    ("Lei Sha", "Lei Sha"),  # Currently: Sha Lei
    ("Lei Yu", "Lei Yu"),  # Currently: Yu Lei
    ("Li Hao", "Li Hao"),  # Currently: Hao Li
    ("Li Jin", "Li Jin"),  # Currently: Jin Li
    ("Li Kuang", "Li Kuang"),  # Currently: Kuang Li
    ("Li Lin", "Li Lin"),  # Currently: Lin Li
    ("Li Ni", "Li Ni"),  # Currently: Ni Li
    ("Li Qing", "Li Qing"),  # Currently: Qing Li
    ("Li Zheng", "Li Zheng"),  # Currently: Zheng Li
    ("Liang Lin", "Liang Lin"),  # Currently: Lin Liang
    ("Liang Pang", "Liang Pang"),  # Currently: Pang Liang
    ("Lin Gui", "Lin Gui"),  # Currently: Gui Lin
    ("Lin Lu", "Lin Lu"),  # Currently: Lu Lin
    ("Lin Mu", "Lin Mu"),  # Currently: Mu Lin
    ("Lin Qiu", "Lin Qiu"),  # Currently: Qiu Lin
    ("Lin Yan", "Lin Yan"),  # Currently: Yan Lin
    ("Liu Guoming", "Liu Guoming"),  # Currently: Guo-Ming Liu
    ("Liu Weichuan", "Liu Weichuan"),  # Currently: Wei-Chuan Liu
    ("Liu Yabin", "Liu Yabin"),  # Currently: Ya-Bin Liu
    ("Long Le", "Long Le"),  # Currently: Le Long
    ("Lu Xiang", "Lu Xiang"),  # Currently: Xiang Lu
    ("Lu Xiangju", "Lu Xiangju"),  # Currently: Xiang-Ju Lu
    ("MA Jianfei", "Ma Jianfei"),  # Currently: Jian-Fei Ma
    ("MA Zhuoheng", "Ma Zhuoheng"),  # Currently: Zhuo-Heng Ma
    ("Man Lan", "Man Lan"),  # Currently: Lan Man
    ("Mo Yu", "Mo Yu"),  # Currently: Yu Mo
    ("Peng Di", "Peng Di"),  # Currently: Di Peng
    ("Peng Qi", "Peng Qi"),  # Currently: Qi Peng
    ("Pi Bu", "Pi Bu"),  # Currently: Bu Pi
    ("Qianlong Du", "Qian-Long Du"),  # Currently: Du Qianlong
    ("Qianlong Wang", "Qian-Long Wang"),  # Currently: Wang Qianlong
    ("Qin Jin", "Qin Jin"),  # Currently: Jin Qin
    ("Qiu Jiantao", "Qiu Jiantao"),  # Currently: Jian-Tao Qiu
    ("Ran Xin", "Ran Xin"),  # Currently: Xin Ran
    ("Rui Ming", "Rui Ming"),  # Currently: Ming Rui
    ("SU Yihang", "Su Yihang"),  # Currently: Yi-Hang Su
    ("Shao Kuan Wei", "Shao-Kuan Wei"),  # Currently: Kuan-Wei Shao
    ("Shi Luohe", "Shi Luohe"),  # Currently: Luo-He Shi
    ("Shi Yu", "Shi Yu"),  # Currently: Yu Shi
    ("Song Huatong", "Song Huatong"),  # Currently: Hua-Tong Song
    ("Su Lu", "Su Lu"),  # Currently: Lu Su
    ("Sun Ao", "Sun Ao"),  # Currently: Ao Sun
    ("Tan Yue", "Tan Yue"),  # Currently: Yue Tan
    ("Tao Gui", "Tao Gui"),  # Currently: Gui Tao
    ("Tao Ji", "Tao Ji"),  # Currently: Ji Tao
    ("Tao Yu", "Tao Yu"),  # Currently: Yu Tao
    ("Teng Xiao", "Teng Xiao"),  # Currently: Xiao Teng
    ("Tian Jin", "Tian Jin"),  # Currently: Jin Tian
    ("Tian Lan", "Tian Lan"),  # Currently: Lan Tian
    ("Tian Qiu", "Tian Qiu"),  # Currently: Qiu Tian
    ("Tian Xia", "Tian Xia"),  # Currently: Xia Tian
    ("Ting Song", "Ting Song"),  # Currently: Song Ting
    ("Ting Xiao", "Ting Xiao"),  # Currently: Xiao Ting
    ("Ting Yao", "Ting Yao"),  # Currently: Yao Ting
    ("Tong Ruan", "Tong Ruan"),  # Currently: Ruan Tong
    ("Tong Xiao", "Tong Xiao"),  # Currently: Xiao Tong
    ("Tong Yu", "Tong Yu"),  # Currently: Yu Tong
    ("Wang Lin", "Wang Lin"),  # Currently: Lin Wang
    ("Wei Chengzhi", "Wei Chengzhi"),  # Currently: Cheng-Zhi Wei
    ("Wei Hao", "Wei Hao"),  # Currently: Hao Wei
    ("Wei Jia", "Wei Jia"),  # Currently: Jia Wei
    ("Wei Ju", "Wei Ju"),  # Currently: Ju Wei
    ("Wei Tao", "Wei Tao"),  # Currently: Tao Wei
    ("Wei Xi", "Wei Xi"),  # Currently: Xi Wei
    ("Wei Xue", "Wei Xue"),  # Currently: Xue Wei
    ("Xia Ning", "Xia Ning"),  # Currently: Ning Xia
    ("Xia Xiao", "Xia Xiao"),  # Currently: Xiao Xia
    ("Xiang Ao", "Xiang Ao"),  # Currently: Ao Xiang
    ("Xiang Fei", "Xiang Fei"),  # Currently: Fei Xiang
    ("Xiao Yu", "Xiao Yu"),  # Currently: Yu Xiao
    ("Xiao Zong", "Xiao Zong"),  # Currently: Zong Xiao
    ("Xie Chen", "Xie Chen"),  # Currently: Chen Xie
    ("Xin Jing", "Xin Jing"),  # Currently: Jing Xin
    ("Xinlei Chen", "Xin-Lei Chen"),  # Currently: Chen Xinlei
    ("Xinlei He", "Xin-Lei He"),  # Currently: He Xinlei
    ("Xu Guo", "Xu Guo"),  # Currently: Guo Xu
    ("Xu Han", "Xu Han"),  # Currently: Han Xu
    ("Xu Miao", "Xu Miao"),  # Currently: Miao Xu
    ("Xu Shen", "Xu Shen"),  # Currently: Shen Xu
    ("Xu Yinghui", "Xu Yinghui"),  # Currently: Ying-Hui Xu
    ("Yang Dai", "Yang Dai"),  # Currently: Dai Yang
    ("Yang Duan", "Yang Duan"),  # Currently: Duan Yang
    ("Yang Feng", "Yang Feng"),  # Currently: Feng Yang
    ("Yang Hou", "Yang Hou"),  # Currently: Hou Yang
    ("Yang Lu", "Yang Lu"),  # Currently: Lu Yang
    ("Yang Nan", "Yang Nan"),  # Currently: Nan Yang
    ("Yang Song", "Yang Song"),  # Currently: Song Yang
    ("Yang Xiang", "Yang Xiang"),  # Currently: Xiang Yang
    ("Yang Xiao", "Yang Xiao"),  # Currently: Xiao Yang
    ("Yang Xu", "Yang Xu"),  # Currently: Xu Yang
    ("Yang You", "Yang You"),  # Currently: You Yang
    ("Yang Yuan", "Yang Yuan"),  # Currently: Yuan Yang
    ("Yang Yuguang", "Yang Yuguang"),  # Currently: Yu-Guang Yang
    ("Yang Zhong", "Yang Zhong"),  # Currently: Zhong Yang
    ("Yao Fu", "Yao Fu"),  # Currently: Fu Yao
    ("Yao Jui-Feng", "Yao Jui-Feng"),  # Currently: Jui-Feng Yao
    ("Yao Mu", "Yao Mu"),  # Currently: Mu Yao
    ("Yao Shi", "Yao Shi"),  # Currently: Shi Yao
    ("Yao Shu", "Yao Shu"),  # Currently: Shu Yao
    ("Yao Wan", "Yao Wan"),  # Currently: Wan Yao
    ("Yao Xiao", "Yao Xiao"),  # Currently: Xiao Yao
    ("Ye Tian", "Ye Tian"),  # Currently: Tian Ye
    ("Ye Yuan", "Ye Yuan"),  # Currently: Yuan Ye
    ("Yi Gui", "Yi Gui"),  # Currently: Gui Yi
    ("Yu Chao", "Yu Chao"),  # Currently: Chao Yu
    ("Yu Fei", "Yu Fei"),  # Currently: Fei Yu
    ("Yu Guan", "Yu Guan"),  # Currently: Guan Yu
    ("Yu Hong", "Yu Hong"),  # Currently: Hong Yu
    ("Yu Kuang", "Yu Kuang"),  # Currently: Kuang Yu
    ("Yu Rong", "Yu Rong"),  # Currently: Rong Yu
    ("Yu Wan", "Yu Wan"),  # Currently: Wan Yu
    ("Yu Xi", "Yu Xi"),  # Currently: Xi Yu
    ("Yu Yan", "Yu Yan"),  # Currently: Yan Yu
    ("Yuan Meng", "Yuan Meng"),  # Currently: Meng Yuan
    ("Yuan Qi", "Yuan Qi"),  # Currently: Qi Yuan
    ("Yuan Sui", "Yuan Sui"),  # Currently: Sui Yuan
    ("Yue Xin", "Yue Xin"),  # Currently: Xin Yue
    ("Yuwen Wang", "Yuwen Wang"),  # Currently: Wang Yuwen
    ("Yuxuan Dong", "Yuxuan Dong"),  # Currently: Dong Yuxuan
    ("Yuxuan Gu", "Yuxuan Gu"),  # Currently: Gu Yuxuan
    ("Zhao Tong", "Zhao Tong"),  # Currently: Tong Zhao
    ("Zhao Yang", "Zhao Yang"),  # Currently: Yang Zhao
    ("Zheng Hui", "Zheng Hui"),  # Currently: Hui Zheng
    ("Zheng Lin", "Zheng Lin"),  # Currently: Lin Zheng
    ("Zheng Wei", "Zheng Wei"),  # Currently: Wei Zheng
    ("Zheng Yuan", "Zheng Yuan"),  # Currently: Yuan Zheng
    ("Zhou Zhao", "Zhou Zhao"),  # Currently: Zhao Zhou
    ("Zhu Cao", "Zhu Cao"),  # Currently: Cao Zhu
    ("Zhu Xu", "Zhu Xu"),  # Currently: Xu Zhu
]


def test_acl_chinese_names():
    """Test that known Chinese names from ACL 2025 are correctly detected."""
    detector = ChineseNameDetector()

    failed_cases = []

    for input_name, expected_output in ACL_CHINESE_NAMES:
        result = detector.is_chinese_name(input_name)
        if not result.success:
            failed_cases.append(f"'{input_name}' should be Chinese but got: {result.error_message}")
        elif result.result != expected_output:
            failed_cases.append(f"'{input_name}' should normalize to '{expected_output}' but got '{result.result}'")

    if failed_cases:
        print("\nFailed Chinese name detections:")
        for case in failed_cases:
            print(f"  {case}")

    assert len(failed_cases) == 0, f"Failed to correctly detect {len(failed_cases)} Chinese names from ACL 2025"


def test_acl_non_chinese_names():
    """Test that known non-Chinese names from ACL 2025 are correctly rejected."""
    detector = ChineseNameDetector()

    failed_cases = []

    for name in ACL_NON_CHINESE_NAMES:
        result = detector.is_chinese_name(name)
        if result.success:
            failed_cases.append(f"'{name}' should be rejected but was detected as Chinese: {result.result}")

    if failed_cases:
        print("\nIncorrectly detected as Chinese:")
        for case in failed_cases:
            print(f"  {case}")

    assert len(failed_cases) == 0, f"Incorrectly detected {len(failed_cases)} non-Chinese names as Chinese"


def test_acl_order_preservation():
    """Test that ACL names in Given-Surname format are not flipped."""
    detector = ChineseNameDetector()

    failed_cases = []

    for input_name, expected_output in ACL_ORDER_PRESERVATION_TEST_CASES:
        result = detector.is_chinese_name(input_name)
        if not result.success:
            failed_cases.append(f"{input_name} should be Chinese but got: {result.error_message}")
        else:
            # Check that order is preserved (surname stays last)
            input_surname = input_name.split()[-1]
            output_surname = result.result.split()[-1]
            if input_surname.lower() != output_surname.lower():
                failed_cases.append(f"{input_name} surname order changed: {input_name} -> {result.result}")

    if failed_cases:
        print("\nOrder preservation failures:")
        for case in failed_cases:
            print(f"  {case}")

    assert len(failed_cases) == 0, f"{len(failed_cases)} ACL names had incorrect order changes"


def analyze_acl_2025_authors():
    """Analyze all ACL 2025 author names (not a test, just analysis)."""
    detector = ChineseNameDetector()

    # Read all author names from the extracted file
    authors_file = Path(__file__).parent.parent / "acl_2025_authors.txt"

    with open(authors_file, encoding="utf-8") as f:
        authors = [line.strip() for line in f if line.strip()]

    chinese_names = []
    non_chinese_names = []
    errors = []

    for author in authors:
        try:
            result = detector.is_chinese_name(author)
            if result.success:
                chinese_names.append((author, result.result))
            else:
                non_chinese_names.append((author, result.error_message))
        except Exception as e:
            errors.append((author, str(e)))

    print("\n=== ACL 2025 Author Name Analysis ===")
    print(f"Total authors processed: {len(authors)}")
    print(f"Chinese names detected: {len(chinese_names)}")
    print(f"Non-Chinese names: {len(non_chinese_names)}")
    print(f"Errors: {len(errors)}")

    if chinese_names:
        print("\n=== Sample Detected Chinese Names ===")
        for original, normalized in sorted(chinese_names)[:25]:  # Show first 25
            print(f"  {original} → {normalized}")
        if len(chinese_names) > 25:
            print(f"  ... and {len(chinese_names) - 25} more")

    if errors:
        print(f"\n=== Errors ({len(errors)}) ===")
        for author, error in errors[:10]:  # Show first 10 errors
            print(f"  {author}: {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")

    return len(errors) == 0  # Return success status


if __name__ == "__main__":
    test_acl_chinese_names()
    test_acl_non_chinese_names()
    test_acl_order_preservation()
    analyze_acl_2025_authors()
