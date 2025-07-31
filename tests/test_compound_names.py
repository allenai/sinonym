"""
Compound Names Test Suite

This module contains tests for compound and multi-syllable Chinese names including:
- Compound given names
- Multi-part surnames (Ouyang, Sima, etc.)
- Three-token given names
- Complex name splitting patterns
"""

import sys
from pathlib import Path

# Add the parent directory to path to import sinonym
sys.path.insert(0, str(Path(__file__).parent.parent))

from sinonym import ChineseNameDetector

# Test cases for compound and multi-syllable names
CHINESE_NAME_TEST_CASES = [
    # Compound surnames
    ("Ouyang Xiaoming", (True, "Xiao-Ming Ouyang")),
    ("Au-Yeung Chun", (True, "Chun Au-Yeung")),
    ("Ouyang Zhen Yu", (True, "Zhen-Yu Ouyang")),  # System uses hyphens
    ("Sima Zhao Hua", (True, "Zhao-Hua Sima")),  # System uses hyphens
    ("Murong Xue", (True, "Xue Murong")),
    ("Duanmu Wenjie", (True, "Wen-Jie Duanmu")),  # Compound surname - preserves compact format
    ("Shangguan Wen", (True, "Wen Shangguan")),
    ("Ouyang Xiu", (True, "Xiu Ouyang")),
    ("Zhu Geliang", (True, "Ge-Liang Zhu")),  # Sounds like "Zhuge Liang"

    # Phase 3 fixes - compound splitting enhancements
    ("Li Zeze", (True, "Ze-Ze Li")),
    ("Li Siran", (True, "Si-Ran Li")),
    ("Chen Niran", (True, "Ni-Ran Chen")),

    # Names with compound given names that were originally failing but now work
    ("Jianying Zhou", (True, "Jian-Ying Zhou")),
    ("Jianping Fan", (True, "Jian-Ping Fan")),
    ("Jiangzhou Wang", (True, "Jiang-Zhou Wang")),
    ("Jianwei Zhang", (True, "Jian-Wei Zhang")),

    # Missing syllable fixes - cases that required adding syllables to PLAUSIBLE_COMPONENTS
    ("Congzuo Li", (True, "Cong-Zuo Li")),  # Added "cong" syllable
    ("Ceyan Wang", (True, "Ce-Yan Wang")),  # Already worked
    ("Suiluan Zhang", (True, "Sui-Luan Zhang")),  # Already worked
    ("Maoqin Chen", (True, "Mao-Qin Chen")),  # Already worked
    ("Chouzhe Liu", (True, "Chou-Zhe Liu")),  # Already worked
    ("Cuanfen Xu", (True, "Cuan-Fen Xu")),  # Added "cuan" syllable
    ("Weibian Zhao", (True, "Wei-Bian Zhao")),  # Added "bian" syllable
    ("Haotian Zhang", (True, "Hao-Tian Zhang")),  # Already worked
    ("Yidian Huang", (True, "Yi-Dian Huang")),  # Already worked
    ("Cuihua Zhang", (True, "Cui-Hua Zhang")),  # Added "cui" syllable

    # Forbidden pattern fix - cases that required fixing the forbidden pattern logic
    ("Dongliang Xu", (True, "Dong-Liang Xu")),  # Fixed forbidden pattern "gl" blocking

    # Three-token given names (common in mainland ID data)
    ("Li Wei Ming Hua", (True, "Wei-Ming-Hua Li")),  # 4 tokens: surname + 3-part given name
    ("Zhang San Ge Zi", (True, "San-Ge-Zi Zhang")),  # 4 tokens: compound hyphenated given name
    ("Chen Yi Er San", (True, "Yi-Er-San Chen")),  # 4 tokens: numerical given name components
    ("Wang A B C", (True, "A-B-C Wang")),  # 4 tokens: single-letter given name components
    ("Liu Xiao Ming Li", (True, "Xiao-Ming-Li Liu")),  # 4 tokens: common 3-part given name
    ("Zhou Da Zhong Xiao", (True, "Da-Zhong-Xiao Zhou")),  # 4 tokens: size-based given name

    # Complex compound patterns
    ("Ou-Ming Li", (True, "Ou-Ming Li")),  # Compound with alias as given name
    ("Jin Wei-Hua", (True, "Wei-Hua Jin")),  # Standard compound pattern

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

    # Compound name splitting
    ("Sun Xiao-long", (True, "Xiao-Long Sun")),
    ("Cai Yun-hui", (True, "Yun-Hui Cai")),
    ("Wang Xueyin", (True, "Xue-Yin Wang")),  # System uses hyphens
    ("Zou Shaoqi", (True, "Shao-Qi Zou")),  # System uses hyphens
    ("Huang Yixuan", (True, "Yi-Xuan Huang")),  # System uses hyphens

    # Rare surnames with compound given names
    ("Sa Beining", (True, "Bei-Ning Sa")),
    ("Dèng Yǎjuān", (True, "Ya-Juan Deng")),
    ("Tú Jīngwěi", (True, "Jing-Wei Tu")),
    ("Cén Lílán", (True, "Li-Lan Cen")),

    # Multi-surname constructions (Rare but Present)
    ("Li 大明", (True, "Da-Ming Li")),  # System uses hyphens

    # Complex syllable patterns
    ("Zhang Xuefeng", (True, "Xue-Feng Zhang")),  # Tests 'xue' syllable preserved in plausible_components
    ("Liu Yuehua", (True, "Yue-Hua Liu")),  # Tests 'yue' syllable preserved
    ("Chen Jueming", (True, "Jue-Ming Chen")),  # Tests 'jue' syllable preserved
    ("Wu Kuaile", (True, "Kuai-Le Wu")),  # Tests 'kuai' syllable preserved
    ("Wang Shuaiming", (True, "Shuai-Ming Wang")),  # Tests 'shuai' syllable preserved
    ("Li Hualiang", (True, "Hua-Liang Li")),  # Tests compound splitting still works

    # Extended compound patterns
    ("Zeng Ming-Li", (True, "Ming-Li Zeng")),  # Compound given name
    ("Hao-Jun Zeng", (True, "Hao-Jun Zeng")),  # Given name first format
    ("Zeng Xiao-Hong", (True, "Xiao-Hong Zeng")),  # Common given name pattern
    ("Yuan Jian-Guo", (True, "Jian-Guo Yuan")),  # Compound given name
    ("Wei-Ming Yuan", (True, "Wei-Ming Yuan")),  # Given name first format
    ("Ou Xiao-Li", (True, "Xiao-Li Ou")),  # Compound given name
    ("Yu-Bin Ou", (True, "Yu-Bin Ou")),  # Given name first format
    ("Jin Li-Ming", (True, "Li-Ming Jin")),  # Compound given name
    ("Xiao-Yu Jin", (True, "Xiao-Yu Jin")),  # Given name first format
    ("Lai Wei-Jun", (True, "Wei-Jun Lai")),  # Compound given name
    ("Ming-Hua Lai", (True, "Ming-Hua Lai")),  # Given name first format
    ("Miao Jian-Wei", (True, "Jian-Wei Miao")),  # Compound given name
    ("Li-Jun Miao", (True, "Li-Jun Miao")),  # Given name first format
    ("Zhai Yu-Ming", (True, "Yu-Ming Zhai")),  # Compound given name
    ("Xiao-Wei Zhai", (True, "Xiao-Wei Zhai")),  # Given name first format
    ("Mo Li-Hua", (True, "Li-Hua Mo")),  # Compound given name
    ("Jun-Ming Mo", (True, "Jun-Ming Mo")),  # Given name first format
    ("Wen Xiao-Jun", (True, "Xiao-Jun Wen")),  # Compound given name
    ("Yu-Li Wen", (True, "Yu-Li Wen")),  # Given name first format

    # Multi-component names with varied patterns
    # Regression test for section 3 compound hyphen processing (GitHub issue: normalize_key bug)

    # Cantonese compound patterns
    ("Chan Tai Man", (True, "Tai-Man Chan")),  # Keep Chan, not Chen
    ("Wong Siu Ming", (True, "Siu-Ming Wong")),
    ("Lee Ka Fai", (True, "Ka-Fai Lee")),
    ("Leung Ka Fai", (True, "Ka-Fai Leung")),
    ("Lau Tak Wah", (True, "Tak-Wah Lau")),
    ("Cheung Hok Yau", (True, "Hok-Yau Cheung")),
    ("Chow Yun Fat", (True, "Yun-Fat Chow")),
    ("Ng Man Tat", (True, "Man-Tat Ng")),
    ("Leung Chiu Wai", (True, "Chiu-Wai Leung")),
    ("Kwok Fu Shing", (True, "Fu-Shing Kwok")),
    ("Lam Ching Ying", (True, "Ching-Ying Lam")),
    ("Yeung Chin Wah", (True, "Chin-Wah Yeung")),
    ("Tsang Chi Wai", (True, "Chi-Wai Tsang")),
    ("Fung Hiu Man", (True, "Hiu-Man Fung")),
    ("Tse Ting Fung", (True, "Ting-Fung Tse")),
    ("Szeto Wah", (True, "Wah Szeto")),
    ("Yip Man", (True, "Man Yip")),
    ("Siu Ming Wong", (True, "Siu-Ming Wong")),
    ("Ka Fai Lee", (True, "Ka-Fai Lee")),
    ("Wong Kit", (True, "Kit Wong")),
    ("Szeto Wai Kin", (True, "Wai-Kin Szeto")),  # Keep Szeto, not Si Tu
    ("Lau Suk Yan", (True, "Suk-Yan Lau")),  # Keep Lau, not Liu

    # Cross-semantic frequency inheritance regression tests
    # These test cases ensure that given name tokens don't inherit problematic
    # surname frequencies through phonetic normalization (e.g., 'fai' → 'hui')
    ("Leung Wai Fai", (True, "Wai-Fai Leung")),  # 'fai' → 'hui' cross-semantic mapping
    ("Cheng Ka Fai", (True, "Ka-Fai Cheng")),  # Different surname with 'fai'
    ("Lam Siu Fai", (True, "Siu-Fai Lam")),  # Cantonese pattern with 'fai'
    ("Tang Wai Fai", (True, "Wai-Fai Tang")),  # Another surname with 'fai'
    ("Ho Ka Fai", (True, "Ka-Fai Ho")),  # Short surname with 'fai'
    ("Liu Ka Man", (True, "Ka-Man Liu")),  # 'man' → 'wen' cross-semantic mapping
    ("Zhao Wai Man", (True, "Wai-Man Zhao")),  # Different surname with 'man'
    ("Xu Siu Man", (True, "Siu-Man Xu")),  # Mandarin surname with Cantonese given name
    ("Chen Ka Hei", (True, "Ka-Hei Chen")),  # 'hei' → 'xi' cross-semantic mapping
    ("Wu Wai Hei", (True, "Wai-Hei Wu")),  # Different surname with 'hei'
    ("Yang Ka Mang", (True, "Ka-Mang Yang")),  # 'mang' → 'meng' cross-semantic mapping
    ("Zhou Wai Mang", (True, "Wai-Mang Zhou")),  # Different surname with 'mang'
    ("Guo Siu Mang", (True, "Siu-Mang Guo")),  # Another pattern with 'mang'
    ("Wang Ka Hui", (True, "Ka-Hui Wang")),  # Control: 'hui' as legitimate given name (wang freq > hui)
    ("Sun Wei Wen", (True, "Wei-Wen Sun")),  # Control: 'wen' as legitimate given name
    ("Ma Xiao Xi", (True, "Xiao-Xi Ma")),  # Control: 'xi' as legitimate given name

    # Given-name-first patterns (real people with these name structures)
    ("Ka Lin Hui", (True, "Ka-Lin Hui")),  # Given-name-first: given="Ka Lin", surname="Hui"
    ("Shih-Hui Lin", (True, "Shih-Hui Lin")),  # Wade-Giles given-name-first: given="Shih-Hui", surname="Lin"
]


def test_compound_names():
    """Test compound and multi-syllable Chinese names."""
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

    assert failed == 0, f"Compound name tests: {failed} failures out of {len(CHINESE_NAME_TEST_CASES)} tests"
    print(f"Compound name tests: {passed} passed, {failed} failed")


if __name__ == "__main__":
    test_compound_names()
