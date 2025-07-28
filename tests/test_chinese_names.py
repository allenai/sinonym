"""
Golden Master Test Suite for Chinese Names Refactoring

This test captures the current behavior of the chinese_names module to ensure
that refactoring doesn't change the public API behavior.

Key fixes tested:
- Tiered confidence system prevents Western name false positives (Julian Lee, Adrian Chen)
- Cultural validation preserves legitimate Chinese names (Zixuan Wang)
- Gold/Silver/Bronze standard classification for name component splitting
- Western name pattern detection (specifically -ian endings without Chinese phonetics)

Additional fixes in this version:
- Missing syllable additions to PLAUSIBLE_COMPONENTS: "cong", "cuan", "bian", "cui"
- Forbidden pattern logic fix: allows Chinese compounds like "Dongliang" while blocking Western names
- Comprehensive coverage of real Chinese names to prevent future regressions
- Enhanced Western name blocking to maintain precision
"""

import pickle
import sys
from pathlib import Path

import pytest

# Add the parent directory to path to import s2and
sys.path.insert(0, str(Path(__file__).parent.parent))

from sinonym import ChineseNameDetector


class GoldenMasterTester:
    """Captures and validates chinese_names behavior."""

    def __init__(self):
        self.golden_file = Path(__file__).parent / "golden_master_chinese_names.pkl"
        self.detector = ChineseNameDetector()

    def capture_golden_master(self, test_cases: list[str]) -> dict[str, tuple[bool, str]]:
        """Capture the current behavior as golden master."""
        results = {}
        for test_case in test_cases:
            try:
                result = self.detector.is_chinese_name(test_case)
                # Convert ParseResult to tuple format for compatibility
                results[test_case] = (result.success, result.result if result.success else result.error_message)
            except Exception as e:
                results[test_case] = (False, f"Exception: {e!s}")
        return results

    def save_golden_master(self, results: dict[str, tuple[bool, str]]) -> None:
        """Save golden master results to disk."""
        with open(self.golden_file, "wb") as f:
            pickle.dump(results, f)

    def load_golden_master(self) -> dict[str, tuple[bool, str]]:
        """Load golden master results from disk."""
        if not self.golden_file.exists():
            return {}
        with open(self.golden_file, "rb") as f:
            return pickle.load(f)

    def validate_against_golden_master(
        self,
        current_results: dict[str, tuple[bool, str]],
        golden_results: dict[str, tuple[bool, str]],
    ) -> None:
        """Validate current results match golden master."""
        mismatches = []

        for test_case, golden_result in golden_results.items():
            if test_case not in current_results:
                mismatches.append(f"Missing test case: {test_case}")
                continue

            current_result = current_results[test_case]
            if current_result != golden_result:
                mismatches.append(
                    f"Mismatch for '{test_case}':\n  Golden:  {golden_result}\n  Current: {current_result}",
                )

        if mismatches:
            raise AssertionError(
                f"Golden master validation failed with {len(mismatches)} mismatches:\n"
                + "\n".join(mismatches[:10]),  # Show first 10 mismatches
            )


# Test cases with expected outcomes from chinese_names.py
CHINESE_NAME_TEST_CASES = [
    ("Yu-Zhong Wei", (True, "Yu-Zhong Wei")),
    ("Yu-zhong Wei", (True, "Yu-Zhong Wei")),
    ("Yuzhong Wei", (True, "Yu-Zhong Wei")),
    ("YuZhong Wei", (True, "Yu-Zhong Wei")),
    ("Yu Zhong Wei", (True, "Yu-Zhong Wei")),
    ("Liu Dehua", (True, "De-Hua Liu")),
    ("Dehua Liu", (True, "De-Hua Liu")),
    ("Zhou Xun", (True, "Xun Zhou")),
    ("Xun Zhou", (True, "Xun Zhou")),
    ("张为民 Wei-min Zhang", (True, "Wei-Min Zhang")),
    ("Wei-min Zhang 张为民", (True, "Wei-Min Zhang")),
    ("Wei Min Zhang", (True, "Wei-Min Zhang")),
    ("Li Ming", (True, "Ming Li")),
    ("Li Na", (True, "Na Li")),
    ("Xiao-Hong Li", (True, "Xiao-Hong Li")),
    ("Xiaohong Li", (True, "Xiao-Hong Li")),
    ("Xiaohong Li 张小红", (True, "Xiao-Hong Li")),
    ("Liu Zhi-guo", (True, "Zhi-Guo Liu")),
    ("Yu Jian-guo", (True, "Jian-Guo Yu")),
    ("He Jian-guo", (True, "Jian-Guo He")),
    ("Zhang Hong-xin", (True, "Hong-Xin Zhang")),
    ("Ouyang Xiaoming", (True, "Xiao-Ming Ouyang")),
    ("Gao Wei", (True, "Wei Gao")),
    ("Zhang Wei", (True, "Wei Zhang")),
    ("Wang Jun", (True, "Jun Wang")),
    ("Jun Wang", (True, "Jun Wang")),
    ("Chen Yu", (True, "Yu Chen")),
    ("Yu Chen", (True, "Yu Chen")),
    ("張Wei Ming", (True, "Wei-Ming Zhang")),
    ("Yu Murong", (True, "Yu Murong")),
    ("Tsai Yu", (True, "Yu Tsai")),
    ("Yu Tsai", (True, "Yu Tsai")),
    # Full-width apostrophe handling (Asian keyboard input)
    ("Ts'ao Ming", (True, "Ming Ts'ao")),  # Full-width apostrophe: preserves Wade-Giles form
    ("Ch'en Wei", (True, "Wei Ch'en")),  # Ch'en → qen → chen via Wade-Giles + SYLLABLE_RULES
    ("K'ung Fu", (True, "Fu K'ung")),  # Full-width apostrophe: preserves Wade-Giles form
    ("T'ang Li", (True, "Li T'ang")),  # Full-width apostrophe: preserves Wade-Giles form
    ("P'eng Yu", (True, "Yu P'eng")),  # Full-width apostrophe: preserves Wade-Giles form
    # Mixed apostrophe types (should work consistently)
    ("Ts'ao Ts'ai", (True, "Ts'ai Ts'ao")),  # Mixed ASCII and full-width apostrophes
    ("Chao（冯超） Feng", (True, "Chao Feng")),
    ("Chen-Hung Huang", (True, "Chen-Hung Huang")),
    ("Cheng-Hung Huang", (True, "Cheng-Hung Huang")),
    ("Chia-Ming Chang", (True, "Chia-Ming Chang")),
    ("Chine-Feng Wu", (True, "Chine-Feng Wu")),
    ("Y. Z. Wei", (True, "Y-Z Wei")),
    ("X.-H. Li", (True, "X-H Li")),
    # Cantonese names
    ("Chan Tai Man", (True, "Tai-Man Chan")),
    ("Wong Siu Ming", (True, "Siu-Ming Wong")),
    ("Lee Ka Fai", (True, "Ka-Fai Lee")),
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
    ("Chan Tai-Man", (True, "Tai-Man Chan")),
    ("Wong Kit", (True, "Kit Wong")),
    ("Au Yeung Chun", (True, "Chun Au-Yeung")),
    # Edge cases
    ("A. I. Lee", (True, "A-I Lee")),
    ("Wei Wei", (True, "Wei Wei")),
    ("Xu Xu", (True, "Xu Xu")),
    ("Chen Chen Yu", (True, "Chen-Yu Chen")),
    ("Wang Li Ming", (True, "Li-Ming Wang")),
    ("Chung Ming Wang", (True, "Chung-Ming Wang")),
    ("Au-Yeung Ka-Ming", (True, "Ka-Ming Au-Yeung")),
    ("Li.Wei.Zhang", (True, "Li-Wei Zhang")),
    ("Xiao Ming-hui Li", (True, "Xiao-Ming-Hui Li")),
    ("Ma Long", (True, "Long Ma")),
    # Cantonese names with overlapping surnames
    ("Choi Suk-Zan", (True, "Suk-Zan Choi")),
    ("Choi Ka-Fai", (True, "Ka-Fai Choi")),
    ("Choi Ming", (True, "Ming Choi")),
    ("Jung Chi-Wai", (True, "Chi-Wai Jung")),
    ("Lim Wai-Kit", (True, "Wai-Kit Lim")),
    ("Im Siu-Ming", (True, "Siu-Ming Im")),
    # Edge case fixes
    ("Lee Min", (True, "Min Lee")),
    ("Lee Jun", (True, "Jun Lee")),
    ("Teo Chee Hean", (True, "Chee-Hean Teo")),
    ("Goh Chok Tong", (True, "Chok-Tong Goh")),
    # Phase 3 fixes - compound splitting enhancements
    ("Li Zeze", (True, "Ze-Ze Li")),
    ("Li Siran", (True, "Si-Ran Li")),
    ("Chen Niran", (True, "Ni-Ran Chen")),
    # Names with initials + Chinese surnames
    ("H Y Tiong", (True, "H-Y Tiong")),
    ("Z D Chen", (True, "Z-D Chen")),
    ("Y Z Wang", (True, "Y-Z Wang")),
    ("H M Zhang", (True, "H-M Zhang")),
    ("P.Y. Huang", (True, "P-Y Huang")),
    ("D. W. Wang", (True, "D-W Wang")),
    # Names with compound given names that were originally failing but now work
    ("Jianying Zhou", (True, "Jian-Ying Zhou")),
    ("Jianping Fan", (True, "Jian-Ping Fan")),
    ("Jiangzhou Wang", (True, "Jiang-Zhou Wang")),
    ("Jianwei Zhang", (True, "Jian-Wei Zhang")),
    # Additional Chinese names with compound syllables
    ("Lianhua Wang", (True, "Lian-Hua Wang")),
    ("Tianjian Li", (True, "Tian-Jian Li")),
    # Session fixes - legitimate Chinese names that should be preserved
    ("Zixuan Wang", (True, "Zi-Xuan Wang")),  # Should pass tiered confidence system
    ("Weiming Zhang", (True, "Wei-Ming Zhang")),  # Gold standard (anchor + anchor)
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
    # Regression test for section 3 compound hyphen processing (GitHub issue: normalize_key bug)
    ("Ou-Yang Wei Ming", (True, "Wei-Ming Ou-Yang")),  # Tests that hyphenated compounds expand correctly
    ("Si-Ma Qian Feng", (True, "Qian-Feng Si-Ma")),  # Tests section 3 vs section 2 parse generation
    ("AuYeung Ka Ming", (True, "Ka-Ming Au-Yeung")),
    ("Zhou Kuihua", (True, "Kui-Hua Zhou")),  # kui syllable (1,293.5 ppm)
    ("Huang Dingyu", (True, "Ding-Yu Huang")),  # ding syllable (1,180.0 ppm)
    # Comma-separated "LAST, First" format tests (academic/professional contexts)
    ("Wei, Yu-Zhong", (True, "Yu-Zhong Wei")),
    ("Liu, Dehua", (True, "De-Hua Liu")),
    ("Zhang, Wei", (True, "Wei Zhang")),
    ("Chen, Yu", (True, "Yu Chen")),
    ("Wang, Li Ming", (True, "Li-Ming Wang")),
    ("Ouyang, Xiaoming", (True, "Xiao-Ming Ouyang")),
    ("Wong, Siu Ming", (True, "Siu-Ming Wong")),
    ("Chan, Tai Man", (True, "Tai-Man Chan")),
    ("Au-Yeung, Ka-Ming", (True, "Ka-Ming Au-Yeung")),
    ("Choi, Suk-Zan", (True, "Suk-Zan Choi")),
    # Test with extra whitespace (should be handled gracefully)
    ("Wei,   Yu-Zhong", (True, "Yu-Zhong Wei")),
    ("Liu,Dehua", (True, "De-Hua Liu")),  # No space after comma
    ("  Zhang  ,  Wei  ", (True, "Wei Zhang")),  # Extra whitespace
    ("Dan-Dan Zhang", (True, "Dan-Dan Zhang")),
    ("Dan Dan Zhang", (True, "Dan-Dan Zhang")),
    ("DAN CHEN", (True, "Dan Chen")),
    ("Juan He", (True, "Juan He")),
    ("Dan Sun", (True, "Dan Sun")),
    ("Nan Huang", (True, "Nan Huang")),
    ("Dan CHEN", (True, "Dan Chen")),
    ("Chen Dan", (True, "Dan Chen")),
    ("He Juan", (True, "Juan He")),
    ("Juan Yu", (True, "Juan Yu")),
    ("Dan Chen", (True, "Dan Chen")),
    ("Shu-Juan Li", (True, "Shu-Juan Li")),
    ("Dan-dan Zhang", (True, "Dan-Dan Zhang")),
    ("Liu Nan", (True, "Nan Liu")),
    ("Sun Dan", (True, "Dan Sun")),
    ("Dan Cheng", (True, "Dan Cheng")),
    ("Shi-Juan Li", (True, "Shi-Juan Li")),
    ("Juan Song", (True, "Juan Song")),
    ("Juan Liang", (True, "Juan Liang")),
    ("DAN SUN", (True, "Dan Sun")),
    ("Huang Nan", (True, "Nan Huang")),
    ("JUAN HE", (True, "Juan He")),
    ("贺娟 He Juan", (True, "Juan He")),  # Mixed Han characters
    ("陈丹 Chen Dan", (True, "Dan Chen")),  # Mixed Han characters
    ("LI Xiao-juan", (True, "Xiao-Juan Li")),
    ("XIAO-JUAN LI", (True, "Xiao-Juan Li")),
    ("Xiao Juan Li", (True, "Xiao-Juan Li")),
    ("Xiao-juan Li", (True, "Xiao-Juan Li")),
    ("Li Xiao-juan", (True, "Xiao-Juan Li")),
    ("Li Xiao-Juan", (True, "Xiao-Juan Li")),
    ("Xiao-Juan Li", (True, "Xiao-Juan Li")),
    ("X. Han", (True, "X Han")),
    ("L Han", (True, "L Han")),
    ("X. F. Han", (True, "X-F Han")),
    ("R. Han", (True, "R Han")),
    ("L. Han", (True, "L Han")),
    ("X F Han", (True, "X-F Han")),
    ("X. -F. Han", (True, "X-F Han")),
    (". X.F.Han", (True, "X-F Han")),
    ("Hu Cha", (True, "Cha Hu")),  # cha moved from Korean-only to overlapping
    ("Feng Cha", (True, "Cha Feng")),  # cha moved from Korean-only to overlapping
    ("He Cha", (True, "Cha He")),  # cha moved from Korean-only to overlapping
    ("Han Jun", (True, "Jun Han")),  # Previously rejected due to Korean overlap
    ("Han jun", (True, "Jun Han")),  # Case variant
    ("Jun Han", (True, "Jun Han")),  # Different order
    ("JUN HAN", (True, "Jun Han")),  # All caps
    ("Min-Hung Lee", (True, "Min-Hung Lee")),  # Korean-style hyphenation but Chinese content
    ("Xuefeng Han", (True, "Xue-Feng Han")),  # Chinese compound + overlapping surname
    ("Xiaojuan Han", (True, "Xiao-Juan Han")),  # Chinese compound + overlapping surname
    ("Ying-Nan P. Chen", (True, "Ying Nan P Chen")),  # TODO
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
    # Wade-Giles edge case
    ("Li Tsu", (True, "Tsu Li")),  # Tests tsu→cu syllable precedence over ts→z prefix
    ("Wang Tseng", (True, "Wang Tseng")),  # Tests ts→z prefix when no syllable match
    ("Chen Tsi", (True, "Tsi Chen")),  # Tests tsi→ci syllable precedence
    ("Wu Tzu", (True, "Tzu Wu")),  # Tests tzu→zi syllable precedence
    ("Zhang Hsien", (True, "Hsien Zhang")),  # Tests hs→x prefix conversion
    ("Huang Hsia", (True, "Hsia Huang")),  # Tests hsia→xia syllable conversion
    ("Zhou Chuang", (True, "Chuang Zhou")),  # Tests chuang→zhuang syllable conversion
    ("Gao Chuai", (True, "Chuai Gao")),  # Tests chuai→zhuai syllable conversion
    ("Sun Chueh", (True, "Chueh Sun")),  # Tests chueh→jue syllable conversion
    ("Ma Chui", (True, "Chui Ma")),  # Tests chui→zhui syllable conversion
    ("Xu Erh", (True, "Erh Xu")),  # Tests erh→er syllable conversion
    ("Fan Chien", (True, "Chien Fan")),  # Tests ien→ian suffix conversion potential
    # Korean surname overlap fix verification
    ("Kong Kung", (True, "Kung Kong")),
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
    # Wade-Giles forms that convert to moved surnames
    ("Wei Kung", (True, "Wei Kung")),  # kung→gong conversion test (name order preserved)
    ("Kung Li", (True, "Li Kung")),  # kung→gong as surname test (name order preserved)
    ("Li Weiwei", (True, "Wei-Wei Li")),
    ("Zhang Weiwei", (True, "Wei-Wei Zhang")),
    ("Wang Weiming", (True, "Wei-Ming Wang")),
    ("Chen Wenjun", (True, "Wen-Jun Chen")),
    # Mixed Han + non-initial roman tokens (parenthetical given names)
    ("张（Wei）Ming", (True, "Ming Zhang")),  # Han surname + roman given name in parentheses
    ("李（Peter）Chen", (True, "Li Chen")),  # Han surname, Western given name stripped
    ("Wang（小明）Zhang", (True, "Zhang Wang")),  # Roman surname + Han given name in parentheses
    ("陈（David）Liu", (True, "Chen Liu")),  # Han surname, Western given name stripped
    ("Zhou（Mary）Li", (True, "Li Zhou")),  # Mixed Han/Roman with Western name stripped
    ("刘（Thomas）Wang", (True, "Liu Wang")),  # Han surname, Western given name stripped
    # Three-token given names (common in mainland ID data)
    ("Li Wei Ming Hua", (True, "Wei-Ming-Hua Li")),  # 4 tokens: surname + 3-part given name
    ("Zhang San Ge Zi", (True, "San-Ge-Zi Zhang")),  # 4 tokens: compound hyphenated given name
    ("Chen Yi Er San", (True, "Yi-Er-San Chen")),  # 4 tokens: numerical given name components
    ("Wang A B C", (True, "A-B-C Wang")),  # 4 tokens: single-letter given name components
    ("Liu Xiao Ming Li", (True, "Xiao-Ming-Li Liu")),  # 4 tokens: common 3-part given name
    ("Zhou Da Zhong Xiao", (True, "Da-Zhong-Xiao Zhou")),  # 4 tokens: size-based given name
    # Wade-Giles syllables with ü (now work correctly with comprehensive diacritical support)
    ("Yü Li", (True, "Yü Li")),  # Wade-Giles yü -> yu conversion works correctly
    ("Li Yü", (True, "Yü Li")),  # Wade-Giles yü -> yu conversion works correctly
    ("Lü Wei", (True, "Wei Lü")),  # Wade-Giles lü -> lu conversion now works correctly
    ("Chü Chen", (True, "Chü Chen")),  # Wade-Giles chü -> ju conversion works correctly
    ("Lü Buwei", (True, "Bu-Wei Lü")),  # Historical Chinese name now works with ü support
    # Concatenated name handling - single token cases (fixed in this session)
    ("LinShu", (True, "Shu Lin")),  # Simple CamelCase → Lin Shu
    ("XIAOChen", (True, "Xiao Chen")),  # Mixed caps → XIAO Chen
    ("LuWANG", (True, "Lu Wang")),  # Mixed caps → Lu WANG
    ("XiaoMing", (True, "Ming Xiao")),  # Chinese CamelCase (surname first format)
    ("ZhangWei", (True, "Wei Zhang")),  # Chinese CamelCase (surname first format)
    # Multi-token cases with concatenated elements
    ("XiaoMing Li", (True, "Xiao-Ming Li")),  # Multi-token with concatenated given name
]

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
]

# Combine all test cases - Chinese with expected outcomes, non-Chinese just names
TEST_CASES = [name for name, expected in CHINESE_NAME_TEST_CASES] + NON_CHINESE_TEST_CASES


@pytest.fixture(scope="session")
def golden_master_tester():
    """Create and return a golden master tester instance."""
    return GoldenMasterTester()


def test_chinese_names_with_expected_results(golden_master_tester):
    """Test Chinese names with their expected exact outputs."""
    passed = 0
    failed = 0
    detector = ChineseNameDetector()

    for input_name, expected in CHINESE_NAME_TEST_CASES:
        result = detector.is_chinese_name(input_name)
        # Convert ParseResult to tuple format for comparison
        result_tuple = (result.success, result.result if result.success else result.error_message)
        if result_tuple == expected:
            passed += 1
        else:
            failed += 1
            print(f"FAILED: '{input_name}': expected {expected}, got {result_tuple}")

    assert failed == 0, f"Chinese name tests: {failed} failures out of {len(CHINESE_NAME_TEST_CASES)} tests"
    print(f"Chinese name tests: {passed} passed, {failed} failed")


def test_non_chinese_names_should_fail():
    """Test that non-Chinese names are correctly rejected."""
    detector = ChineseNameDetector()

    for input_name in NON_CHINESE_TEST_CASES:
        result = detector.is_chinese_name(input_name)
        assert result.success is False, f"Failed for '{input_name}': expected False, got {result.success}"

    print(f"Non-Chinese name tests: {len(NON_CHINESE_TEST_CASES)} passed")
