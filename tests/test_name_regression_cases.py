"""Name regressions from ORCID-verified corpus names (tests/data/name_regression_cases.csv).

Each row is a source name, the expected output and the 0.6.3 output it replaces. ``case_class`` is the bug the row
guards (``split``, ``accent``, ``particle_flip``, ``given_phrase``, ``bound_part``) or a control whose output must not
change (``chinese_control``, ``bound_part_control``). Add rows to extend the suite.
"""
import csv
from pathlib import Path

import pytest

from sinonym import ChineseNameDetector
from sinonym.timo.interface import Instance, Predictor, PredictorConfig, SourceAuthorFields
from sinonym.utils import unsplit_given_words

CASES = list(csv.DictReader((Path(__file__).parent / "data" / "name_regression_cases.csv").read_text(encoding="utf-8").splitlines()))

EDGE_CASES = [
    # a listed word inside an author-written hyphen or glued token does not block the Chinese reading
    (("Li", None, "Ta-Tsien"), ("Ta-Tsien", "", "Li")),
    (("Sheau-yueh", "J.", "Chao"), ("Sheau-Yueh", "J.", "Chao")),
    (("Lina", None, "Wang"), ("Lina", "", "Wang")),
    # ...but not when the bound word is the whole name, or the boundary is punctuation
    ((None, None, "YuLina"), ("", "", "YuLina")),
    ((None, None, "Sheau-yueh"), ("", "", "Sheau-yueh")),
    (("Ju-yeon", None, None), ("", "", "Ju-yeon")),
    ((None, None, "Mei-lina"), ("", "", "Mei-lina")),
    (("Wang.Lina", None, None), ("", "", "Wang.Lina")),
    (("Man\u2013jiSUN", None, "Lee"), ("Man-jiSUN", "", "Lee")),
    (("Man\u2212jiSUN", None, "Lee"), ("Man-jiSUN", "", "Lee")),
    # a bound word whose parts are all never-split words is not Chinese: the fields keep their order
    (("Lin", None, "Hansen-Lange"), ("Lin", "", "Hansen-Lange")),
    (("Wang", None, "Pinto-Neto"), ("Wang", "", "Pinto-Neto")),
    (("Chen", None, "Jensen-Hansen"), ("Chen", "", "Jensen-Hansen")),
    (("Mayer-rene", None, "Li"), ("Mayer-rene", "", "Li")),
    # all caps reads the same as mixed case
    (("TA-TSIEN", None, "LI"), ("Ta-Tsien", "", "Li")),
    (("Li", None, "TA-TSIEN"), ("Ta-Tsien", "", "Li")),
    (("MAN-jiSUN", None, "Lee"), ("MAN-jiSUN", "", "Lee")),
    # Yuchi as the compound surname: its normalized form yuji is a never-split word
    (("Zhiguang", None, "Yuchi"), ("Zhi-Guang", "", "Yuchi")),
    (("Yuchi", None, "Zhiguang"), ("Zhi-Guang", "", "Yuchi")),
    (("Xuebiao", None, "Yuchi"), ("Xue-Biao", "", "Yuchi")),
    (("Yinghao", None, "Yuchi"), ("Ying-Hao", "", "Yuchi")),
    (("Weiran", None, "Yuchi"), ("Wei-Ran", "", "Yuchi")),
    # Yuchi as a given name after a Chinese surname, both orders
    (("Zhang", None, "Yuchi"), ("Yu-Chi", "", "Zhang")),
    (("Yuchi", None, "Zhang"), ("Yu-Chi", "", "Zhang")),
    (("Wu", None, "Yuchi"), ("Yu-Chi", "", "Wu")),
    (("\u5434\u7389\u8fdf", "Wu", "Yuchi"), ("Yu-Chi", "", "Wu")),
    # Yuchi as a Japanese surname, or unclear: kept as written
    (("Akio", None, "Yuchi"), ("Akio", "", "Yuchi")),
    (("Yunosuke", None, "Yuchi"), ("Yunosuke", "", "Yuchi")),
    (("Yuchi", None, "Tanaka"), ("Yuchi", "", "Tanaka")),
    (("Ming", None, "Yuchi"), ("Ming", "", "Yuchi")),
    # the Yuchi evidence comes from the compound surname only, not from any word that normalizes like it
    (("Taishi", None, "Mochi"), ("Taishi", "", "Mochi")),
    (("Taishi", None, "Kochi"), ("Taishi", "", "Kochi")),
    (("Kochi", None, "Taishi"), ("Kochi", "", "Taishi")),
    # surname scoring ignores the never-split list only for a normalized form the source did not write
    (("SongHo", None, "Moon"), ("SongHo", "", "Moon")),
    (("Quang", None, "Yuchi"), ("Quang", "", "Yuchi")),
    # other compound surnames, both orders
    (("Xiaoqian", None, "Ouyang"), ("Xiao-Qian", "", "Ouyang")),
    (("Ouyang", None, "Xiaoqian"), ("Xiao-Qian", "", "Ouyang")),
    (("Zhuge", None, "Jianguo"), ("Jian-Guo", "", "Zhuge")),
    (("Xiaoming", None, "Situ"), ("Xiao-Ming", "", "Situ")),
    (("Shangguan", None, "Xiaoming"), ("Xiao-Ming", "", "Shangguan")),
    # an author-written hyphen or CamelCase boundary still splits a listed word
    (("An-Nie", None, "Chen"), ("An-Nie", "", "Chen")),
    (("AnNie", None, "Chen"), ("An-Nie", "", "Chen")),
    # off the Chinese path, a Korean or Cantonese given syllable stays out of the surname
    (("Chae", "Bin", "Lee"), ("Chae-Bin", "", "Lee")),
    (("Ben", "Bin", "Xu"), ("Ben", "Bin", "Xu")),
    (("Ahmad", "Bin", "Ismail"), ("Ahmad", "", "Bin Ismail")),
    (("Mohd", "bin", "Abdullah"), ("Mohd", "", "bin Abdullah")),
    (("Awang", "Bin", "Long"), ("Awang", "", "Bin Long")),
    (("Yusof", "Khamaruzaman Bin", "Wan"), ("Yusof", "Khamaruzaman", "Bin Wan")),
    (("Nike", "Kwai Cheung", "Lau"), ("Nike", "Kwai Cheung", "Lau")),
    (("Mary", "Cheung", "Lau"), ("Mary", "", "Cheung Lau")),
    (("Malcolm", "Ng Mou", "Kehn"), ("Malcolm", "", "Ng Mou Kehn")),
    (("Mary", "Yeung Lai", "Wah"), ("Mary", "", "Yeung Lai Wah")),
    (("Gert", "Jan te", "Riele"), ("Gert", "Jan", "Te Riele")),
    (("Lars", "Jan Bak", "Nielsen"), ("Lars", "Jan", "Bak Nielsen")),
    (("Sergio", "San Juan", "Dertkigil"), ("Sergio", "", "San Juan Dertkigil")),
    # surname-first field layout: a common Chinese surname before a given-name word is read given-first,
    # a surname word after Lee keeps the order the fields give
    (("Chen", None, "Annie"), ("Annie", "", "Chen")),
    (("Wang", None, "Lina"), ("Lina", "", "Wang")),
    (("Su", None, "Anna"), ("Anna", "", "Su")),
    (("Lee", None, "Baer"), ("Lee", "", "Baer")),
    # all caps
    (("ANNIE", None, "CHEN"), ("Annie", "", "Chen")),
    (("DANA", None, "CHEN"), ("Dana", "", "Chen")),
    (("XIAOMING", None, "WANG"), ("Xiao-Ming", "", "Wang")),
    # NFD input
    (("Rene\u0301", None, "Kok"), ("Ren\u00e9", "", "Kok")),
    (("Jan", None, "S\u030ci\u0301ma"), ("Jan", "", "\u0160\u00edma")),
    # Chinese order handling and tone marks are unchanged
    (("Wang", None, "Xiaoming"), ("Xiao-Ming", "", "Wang")),
    (("Xiāomíng", None, "Wáng"), ("Xiao-Ming", "", "Wang")),
    (("Lü", None, "Xiaoming"), ("Xiao-Ming", "", "Lü")),
    # mixed script: a Han surname beside a non-Chinese given word is left as written
    (("Annie", None, "陈"), ("Annie", "", "陈")),
]


@pytest.fixture(scope="module")
def predictor() -> Predictor:
    return Predictor(PredictorConfig(parallel="never"), ".")


def resolve(predictor: Predictor, first, middle, last):
    source = SourceAuthorFields(first_name=first or None, middle_names=middle or None, last_name=last)
    author = predictor.predict_batch([Instance(pp_authors=[source])])[0].authors[0]
    return author.first_name, author.middle_names, author.last_name


@pytest.mark.parametrize("case", CASES, ids=[f"{c['case_class']}:{c['first']} {c['middle']} {c['last']}" for c in CASES])
def test_name_regression_case(predictor, case):
    expected = (case["exp_first"], case["exp_middle"], case["exp_last"])
    assert resolve(predictor, case["first"], case["middle"], case["last"]) == expected, case["reason"]


@pytest.mark.parametrize(("source", "expected"), EDGE_CASES)
def test_name_regression_edge_case(predictor, source, expected):
    assert resolve(predictor, *source) == expected


DETECTOR_CASES = [
    ("Li Ta-Tsien", "Ta-Tsien Li"),
    ("LI TA-TSIEN", "Ta-Tsien Li"),
    ("Sheau-yueh", None),
    ("Ta-tsien", None),
    ("Chen,Annie", None),
    ("Lina.Wang", None),
    ("Wang\uff0cLina", None),
    ("Man\u2013jiSUN Lee", None),
]


@pytest.fixture(scope="module")
def detector() -> ChineseNameDetector:
    return ChineseNameDetector()


@pytest.mark.parametrize(("raw", "expected"), DETECTOR_CASES)
def test_bound_part_detector_and_batch(detector, raw, expected):
    single = detector.normalize_name(raw)
    batch = detector.analyze_name_batch([raw, "Wang Xiaoming"]).results[0]
    for result in (single, batch):
        assert result.success is (expected is not None)
        if expected is not None:
            assert result.result == expected
    assert not unsplit_given_words.is_listed_bound_part("Lina")


def test_standalone_words_reset_after_exception():
    def fail_inside():
        with unsplit_given_words.standalone_words("Chen Xiao-Lina"):
            assert unsplit_given_words.is_listed_bound_part("Lina")
            raise RuntimeError

    pytest.raises(RuntimeError, fail_inside)
    assert not unsplit_given_words.is_listed_bound_part("Lina")
