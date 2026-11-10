"""Name regressions from ORCID-verified corpus names (tests/data/name_regression_cases.csv).

Each row is a source name, the expected output and the 0.6.3 output it replaces. ``case_class`` is the bug the row
guards (``split``, ``accent``, ``particle_flip``, ``given_phrase``) or ``chinese_control``, a Chinese name whose
normalization must not change. Add rows to extend the suite.
"""
import csv
from pathlib import Path

import pytest

from sinonym.timo.interface import Instance, Predictor, PredictorConfig, SourceAuthorFields

CASES = list(csv.DictReader((Path(__file__).parent / "data" / "name_regression_cases.csv").read_text(encoding="utf-8").splitlines()))

EDGE_CASES = [
    # a listed word inside an author-written hyphen or glued token does not block the Chinese reading
    (("Li", None, "Ta-Tsien"), ("Ta-Tsien", "", "Li")),
    (("Sheau-yueh", "J.", "Chao"), ("Sheau-Yueh", "J.", "Chao")),
    ((None, None, "YuLina"), ("Lina", "", "Yu")),
    (("Lina", None, "Wang"), ("Lina", "", "Wang")),
    # an author-written hyphen or CamelCase boundary still splits a listed word
    (("An-Nie", None, "Chen"), ("An-Nie", "", "Chen")),
    (("AnNie", None, "Chen"), ("An-Nie", "", "Chen")),
    # surname-first field layout: a common Chinese surname before a given-name word is read given-first,
    # a surname word after Lee keeps the order the fields give
    (("Chen", None, "Annie"), ("Annie", "", "Chen")),
    (("Wang", None, "Lina"), ("Lina", "", "Wang")),
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
