"""Source-field repairs (services/source_repairs.py) on the cases in tests/data/source_repair_cases.csv.

Each row is a source name, the expected output and the 0.6.4 output. ``kind`` is ``bug`` (0.6.4 kept the defect),
``surname`` (0.6.4 also got the surname wrong) or ``control`` (a real name that looks similar and must stay as 0.6.4
wrote it). Bug rows must fall back to the 0.6.4 output when their rule is switched off, so each row tests its rule.
Cases come from ORCID-linked people and from corpus names reviewed blind. Add rows to extend the suite.
"""

import csv
from pathlib import Path

import pytest

from sinonym.services import source_repairs
from sinonym.timo.interface import Instance, Predictor, SourceAuthorFields
from sinonym.utils.korean_given_names import korean_given_form

CASES = list(
    csv.DictReader((Path(__file__).parent / "data" / "source_repair_cases.csv").read_text(encoding="utf-8").splitlines()),
)
FIXES = [case for case in CASES if case["kind"] != "control"]
CONTROLS = [case for case in CASES if case["kind"] == "control"]


def _id(case: dict) -> str:
    return f"{case['rule']}:{case['kind']}:{case['first']}|{case['middle']}|{case['last']}"


def resolve(predictor: Predictor, case: dict) -> tuple[str, str, str]:
    source = SourceAuthorFields(first_name=case["first"] or None, middle_names=case["middle"] or None, last_name=case["last"])
    author = predictor.predict_batch([Instance(pp_authors=[source])])[0].authors[0]
    return author.first_name, author.middle_names, author.last_name


def expected(case: dict) -> tuple[str, str, str]:
    return case["exp_first"], case["exp_middle"], case["exp_last"]


def was_064(case: dict) -> tuple[str, str, str]:
    return case["was_064_first"], case["was_064_middle"], case["was_064_last"]


@pytest.mark.parametrize("case", CASES, ids=[_id(c) for c in CASES])
def test_source_repair_case(predictor: Predictor, case: dict) -> None:
    assert resolve(predictor, case) == expected(case), case["reason"]


@pytest.mark.parametrize("case", FIXES, ids=[_id(c) for c in FIXES])
def test_fix_depends_on_its_rule(predictor: Predictor, case: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    rules = tuple((name, rule) for name, rule in source_repairs.RULES if name != case["rule"])
    monkeypatch.setattr(source_repairs, "RULES", rules)
    assert was_064(case) != expected(case)
    assert resolve(predictor, case) == was_064(case)


@pytest.mark.parametrize("case", CONTROLS, ids=[_id(c) for c in CONTROLS])
def test_control_is_not_repaired(case: dict) -> None:
    korean = korean_given_form(*was_064(case))
    assert expected(case) in {was_064(case), (korean, "", was_064(case)[2])}
    assert source_repairs.repair_source_fields(case["first"] or None, case["middle"] or None, case["last"], None) is None


def test_repair_applies_on_the_pp_vys_path(predictor: Predictor) -> None:
    source = SourceAuthorFields(first_name="DamianG.", last_name="Downey")
    instance = Instance(pp_authors=[source], vys_other_names=["Damian G. Downey", "Michael M. Tunney", "J. Stuart Elborn"])
    author = predictor.predict_batch([instance])[0].authors[0]
    assert (author.first_name, author.middle_names, author.last_name) == ("Damian", "G.", "Downey")


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        # with the word in the prefix table, the guard itself must hold
        (("CityU", None, "Library", None), None),
        (("UniKL", None, "Library", None), None),
        (("PharmD", None, "Estes", None), None),
        (("MedJ.", None, "Estes", None), None),
        (("JeongA", None, "Lee", None), None),
        (("DamianG.", None, "Downey", "Jr."), None),
        (("ArunaS.", None, "Rao", None), ("Aruna", "S.")),
    ],
)
def test_glued_initial_guards_hold_for_listed_words(args, expected, monkeypatch: pytest.MonkeyPatch) -> None:
    table = {word: ("", "") for word in ("City", "Uni", "Pharm", "Med", "Jeong", "Aruna", "Damian")}
    monkeypatch.setattr(source_repairs, "_glued_prefixes", lambda: table)
    assert source_repairs.unglue_given_initial(*args) == expected


def test_predictor_loads_the_data_tables() -> None:
    from sinonym.timo.interface import PredictorConfig
    from sinonym.utils import korean_given_names, unsplit_given_words

    for cached in (source_repairs._glued_prefixes, unsplit_given_words._words, korean_given_names._words):
        cached.cache_clear()
    Predictor(PredictorConfig(parallel="never"), ".")
    assert all(
        cached.cache_info().currsize == 1
        for cached in (source_repairs._glued_prefixes, unsplit_given_words._words, korean_given_names._words)
    )
