"""Source-field repairs (services/source_repairs.py) on the cases in tests/data/source_repair_cases.csv.

Each row is a source name, the expected output and the 0.6.4 output. ``kind`` is ``bug`` (0.6.4 kept the defect),
``surname`` (0.6.4 also got the surname wrong) or ``control`` (a real name that looks similar and must stay as 0.6.4
wrote it). Bug rows must fall back to the 0.6.4 output when their rule is switched off, so each row tests its rule.
For ``credential`` rows (added in 0.6.9, made-up names) the ``was_064`` columns hold the 0.6.8 output.
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
        (("ClinicalJ.", None, "Estes", None), None),
        (("JeongA", None, "Lee", None), None),
        (("DamianG.", None, "Downey", "Jr."), None),
        (("ArunaS.", None, "Rao", None), ("Aruna", "S.")),
    ],
)
def test_glued_initial_guards_hold_for_listed_words(args, expected, monkeypatch: pytest.MonkeyPatch) -> None:
    table = {word: ("", "") for word in ("City", "Uni", "Pharm", "Clinical", "Jeong", "Aruna", "Damian")}
    monkeypatch.setattr(source_repairs, "_glued_prefixes", lambda: table)
    assert source_repairs.unglue_given_initial(*args) == expected


def test_predictor_loads_the_data_tables() -> None:
    from sinonym.timo.interface import PredictorConfig
    from sinonym.utils import allcaps_given_words, given_name_words, korean_given_names, unsplit_given_words

    tables = (
        source_repairs._glued_prefixes,
        unsplit_given_words._words,
        korean_given_names._words,
        allcaps_given_words._words,
        given_name_words._words,
    )
    for cached in tables:
        cached.cache_clear()
    Predictor(PredictorConfig(parallel="never"), ".")
    assert all(cached.cache_info().currsize == 1 for cached in tables)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        # a title still has its initial split off, so title stripping leaves the initial
        (("FatherJ.", None, "Brown"), ("J.", "", "Brown")),
        (("CaptainJ.", None, "Brown"), ("J.", "", "Brown")),
        (("ReverendJ.", None, "Brown"), ("J.", "", "Brown")),
        (("HonorableJ.", None, "Brown"), ("J.", "", "Brown")),
        (("MmeC.", None, "Dupont"), ("Mme", "C.", "Dupont")),
        (("DiplK.", None, "Schmidt"), ("Dipl", "K.", "Schmidt")),
        (("DraM.", None, "Garcia"), ("Dra", "M.", "Garcia")),
        # given names that also open organization strings
        (("AnonK.", None, "Chaiyasit"), ("Anon", "K.", "Chaiyasit")),
        (("RoyalT.", None, "Smith"), ("Royal", "T.", "Smith")),
        (("WhoS.", None, "Lee"), ("Who", "S.", "Lee")),
        (("MedH.", None, "Bayon"), ("Med", "H.", "Bayon")),
        # organization words keep the glued capital
        (("AmericanJ.", None, "Smith"), ("AmericanJ", "", "Smith")),
    ],
)
def test_glued_initial_after_title_or_shared_word(predictor: Predictor, source, expected) -> None:
    first, middle, last = source
    fields = SourceAuthorFields(first_name=first, middle_names=middle, last_name=last)
    author = predictor.predict_batch([Instance(pp_authors=[fields])])[0].authors[0]
    assert (author.first_name, author.middle_names, author.last_name) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (("R", "C M Edward Ph.D.", "Udovic"), ("R.", "C. M. Edward", "Udovic")),
        (("George", "B. Louis Ph.D.", "Arner"), ("George", "B. Louis", "Arner")),
        (("Jr.", "PharmD", "H. Glenn Anderson"), ("Jr.", "H. Glenn", "Anderson")),
        # a two-word surname field the repaired parse keeps whole stays whole
        (("A.", "K. M. PhD", "Fazlul Haque"), ("A.", "K. M.", "Fazlul Haque")),
        (("K", "S R PhD", "Rama Rao"), ("K.", "S. R.", "Rama Rao")),
    ],
)
def test_credential_repair_keeps_the_given_words(predictor: Predictor, source, expected) -> None:
    first, middle, last = source
    fields = SourceAuthorFields(first_name=first, middle_names=middle, last_name=last)
    author = predictor.predict_batch([Instance(pp_authors=[fields])])[0].authors[0]
    assert (author.first_name, author.middle_names, author.last_name) == expected


def test_credential_repair_on_one_author_of_several(predictor: Predictor) -> None:
    sources = [
        SourceAuthorFields(first_name="R", middle_names="C M Edward Ph.D.", last_name="Udovic"),
        SourceAuthorFields(first_name="John", last_name="Smith"),
        SourceAuthorFields(first_name="K", middle_names="S R PhD", last_name="Rama Rao"),
    ]
    authors = predictor.predict_batch([Instance(pp_authors=sources)])[0].authors
    assert [(a.first_name, a.middle_names, a.last_name) for a in authors] == [
        ("R.", "C. M. Edward", "Udovic"),
        ("John", "", "Smith"),
        ("K.", "S. R.", "Rama Rao"),
    ]


@pytest.mark.parametrize(
    ("source", "surname"),
    [
        (("R", "C M Edward Ph.D.", "Udovic"), "Udovic"),
        (("S", "R N Timothy Ph.D.", "Bredow"), "Bredow"),
        (("E", "C M John Ph.D.", "Rybolt"), "Rybolt"),
        (("George", "B. Louis Ph.D.", "Arner"), "Arner"),
        (("Vito", "J. Counsel Esq.", "Petitti"), "Petitti"),
        (("Millard", "E. Colonel Ed.D.", "Moon"), "Moon"),
        (("Robert", "J. Jr PhD", "Belton"), "Belton"),
        (("Kathryn", "S. McClelland Ph.D", "Humphrey"), "Humphrey"),
        (("S.", "H. M.Hum. Ll.M.", "Paripurna"), "Paripurna"),
        (("S.H.M.Hum.", "Ll.M", "Paripurna"), "Paripurna"),
        (("M.A.Edin.", "B.Sc", "Fothergill"), "Fothergill"),
        (("Nathalie", "C. Mdcm Frcpc", "Zeitouni"), "Zeitouni"),
        (("Ebenezer", "A. Face Facp", "Nyenwe"), "Nyenwe"),
        # the whole name in the surname field
        (("FRCPA", "FRCPath", "C.  Soon  Lee  MD"), "Lee"),
        (("Jr.", "PharmD", "H. Glenn Anderson"), "Anderson"),
        (("S.", "T. M.Eng.", "I Wayan Mustika"), "Mustika"),
        (("DNSc", "Aprn Ccrn", "A. Elaine Bond"), "Bond"),
        (("MChir", "FRCS", "D. Innes Williams MD"), "Williams"),
        (("Msph", "DrPH", "C. Perry Brown"), "Brown"),
        (("Mphil.", "Ir. H Agoes", "Santoso"), "Santoso"),
        (("Leslie", "A. PhD", "Burton"), "Burton"),
    ],
)
def test_credential_repair_keeps_the_source_surname(predictor: Predictor, source, surname) -> None:
    first, middle, last = source
    fields = SourceAuthorFields(first_name=first, middle_names=middle, last_name=last)
    assert predictor.predict_batch([Instance(pp_authors=[fields])])[0].authors[0].last_name == surname


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (("Sunita", "Rao", "Mbbs"), ("Sunita", None, "Rao")),
        (("Jessica", "Reyes Bsn", "Rn"), ("Jessica", None, "Reyes")),
        (("Holly Powell", None, "Cnm"), ("Holly", None, "Powell")),
        (("Kyle", "M Langston", "Pa-C"), ("Kyle", "M", "Langston")),
        (("Msn", "Anne McAdam", "Rn"), ("Anne", None, "McAdam")),
        (("Cde", "Linda Ross", "Rn"), ("Linda", None, "Ross")),
        (("Jan", "van Dijk", "PhD"), ("Jan", None, "van Dijk")),
        (("Anna", "Lee", "M.B.B.S."), ("Anna", None, "Lee")),
        (("Anna", "Lee", "Mb Bs"), ("Anna", None, "Lee")),
        # surnames and given names that are also on the credential list
        (("Nadia", "Karam", "Fahs"), None),
        (("Marie", "Pierre", "Fache"), None),
        (("Rd.", "Budi Santoso", "Fasa"), None),
        (("Debra", "Hale", "Ches"), None),
        (("Faan", "Joan Quinn", "Msn"), ("Faan", "Joan", "Quinn")),
        (("Ay", "Senur", "Pac"), None),
        (("Haro", "Lopez", "C. de"), None),
        # initials: spelled out or RN in capitals in the surname field, dotted in the first field before two words
        (("Ravi", "Kumar", "C N S"), None),
        (("Ravi Kumar", None, "C.N.S"), None),
        (("Prema", "Selvi", "D.N.P"), None),
        (("Amy", "Stone", "M.S.W."), None),
        (("Ravi", "Kumar", "RN"), None),
        (("D.R.", "Anand Kumar", "PhD"), None),
        (("M.R.S.", "Lakshmi Priya", "Phd"), None),
        (("CNS", "Ravi Kumar", "Phd"), ("CNS", "Ravi", "Kumar")),
        (("R.A.S.", "Ahmad", "Mrcp"), ("R.A.S.", None, "Ahmad")),
        (("MA", "Carol Hart", "Rn"), ("MA", "Carol", "Hart")),
        (("A.", "B. Wood", "Frcs"), ("A.", "B.", "Wood")),
        # a new surname that is a particle, a degree or a fragment
        (("Ali", "Al", "Phd"), None),
        (("Arun", "Kiran Das", "D.Sc."), ("Arun", "Kiran", "Das")),
        (("Anna", "Lee et al", "Mbbs"), None),
        (("Laura", "Wood MEd", "Rrt"), None),
        (("Garcia", "Torrente Arquitectos", "Slp"), None),
        (("Anna", "Lee", "Ph D"), ("Anna", None, "Lee")),
        # surname-first source, particle-only first field, a lone given word, a reviewed comma tail
        (("Okafor,", "O. Vincent.", "Ph.D"), None),
        (("De", "Jong", "Rn"), None),
        (("Lena", None, "MSW"), None),
        (("Lalit", "Kalra", "PhD, MRCP"), None),
    ],
)
def test_strip_last_field_credentials(source, expected) -> None:
    assert source_repairs.strip_last_field_credentials(*source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (("Sunita", "Rao", "Mbbs"), ("Sunita", "", "Rao")),
        (("Jessica", "Reyes Bsn", "Rn"), ("Jessica", "", "Reyes")),
        (("Nadia", "Karam", "Fahs"), ("Nadia", "Karam", "Fahs")),
        (("Hon", "Kit Wong", "PhD"), ("Hon-Kit", "", "Wong")),
        (("Dr", "Anne McAdam", "Rn"), ("Anne", "", "McAdam")),
        (("Marie", "Pierre", "Fache"), ("Marie", "Pierre", "Fache")),
    ],
)
def test_last_field_credentials_on_the_predictor(predictor: Predictor, source, expected) -> None:
    first, middle, last = source
    fields = SourceAuthorFields(first_name=first, middle_names=middle, last_name=last)
    author = predictor.predict_batch([Instance(pp_authors=[fields])])[0].authors[0]
    assert (author.first_name, author.middle_names, author.last_name) == expected
