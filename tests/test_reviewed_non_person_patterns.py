"""Tests for source patterns that prove a routed row is not a person."""

import pytest

from sinonym import ChineseNameDetector
from sinonym.services.non_person import reviewed_non_person_source_pattern, reviewed_non_person_text_pattern

PUBLIC_STRUCTURED_NON_PERSON_CASES = [
    ("STADT", None, "NÜRNBERG"),
    ("UNDANG-UNDANG", None, "HAK CIPTA"),
    ("PENDIDIKAN", None, "AGAMA"),
    ("Pendidikan", None, "adalah perlu"),
    ("Pendidikan", None, "FIP UNY"),
    ("Dr", None, "Research Scholar"),
    ("DER", None, "REFORMATION UND"),
    ("M.Si", None, "PhD"),
    ("Not Available", None, "Not Available"),
    ("None", None, "None"),
    ("Unknown", None, "Author"),
    ("undefined", "No authorship", "indicated"),
    (None, None, "January-February"),
    (None, None, "Wku Libraries"),
    (None, None, "Kabupaten Kendal"),
    (None, None, "Anthony C. Laborte, Marissa C. Hitalia*"),
    ("Array", None, "BioPharma"),
    ("Professur", None, "Fördertechnik"),
    (None, None, "Petroleum Geo-Services"),
    (None, None, "services-customer support"),
    (None, None, "대한전자공학회"),
    (None, None, "부산외국어대학교 중국학부"),
    (None, None, "한국연구소"),
    (None, None, "연구원자료"),
    (None, None, "상임위원회"),
    (None, None, "北京大学"),
    (None, None, "國立臺灣大學"),
    (None, None, "北京⼤学"),
]


@pytest.mark.parametrize(
    ("source", "expected_pattern"),
    [
        (("STADT", None, "NÜRNBERG"), "municipal_text"),
        (("UNDANG-UNDANG", None, "HAK CIPTA"), "legal_text"),
        (("PENDIDIKAN", None, "AGAMA"), "education_text"),
        (("Pendidikan", None, "adalah perlu"), "education_text"),
        (("Pendidikan", None, "FIP UNY"), "education_text"),
        (("Dr", None, "Research Scholar"), "research_scholar_metadata"),
        (("DER", None, "REFORMATION UND"), "reformation_fragment"),
        (("M.Si", None, "PhD"), "credential_only"),
        (("Not Available", None, "Not Available"), "placeholder_literal"),
        (("None", None, "None"), "placeholder_literal"),
        (("Unknown", None, "Author"), "placeholder_literal"),
        (("undefined", "No authorship", "indicated"), "placeholder_literal"),
        ((None, None, "January-February"), "month_range"),
        ((None, None, "Wku Libraries"), "organization_token"),
        ((None, None, "Kabupaten Kendal"), "organization_token"),
        ((None, None, "Anthony C. Laborte, Marissa C. Hitalia*"), "non_person_literal"),
        (("Array", None, "BioPharma"), "non_person_literal"),
        (("Professur", None, "Fördertechnik"), "non_person_literal"),
        ((None, None, "Petroleum Geo-Services"), "hyphenated_services"),
        ((None, None, "services-customer support"), "hyphenated_services"),
        ((None, None, "대한전자공학회"), "hangul_organization_marker"),
        ((None, None, "부산외국어대학교 중국학부"), "hangul_organization_marker"),
        ((None, None, "한국연구소"), "hangul_organization_marker"),
        ((None, None, "연구원자료"), "hangul_organization_marker"),
        ((None, None, "상임위원회"), "hangul_organization_marker"),
        ((None, None, "北京大学"), "han_institution_marker"),
        ((None, None, "國立臺灣大學"), "han_institution_marker"),
        ((None, None, "北京⼤学"), "han_institution_marker"),
        (("Pendidikan", None, "Mardiah Astuti"), None),
        (("Manish", None, "Goyal Research Scholar"), None),
        (("Olivier", None, "Company"), None),
        (("BEng", None, "Robert McManus"), None),
        (("Babak", None, "Esmaeili"), None),
        (("Horiguchi", None, "Daigaku"), None),
        (("STADT", None, "ß"), None),
        (("Unknown", None, "Authors"), None),
        (("January", None, "February"), None),
        (("Nan", None, "Hao"), None),
        (("Syafira", "Elfithri", "Universitas"), None),
        (("Benjamín", "Cristian", "Corona-Comunidad"), None),
        (("Roque", "A.", "Comunidad-Bonilla"), None),
        (("E.", "Slovenská poľnohospodárska univerzita v Nitre", "Hazuchová"), None),
        (("Miguel", "Angel Clínica Universitaria de Navarra", "Monge"), None),
        (("M.", "D", "Services-REGINALD M. ATWATER"), None),
        (("A.", "Hosp. Clínico Universitario Lozano Bles", "Angusto"), None),
        (("北京", None, "大学"), None),
        ((None, "中国科学院", "大学"), None),
        ((None, None, "北京大学", "Jr."), None),
        (("STADT", None, "NÜRNBERG", "Jr."), None),
    ],
)
def test_reviewed_non_person_patterns_match_the_complete_review_matrix(
    source: tuple[str | None, ...],
    expected_pattern: str | None,
) -> None:
    assert reviewed_non_person_source_pattern(*source) == expected_pattern


@pytest.mark.parametrize(
    ("raw_name", "expected_pattern"),
    [
        (" Unknown Author ", "placeholder_literal"),
        ("Unknown\t\nAuthor", "placeholder_literal"),
        (" January-February ", "month_range"),
        (" Anthony C. Laborte,  Marissa C. Hitalia* ", "non_person_literal"),
        (" services-customer support ", "hyphenated_services"),
        (" 서울대학교 ", "hangul_organization_marker"),
        (" 부산외국어대학교   중국학부 ", "hangul_organization_marker"),
    ],
)
def test_reviewed_non_person_text_patterns_ignore_incidental_whitespace(
    raw_name: str,
    expected_pattern: str,
) -> None:
    assert reviewed_non_person_text_pattern(raw_name) == expected_pattern


@pytest.mark.parametrize(
    "raw_name",
    [
        "unknown author",
        "Unknown-Author",
        "Unknown Author Jr.",
        "January / February",
        "홍길동",
        "홍길동 서울대학교",
        "서울대학교 홍길동",
    ],
)
def test_reviewed_non_person_text_patterns_keep_lexical_matching_exact(raw_name: str) -> None:
    assert reviewed_non_person_text_pattern(raw_name) is None


def test_reviewed_non_person_raw_and_structured_whitespace_have_parity() -> None:
    assert reviewed_non_person_text_pattern(" Unknown   Author ") == reviewed_non_person_source_pattern(
        " Unknown ",
        None,
        " Author ",
    )


def test_last_only_han_institution_declines_middle_dot_person_affiliation() -> None:
    assert reviewed_non_person_source_pattern(None, None, "陳大文·香港大學") is None


@pytest.mark.parametrize(("first_name", "middle_name", "last_name"), PUBLIC_STRUCTURED_NON_PERSON_CASES)
def test_public_structured_ingress_rejects_every_reviewed_non_person_pattern(
    detector: ChineseNameDetector,
    first_name: str | None,
    middle_name: str | None,
    last_name: str | None,
) -> None:
    assert (
        detector.normalize_person_name_components(
            first_name=first_name,
            middle_name=middle_name,
            last_name=last_name,
        )
        is None
    )


@pytest.mark.parametrize(
    "raw_name",
    ["서울대학교", "대한전자공학회", "부산외국어대학교 중국학부", "한국연구소", "연구원자료", "상임위원회"],
)
def test_raw_hangul_organizations_are_rejected_at_public_ingress(
    detector: ChineseNameDetector,
    raw_name: str,
) -> None:
    assert detector.normalize_person_name(raw_name) is None
    assert detector.normalize_name(raw_name).canonical_name is None


@pytest.mark.parametrize("raw_name", ["홍길동", "김민수"])
def test_raw_hangul_person_controls_remain_people(
    detector: ChineseNameDetector,
    raw_name: str,
) -> None:
    assert detector.normalize_person_name(raw_name) is not None
