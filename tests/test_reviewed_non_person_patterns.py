"""Tests for source patterns that prove a routed row is not a person."""

import pytest

from sinonym.services.non_person import reviewed_non_person_source_pattern, reviewed_non_person_text_pattern


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
        ((None, None, "대한전자공학회"), "hangul_organization_marker"),
        ((None, None, "부산외국어대학교 중국학부"), "hangul_organization_marker"),
        ((None, None, "한국연구소"), "hangul_organization_marker"),
        ((None, None, "연구원자료"), "hangul_organization_marker"),
        ((None, None, "상임위원회"), "hangul_organization_marker"),
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
