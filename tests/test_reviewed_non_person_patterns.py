"""Tests for source patterns that prove a routed row is not a person."""

import pytest

from sinonym.services.non_person import reviewed_non_person_source_pattern


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
    ],
)
def test_reviewed_non_person_patterns_match_complete_reviewed_classes(
    source: tuple[str | None, str | None, str | None],
    expected_pattern: str,
) -> None:
    assert reviewed_non_person_source_pattern(*source) == expected_pattern


@pytest.mark.parametrize(
    "source",
    [
        ("Pendidikan", None, "Mardiah Astuti"),
        ("Manish", None, "Goyal Research Scholar"),
        ("Olivier", None, "Company"),
        ("BEng", None, "Robert McManus"),
        ("Babak", None, "Esmaeili"),
        ("Horiguchi", None, "Daigaku"),
        ("STADT", None, "ß"),
        ("Unknown", None, "Authors"),
        ("January", None, "February"),
        ("Nan", None, "Hao"),
        ("Syafira", "Elfithri", "Universitas"),
        ("Benjamín", "Cristian", "Corona-Comunidad"),
        ("Roque", "A.", "Comunidad-Bonilla"),
        ("E.", "Slovenská poľnohospodárska univerzita v Nitre", "Hazuchová"),
        ("Miguel", "Angel Clínica Universitaria de Navarra", "Monge"),
        ("M.", "D", "Services-REGINALD M. ATWATER"),
        ("A.", "Hosp. Clínico Universitario Lozano Bles", "Angusto"),
    ],
)
def test_reviewed_non_person_patterns_exclude_person_controls(
    source: tuple[str | None, str | None, str | None],
) -> None:
    assert reviewed_non_person_source_pattern(*source) is None


def test_reviewed_non_person_patterns_decline_a_nonempty_source_suffix() -> None:
    assert reviewed_non_person_source_pattern("STADT", None, "NÜRNBERG", "Jr.") is None
