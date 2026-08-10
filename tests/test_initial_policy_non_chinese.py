"""Focused contract tests for non-Chinese personal-name initials."""

from __future__ import annotations

import pytest

from sinonym.services.person_name_normalization import (
    PersonNameNormalizationResult,
    PersonNameNormalizationService,
    PersonNameOutcome,
)


@pytest.fixture
def normalizer() -> PersonNameNormalizationService:
    """Return the dependency-free non-Chinese normalizer."""
    return PersonNameNormalizationService()


def assert_person_components(
    result: PersonNameNormalizationResult,
    expected: tuple[str, str, str],
) -> None:
    """Assert one successful canonical component assignment."""
    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None
    normalized = result.canonical_name.normalized
    assert (normalized.given_name, normalized.middle_name, normalized.surname) == expected


@pytest.mark.parametrize(
    "raw_name",
    ["A.D. Smith", "A.D.Smith", "A D Smith", "A. D. Smith", "A.D Smith", "a d Smith"],
)
def test_two_initial_typography_variants_converge(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert_person_components(result, ("A.", "D.", "Smith"))
    assert result.canonical_name is not None
    assert result.canonical_name.text == "A. D. Smith"
    assert result.canonical_name.source_text == raw_name


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("A.B.C. Smith", ("A.", "B. C.", "Smith")),
        ("A B C Smith", ("A.", "B. C.", "Smith")),
        ("P.M D'Mello", ("P.", "M.", "D'Mello")),
        ("A.S.V.L.Sandhya", ("A.", "S. V. L.", "Sandhya")),
    ],
)
def test_longer_initial_sequences_use_first_as_given_and_rest_as_middle(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
    expected: tuple[str, str, str],
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert_person_components(result, expected)
    assert result.canonical_name is not None
    assert result.canonical_name.source_text == raw_name


@pytest.mark.parametrize(
    "raw_name",
    ["John A B Smith", "John A. B. Smith", "John A.B. Smith"],
)
def test_full_given_name_keeps_all_following_initials_in_middle(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert_person_components(result, ("John", "A. B.", "Smith"))


@pytest.mark.parametrize(
    "raw_name",
    [
        "John A B Llibre Rodriguez",
        "John A.B. Llibre Rodriguez",
        "A B C Llibre Rodriguez",
        "A.B.C. Llibre Rodriguez",
        "A.B.C.Llibre Rodriguez",
    ],
)
def test_initial_expansion_does_not_change_compound_surname_floor(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.canonical_name is not None
    assert result.canonical_name.normalized.surname == "Llibre Rodriguez"
    assert result.canonical_name.normalized.middle_name.endswith(("B.", "C."))


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("AARCHA S S", ("Aarcha", "S.", "S.", "Aarcha S. S.")),
        ("Masterov R. A.", ("Masterov", "R.", "A.", "Masterov R. A.")),
        ("MASTEROV R.A.", ("Masterov", "R.", "A.", "Masterov R. A.")),
    ],
)
def test_comma_free_full_name_and_initial_tail_preserves_source_order(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
    expected: tuple[str, str, str, str],
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert_person_components(result, expected[:3])
    assert result.canonical_name is not None
    assert result.canonical_name.text == expected[3]
    assert result.canonical_name.source.order[0] == "given"


@pytest.mark.parametrize(
    "last_name",
    [
        "Masterov R. A.",
        "Masterov R.A.",
        "MASTEROV R A",
    ],
)
def test_structured_last_name_field_can_repair_surname_first_initial_tail(
    normalizer: PersonNameNormalizationService,
    last_name: str,
) -> None:
    result = normalizer.normalize_components(last_name=last_name)

    assert_person_components(result, ("R.", "A.", "Masterov"))
    assert result.canonical_name is not None
    assert result.canonical_name.source.surname == last_name


@pytest.mark.parametrize(
    "components",
    [
        {"first_name": "A.D.", "last_name": "Smith"},
        {"first_name": "A D", "last_name": "Smith"},
        {"first_name": "A.", "middle_name": "D.", "last_name": "Smith"},
        {"last_name": "A.D.Smith"},
    ],
)
def test_structured_variants_match_raw_policy_and_preserve_source_fields(
    normalizer: PersonNameNormalizationService,
    components: dict[str, str],
) -> None:
    result = normalizer.normalize_components(**components)

    assert_person_components(result, ("A.", "D.", "Smith"))
    assert result.canonical_name is not None
    if "first_name" in components:
        assert result.canonical_name.source.given_name == components["first_name"]
    if "middle_name" in components:
        assert result.canonical_name.source.middle_name == components["middle_name"]
    if "last_name" in components:
        assert result.canonical_name.source.surname == components["last_name"]


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("AD Smith", "AD Smith"),
        ("A.DSmith", "A.DSmith"),
        ("St.John Smith", "St.John Smith"),
    ],
)
def test_initial_policy_does_not_split_undelimited_or_transliteration_tokens(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
    expected: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.canonical_name is not None
    assert result.canonical_name.text == expected


def test_explicit_comma_establishes_surname_first_order(
    normalizer: PersonNameNormalizationService,
) -> None:
    raw_name = "SMITH, A D"
    result = normalizer.normalize_text(raw_name)

    assert_person_components(result, ("A.", "D.", "Smith"))
    assert result.canonical_name is not None
    assert result.canonical_name.source_text == raw_name


def test_explicit_component_boundaries_are_not_reallocated(
    normalizer: PersonNameNormalizationService,
) -> None:
    comma = normalizer.normalize_text("J. K., Aarcha")
    structured = normalizer.normalize_components(first_name="Aarcha", last_name="J. K.")

    for result in (comma, structured):
        assert_person_components(result, ("Aarcha", "", "J. K."))
