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
    *,
    given: str,
    middle: str,
    surname: str,
) -> None:
    """Assert one successful canonical component assignment."""
    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None
    assert result.canonical_name.normalized.given_name == given
    assert result.canonical_name.normalized.middle_name == middle
    assert result.canonical_name.normalized.surname == surname


@pytest.mark.parametrize(
    "raw_name",
    [
        "A.D. Smith",
        "A.D.Smith",
        "A D Smith",
        "A. D. Smith",
        "A.D Smith",
        "a d Smith",
    ],
)
def test_two_initial_typography_variants_converge(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert_person_components(result, given="A.", middle="D.", surname="Smith")
    assert result.canonical_name is not None
    assert result.canonical_name.text == "A. D. Smith"
    assert result.canonical_name.source_text == raw_name


@pytest.mark.parametrize(
    ("raw_name", "middle", "surname"),
    [
        ("A.B.C. Smith", "B. C.", "Smith"),
        ("A B C Smith", "B. C.", "Smith"),
        ("P.M D'Mello", "M.", "D'Mello"),
        ("A.S.V.L.Sandhya", "S. V. L.", "Sandhya"),
    ],
)
def test_longer_initial_sequences_use_first_as_given_and_rest_as_middle(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
    middle: str,
    surname: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    expected_given = "P." if raw_name.startswith("P") else "A."
    assert_person_components(result, given=expected_given, middle=middle, surname=surname)
    assert result.canonical_name is not None
    assert result.canonical_name.source_text == raw_name


@pytest.mark.parametrize(
    "raw_name",
    [
        "John A B Smith",
        "John A. B. Smith",
        "John A.B. Smith",
    ],
)
def test_full_given_name_keeps_all_following_initials_in_middle(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert_person_components(result, given="John", middle="A. B.", surname="Smith")


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
    assert result.canonical_name.normalized.middle_name.endswith("B.") or result.canonical_name.normalized.middle_name.endswith(
        "C.",
    )


@pytest.mark.parametrize("raw_name", ["Masterov R. A.", "Masterov R A", "Masterov R.A."])
def test_surname_first_initial_tails_accept_dotted_and_undotted_forms(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert_person_components(result, given="R.", middle="A.", surname="Masterov")
    assert result.canonical_name is not None
    assert result.canonical_name.text == "R. A. Masterov"
    assert result.canonical_name.source.order[0] == "surname"


@pytest.mark.parametrize("last_name", ["Masterov R. A.", "Masterov R A", "Masterov R.A."])
def test_structured_surname_first_initial_tails_match_raw_policy(
    normalizer: PersonNameNormalizationService,
    last_name: str,
) -> None:
    result = normalizer.normalize_components(last_name=last_name)

    assert_person_components(result, given="R.", middle="A.", surname="Masterov")
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

    assert_person_components(result, given="A.", middle="D.", surname="Smith")
    assert result.canonical_name is not None
    if "first_name" in components:
        assert result.canonical_name.source.given_name == components["first_name"]
    if "middle_name" in components:
        assert result.canonical_name.source.middle_name == components["middle_name"]
    if "last_name" in components:
        assert result.canonical_name.source.surname == components["last_name"]


@pytest.mark.parametrize("raw_name", ["J-D Smith", "J.-D. Smith"])
def test_explicitly_hyphenated_initials_remain_one_compound_given_name(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert_person_components(result, given="J.-D.", middle="", surname="Smith")


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("AD Smith", "AD Smith"),
        ("Gordon CS Smith", "Gordon CS Smith"),
        ("A.DSmith", "A.DSmith"),
        ("St.John Smith", "St.John Smith"),
        ("P.Sh. Ibragimov", "P.Sh. Ibragimov"),
        ("Alekseeva M.Yu. Alekseeva", "Alekseeva M.Yu. Alekseeva"),
        ("MM. Cunningham", "MM. Cunningham"),
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


def test_credential_cleanup_precedes_initial_expansion(
    normalizer: PersonNameNormalizationService,
) -> None:
    result = normalizer.normalize_text("M.A. E. Zayas")

    assert_person_components(result, given="E.", middle="", surname="Zayas")
    assert result.canonical_name is not None
    assert result.canonical_name.text == "E. Zayas"
    assert [token.text for token in result.dropped_tokens] == ["M.A."]


def test_comma_form_uses_the_same_initial_policy(
    normalizer: PersonNameNormalizationService,
) -> None:
    result = normalizer.normalize_text("Smith, A.D.")

    assert_person_components(result, given="A.", middle="D.", surname="Smith")
    assert result.canonical_name is not None
    assert result.canonical_name.source_text == "Smith, A.D."
