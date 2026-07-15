"""Focused tests for canonical person-name normalization."""

from dataclasses import FrozenInstanceError

import pytest

from sinonym.services import person_name_normalization
from sinonym.services.person_name_normalization import (
    DropReason,
    PersonNameNormalizationService,
    PersonNameOutcome,
)


@pytest.fixture
def normalizer() -> PersonNameNormalizationService:
    """Return the dependency-free canonical name normalizer."""
    return PersonNameNormalizationService()


@pytest.mark.parametrize(
    "dash",
    ["-", "\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2015", "\u2043", "\u2212", "\ufe58", "\ufe63", "\uff0d"],
)
def test_normalize_text_converts_dash_like_joiners(
    normalizer: PersonNameNormalizationService,
    dash: str,
) -> None:
    result = normalizer.normalize_text(f"anne {dash} marie smith")

    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None
    assert result.canonical_name.text == "Anne-Marie Smith"
    assert result.canonical_name.normalized.given_name == "Anne-Marie"


@pytest.mark.parametrize(
    "apostrophe",
    [
        "'",
        "\u2018",
        "\u2019",
        "\u201a",
        "\u201b",
        "\u2032",
        "\u2035",
        "\u02bb",
        "\u02bc",
        "\u02b9",
        "\ua78b",
        "\ua78c",
        "\uff07",
        "`",
        "\uff40",
        "\u00b4",
    ],
)
def test_normalize_text_converts_apostrophe_like_joiners(
    normalizer: PersonNameNormalizationService,
    apostrophe: str,
) -> None:
    result = normalizer.normalize_text(f"sean o {apostrophe} connor")

    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None
    assert result.canonical_name.text == "Sean O'Connor"
    assert result.canonical_name.normalized.surname == "O'Connor"


def test_nfkc_apostrophe_expansion_preserves_its_letter(
    normalizer: PersonNameNormalizationService,
) -> None:
    result = normalizer.normalize_text("sean o\u0149eill")

    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None
    assert result.canonical_name.text == "Sean O'Neill"


def test_ascii_surface_fast_path_skips_unicode_scans(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_call(*_args: object, **_kwargs: object) -> str:
        raise AssertionError

    class UnexpectedUnicodeData:
        normalize = staticmethod(unexpected_call)

    monkeypatch.setattr(person_name_normalization, "unicodedata", UnexpectedUnicodeData)
    monkeypatch.setattr(PersonNameNormalizationService, "_normalize_joiner", staticmethod(unexpected_call))

    assert (
        PersonNameNormalizationService._normalize_surface(  # noqa: SLF001
            "  sean  o ` connor  ",
        )
        == "sean o'connor"
    )


@pytest.mark.parametrize("raw_name", ["John Smith", "JOHN SMITH", "jOhN mCdonald", "A Li", "CS Smith", "Ben Sherwood"])
def test_simple_two_token_fast_path_matches_full_pipeline(
    normalizer: PersonNameNormalizationService,
    monkeypatch: pytest.MonkeyPatch,
    raw_name: str,
) -> None:
    fast_result = normalizer._normalize_simple_two_token_text(raw_name)  # noqa: SLF001
    assert fast_result is not None

    monkeypatch.setattr(
        PersonNameNormalizationService,
        "_normalize_simple_two_token_text",
        lambda _self, _raw_name: None,
    )

    assert normalizer.normalize_text(raw_name) == fast_result


@pytest.mark.parametrize(
    "raw_name",
    ["John Jr", "John Phd", "Phd Smith", "PD Dr", "John University", "John AND", "John II"],
)
def test_simple_two_token_fast_path_abstains_on_policy_tokens(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
) -> None:
    assert normalizer._normalize_simple_two_token_text(raw_name) is None  # noqa: SLF001


def test_normalize_text_strips_stacked_title_and_credentials_at_boundaries(
    normalizer: PersonNameNormalizationService,
) -> None:
    result = normalizer.normalize_text("Dr. Steve Marsh Blando, Ph.D., M.S.")

    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None
    assert result.canonical_name.text == "Steve Marsh Blando"
    assert result.canonical_name.normalized.given_name == "Steve"
    assert result.canonical_name.normalized.middle_name == "Marsh"
    assert result.canonical_name.normalized.surname == "Blando"
    assert [(token.text, token.source_role, token.reason) for token in result.dropped_tokens] == [
        ("Dr.", "given", DropReason.TITLE),
        ("Ph.D.", "suffix", DropReason.CREDENTIAL),
        ("M.S.", "suffix", DropReason.CREDENTIAL),
    ]


@pytest.mark.parametrize(
    ("first_name", "middle_name", "last_name", "expected", "expected_first", "expected_last"),
    [
        ("Chaplain", "David", "Walden", "David Walden", "David", "Walden"),
        ("BSc", "", "Millie Chau", "Millie Chau", "Millie", "Chau"),
        ("Frau", "", "Schäfer-Graf", "Schäfer-Graf", "", "Schäfer-Graf"),
    ],
)
def test_structured_titles_and_credentials_repair_emptied_given_boundary(  # noqa: PLR0913
    normalizer: PersonNameNormalizationService,
    first_name: str,
    middle_name: str,
    last_name: str,
    expected: str,
    expected_first: str,
    expected_last: str,
) -> None:
    result = normalizer.normalize_components(
        first_name=first_name,
        middle_name=middle_name,
        last_name=last_name,
    )

    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None
    assert result.canonical_name.text == expected
    assert result.canonical_name.normalized.given_name == expected_first
    assert result.canonical_name.normalized.surname == expected_last


@pytest.mark.parametrize(
    ("last_name", "expected_first", "expected_middle", "expected_last"),
    [
        ("Mifta Rezki", "Mifta", "", "Rezki"),
        ("Rajashree Vishnoo Naik", "Rajashree", "Vishnoo", "Naik"),
        ("Njarasoa Charlette RANDRIAMALALA", "Njarasoa Charlette", "", "Randriamalala"),
        ("Sathya S.", "Sathya", "", "S."),
    ],
)
def test_last_only_complete_names_repair_structurally_empty_given_boundary(
    normalizer: PersonNameNormalizationService,
    last_name: str,
    expected_first: str,
    expected_middle: str,
    expected_last: str,
) -> None:
    result = normalizer.normalize_components(last_name=last_name)

    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None
    assert result.canonical_name.normalized.given_name == expected_first
    assert result.canonical_name.normalized.middle_name == expected_middle
    assert result.canonical_name.normalized.surname == expected_last


@pytest.mark.parametrize("marker", ["†", "‡", "*", "§"])
def test_boundary_symbol_and_fused_initial_surname_are_repaired(
    normalizer: PersonNameNormalizationService,
    marker: str,
) -> None:
    result = normalizer.normalize_components(first_name=marker, last_name="W.Tsujita")

    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None
    assert result.canonical_name.text == "W. Tsujita"
    assert [(token.text, token.reason) for token in result.dropped_tokens] == [(marker, DropReason.CONNECTOR)]


def test_last_only_particle_surname_is_not_split(
    normalizer: PersonNameNormalizationService,
) -> None:
    result = normalizer.normalize_components(last_name="van der Waals")

    assert result.canonical_name is not None
    assert result.canonical_name.normalized.given_name == ""
    assert result.canonical_name.normalized.surname == "van der Waals"


def test_terminal_transliteration_apostrophe_is_preserved(
    normalizer: PersonNameNormalizationService,
) -> None:
    result = normalizer.normalize_components(first_name="P", middle_name="V", last_name="Bigar'")

    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None
    assert result.canonical_name.text == "P V Bigar'"


@pytest.mark.parametrize(
    ("first_name", "middle_name", "last_name", "expected"),
    [
        ("Edilene", "Natália", "Araújo das Graças", "Edilene Natália Araújo das Graças"),
        ("Frederico", "Ferreira de", "Oliveira", "Frederico Ferreira de Oliveira"),
        ("ROBERT", "C.", "McKEAN", "Robert C. McKean"),
        ("O.", "", "LöWENSTEIN", "O. Löwenstein"),
    ],
)
def test_source_particles_and_mixed_ocr_case_are_normalized(
    normalizer: PersonNameNormalizationService,
    first_name: str,
    middle_name: str,
    last_name: str,
    expected: str,
) -> None:
    result = normalizer.normalize_components(
        first_name=first_name,
        middle_name=middle_name,
        last_name=last_name,
    )

    assert result.canonical_name is not None
    assert result.canonical_name.text == expected


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("H.-J. Pompino", "H.-J. Pompino"),
        ("Safiye ŞAHİN", "Safiye Şahin"),
        ("Mihajlo (Michael) B Jakovljevic", "Mihajlo (Michael) B Jakovljevic"),
    ],
)
def test_compound_initial_unicode_case_and_parenthetical_name_are_preserved(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
    expected: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None
    assert result.canonical_name.text == expected


def test_normalize_text_extracts_true_suffix(normalizer: PersonNameNormalizationService) -> None:
    result = normalizer.normalize_text("Steve Blando IV")

    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None
    assert result.canonical_name.text == "Steve Blando IV"
    assert result.canonical_name.normalized.given_name == "Steve"
    assert result.canonical_name.normalized.middle_name == ""
    assert result.canonical_name.normalized.surname == "Blando"
    assert result.canonical_name.normalized.suffix == "IV"


def test_comma_suffix_does_not_trigger_family_first_parsing(
    normalizer: PersonNameNormalizationService,
) -> None:
    result = normalizer.normalize_text("Thomas L. Duvall, Jr.")

    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None
    assert result.canonical_name.text == "Thomas L. Duvall Jr."
    assert result.canonical_name.normalized.given_name == "Thomas"
    assert result.canonical_name.normalized.middle_name == "L."
    assert result.canonical_name.normalized.surname == "Duvall"
    assert result.canonical_name.normalized.suffix == "Jr."


def test_mixed_case_roman_looking_surname_is_not_a_suffix(
    normalizer: PersonNameNormalizationService,
) -> None:
    result = normalizer.normalize_text("Hiroshi Ii")

    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None
    assert result.canonical_name.text == "Hiroshi Ii"
    assert result.canonical_name.normalized.surname == "Ii"
    assert result.canonical_name.normalized.suffix == ""


def test_explicit_comma_parses_family_first_and_preserves_family_particles(
    normalizer: PersonNameNormalizationService,
) -> None:
    result = normalizer.normalize_text("de la cruz, maría elena")

    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None
    assert result.canonical_name.text == "María Elena de la Cruz"
    assert result.canonical_name.normalized.given_name == "María"
    assert result.canonical_name.normalized.middle_name == "Elena"
    assert result.canonical_name.normalized.surname == "de la Cruz"
    assert result.canonical_name.normalized.order == ("given", "middle", "surname", "surname", "surname")
    assert result.canonical_name.source.order == ("surname", "surname", "surname", "given", "middle")


def test_trailing_family_particles_are_kept_with_surname(
    normalizer: PersonNameNormalizationService,
) -> None:
    result = normalizer.normalize_text("ludwig van der waals")

    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None
    assert result.canonical_name.text == "Ludwig van der Waals"
    assert result.canonical_name.normalized.middle_name == ""
    assert result.canonical_name.normalized.surname == "van der Waals"


def test_structured_components_reinfer_roles_after_boundary_tokens_are_dropped(
    normalizer: PersonNameNormalizationService,
) -> None:
    result = normalizer.normalize_components(
        first_name="dr steve",
        middle_name="marsh",
        last_name="phd",
    )

    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None
    assert result.canonical_name.text == "Steve Marsh"
    assert result.canonical_name.source.given_name == "dr steve"
    assert result.canonical_name.source.middle_name == "marsh"
    assert result.canonical_name.source.surname == "phd"
    assert result.canonical_name.normalized.given_name == "Steve"
    assert result.canonical_name.normalized.middle_name == ""
    assert result.canonical_name.normalized.surname == "Marsh"
    assert result.canonical_name.normalized.suffix == ""
    assert [(token.text, token.source_role, token.reason) for token in result.dropped_tokens] == [
        ("dr", "given", DropReason.TITLE),
        ("phd", "surname", DropReason.CREDENTIAL),
    ]


def test_particle_only_surname_expansion_cannot_cross_explicit_middle(
    normalizer: PersonNameNormalizationService,
) -> None:
    result = normalizer.normalize_components(first_name="An", middle_name="Van", last_name="Nguyen")

    assert result.canonical_name is not None
    assert result.canonical_name.normalized.given_name == "An"
    assert result.canonical_name.normalized.middle_name == "Van"
    assert result.canonical_name.normalized.surname == "Nguyen"


@pytest.mark.parametrize(
    ("first_name", "middle_name", "last_name"),
    [
        ("Byung", "Chan", "Lee"),
        ("Ricardo", "Burciaga", "Castañeda"),
        ("Doménica", "Alejandra", "Delgado González"),
    ],
)
def test_structured_source_roles_are_preserved_without_mechanical_corruption(
    normalizer: PersonNameNormalizationService,
    first_name: str,
    middle_name: str,
    last_name: str,
) -> None:
    result = normalizer.normalize_components(
        first_name=first_name,
        middle_name=middle_name,
        last_name=last_name,
    )

    assert result.canonical_name is not None
    assert result.canonical_name.normalized.given_name == first_name
    assert result.canonical_name.normalized.middle_name == middle_name
    assert result.canonical_name.normalized.surname == last_name


def test_ambiguous_degree_key_in_surname_case_is_preserved(
    normalizer: PersonNameNormalizationService,
) -> None:
    result = normalizer.normalize_text("Bao Do")

    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None
    assert result.canonical_name.text == "Bao Do"
    assert result.canonical_name.normalized.surname == "Do"
    assert result.dropped_tokens == ()


def test_uppercase_leading_credential_is_dropped(
    normalizer: PersonNameNormalizationService,
) -> None:
    result = normalizer.normalize_text("MD Jane Smith")

    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None
    assert result.canonical_name.text == "Jane Smith"
    assert [(token.text, token.reason) for token in result.dropped_tokens] == [
        ("MD", DropReason.CREDENTIAL),
    ]


@pytest.mark.parametrize(
    ("raw_name", "expected", "expected_given", "expected_surname"),
    [
        ("JOHN MA", "John Ma", "John", "Ma"),
        ("MA SMITH", "Ma Smith", "Ma", "Smith"),
        ("Smith, Ma", "Ma Smith", "Ma", "Smith"),
        ("SMITH, MA", "Ma Smith", "Ma", "Smith"),
    ],
)
def test_ambiguous_uppercase_credential_token_is_preserved_when_required_as_name(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
    expected: str,
    expected_given: str,
    expected_surname: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None
    assert result.canonical_name.text == expected
    assert result.canonical_name.normalized.given_name == expected_given
    assert result.canonical_name.normalized.surname == expected_surname
    assert result.dropped_tokens == ()


def test_ambiguous_uppercase_credential_is_still_dropped_from_complete_name(
    normalizer: PersonNameNormalizationService,
) -> None:
    results = [
        normalizer.normalize_text("JOHN SMITH MA"),
        normalizer.normalize_text("John Smith, MA"),
    ]

    for result in results:
        assert result.outcome is PersonNameOutcome.PERSON
        assert result.canonical_name is not None
        assert result.canonical_name.text == "John Smith"
        assert [(token.text, token.reason) for token in result.dropped_tokens] == [
            ("MA", DropReason.CREDENTIAL),
        ]


def test_meng_surname_is_preserved_while_meng_degree_is_dropped(
    normalizer: PersonNameNormalizationService,
) -> None:
    surname = normalizer.normalize_text("Wenhua Meng")
    credential = normalizer.normalize_text("John Smith MEng")

    assert surname.canonical_name is not None
    assert surname.canonical_name.text == "Wenhua Meng"
    assert surname.canonical_name.normalized.surname == "Meng"
    assert surname.dropped_tokens == ()
    assert credential.canonical_name is not None
    assert credential.canonical_name.text == "John Smith"
    assert [(token.text, token.reason) for token in credential.dropped_tokens] == [
        ("MEng", DropReason.CREDENTIAL),
    ]


@pytest.mark.parametrize("raw_name", ["Dr Smith", "Ms Smith"])
def test_short_unambiguous_titles_are_still_dropped(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.canonical_name is not None
    assert result.canonical_name.text == "Smith"
    assert [(token.text, token.reason) for token in result.dropped_tokens] == [
        (raw_name.split()[0], DropReason.TITLE),
    ]


def test_structured_explicit_suffix_supports_single_roman_numeral(
    normalizer: PersonNameNormalizationService,
) -> None:
    result = normalizer.normalize_components(first_name="john", last_name="smith", suffix="v")

    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None
    assert result.canonical_name.text == "John Smith V"
    assert result.canonical_name.normalized.suffix == "V"


@pytest.mark.parametrize(
    ("given_name", "roman_surname"),
    [("John", "Vi"), ("John", "Iv"), ("John", "Iii"), ("John", "Vii"), ("Malcolm", "X")],
)
def test_two_token_raw_roman_numeral_is_a_surname_but_explicit_is_a_suffix(
    normalizer: PersonNameNormalizationService,
    given_name: str,
    roman_surname: str,
) -> None:
    raw = normalizer.normalize_text(f"{given_name} {roman_surname}")
    structured = normalizer.normalize_components(first_name=given_name, suffix=roman_surname)

    assert raw.canonical_name is not None
    assert raw.canonical_name.text == f"{given_name} {roman_surname}"
    assert raw.canonical_name.normalized.surname == roman_surname
    assert raw.canonical_name.normalized.suffix == ""
    assert structured.canonical_name is not None
    assert structured.canonical_name.text == f"{given_name} {roman_surname.upper()}"
    assert structured.canonical_name.normalized.surname == ""
    assert structured.canonical_name.normalized.suffix == roman_surname.upper()


@pytest.mark.parametrize(
    ("raw_name", "expected", "expected_suffix"),
    [
        ("Steve Blando IV", "Steve Blando IV", "IV"),
        ("John Smith Vi", "John Smith VI", "VI"),
        ("Smith, John IV", "John Smith IV", "IV"),
    ],
)
def test_raw_roman_suffix_requires_complete_name_context(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
    expected: str,
    expected_suffix: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.canonical_name is not None
    assert result.canonical_name.text == expected
    assert result.canonical_name.normalized.suffix == expected_suffix


def test_structured_missing_surname_is_reinferred_from_one_split_component(
    normalizer: PersonNameNormalizationService,
) -> None:
    result = normalizer.normalize_components(first_name="Dr. Mary Ann O\u2019Neill", last_name="MS")

    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None
    assert result.canonical_name.text == "Mary Ann O'Neill"
    assert result.canonical_name.normalized.given_name == "Mary"
    assert result.canonical_name.normalized.middle_name == "Ann"
    assert result.canonical_name.normalized.surname == "O'Neill"


def test_affiliation_digits_are_dropped_without_losing_name_token(
    normalizer: PersonNameNormalizationService,
) -> None:
    result = normalizer.normalize_text("Anna Spießl1, 4")

    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None
    assert result.canonical_name.text == "Anna Spießl"
    assert [(token.text, token.source_role, token.reason) for token in result.dropped_tokens] == [
        ("1", "surname", DropReason.AFFILIATION),
        ("4", "suffix", DropReason.AFFILIATION),
    ]


@pytest.mark.parametrize(
    ("raw_name", "expected_outcome"),
    [
        ("", PersonNameOutcome.INVALID),
        ("---", PersonNameOutcome.INVALID),
        ("John Smith and Jane Doe", PersonNameOutcome.NON_PERSON),
        ("John Smith, Jane Doe", PersonNameOutcome.NON_PERSON),
        ("Stanford University", PersonNameOutcome.NON_PERSON),
    ],
)
def test_normalize_text_returns_typed_non_person_and_invalid_outcomes(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
    expected_outcome: PersonNameOutcome,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.outcome is expected_outcome
    assert result.canonical_name is None
    assert result.reason


def test_result_and_dropped_token_lineage_are_immutable(
    normalizer: PersonNameNormalizationService,
) -> None:
    result = normalizer.normalize_text("Mr. Steve Blando PhD")

    assert result.canonical_name is not None
    reason_attribute = "reason"
    text_attribute = "text"
    with pytest.raises(FrozenInstanceError):
        setattr(result, reason_attribute, "changed")
    with pytest.raises(FrozenInstanceError):
        setattr(result.dropped_tokens[0], text_attribute, "changed")


@pytest.mark.parametrize(
    "raw_name",
    [
        "Dr. Steve Marsh Blando, Ph.D.",
        "de la Cruz, María Elena",
        "Thomas L. Duvall, Jr.",
        "Anne-Marie O'Connor",
    ],
)
def test_component_orders_contain_exactly_one_role_per_token(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.canonical_name is not None
    for components in (result.canonical_name.source, result.canonical_name.normalized):
        token_count = sum(
            len(tokens)
            for tokens in (
                components.given_tokens,
                components.middle_tokens,
                components.surname_tokens,
                components.suffix_tokens,
            )
        )
        assert len(components.order) == token_count


def test_structured_non_person_and_invalid_results_have_no_canonical_name(
    normalizer: PersonNameNormalizationService,
) -> None:
    non_person = normalizer.normalize_components(first_name="Stanford", last_name="University")
    invalid = normalizer.normalize_components()

    assert non_person.outcome is PersonNameOutcome.NON_PERSON
    assert non_person.canonical_name is None
    assert invalid.outcome is PersonNameOutcome.INVALID
    assert invalid.canonical_name is None


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("Vincent El Ghouzzi", "Vincent El Ghouzzi"),
        ("Monica Da Costa", "Monica Da Costa"),
        ("P. T. d'Orbán", "P. T. d'Orbán"),
        ("de la cruz, maría elena", "María Elena de la Cruz"),
    ],
)
def test_family_particles_preserve_credible_source_case(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
    expected: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.canonical_name is not None
    assert result.canonical_name.text == expected


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("E V Usol'tseva", "E V Usol'tseva"),
        ("Wa\u2019el Tuqan", "Wa'el Tuqan"),
        ("sean o \u2019 connor", "Sean O'Connor"),
        ("Michael F. O''Rourke", "Michael F. O'Rourke"),
        ("Carol O\u2019sullivan", "Carol O'Sullivan"),
        ("Ruth D'arcy Hart", "Ruth D'Arcy Hart"),
    ],
)
def test_apostrophe_normalization_preserves_mixed_case_and_collapses_duplicates(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
    expected: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.canonical_name is not None
    assert result.canonical_name.text == expected


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("' J. LEECH", "J. Leech"),
        ("Ph. R. Hénon", "Ph. R. Hénon"),
        ("Zd. Servit", "Zd. Servit"),
        ("A. D\u2018A. BELLAIRS", "A. D'A. Bellairs"),
    ],
)
def test_stray_joiners_and_abbreviated_tokens_are_normalized(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
    expected: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.canonical_name is not None
    assert result.canonical_name.text == expected


def test_additional_boundary_credentials_and_credential_like_surnames(
    normalizer: PersonNameNormalizationService,
) -> None:
    mpa = normalizer.normalize_text("Amina Helmi MPA")
    spaced_md = normalizer.normalize_text("William A. Horwitz M. D.")
    surname = normalizer.normalize_text("E C Mba")

    assert mpa.canonical_name is not None
    assert mpa.canonical_name.text == "Amina Helmi"
    assert spaced_md.canonical_name is not None
    assert spaced_md.canonical_name.text == "William A. Horwitz"
    assert surname.canonical_name is not None
    assert surname.canonical_name.text == "E C Mba"


@pytest.mark.parametrize("credential", ["Ph. D.", "M. Sc."])
def test_spaced_credentials_are_dropped_in_raw_and_structured_suffix_forms(
    normalizer: PersonNameNormalizationService,
    credential: str,
) -> None:
    raw = normalizer.normalize_text(f"John Smith {credential}")
    structured = normalizer.normalize_components(first_name="John", last_name="Smith", suffix=credential)

    for result in (raw, structured):
        assert result.outcome is PersonNameOutcome.PERSON
        assert result.canonical_name is not None
        assert result.canonical_name.text == "John Smith"
        assert all(token.reason is DropReason.CREDENTIAL for token in result.dropped_tokens)


@pytest.mark.parametrize(
    ("raw_name", "expected_suffix"),
    [("McKinley Glover Iv", "IV"), ("N David Yanez Iii", "III"), ("Hiroshi Ii", "")],
)
def test_mixed_case_long_roman_suffixes_are_unambiguous(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
    expected_suffix: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.canonical_name is not None
    assert result.canonical_name.normalized.suffix == expected_suffix


def test_stacked_academic_titles_and_parenthetical_duplicate_are_removed(
    normalizer: PersonNameNormalizationService,
) -> None:
    titled = normalizer.normalize_text("Univ.-Prof. Dr. med. Prof. honoraire Dr. h.c. C. C. Zouboulis")
    parenthetical = normalizer.normalize_text("Alan (Alan B.) Cantor")

    assert titled.canonical_name is not None
    assert titled.canonical_name.text == "C. C. Zouboulis"
    assert parenthetical.canonical_name is not None
    assert parenthetical.canonical_name.text == "Alan B. Cantor"


def test_name_glued_to_affiliation_is_rejected_not_salvaged(
    normalizer: PersonNameNormalizationService,
) -> None:
    # Previously this salvaged "Rachel Webster" by stripping the trailing affiliation.
    # But the same salvage turned tens of thousands of pure organizations into garbage
    # person-fragments ("Max Planck Institute ..." -> "Max Planck", "Niels Bohr
    # Institute" -> "Niels Bohr"), so the whole contaminated string is now rejected.
    # The real researcher is still captured via their clean (affiliation-free) entries.
    result = normalizer.normalize_text("Rachel Webster University of New South Wales")

    assert result.outcome is PersonNameOutcome.NON_PERSON
    assert result.canonical_name is None


def test_two_token_particle_like_given_name_is_not_duplicated(
    normalizer: PersonNameNormalizationService,
) -> None:
    result = normalizer.normalize_text("Ben Sherwood")

    assert result.canonical_name is not None
    assert result.canonical_name.text == "Ben Sherwood"
    assert result.canonical_name.normalized.given_name == "Ben"
    assert result.canonical_name.normalized.surname == "Sherwood"


def test_uppercase_initial_clusters_remain_uppercase(
    normalizer: PersonNameNormalizationService,
) -> None:
    result = normalizer.normalize_text("Gordon CS Smith")

    assert result.canonical_name is not None
    assert result.canonical_name.text == "Gordon CS Smith"


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("N.p Sunil-Chandra", "N.P Sunil-Chandra"),
        ("G.Y Minuk", "G.Y Minuk"),
        ("K.D.D.I Kodithuwakku", "K.D.D.I Kodithuwakku"),
        ("Alekseeva M.Yu. Alekseeva", "Alekseeva M.Yu. Alekseeva"),
        ("P.Sh. Ibragimov", "P.Sh. Ibragimov"),
        ("MM. Cunningham", "MM. Cunningham"),
    ],
)
def test_period_policy_distinguishes_initial_clusters_from_transliteration_abbreviations(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
    expected: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.canonical_name is not None
    assert result.canonical_name.text == expected


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("Dr.Wenjun Zhang", "Wenjun Zhang"),
        ("Mrs.E. Sumathi", "E. Sumathi"),
        ("Dr.AARCHA S S", "Aarcha S S"),
        ("PD Dr. M. Mengel", "M. Mengel"),
    ],
)
def test_attached_and_stacked_titles_are_removed_without_dropping_the_name(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
    expected: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.canonical_name is not None
    assert result.canonical_name.text == expected
    assert all(token.reason is DropReason.TITLE for token in result.dropped_tokens)
    expected_first_key = "".join(character.casefold() for character in expected.split()[0] if character.isalnum())
    source_first_count = sum(
        "".join(character.casefold() for character in token if character.isalnum()) == expected_first_key
        for token in result.canonical_name.source.given_tokens
    )
    assert source_first_count == 1


def test_period_context_preserves_md_given_abbreviation_and_ms_initials(
    normalizer: PersonNameNormalizationService,
) -> None:
    md = normalizer.normalize_text("Md. Abu Bakar Siddiq")
    initials = normalizer.normalize_text("M.S. Crouse")
    credential = normalizer.normalize_text("Jaume Bosch M.D.")

    assert md.canonical_name is not None
    assert md.canonical_name.text == "Md. Abu Bakar Siddiq"
    assert initials.canonical_name is not None
    assert initials.canonical_name.text == "M.S. Crouse"
    assert credential.canonical_name is not None
    assert credential.canonical_name.text == "Jaume Bosch"


def test_terminal_and_standalone_sentence_periods_are_removed(
    normalizer: PersonNameNormalizationService,
) -> None:
    terminal = normalizer.normalize_text("Subramanian. A")
    standalone = normalizer.normalize_text("A.Kalaikannan .")

    assert terminal.canonical_name is not None
    assert terminal.canonical_name.text == "Subramanian A"
    assert standalone.canonical_name is not None
    assert standalone.canonical_name.text == "A.Kalaikannan"


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("DAVID NG", "David Ng"),
        ("JUAN DE LA CRUZ", "Juan de la Cruz"),
    ],
)
def test_uppercase_two_letter_surnames_and_particles_use_name_case(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
    expected: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.canonical_name is not None
    assert result.canonical_name.text == expected


@pytest.mark.parametrize(
    ("raw_name", "expected_middle", "expected_surname"),
    [
        ("Juan J Llibre Rodriguez", "J", "Llibre Rodriguez"),
        ("Carlos A. Henríquez Q.", "A.", "Henríquez Q."),
        ("Landys A. Lopez Quezada", "A.", "Lopez Quezada"),
    ],
)
def test_initial_followed_by_full_token_preserves_two_token_family_name(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
    expected_middle: str,
    expected_surname: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.canonical_name is not None
    assert result.canonical_name.normalized.middle_name == expected_middle
    assert result.canonical_name.normalized.surname == expected_surname


def test_ordinary_three_token_name_keeps_middle_name(
    normalizer: PersonNameNormalizationService,
) -> None:
    result = normalizer.normalize_text("John Michael Smith")

    assert result.canonical_name is not None
    assert result.canonical_name.normalized.middle_name == "Michael"
    assert result.canonical_name.normalized.surname == "Smith"


def test_structured_middle_drops_only_surname_copy_with_lowercase_marker(
    normalizer: PersonNameNormalizationService,
) -> None:
    result = normalizer.normalize_components(
        first_name="Mery",
        middle_name="Luz Rojas Aire z",
        last_name="Rojas Aire",
    )

    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None
    assert result.canonical_name.text == "Mery Luz Rojas Aire"
    assert result.canonical_name.source.middle_name == "Luz Rojas Aire z"
    assert result.canonical_name.normalized.given_name == "Mery"
    assert result.canonical_name.normalized.middle_name == "Luz"
    assert result.canonical_name.normalized.surname == "Rojas Aire"
    assert [(token.text, token.source_role, token.reason) for token in result.dropped_tokens] == [
        ("Rojas", "middle", DropReason.DUPLICATE),
        ("Aire", "middle", DropReason.DUPLICATE),
        ("z", "middle", DropReason.CONNECTOR),
    ]


@pytest.mark.parametrize(
    ("middle_name", "last_name", "expected_middle"),
    [
        ("Smith", "Smith", "Smith"),
        ("Luz Rojas Aire", "Rojas Aire", "Luz Rojas Aire"),
        ("Luz Rojas Aire Z", "Rojas Aire", "Luz Rojas Aire Z"),
        ("Luz Rojas Air z", "Rojas Aire", "Luz Rojas Air Z"),
    ],
)
def test_structured_middle_does_not_broadly_deduplicate_surnames(
    normalizer: PersonNameNormalizationService,
    middle_name: str,
    last_name: str,
    expected_middle: str,
) -> None:
    result = normalizer.normalize_components(
        first_name="Mery",
        middle_name=middle_name,
        last_name=last_name,
    )

    assert result.canonical_name is not None
    assert result.canonical_name.normalized.middle_name == expected_middle
    assert result.canonical_name.normalized.surname == last_name


def test_exact_leading_ma_abbreviation_is_preserved_before_initial_and_surname(
    normalizer: PersonNameNormalizationService,
) -> None:
    results = [
        normalizer.normalize_text("Ma. E. Zayas"),
        normalizer.normalize_components(first_name="Ma.", middle_name="E.", last_name="Zayas"),
    ]

    for result in results:
        assert result.outcome is PersonNameOutcome.PERSON
        assert result.canonical_name is not None
        assert result.canonical_name.text == "Ma. E. Zayas"
        assert result.canonical_name.normalized.given_name == "Ma."
        assert result.canonical_name.normalized.middle_name == "E."
        assert result.canonical_name.normalized.surname == "Zayas"
        assert result.dropped_tokens == ()


@pytest.mark.parametrize("raw_name", ["M.A. E. Zayas", "Jane Smith M.A."])
def test_ma_credential_forms_are_still_dropped(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.canonical_name is not None
    assert "M.A." not in result.canonical_name.text
    assert [(token.text, token.reason) for token in result.dropped_tokens] == [
        ("M.A.", DropReason.CREDENTIAL),
    ]


def test_lowercase_marker_fused_to_leading_initial_is_split_with_lineage(
    normalizer: PersonNameNormalizationService,
) -> None:
    results = [
        normalizer.normalize_text("tE. L. Winkelman"),
        normalizer.normalize_components(first_name="tE.", middle_name="L.", last_name="Winkelman"),
    ]

    for result in results:
        assert result.outcome is PersonNameOutcome.PERSON
        assert result.canonical_name is not None
        assert result.canonical_name.text == "E. L. Winkelman"
        assert result.canonical_name.source.given_name == "tE."
        assert result.canonical_name.normalized.given_name == "E."
        assert result.canonical_name.normalized.middle_name == "L."
        assert result.canonical_name.normalized.surname == "Winkelman"
        assert [(token.text, token.source_role, token.reason) for token in result.dropped_tokens] == [
            ("t", "given", DropReason.CONNECTOR),
        ]


@pytest.mark.parametrize("raw_name", ["tE. Louis Winkelman", "eBay L. Smith", "JoAnn L. Smith"])
def test_leading_mixed_case_names_are_not_generically_split(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.canonical_name is not None
    assert result.canonical_name.text == raw_name
    assert result.dropped_tokens == ()


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("Dr Li", "Li"),
        ("Doctor Li", "Li"),
        ("Pastor John Smith", "John Smith"),
        ("Frau Ng", "Ng"),
        ("Rabbi Ng", "Ng"),
        ("Lord Ng", "Ng"),
        ("Pastor Ng", "Ng"),
    ],
)
def test_unambiguous_title_contexts_are_still_dropped(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
    expected: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.canonical_name is not None
    assert result.canonical_name.text == expected
    assert len(result.dropped_tokens) == 1
    assert result.dropped_tokens[0].reason is DropReason.TITLE


def test_separated_compound_initials_join_only_at_visible_initial_boundary(
    normalizer: PersonNameNormalizationService,
) -> None:
    raw = normalizer.normalize_text("H. -J. Schneider")
    structured = normalizer.normalize_components(first_name="H.", middle_name="-J.", last_name="Schneider")

    for result in (raw, structured):
        assert result.outcome is PersonNameOutcome.PERSON
        assert result.canonical_name is not None
        assert result.canonical_name.text == "H.-J. Schneider"
        assert result.canonical_name.normalized.given_name == "H.-J."
        assert result.canonical_name.normalized.middle_name == ""
        assert result.canonical_name.normalized.surname == "Schneider"

    assert structured.canonical_name is not None
    assert structured.canonical_name.source.given_name == "H."
    assert structured.canonical_name.source.middle_name == "-J."


def test_compound_initial_join_does_not_change_normal_initial_order_or_words(
    normalizer: PersonNameNormalizationService,
) -> None:
    normal = normalizer.normalize_text("H. J. Schneider")
    word = normalizer.normalize_components(first_name="H.", middle_name="-Jean", last_name="Schneider")

    assert normal.canonical_name is not None
    assert normal.canonical_name.text == "H. J. Schneider"
    assert normal.canonical_name.normalized.middle_name == "J."
    assert word.outcome is PersonNameOutcome.INVALID


def test_leading_superscript_affiliation_marker_is_removed_with_lineage(
    normalizer: PersonNameNormalizationService,
) -> None:
    raw = normalizer.normalize_text("\u00b9Matias Julyus")
    structured = normalizer.normalize_components(first_name="\u00b9Matias", last_name="Julyus")

    for result in (raw, structured):
        assert result.outcome is PersonNameOutcome.PERSON
        assert result.canonical_name is not None
        assert result.canonical_name.text == "Matias Julyus"
        assert result.canonical_name.source.given_name == "\u00b9Matias"
        assert result.canonical_name.normalized.given_name == "Matias"
        assert result.canonical_name.normalized.surname == "Julyus"
        assert [(token.text, token.source_role, token.reason) for token in result.dropped_tokens] == [
            ("\u00b9", "given", DropReason.AFFILIATION),
        ]


@pytest.mark.parametrize("raw_name", ["1Matias Julyus", "Ma\u00b9tias Julyus"])
def test_ordinary_digits_and_internal_superscripts_are_not_affiliation_prefixes(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.outcome is PersonNameOutcome.INVALID


@pytest.mark.parametrize(
    ("first_name", "middle_name", "last_name", "expected"),
    [
        ("\u00c9va", "d. H.", "Alm\u00e1si", "\u00c9va d. H. Alm\u00e1si"),
        ("L.", "M. b.", "Hoskins", "L. M. b. Hoskins"),
        ("F.", "E. Soares e", "Silva", "F. E. Soares e Silva"),
        ("Yuldosheva", "Zulfizar Sobir", "kizi", "Yuldosheva Zulfizar Sobir kizi"),
        ("Aysel", "Mammad", "qizi", "Aysel Mammad qizi"),
    ],
)
def test_lowercase_relational_tokens_preserve_source_case_by_role(
    normalizer: PersonNameNormalizationService,
    first_name: str,
    middle_name: str,
    last_name: str,
    expected: str,
) -> None:
    result = normalizer.normalize_components(
        first_name=first_name,
        middle_name=middle_name,
        last_name=last_name,
    )

    assert result.canonical_name is not None
    assert result.canonical_name.text == expected


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("d. Smith", "D. Smith"),
        ("kizi Smith", "Kizi Smith"),
        ("McDONALD Smith", "McDonald Smith"),
        ("McDonald Smith", "McDonald Smith"),
        ("FitzGerald Smith", "FitzGerald Smith"),
    ],
)
def test_relational_case_rule_does_not_preserve_given_or_ocr_casing(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
    expected: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.canonical_name is not None
    assert result.canonical_name.text == expected


def test_known_alphabetic_name_abbreviation_preserves_meaningful_period(
    normalizer: PersonNameNormalizationService,
) -> None:
    raw = normalizer.normalize_text("Most. Sumaiya Khatun Kali")
    structured = normalizer.normalize_components(first_name="Most.", middle_name="Sumaiya Khatun", last_name="Kali")

    for result in (raw, structured):
        assert result.canonical_name is not None
        assert result.canonical_name.text == "Most. Sumaiya Khatun Kali"
        assert result.canonical_name.normalized.given_name == "Most."


@pytest.mark.parametrize(("raw_name", "expected"), [("Mark. Smith", "Mark Smith"), ("Subramanian. A", "Subramanian A")])
def test_sentence_like_periods_are_not_preserved_as_abbreviations(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
    expected: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.canonical_name is not None
    assert result.canonical_name.text == expected


def test_packed_surname_first_with_two_trailing_initials_is_reordered(
    normalizer: PersonNameNormalizationService,
) -> None:
    raw = normalizer.normalize_text("Masterov R. A.")
    structured = normalizer.normalize_components(last_name="Masterov R. A.")

    for result in (raw, structured):
        assert result.canonical_name is not None
        assert result.canonical_name.text == "R. A. Masterov"
        assert result.canonical_name.normalized.given_name == "R."
        assert result.canonical_name.normalized.middle_name == "A."
        assert result.canonical_name.normalized.surname == "Masterov"

    assert raw.canonical_name is not None
    assert raw.canonical_name.source.order == ("surname", "given", "middle")


def test_packed_surname_first_gate_preserves_normal_order_and_source_last_floor(
    normalizer: PersonNameNormalizationService,
) -> None:
    normal = normalizer.normalize_text("R. A. Masterov")
    one_initial = normalizer.normalize_text("Masterov R.")
    source_last = normalizer.normalize_components(first_name="John", last_name="Masterov R. A.")

    assert normal.canonical_name is not None
    assert normal.canonical_name.text == "R. A. Masterov"
    assert normal.canonical_name.normalized.surname == "Masterov"
    assert one_initial.canonical_name is not None
    assert one_initial.canonical_name.text == "Masterov R."
    assert one_initial.canonical_name.normalized.given_name == "Masterov"
    assert source_last.canonical_name is not None
    assert source_last.canonical_name.normalized.given_name == "John"
    assert source_last.canonical_name.normalized.surname == "Masterov R. A."


@pytest.mark.parametrize(
    ("raw_name", "expected_first", "expected_middle", "expected_last"),
    [
        ("M. Calder\u00f3n de la Barca S\u00e1nchez", "M.", "", "Calder\u00f3n de la Barca S\u00e1nchez"),
        ("Roberto Jos\u00e9 Carvalho da Silva", "Roberto", "Jos\u00e9", "Carvalho da Silva"),
        ("Carvalho da Silva Roberto Jos\u00e9", "Roberto", "Jos\u00e9", "Carvalho da Silva"),
    ],
)
def test_strong_particle_spans_expand_compound_surname_floor(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
    expected_first: str,
    expected_middle: str,
    expected_last: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.canonical_name is not None
    assert result.canonical_name.normalized.given_name == expected_first
    assert result.canonical_name.normalized.middle_name == expected_middle
    assert result.canonical_name.normalized.surname == expected_last


def test_structured_strong_particle_repairs_cover_floor_and_mechanical_shift(
    normalizer: PersonNameNormalizationService,
) -> None:
    calderon = normalizer.normalize_components(
        first_name="M.",
        middle_name="Calder\u00f3n de la Barca",
        last_name="S\u00e1nchez",
    )
    carvalho = normalizer.normalize_components(
        first_name="Carvalho",
        middle_name="da Silva Roberto",
        last_name="Jos\u00e9",
    )

    assert calderon.canonical_name is not None
    assert calderon.canonical_name.text == "M. Calder\u00f3n de la Barca S\u00e1nchez"
    assert calderon.canonical_name.normalized.middle_name == ""
    assert calderon.canonical_name.normalized.surname == "Calder\u00f3n de la Barca S\u00e1nchez"
    assert carvalho.canonical_name is not None
    assert carvalho.canonical_name.text == "Roberto Jos\u00e9 Carvalho da Silva"
    assert carvalho.canonical_name.normalized.given_name == "Roberto"
    assert carvalho.canonical_name.normalized.middle_name == "Jos\u00e9"
    assert carvalho.canonical_name.normalized.surname == "Carvalho da Silva"


def test_particle_floor_does_not_override_weak_context_or_source_last(
    normalizer: PersonNameNormalizationService,
) -> None:
    weak = normalizer.normalize_components(first_name="John", middle_name="Michael de la", last_name="Smith")
    incomplete_shift = normalizer.normalize_components(first_name="Carvalho", middle_name="da Silva", last_name="Jos\u00e9")
    title_case_surname = normalizer.normalize_text("H. K. Das Gupta")

    assert weak.canonical_name is not None
    assert weak.canonical_name.normalized.given_name == "John"
    assert weak.canonical_name.normalized.middle_name == "Michael de la"
    assert weak.canonical_name.normalized.surname == "Smith"
    assert incomplete_shift.canonical_name is not None
    assert incomplete_shift.canonical_name.normalized.given_name == "Carvalho"
    assert incomplete_shift.canonical_name.normalized.middle_name == "da Silva"
    assert incomplete_shift.canonical_name.normalized.surname == "Jos\u00e9"
    assert title_case_surname.canonical_name is not None
    assert title_case_surname.canonical_name.normalized.middle_name == "K."
    assert title_case_surname.canonical_name.normalized.surname == "Das Gupta"


@pytest.mark.parametrize(
    ("raw_name", "expected_text"),
    [
        # A dangling author-list connector "and" (leading/trailing, from a truncated
        # author list) is stripped, recovering the one real person's name. Real DB rows.
        ("Alexander Campbell and", "Alexander Campbell"),
        ("Benjamin R. Knoll and", "Benjamin R. Knoll"),
        ("Sven Björkman and", "Sven Björkman"),
        ("and Ariel Feldman", "Ariel Feldman"),
        ("and Jürg Hutter", "Jürg Hutter"),
        ("W. L. HAFLEY AND", "W. L. Hafley"),  # all-caps AND connector
        ("Susana Patricia Cabrera Huerta, and", "Susana Patricia Cabrera Huerta"),  # comma + and
        # A real surname "And" (title-case) or a hyphenated "-And" is NOT a connector.
        ("Metin And", "Metin And"),
        ("Lars And", "Lars And"),
        ("K. Jon-And", "K. Jon-And"),
    ],
)
def test_dangling_and_connector_stripped_but_real_and_surname_kept(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
    expected_text: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None
    assert result.canonical_name.text == expected_text


@pytest.mark.parametrize(
    "raw_name",
    [
        "Smith and Jones",  # two people joined by a mid connector
        "Anna Smith and Bob Jones",
        "Computational and Mathematical Methods in Medicine",  # journal with mid "and"
    ],
)
def test_mid_and_connector_between_names_is_non_person(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.outcome is PersonNameOutcome.NON_PERSON
    assert result.canonical_name is None


@pytest.mark.parametrize(
    ("raw_name", "expected_text", "expected_surname"),
    [
        # Dutch "ten", Urdu "ur"/"ud" are family particles: kept lowercase and grouped
        # into the surname (previously "ten"/"ur" were Title-cased and mis-split).
        ("Henk ten Have", "Henk ten Have", ("ten", "Have")),
        ("Peter ten Dijke", "Peter ten Dijke", ("ten", "Dijke")),
        ("Zia ur Rehman", "Zia ur Rehman", ("ur", "Rehman")),
        ("Ali ud Din", "Ali ud Din", ("ud", "Din")),
        # Control: an established particle name is unchanged.
        ("Jan van der Berg", "Jan van der Berg", ("van", "der", "Berg")),
    ],
)
def test_dutch_urdu_particles_lowercased_and_grouped(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
    expected_text: str,
    expected_surname: tuple[str, ...],
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None
    assert result.canonical_name.text == expected_text
    assert result.canonical_name.normalized.surname_tokens == expected_surname


@pytest.mark.parametrize(
    "raw_name",
    ["K. Dem'yankov", "Ol'ga V. Dem'yanova", "Yu. K. Dem'yanovich"],
)
def test_apostrophe_surname_not_lowercased_as_particle(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
) -> None:
    # Regression guard: a surname starting "Dem'" must NOT be treated as the
    # German particle "dem" (which is why "dem" is intentionally NOT in the set).
    result = normalizer.normalize_text(raw_name)

    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None
    assert "dem" not in result.canonical_name.text.split()
    assert result.canonical_name.text[0] == raw_name[0]  # leading capital preserved


@pytest.mark.parametrize(
    ("raw_name", "expected_given", "expected_surname"),
    [
        # "Van"/"Ten" are common real given names (Van Morrison, Van Jones). Adding
        # "ten" to the particle set must NOT swallow a Title-cased LEADING one:
        # casing + position disambiguate — Title-case leading token = given name.
        ("Van Jones", "Van", ("Jones",)),
        ("Van Morrison", "Van", ("Morrison",)),
        ("Van A. Smith", "Van", ("Smith",)),
        ("Ten Berge", "Ten", ("Berge",)),
    ],
)
def test_titlecase_leading_van_ten_is_given_not_particle(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
    expected_given: str,
    expected_surname: tuple[str, ...],
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None
    assert result.canonical_name.text == raw_name
    assert result.canonical_name.normalized.given_name == expected_given
    assert result.canonical_name.normalized.surname_tokens == expected_surname


@pytest.mark.parametrize(
    ("raw_name", "expected_surname"),
    [
        # A Title-cased "Van" in the MIDDLE is a real Dutch-American compound surname
        # (Van Dyke, Van Winkle) — grouped whole into the surname, casing preserved.
        ("John Van Dyke", ("Van", "Dyke")),
        ("Rob Van Winkle", ("Van", "Winkle")),
    ],
)
def test_titlecase_medial_van_is_compound_surname(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
    expected_surname: tuple[str, ...],
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None
    assert result.canonical_name.text == raw_name
    assert result.canonical_name.normalized.surname_tokens == expected_surname


@pytest.mark.parametrize(
    ("raw_name", "expected_text"),
    [
        # HTML entities were previously left as literal tokens, so every one of these
        # real people was REJECTED (canonical_name=None). Decoding recovers them.
        ("Martin G&#x00F6;tz", "Martin Götz"),          # hex numeric char ref -> ö
        ("Benjamin I&#x00F1;iguez", "Benjamin Iñiguez"),
        ("G. Kope&#263;", "G. Kopeć"),                    # decimal numeric char ref -> ć
        ("A. Doboszy&#324;ska", "A. Doboszyńska"),
        ("Mitch D&#39;Arcy", "Mitch D'Arcy"),            # named-ish decimal ref -> apostrophe
        ("Fr&#x00E9;d&#x00E9;ric Sirois", "Frédéric Sirois"),  # multiple entities in one name
        ("Jos&#x00E9; Rodr&#x00ED;guez", "José Rodríguez"),
        ("Jos&eacute; Silva", "José Silva"),             # named entity -> é
        ("Andr&eacute; M&uuml;ller", "André Müller"),    # two named entities
    ],
)
def test_html_entities_decoded_and_name_recovered(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
    expected_text: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None
    assert result.canonical_name.text == expected_text
    assert "&" not in result.canonical_name.text  # no entity remnants


@pytest.mark.parametrize(
    "raw_name",
    [
        # "&amp;" decodes to "&", which is a name separator: an org or a two-person
        # string must still be rejected, never accepted as a single person.
        "Texas A &amp; M",
        "John &amp; Jane Smith",
        "Smith &amp; Wesson",
    ],
)
def test_amp_entity_decodes_to_separator_and_rejects(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.outcome is PersonNameOutcome.NON_PERSON
    assert result.canonical_name is None


def test_html_entity_decoded_in_long_multi_token_string(
    normalizer: PersonNameNormalizationService,
) -> None:
    # A long author-list blob with an embedded entity: decoding must still apply
    # (no "&#..." remnant, "Agnès" recovered). Whether such a blob is ultimately a
    # person is a separate over-acceptance concern, orthogonal to entity decoding.
    raw = (
        "Gilles Julien Diego Anne Agn&#xe8;s Jacques Anne Florent  "
        "Blancho Branchereau Cantarovich Cesbron Chapelet D"
    )
    result = normalizer.normalize_text(raw)

    assert result.canonical_name is not None
    assert "&#" not in result.canonical_name.text
    assert "Agnès" in result.canonical_name.text


def test_name_without_entity_is_unaffected_by_decode(
    normalizer: PersonNameNormalizationService,
) -> None:
    # Regression guard: names with no "&" skip the decode path entirely.
    result = normalizer.normalize_text("John Smith")

    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None
    assert result.canonical_name.text == "John Smith"


@pytest.mark.parametrize(
    ("raw_name", "expected_surname", "lead_kept"),
    [
        # Leading given/initials that collide with a credential/title acronym were
        # dropped, collapsing the name (surname left EMPTY: "MS Islam" -> given "Islam").
        # Now the leading token is kept and the surname is restored.
        ("MS Islam", ("Islam",), "Ms"),          # all-caps initials, not the "Ms" honorific
        ("MS Ali", ("Ali",), "Ms"),
        ("M-S. Barisits", ("Barisits",), "M-S."),
        ("M-S Barisits", ("Barisits",), "M-S"),
        ("Edd Gent", ("Gent",), "Edd"),           # "Edd" given, not the "EdD" degree
        ("Edd A. Blekkan", ("Blekkan",), "Edd"),
        ("Ma. Lucila Lapar", ("Lapar",), "Ma."),  # "Ma." = María, not the "MA" degree
        ("Ma. Mercedes T. Rodrigo", ("Rodrigo",), "Ma."),
        ("Ma. Elena Medina-Mora", ("Medina-Mora",), "Ma."),
    ],
)
def test_leading_name_token_not_dropped_as_credential(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
    expected_surname: tuple[str, ...],
    lead_kept: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None
    assert result.canonical_name.normalized.surname_tokens == expected_surname
    assert result.canonical_name.text.split()[0] == lead_kept  # leading token preserved


@pytest.mark.parametrize(
    ("raw_name", "expected_text"),
    [
        # Genuine honorifics (Title-case, with or without a trailing dot) must STILL
        # be dropped — casing is the signal: all-caps "MS" = initials, "Ms"/"Ms." = title.
        ("Ms Smith", "Smith"),
        ("Ms. Islam", "Islam"),
        ("Mr Jones", "Jones"),
        ("Dr Watson", "Watson"),
    ],
)
def test_titlecase_honorific_still_dropped(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
    expected_text: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.canonical_name is not None
    assert result.canonical_name.text == expected_text


@pytest.mark.parametrize(
    ("raw_name", "expected_text", "dropped_token"),
    [
        # Trailing credentials (and the dotted "M.A." degree convention) still drop.
        ("John Smith PhD", "John Smith", "PhD"),
        ("Robert Jones MD", "Robert Jones", "MD"),
        ("Jane Doe EdD", "Jane Doe", "EdD"),      # mixed-case degree dropped...
        ("Jane Doe EDD", "Jane Doe", "EDD"),      # ...and its all-caps form
        ("M.A. E. Zayas", "E. Zayas", "M.A."),    # dotted degree acronym (ambiguous; repo convention = drop)
    ],
)
def test_trailing_and_dotted_credentials_still_dropped(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
    expected_text: str,
    dropped_token: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.canonical_name is not None
    assert result.canonical_name.text == expected_text
    assert dropped_token not in result.canonical_name.text


def test_credential_collision_hard_and_ambiguous_cases_characterization(
    normalizer: PersonNameNormalizationService,
) -> None:
    """HARD / AMBIGUOUS cases for the leading-credential fix — documented for review.

    These are NOT clean wins; they record current behavior so a reviewer can judge.
    None is a regression vs the pre-fix baseline (which dropped the leading token
    entirely and produced an empty surname or lost the María given).
    """
    # (1) "MS" is kept (surname restored) but re-cased to Title-case "Ms" downstream,
    #     so the given reads like the honorific. Cosmetic; the surname is now correct.
    r = normalizer.normalize_text("MS Islam")
    assert r.canonical_name.text == "Ms Islam"
    assert r.canonical_name.normalized.surname_tokens == ("Islam",)

    # (2) "Ma. de los Ángeles Martínez Ortega": a Mexican compound. The strong particle
    #     span "de los" pulls "Ma. de los Ángeles" into the surname role and makes
    #     "Martínez" the given — mis-parsed, and rendered surname-first.
    #     Baseline was also wrong (dropped "Ma." -> "de los Ángeles Martínez Ortega").
    r = normalizer.normalize_text("Ma. de los Ángeles Martínez Ortega")
    assert r.canonical_name.text == "Martínez Ortega Ma. de los Ángeles"

    # (3) "Ma. del Mar Delgado": "Ma." kept as given; "del Mar" ends up in the surname
    #     rather than the given "María del Mar". Imperfect but better than dropping María.
    r = normalizer.normalize_text("Ma. del Mar Delgado")
    assert r.canonical_name.text == "Ma. del Mar Delgado"
    assert r.canonical_name.normalized.given_name == "Ma."


@pytest.mark.parametrize(
    "raw_name",
    [
        # "Senior" is the SURNAME here (English given + Senior). It was being eaten as
        # the "Sr." suffix, leaving an EMPTY surname ("Roxy Senior" -> given Roxy, s='').
        "Roxy Senior",
        "Kathryn Senior",
        "Jane Senior",
        "Carl Senior",
        "Peter A. Senior",  # 3 tokens but only one non-initial precedes -> still a surname
    ],
)
def test_trailing_senior_kept_as_surname_when_no_other_surname(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None
    assert result.canonical_name.normalized.surname_tokens == ("Senior",)
    assert result.canonical_name.normalized.suffix == ""


@pytest.mark.parametrize(
    ("raw_name", "expected_surname", "expected_suffix"),
    [
        # A real generational suffix: a surname survives the removal, so Senior/Junior
        # is correctly demoted to Sr./Jr.
        ("Geraldo Bezerra da Silva Junior", ("da", "Silva"), "Jr."),
        ("John Smith Jr", ("Smith",), "Jr."),
        ("John Smith Sr.", ("Smith",), "Sr."),
        ("Robert Downey Jr.", ("Downey",), "Jr."),
        ("Mary Ann Evans Senior", ("Evans",), "Sr."),
        ("Henry Ford III", ("Ford",), "III"),  # roman-numeral suffix unaffected
    ],
)
def test_senior_junior_still_suffix_when_surname_survives(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
    expected_surname: tuple[str, ...],
    expected_suffix: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.canonical_name is not None
    assert result.canonical_name.normalized.surname_tokens == expected_surname
    assert result.canonical_name.normalized.suffix == expected_suffix


def test_suffix_senior_junior_hard_and_ambiguous_cases_characterization(
    normalizer: PersonNameNormalizationService,
) -> None:
    """HARD / AMBIGUOUS Senior/Junior cases — documented for review.

    Bare "Surname Junior" (no given) is genuinely ambiguous between the Brazilian
    suffix (surname Silva, suffix Jr.) and a two-part name (given Silva, surname
    Junior). We keep it as a surname so the surname is never emptied; this matches
    the source DB's own first/last split and is not a regression (baseline produced
    an empty surname with suffix Jr.).
    """
    r = normalizer.normalize_text("Silva Junior")
    assert r.canonical_name.normalized.surname_tokens == ("Junior",)
    assert r.canonical_name.normalized.suffix == ""

    # A single-initial given followed by "Senior": surname is still preserved.
    r = normalizer.normalize_text("A Senior")
    assert r.canonical_name.normalized.surname_tokens == ("Senior",)


@pytest.mark.parametrize(
    "raw_name",
    [
        # Pure organizations were accepted as persons (the old code even salvaged a
        # garbage name-fragment from them). They are now rejected as a whole.
        "Niels Bohr Institute",
        "Jet Propulsion Laboratory",
        "Max Planck Institute for Astronomy",
        "Editorial Board",
        "Editorial Office",
        "Proceedings of SPIE",
        "Journal of Healthcare Engineering",
        "Institut Agama",                       # German/Indonesian "Institut" (no trailing e)
        "Universitat Politecnica de Valencia",  # Catalan "Universitat"
        "Faculty Senate",
        "Proteomics Initiative",
        "Ministry of Education",
    ],
)
def test_organization_strings_rejected(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.outcome is PersonNameOutcome.NON_PERSON
    assert result.canonical_name is None


@pytest.mark.parametrize("raw_name", ["null", "null null", "of the", "of", "the"])
def test_null_literal_and_function_word_only_rejected(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.outcome is PersonNameOutcome.NON_PERSON
    assert result.canonical_name is None


@pytest.mark.parametrize(
    "raw_name",
    [
        # Real people whose surname collides with an org word must NOT be rejected.
        # These surnames are deliberately EXCLUDED from the org lexicon.
        "Philip G. Board",       # Board (excluded: many real "* Board" people)
        "Johnathan Board",
        "Frank Press",           # Press (William H. Press, Frank Press the geophysicist)
        "William H. Press",
        "Christophe Bureau",     # Bureau (common French surname)
        "Martin Bureau",
        "Gary Null",             # Null is a real surname; "null" rejects only whole-name
        "Cynthia H. Null",
        "Linda M. Null",
        # Org word only as a SUBSTRING of a real name (whole-token matching keeps these).
        "John Boardman",
        "F. Anulli",
        "Luigi Canullo",
        "Amanullah",
        # Org word inside a hyphenated compound surname (not split into its own token).
        "M. Huertas-Company",    # Marc Huertas-Company, astronomer
        "J. G. Beebe-Center",
    ],
)
def test_real_people_with_org_like_surnames_kept(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None


def test_organization_hard_and_ambiguous_cases_characterization(
    normalizer: PersonNameNormalizationService,
) -> None:
    """HARD / AMBIGUOUS org-vs-person cases — documented for review.

    (A) Human name GLUED to an affiliation: we no longer salvage the name — the whole
    contaminated string is rejected. The researcher is captured via their clean
    (affiliation-free) entries elsewhere, so freq barely moves; meanwhile this avoids
    turning tens of thousands of pure orgs into garbage name-fragments.
    """
    for org_glued in (
        "Rachel Webster University of New South Wales",
        "L. D. Landau Institute for Theoretical Physics",  # org named after a person
        "C. W. Chu Department of Physics",                 # real researcher + affiliation
    ):
        assert normalizer.normalize_text(org_glued).outcome is PersonNameOutcome.NON_PERSON

    # (B) An org word fused into a hyphenated token is now caught, but ONLY for words that
    # are never part of a real surname (institut/university/journal/centre/team). So the org
    # "Robert Koch-Institut" is rejected while the real surname "Beebe-Center" is kept.
    #
    # center/company/hospital/bureau/press are deliberately EXCLUDED from the hyphen split
    # because each is a genuine (usually Catalan/French) compound-surname element with real
    # people in the corpus, so splitting on them would false-reject:
    #   company  -> Torres-Company, Huertas-Company, Company-Quiroga   (Catalan "Company")
    #   hospital -> Gómez-Hospital, Hospital-Benito, Tirado-Hospital   (Catalan "Hospital")
    #   bureau   -> Plu-Bureau, Bureau-Point, Bureau-Franz             (French "Bureau")
    #   press    -> Broniarz-Press                                      ("Press" surname)
    #   center   -> Beebe-Center                                        ("Center" surname)
    # Per the 1M/full-corpus data these lose more real people than they gain orgs.
    assert normalizer.normalize_text("Robert Koch-Institut").outcome is PersonNameOutcome.NON_PERSON
    assert normalizer.normalize_text("J. G. Beebe-Center").outcome is PersonNameOutcome.PERSON


@pytest.mark.parametrize(
    ("raw_name", "given", "middle", "surname"),
    [
        # A leading initial before a STRONG particle span was copied into the surname
        # and duplicated ("M. van der Klis" -> "M. M. van der Klis"). Now the initial
        # stays the given and the surname starts at the particle.
        ("M. van der Klis", "M.", "", ("van", "der", "Klis")),
        ("A.F.B. van der Poel", "A.F.B.", "", ("van", "der", "Poel")),
        ("J.M.F. dos Santos", "J.M.F.", "", ("dos", "Santos")),
        ("G. van der Laan", "G.", "", ("van", "der", "Laan")),
        ("S. de la Torre", "S.", "", ("de", "la", "Torre")),
        ("H. von der Schmitt", "H.", "", ("von", "der", "Schmitt")),
        # A middle initial before the particle is placed in the middle, not the surname.
        ("M. F. van der Berg", "M.", "F.", ("van", "der", "Berg")),
        ("Nelson L. S. da Fonseca", "Nelson", "L. S.", ("da", "Fonseca")),
    ],
)
def test_leading_initial_not_duplicated_into_particle_surname(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
    given: str,
    middle: str,
    surname: tuple[str, ...],
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None
    n = result.canonical_name.normalized
    assert n.given_name == given
    assert n.middle_name == middle
    assert n.surname_tokens == surname
    assert result.canonical_name.text == raw_name  # no duplicated token


@pytest.mark.parametrize(
    ("raw_name", "surname"),
    [
        # Controls: a real surname-head token before the particle IS kept in the surname,
        # single-particle names were never affected, and full given names are unchanged.
        ("M. Carvalho da Silva", ("Carvalho", "da", "Silva")),
        ("M. de Boer", ("de", "Boer")),
        ("J. van den Berg", ("van", "den", "Berg")),
        ("Anna van der Berg", ("van", "der", "Berg")),
    ],
)
def test_particle_surname_controls_unchanged(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
    surname: tuple[str, ...],
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.canonical_name is not None
    assert result.canonical_name.normalized.surname_tokens == surname
    assert result.canonical_name.text == raw_name


@pytest.mark.parametrize(
    "raw_name",
    [
        # An org word FUSED into a hyphenated token is now detected (only for words that
        # are never part of a real surname: institut/university/journal/laboratory/centre/team).
        "Robert Koch-Institut",
        "Ruhr-Universität Bochum",
        "Louisville Courier-Journal",
        "MGIMO-University",
        "Goethe-Institut Glasgow",
        "ASDEX-Team",
    ],
)
def test_hyphenated_org_word_rejected(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.outcome is PersonNameOutcome.NON_PERSON
    assert result.canonical_name is None


@pytest.mark.parametrize(
    "raw_name",
    [
        # Real hyphenated compound surnames whose second element is an org-ish word are
        # KEPT — those words (company/hospital/bureau/press/center/board) are deliberately
        # NOT in the hyphen-split set because they are genuine surnames.
        "Victor Torres-Company",     # Catalan "Company"
        "Marc Huertas-Company",
        "Joan Antoni Gómez-Hospital",
        "Geneviève Plu-Bureau",      # French "Bureau"
        "Lubomira Broniarz-Press",
        "J. G. Beebe-Center",
    ],
)
def test_hyphenated_surname_with_org_element_kept(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None


@pytest.mark.parametrize(
    "raw_name",
    [
        # Curated org/section nouns (empirically org-only) — rejected as non-person.
        "Gene Therapy Program",
        "Aaron Blake Publishers",
        "Marketplace Services",
        "Intelligence Division",
        "World Health Organization",
        "World Health Organisation",
        "National Academy of Sciences",
        "Meteorological Office",
        "Climate Network",
        "European Bulletin",
    ],
)
def test_additional_org_section_nouns_rejected(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.outcome is PersonNameOutcome.NON_PERSON
    assert result.canonical_name is None


@pytest.mark.parametrize(
    "raw_name",
    [
        # Words deliberately EXCLUDED from the org lexicon because they are real
        # surnames — the people must be kept (verified against the corpus).
        "Anne Cathrine Staff",   # "Staff" is a real surname
        "Jeremy Staff",
        "Ilene Staff",
    ],
)
def test_excluded_org_words_that_are_real_surnames_kept(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None


@pytest.mark.parametrize(
    "raw_name",
    [
        # Real high-occurrence organization strings from the corpus (occ in comments) —
        # every one was previously ACCEPTED as a person; all now rejected.
        "Proceedings of SPIE",                          # 2645
        "Proteomics Initiative",                        # 1747
        "journals Iosr",                                # 1353
        "Journal of Healthcare Engineering",            # 973
        "Faculty Senate",                               # 955
        "Editorial Board",                              # 721
        "Institut Agama",                               # 650
        "Editors Archiv für katholisches Kirchenrech",  # 3843
        "Ludwig-Maximilians-Universität München",       # 505
        "Kapteyn Astronomical Institute",               # 239
        "Robert Koch-Institut",                         # 1693
    ],
)
def test_real_corpus_organization_strings_rejected(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.outcome is PersonNameOutcome.NON_PERSON
    assert result.canonical_name is None


@pytest.mark.parametrize(
    "raw_name",
    [
        # Non-English org-only nouns (FR/ES/IT/PT/NL/DE). The English-centric org
        # lexicon let these through as "persons"; all are real corpus strings (occ in
        # comments) previously ACCEPTED, now rejected. Sized at ~30k names / 103k occ.
        "Ministerio de Educación",                    # 419  ES
        "ministère du Travail",                       # 330  FR
        "Sächsische Akademie der Wissenschaften zu Leipzig",  # 13  DE
        "Société des Auxiliaires des Missions",       # 12  FR
        "Gesellschaft der Musikfreunde in Wien",      # 10  DE
        "Sociedade Brasileira de Comportamento Motor",  # 5  PT
        "Universidad Autonoma Metropolitana",         # 1  ES
        "Istituto di Cosmogeofisica",                 # 1  IT
        "Freudenthal Instituut",                      # 3  NL
        "Stichting Nedeco",                           # 1  NL
        "Federación Internacional Farmacéutica",      # 1  ES
        "Ministerie van Onderwijs",                   # 1  NL
        "Associazione Euratom-Enea",                  # 1  IT
    ],
)
def test_non_english_org_strings_rejected(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.outcome is PersonNameOutcome.NON_PERSON
    assert result.canonical_name is None


@pytest.mark.parametrize(
    "raw_name",
    [
        # The prepositions "für" (DE) / "voor" (NL) mark an org ONLY when a token follows
        # them (leading/mid position). All real corpus org strings, previously accepted.
        "Bundesministerium für Bildung und Forschung",  # compound noun caught only via "für"
        "Zentrum für Orthopädie",
        "Institut für Physik",
        "Voor Numismatiek",                             # NL, leading "voor"
        "Vereniging voor Natuurwetenschappen",
    ],
)
def test_org_preposition_nonfinal_rejected(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.outcome is PersonNameOutcome.NON_PERSON
    assert result.canonical_name is None


@pytest.mark.parametrize(
    "raw_name",
    [
        # Real people whose surname IS the preposition token but as the FINAL token —
        # "Für" (Hungarian) / "Voor" (Estonian/Dutch) are genuine surnames and must be kept.
        # The positional rule (org only when NOT final) preserves these; a bare-token
        # reject would have lost ~72 "Voor" + ~104 "Für" real people from the corpus.
        "Gabriella Für",         # Hungarian surname Für
        "Csilla Sepsey Für",
        "Michael J. Voor",       # highest-occ real "Voor" author
        "Tiia Voor",
        "Ivo Voor",
        # Contamination guards: "para"/"pour"/"und" are deliberately NOT org markers —
        # they collide with real givens and noble compound surnames.
        "Per Andersson",         # "Per" is a Scandinavian given name, not "per"
        "Para Chandrasoma",      # leading "Para" is a real given name
        "O. von Bohlen und Halbach",   # noble compound surname with "und"
        "Marco von Strauss und Torney",
    ],
)
def test_org_preposition_final_and_guards_kept(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None


@pytest.mark.parametrize(
    "raw_name",
    [
        # Real people from the corpus whose compound surname ends in an org-ish word —
        # these are exactly why company/hospital/bureau/press/center are NOT hyphen-split.
        "Marc Huertas-Company",       # astronomer
        "Victor Torres-Company",
        "Jaime Company-Quiroga",
        "Joan Antoni Gómez-Hospital",
        "Daniel Hospital-Benito",
        "Juan Luis Tirado-Hospital",
        "Geneviève Plu-Bureau",
        "Eve Bureau-Point",
        "Lubomira Broniarz-Press",
        "J. G. Beebe-Center",
    ],
)
def test_real_corpus_hyphenated_compound_surnames_kept(
    normalizer: PersonNameNormalizationService,
    raw_name: str,
) -> None:
    result = normalizer.normalize_text(raw_name)

    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None
