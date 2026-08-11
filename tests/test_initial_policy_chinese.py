"""Focused regressions for the Chinese given-name and initial policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from sinonym.coretypes import CanonicalName


def _canonical_fields(canonical: CanonicalName | None) -> tuple[str, str, str, str]:
    """Return the canonical text and normalized person fields."""
    assert canonical is not None
    normalized = canonical.normalized
    return canonical.text, normalized.given_name, normalized.middle_name, normalized.surname


@pytest.mark.parametrize(
    "raw",
    ["A. S. Wang", "A.S Wang", "A.S. Wang", "A S Wang", "A.S.Wang", "Wang A. S."],
)
def test_chinese_all_initial_variants_are_one_dotted_compound_given(detector, raw):
    result = detector.normalize_name(raw)
    canonical = detector.normalize_person_name(raw)

    assert result.success
    assert result.result == "A.-S. Wang"
    assert result.parsed.given_name == "A.-S."
    assert result.parsed.given_tokens == ["A.", "S."]
    assert result.parsed.middle_name == ""
    assert result.parsed.middle_tokens == []
    assert _canonical_fields(canonical) == ("A.-S. Wang", "A.-S.", "", "Wang")


@pytest.mark.parametrize("raw", ["A.B.C. Wang", "A B C Wang", "A.B.C.Wang"])
def test_long_chinese_initial_sequences_remain_one_compound_given(detector, raw):
    result = detector.normalize_name(raw)

    assert result.success
    assert result.result == "A.-B.-C. Wang"
    assert result.parsed.given_name == "A.-B.-C."
    assert result.parsed.given_tokens == ["A.", "B.", "C."]
    assert result.parsed.middle_tokens == []


@pytest.mark.parametrize("raw", ["BC Wang", "Wang BC"])
def test_reviewed_chinese_compact_initial_bundle_reaches_canonical_api(detector, raw):
    legacy = detector.normalize_name(raw)
    canonical = detector.normalize_person_name(raw)

    assert legacy.success
    assert _canonical_fields(canonical)[:3] == ("B.-C. Wang", "B.-C.", "")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Lee Tsz", "Tsz Lee"),
        ("Tsz Lee", "Tsz Lee"),
        ("LEE TSZ", "Tsz Lee"),
        ("Lee Jyh", "Jyh Lee"),
        ("Jyh Lee", "Jyh Lee"),
        ("LEE JYH", "Jyh Lee"),
        ("Lee Ng", "Ng Lee"),
        ("Ng Lee", "Ng Lee"),
        ("LEE NG", "Ng Lee"),
    ],
)
def test_attested_remapped_short_given_syllable_remains_atomic(detector, raw, expected):
    """Romanization aliases attested as syllables must not become initials."""
    result = detector.normalize_name(raw)

    assert result.success
    assert result.result == expected
    assert result.parsed.given_tokens == [expected.split()[0]]
    assert result.parsed.middle_tokens == []


def test_dotted_short_bundle_remains_initial_only_evidence(detector):
    """Explicit punctuation retains initial semantics despite a matching letter run."""
    assert not detector.normalize_name("Lee T.S.Z.").success


def test_undelimited_unrecognized_letters_remain_atomic(detector):
    legacy = detector.normalize_name("AD Wang")
    canonical = detector.normalize_person_name("AD Wang")

    assert not legacy.success
    assert _canonical_fields(canonical)[0] == "AD Wang"


@pytest.mark.parametrize(
    ("raw", "formatted", "given", "middle", "source_order"),
    [
        ("Wei Ming Wang", "Wei-Ming Wang", "Wei-Ming", "", ["given", "surname"]),
        ("Wei M. Wang", "Wei M. Wang", "Wei", "M.", ["given", "middle", "surname"]),
        (
            "Yi J Xiang Wang",
            "Yi-Xiang J. Wang",
            "Yi-Xiang",
            "J.",
            ["given", "middle", "given", "surname"],
        ),
        (
            "J Yi K Xiang Wang",
            "Yi-Xiang J. K. Wang",
            "Yi-Xiang",
            "J. K.",
            ["middle", "given", "middle", "given", "surname"],
        ),
        (
            "Yi J Xiang K Wang",
            "Yi-Xiang J. K. Wang",
            "Yi-Xiang",
            "J. K.",
            ["given", "middle", "given", "middle", "surname"],
        ),
    ],
)
def test_chinese_mixed_spelled_and_initial_components(  # noqa: PLR0913 - explicit policy matrix columns
    detector,
    raw,
    formatted,
    given,
    middle,
    source_order,
):
    result = detector.normalize_name(raw)

    assert result.success
    assert result.result == formatted
    assert result.parsed.given_name == given
    assert result.parsed.given_tokens == given.split("-")
    assert result.parsed.middle_name == middle
    assert result.parsed.middle_tokens == middle.split()
    assert result.parsed_original_order.order == source_order


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Wang Guang-y", "Guang-Y. Wang"),
        ("Guang-y Wang", "Guang-Y. Wang"),
    ],
)
def test_explicit_hyphen_binds_given_parts_while_true_initials_remain_dotted(detector, raw, expected):
    result = detector.normalize_name(raw)

    assert result.success
    assert result.result == expected
    assert result.parsed.middle_tokens == []


def test_adopted_english_given_remains_outside_legacy_chinese_recognition(detector):
    result = detector.normalize_name("Olivia Y Wang")
    canonical = detector.normalize_person_name("Olivia Y Wang")

    assert not result.success
    assert _canonical_fields(canonical)[1:] == ("Olivia", "Y.", "Wang")


@pytest.mark.parametrize(
    "raw",
    ["Zhang \u5f20 Wei \u4f1f A \u963f", "\u5f20\u4f1f\u963f"],
)
def test_native_a_syllable_overrides_mixed_initial_fallback(detector, raw):
    result = detector.normalize_name(raw)

    assert result.success
    assert result.result == "Wei-A Zhang"
    assert result.parsed.given_name == "Wei-A"
    assert result.parsed.given_tokens == ["Wei", "A"]
    assert result.parsed.middle_tokens == []


def test_chinese_initial_policy_has_scalar_batch_component_parity(detector):
    names = [
        "A. S. Wang",
        "Wei M. Wang",
        "Yi J Xiang Wang",
        "Zhang \u5f20 Wei \u4f1f A \u963f",
    ]
    scalar = [detector.normalize_name(name) for name in names]
    batch = detector.analyze_name_batch(names).results

    assert [result.result for result in batch] == [result.result for result in scalar]
    assert [result.parsed for result in batch] == [result.parsed for result in scalar]
    assert [result.parsed_original_order for result in batch] == [result.parsed_original_order for result in scalar]


@pytest.mark.parametrize(
    ("raw", "given", "middle"),
    [
        ("Wei Ming Wang", "Wei-Ming", ""),
        ("Wei M. Wang", "Wei", "M."),
        ("A. S. Wang", "A.-S.", ""),
    ],
)
def test_raw_person_canonical_uses_affirmative_chinese_normalization_and_generic_source(
    detector,
    raw,
    given,
    middle,
):
    generic = detector._person_name_normalizer.normalize_text(raw).canonical_name  # noqa: SLF001
    canonical = detector.normalize_person_name(raw)

    assert generic is not None
    assert canonical is not None
    assert _canonical_fields(canonical)[:3] == (" ".join(filter(None, (given, middle, "Wang"))), given, middle)
    assert canonical.source == generic.source


@pytest.mark.parametrize(
    ("first", "middle", "normalized_first", "normalized_middle"),
    [
        ("Wei Ming", None, "Wei-Ming", ""),
        ("Wei M.", None, "Wei", "M."),
        ("A. S.", None, "A.-S.", ""),
        ("BC", None, "B.-C.", ""),
        ("Olivia", "Y.", "Olivia", "Y."),
    ],
)
def test_structured_person_canonical_uses_chinese_policy_without_rewriting_source(
    detector,
    first,
    middle,
    normalized_first,
    normalized_middle,
):
    generic = detector._person_name_normalizer.normalize_components(  # noqa: SLF001
        first_name=first,
        middle_name=middle,
        last_name="Wang",
    ).canonical_name
    canonical = detector.normalize_person_name_components(
        first_name=first,
        middle_name=middle,
        last_name="Wang",
    )

    assert generic is not None
    assert canonical is not None
    assert _canonical_fields(canonical)[:3] == (
        " ".join(filter(None, (normalized_first, normalized_middle, "Wang"))),
        normalized_first,
        normalized_middle,
    )
    assert canonical.source == generic.source


@pytest.mark.parametrize(
    ("first", "expected_first", "expected_middle"),
    [
        ("Wei Ming", "Wei-Ming", ""),
        ("Wei M", "Wei", "M."),
    ],
)
def test_cross_cultural_surname_accepts_direct_chinese_given_evidence_for_raw_and_structured_input(
    detector,
    first,
    expected_first,
    expected_middle,
):
    raw = detector.normalize_person_name(f"{first} Lee")
    structured = detector.normalize_person_name_components(first_name=first, last_name="Lee")

    for canonical in (raw, structured):
        assert _canonical_fields(canonical)[:3] == (
            " ".join(filter(None, (expected_first, expected_middle, "Lee"))),
            expected_first,
            expected_middle,
        )
