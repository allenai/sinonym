"""Focused regressions for the Chinese given-name and initial policy."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "raw",
    [
        "A. S. Wang",
        "A.S Wang",
        "A.S. Wang",
        "A S Wang",
        "A.S.Wang",
        "Wang A. S.",
    ],
)
def test_chinese_all_initial_variants_are_one_dotted_compound_given(detector, raw):
    result = detector.normalize_name(raw)

    assert result.success
    assert result.result == "A.-S. Wang"
    assert result.parsed.given_name == "A.-S."
    assert result.parsed.given_tokens == ["A.", "S."]
    assert result.parsed.middle_name == ""
    assert result.parsed.middle_tokens == []


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
    assert canonical is not None
    assert canonical.text == "B.-C. Wang"
    assert canonical.normalized.given_name == "B.-C."
    assert canonical.normalized.middle_name == ""


def test_undelimited_unrecognized_letters_remain_atomic(detector):
    legacy = detector.normalize_name("AD Wang")
    canonical = detector.normalize_person_name("AD Wang")

    assert not legacy.success
    assert canonical is not None
    assert canonical.text == "AD Wang"


@pytest.mark.parametrize(
    ("raw", "formatted", "given", "given_tokens", "middle", "middle_tokens", "source_order"),
    [
        ("Wei Ming Wang", "Wei-Ming Wang", "Wei-Ming", ["Wei", "Ming"], "", [], ["given", "surname"]),
        ("Wei M. Wang", "Wei M. Wang", "Wei", ["Wei"], "M.", ["M."], ["given", "middle", "surname"]),
        (
            "Yi J Xiang Wang",
            "Yi-Xiang J. Wang",
            "Yi-Xiang",
            ["Yi", "Xiang"],
            "J.",
            ["J."],
            ["given", "middle", "given", "surname"],
        ),
        (
            "J Yi K Xiang Wang",
            "Yi-Xiang J. K. Wang",
            "Yi-Xiang",
            ["Yi", "Xiang"],
            "J. K.",
            ["J.", "K."],
            ["middle", "given", "middle", "given", "surname"],
        ),
        (
            "Yi J Xiang K Wang",
            "Yi-Xiang J. K. Wang",
            "Yi-Xiang",
            ["Yi", "Xiang"],
            "J. K.",
            ["J.", "K."],
            ["given", "middle", "given", "middle", "surname"],
        ),
    ],
)
def test_chinese_mixed_spelled_and_initial_components(  # noqa: PLR0913 - explicit policy matrix columns
    detector,
    raw,
    formatted,
    given,
    given_tokens,
    middle,
    middle_tokens,
    source_order,
):
    result = detector.normalize_name(raw)

    assert result.success
    assert result.result == formatted
    assert result.parsed.given_name == given
    assert result.parsed.given_tokens == given_tokens
    assert result.parsed.middle_name == middle
    assert result.parsed.middle_tokens == middle_tokens
    assert result.parsed_original_order.order == source_order


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Wang Guang-y", "Guang-Y. Wang"),
        ("Guang-y Wang", "Guang-Y. Wang"),
        ("A-wei Zhang", "A-Wei Zhang"),
        ("Wei-A Zhang", "Wei-A Zhang"),
        ("XiangE Sun", "Xiang-E Sun"),
        ("Li Guo-e", "Guo-E Li"),
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
    assert canonical is not None
    assert canonical.normalized.given_name == "Olivia"
    assert canonical.normalized.middle_name == "Y."
    assert canonical.normalized.surname == "Wang"


@pytest.mark.parametrize(
    "raw",
    [
        "Zhang \u5f20 Wei \u4f1f A \u963f",
        "\u5f20\u4f1f\u963f",
    ],
)
def test_native_a_syllable_overrides_mixed_initial_fallback(detector, raw):
    result = detector.normalize_name(raw)

    assert result.success
    assert result.result == "Wei-A Zhang"
    assert result.parsed.given_name == "Wei-A"
    assert result.parsed.given_tokens == ["Wei", "A"]
    assert result.parsed.middle_tokens == []


def test_aligned_multisyllable_roman_token_keeps_native_one_letter_syllable_in_given(detector):
    result = detector.normalize_name("Xiaoe \u5c0f\u5a25 Li \u674e")

    assert result.success
    assert result.result == "Xiao-E Li"
    assert result.parsed.given_tokens == ["Xiao", "E"]
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
    ("raw", "text", "given", "middle"),
    [
        ("Wei Ming Wang", "Wei-Ming Wang", "Wei-Ming", ""),
        ("Wei M. Wang", "Wei M. Wang", "Wei", "M."),
        ("A. S. Wang", "A.-S. Wang", "A.-S.", ""),
    ],
)
def test_raw_person_canonical_uses_affirmative_chinese_normalization_and_generic_source(
    detector,
    raw,
    text,
    given,
    middle,
):
    generic = detector._person_name_normalizer.normalize_text(raw).canonical_name  # noqa: SLF001
    canonical = detector.normalize_person_name(raw)

    assert generic is not None
    assert canonical is not None
    assert canonical.text == text
    assert canonical.normalized.given_name == given
    assert canonical.normalized.middle_name == middle
    assert canonical.source == generic.source


@pytest.mark.parametrize(
    ("first", "middle", "last", "text", "normalized_first", "normalized_middle"),
    [
        ("Wei Ming", None, "Wang", "Wei-Ming Wang", "Wei-Ming", ""),
        ("Wei M.", None, "Wang", "Wei M. Wang", "Wei", "M."),
        ("A. S.", None, "Wang", "A.-S. Wang", "A.-S.", ""),
        ("BC", None, "Wang", "B.-C. Wang", "B.-C.", ""),
        ("Olivia", "Y.", "Wang", "Olivia Y. Wang", "Olivia", "Y."),
    ],
)
def test_structured_person_canonical_uses_chinese_policy_without_rewriting_source(  # noqa: PLR0913
    detector,
    first,
    middle,
    last,
    text,
    normalized_first,
    normalized_middle,
):
    generic = detector._person_name_normalizer.normalize_components(  # noqa: SLF001
        first_name=first,
        middle_name=middle,
        last_name=last,
    ).canonical_name
    canonical = detector.normalize_person_name_components(
        first_name=first,
        middle_name=middle,
        last_name=last,
    )

    assert generic is not None
    assert canonical is not None
    assert canonical.text == text
    assert canonical.normalized.given_name == normalized_first
    assert canonical.normalized.middle_name == normalized_middle
    assert canonical.source == generic.source


@pytest.mark.parametrize(
    ("raw", "expected_text", "expected_first", "expected_middle"),
    [
        ("Wei Ming Lee", "Wei-Ming Lee", "Wei-Ming", ""),
        ("Wei M Lee", "Wei M. Lee", "Wei", "M."),
    ],
)
def test_cross_cultural_surname_accepts_direct_chinese_given_evidence(
    detector,
    raw,
    expected_text,
    expected_first,
    expected_middle,
):
    canonical = detector.normalize_person_name(raw)

    assert canonical is not None
    assert canonical.text == expected_text
    assert canonical.normalized.given_name == expected_first
    assert canonical.normalized.middle_name == expected_middle


@pytest.mark.parametrize(
    ("first", "expected_text", "expected_first", "expected_middle"),
    [
        ("Wei Ming", "Wei-Ming Lee", "Wei-Ming", ""),
        ("Wei M", "Wei M. Lee", "Wei", "M."),
    ],
)
def test_structured_cross_cultural_surname_accepts_direct_chinese_given_evidence(
    detector,
    first,
    expected_text,
    expected_first,
    expected_middle,
):
    canonical = detector.normalize_person_name_components(first_name=first, last_name="Lee")

    assert canonical is not None
    assert canonical.text == expected_text
    assert canonical.normalized.given_name == expected_first
    assert canonical.normalized.middle_name == expected_middle
