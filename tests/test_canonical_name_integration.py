"""Public canonical-name behavior without changing legacy Chinese recognition."""

from __future__ import annotations

import pytest

from sinonym.name_punctuation import ROMAN_HYPHEN_LIKE


def test_token_lookup_normalization_removes_all_reviewed_roman_hyphens(detector):
    baseline = "jiae"

    for hyphen in ROMAN_HYPHEN_LIKE:
        token = f"Ji{hyphen}Ae"
        assert detector._normalizer.norm(token) == baseline  # noqa: SLF001
        assert detector._normalizer.norm_light(token) == baseline  # noqa: SLF001


def test_non_chinese_result_surfaces_canonical_name_without_legacy_success(detector):
    result = detector.normalize_name("Dr. Steve Marsh PhD")

    assert not result.success
    assert result.result == ""
    assert result.parsed is None
    assert result.canonical_name is not None
    assert result.canonical_name.text == "Steve Marsh"
    assert result.canonical_name.normalized.given_name == "Steve"
    assert result.canonical_name.normalized.middle_name == ""
    assert result.canonical_name.normalized.surname == "Marsh"
    assert result.canonical_name.normalized.suffix == ""


def test_canonical_name_normalizes_all_joiner_variants(detector):
    result = detector.normalize_name("Ms. Ana\u2013Maria O\u2019Neill MS")

    assert not result.success
    assert result.canonical_name is not None
    assert result.canonical_name.text == "Ana-Maria O'Neill"
    assert result.canonical_name.normalized.given_tokens == ("Ana-Maria",)
    assert result.canonical_name.normalized.surname_tokens == ("O'Neill",)


def test_canonical_name_moves_true_suffix_to_suffix_component(detector):
    result = detector.normalize_name("Steve Blando IV")

    assert not result.success
    assert result.canonical_name is not None
    assert result.canonical_name.text == "Steve Blando IV"
    assert result.canonical_name.normalized.given_name == "Steve"
    assert result.canonical_name.normalized.surname == "Blando"
    assert result.canonical_name.normalized.suffix == "IV"
    assert result.canonical_name.normalized.order == ("given", "surname", "suffix")


def test_two_token_suffixes_and_credentials_use_full_canonical_pipeline(detector):
    cases = (
        ("John Jr", "John Jr.", "Jr."),
        # Spelled-out "Senior" is a real surname: with no other surname to fall back on
        # it is kept as the surname (not demoted to "Sr."), avoiding an empty surname.
        # The abbreviated "Jr."/"Sr." forms above still behave as suffixes.
        ("John Senior", "John Senior", ""),
        ("John Phd", "John", ""),
        ("Phd Smith", "Smith", ""),
        ("PD Dr", None, None),
    )

    for raw_name, expected_text, expected_suffix in cases:
        public = detector.normalize_name(raw_name).canonical_name
        generic = detector.normalize_person_name(raw_name)

        assert public == generic
        if expected_text is None:
            assert public is None
            continue
        assert public is not None
        assert public.text == expected_text
        assert public.normalized.suffix == expected_suffix


def test_structured_component_normalization_repairs_roles_after_drops(detector):
    canonical = detector.normalize_person_name_components(
        first_name="dr steve",
        middle_name="marsh",
        last_name="phd",
    )

    assert canonical is not None
    assert canonical.text == "Steve Marsh"
    assert canonical.normalized.given_name == "Steve"
    assert canonical.normalized.middle_name == ""
    assert canonical.normalized.surname == "Marsh"
    assert canonical.normalized.suffix == ""


def test_explicit_family_first_comma_is_rendered_in_canonical_order(detector):
    canonical = detector.normalize_person_name("Smith, John Q. Jr.")

    assert canonical is not None
    assert canonical.text == "John Q. Smith Jr."
    assert canonical.source.order == ("surname", "given", "middle", "suffix")
    assert canonical.normalized.order == ("given", "middle", "surname", "suffix")


def test_failed_scalar_reuses_its_chinese_result_for_canonical_attachment(detector, monkeypatch):
    original = detector._normalize_chinese_name  # noqa: SLF001
    calls = 0

    def counted(raw_name):
        nonlocal calls
        calls += 1
        return original(raw_name)

    monkeypatch.setattr(detector, "_normalize_chinese_name", counted)

    result = detector.normalize_name("John Smith")

    assert not result.success
    assert result.canonical_name is not None
    assert result.canonical_name.text == "John Smith"
    assert calls == 1


def test_chinese_result_canonical_name_matches_selected_parse(detector):
    result = detector.normalize_name("Wei Zhu Ge Ming")

    assert result.success
    assert result.result == "Wei-Ming Zhu Ge"
    assert result.parsed is not None
    assert result.canonical_name is not None
    assert result.canonical_name.text == result.result
    assert result.canonical_name.normalized.given_name == result.parsed.given_name
    assert result.canonical_name.normalized.surname == result.parsed.surname
    assert result.canonical_name.normalized.given_tokens == tuple(result.parsed.given_tokens)
    assert result.canonical_name.normalized.surname_tokens == tuple(result.parsed.surname_tokens)


def test_chinese_canonical_source_preserves_fused_token_lineage(detector):
    result = detector.normalize_name("Wang Weiming")

    assert result.canonical_name is not None
    assert result.canonical_name.source.given_tokens == ("Weiming",)
    assert result.canonical_name.source.surname_tokens == ("Wang",)
    assert result.canonical_name.source.order == ("surname", "given")
    assert result.canonical_name.normalized.given_tokens == ("Wei", "Ming")


def test_chinese_canonical_source_excludes_dropped_leading_citation_tokens(detector):
    result = detector.normalize_name("Et al. zHaNg wEi")

    assert result.success
    assert result.canonical_name is not None
    assert result.canonical_name.source_text == "Et al. zHaNg wEi"
    assert result.canonical_name.source.given_tokens == ("wEi",)
    assert result.canonical_name.source.surname_tokens == ("zHaNg",)
    assert result.canonical_name.source.order == ("surname", "given")


@pytest.mark.parametrize("surname", ["Li", "Wang"])
def test_chinese_canonical_source_uses_expanded_roles_for_repeated_tokens(detector, surname):
    """Equal source tokens inherit their positional role rather than the first match."""
    result = detector.normalize_name(f"{surname} Wei {surname}")

    assert result.success
    assert result.canonical_name is not None
    assert result.canonical_name.source.given_tokens == (surname, "Wei")
    assert result.canonical_name.source.surname_tokens == (surname,)
    assert result.canonical_name.source.order == ("given", "given", "surname")


@pytest.mark.parametrize(
    ("raw", "display", "expected_source"),
    [
        ("Ouyang Ou Yang", "Ou-Yang Ouyang", (("Ouyang",), ("Ou", "Yang"), ("surname", "given", "given"))),
        ("Ou Yang Ouyang", "Ou-Yang Ouyang", (("Ouyang",), ("Ou", "Yang"), ("given", "given", "surname"))),
        ("Zhuge Zhu Ge", "Zhu-Ge Zhuge", (("Zhuge",), ("Zhu", "Ge"), ("surname", "given", "given"))),
        ("Zhu Ge Zhuge", "Zhu-Ge Zhuge", (("Zhuge",), ("Zhu", "Ge"), ("given", "given", "surname"))),
    ],
)
def test_chinese_canonical_source_aligns_fused_compound_occurrences(
    detector,
    raw,
    display,
    expected_source,
):
    """A fused source token consumes its expanded normalized compound span."""
    result = detector.normalize_name(raw)

    assert result.success
    assert result.result == display
    assert result.canonical_name is not None
    source_surname, source_given, source_order = expected_source
    assert result.canonical_name.source.surname_tokens == source_surname
    assert result.canonical_name.source.given_tokens == source_given
    assert result.canonical_name.source.order == source_order
    assert result.canonical_name.normalized.surname_tokens == source_given
    assert result.canonical_name.normalized.given_tokens == source_given


@pytest.mark.parametrize(
    ("raw", "display", "expected_source"),
    [
        (
            "\u4e0a\u5b98 \u5a49\u513f",
            "Wan-Er Shang Guan",
            (("\u4e0a\u5b98",), ("\u5a49\u513f",), ("surname", "given")),
        ),
        (
            "\u529f\u534e \u5f20",
            "Gong-Hua Zhang",
            (("\u5f20",), ("\u529f\u534e",), ("given", "surname")),
        ),
        (
            "\u6b27\u9633 \u4fee \u6587",
            "Xiu-Wen Ou Yang",
            (("\u6b27\u9633",), ("\u4fee", "\u6587"), ("surname", "given", "given")),
        ),
    ],
)
def test_chinese_canonical_source_keeps_generic_han_component_fallback(detector, raw, display, expected_source):
    """Han source tokens retain parsed component order without Roman span alignment."""
    result = detector.normalize_name(raw)

    assert result.success
    assert result.result == display
    assert result.canonical_name is not None
    source_surname, source_given, source_order = expected_source
    assert result.canonical_name.source.surname_tokens == source_surname
    assert result.canonical_name.source.given_tokens == source_given
    assert result.canonical_name.source.order == source_order


def test_batch_results_surface_canonical_name_after_final_selection(detector):
    batch = detector.analyze_name_batch(["Li Wei", "John Smith"])

    chinese, western = batch.results
    assert chinese.parsed is not None
    assert chinese.canonical_name is not None
    assert chinese.canonical_name.text == chinese.result
    assert chinese.canonical_name.normalized.given_name == chinese.parsed.given_name
    assert chinese.canonical_name.normalized.surname == chinese.parsed.surname
    assert western.canonical_name is not None
    assert western.canonical_name.text == "John Smith"
    assert not western.success
    assert western.parsed is None


def test_invalid_and_obvious_non_person_inputs_have_no_canonical_name(detector):
    assert detector.normalize_name("").canonical_name is None
    assert detector.normalize_name("---").canonical_name is None
    assert detector.normalize_name("Veecon Music & Entertainment").canonical_name is None
    assert detector.normalize_name("北京大学").canonical_name is None


def test_korean_native_name_surfaces_semantic_family_first_canonical_name(detector):
    result = detector.normalize_name("김민준")

    assert not result.success
    assert result.canonical_name is not None
    assert result.canonical_name.text == "민준 김"
    assert result.canonical_name.source.order == ("surname", "given")
