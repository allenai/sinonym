"""Public canonical-name behavior without changing legacy Chinese recognition."""

from __future__ import annotations


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
