# ruff: noqa: PLR2004, SLF001
"""Regression tests that implement the proposals tracked in TEST_PROPOSAL.md."""

from __future__ import annotations

import inspect
import logging
import math
import re
import unicodedata
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import ClassVar

import pytest

from sinonym.chinese_names_data import COMPOUND_VARIANTS
from sinonym.coretypes import NameFormat, ParseCandidate
from sinonym.resources import open_csv_reader, resource_path
from sinonym.services import ethnicity
from sinonym.services.batch_analysis import LATIN_ONLY_REPRESENTATION, BatchAnalysisService, BatchCandidateEntry
from sinonym.services.non_person import NON_PERSON_FAILURE_REASON
from sinonym.services.normalization import CompoundMetadata
from sinonym.utils.string_manipulation import StringManipulationUtils
from tests import conftest as tests_conftest
from tests import test_performance


def _build_given_frequencies_from_csv() -> dict[str, float]:
    """Reconstruct the finalized given-frequency map used for log probabilities."""
    given_frequencies: dict[str, float] = {}

    for row in open_csv_reader("givenname_orcid.csv"):
        han_char = row["character"]
        normalized = unicodedata.normalize("NFKD", row["pinyin"])
        pinyin = "".join(c for c in normalized if not unicodedata.combining(c))
        pinyin = "".join(c for c in pinyin if not c.isdigit()).lower()

        ppm = float(row.get("ppm", 0))
        if ppm > 0:
            given_frequencies[han_char] = max(given_frequencies.get(han_char, 0.0), ppm)
            given_frequencies[pinyin] = max(given_frequencies.get(pinyin, 0.0), ppm)

    return given_frequencies


def _build_compound_surnames(surnames_raw: set[str]) -> frozenset[str]:
    compounds = {surname for surname in surnames_raw if " " in surname}
    compounds.update(COMPOUND_VARIANTS.values())
    return frozenset(compounds)


def _batch_signature(detector, names: list[str], threshold: float) -> tuple:
    pattern = detector.detect_batch_format(names, format_threshold=threshold)
    batch = detector.analyze_name_batch(names, format_threshold=threshold)
    results = tuple(result.result if result.success else f"ERR:{result.error_message}" for result in batch.results)
    return (
        pattern.threshold_met,
        pattern.total_count,
        batch.format_pattern.threshold_met,
        batch.format_pattern.total_count,
        results,
    )


def test_korean_hyphenated_overlap_regression(detector):
    accepted = detector.normalize_name("Min-Hung Lee")
    assert accepted.success
    assert accepted.result == "Min-Hung Lee"

    rejected = detector.normalize_name("Kim Min-jun")
    assert not rejected.success


def test_korean_curated_pairs_are_format_normalized(detector):
    rejected_names = [
        "Son Ye-jin",
        "Ye-Jin Son",
        "Son Ye Jin",
        "Ye Jin Son",
        "Son Heung-min",
        "Heung-Min Son",
        "Son Heung Min",
        "Heung Min Son",
        "Ha Young Lee",
        "Sung Soo Jung",
    ]

    for name in rejected_names:
        result = detector.normalize_name(name)
        assert not result.success, name
        assert result.error_message == "Korean structural patterns detected"

    accepted_controls = {
        "Min-Hung Lee": "Min-Hung Lee",
        "Jung Chi-Wai": "Chi-Wai Jung",
    }
    for name, expected in accepted_controls.items():
        result = detector.normalize_name(name)
        assert result.success, name
        assert result.result == expected


def test_trailing_na_surname_last_examples_keep_given_first_order(detector):
    normalized = detector._normalizer.apply("Yu Bin Na")
    analysis = detector._ethnicity_service._analyze_tokens_for_patterns(normalized.roman_tokens)

    assert analysis["surname_type"] == "chinese_only"
    assert "bin" in analysis["korean_ambiguous_tokens"]
    assert analysis["korean_given_pairs"] == []

    examples = {
        "Yu Bin Na": "Yu-Bin Na",
        "Sang Min Na": "Sang-Min Na",
        "Sang Ho Na": "Sang-Ho Na",
    }
    for raw_name, expected in examples.items():
        result = detector.normalize_name(raw_name)
        assert result.success
        assert result.result == expected
        assert result.parsed_original_order.order == ["given", "surname"]


def test_trailing_na_bonus_does_not_flip_chinese_surname_first_name(detector):
    result = detector.normalize_name("Peng Wen Na")

    assert result.success
    assert result.result == "Wen-Na Peng"
    assert result.parsed_original_order.order == ["surname", "given"]


def test_korean_overlapping_ambiguous_branch_stays_bounded(detector):
    normalized = detector._normalizer.apply("Ha Min Lee")
    analysis = detector._ethnicity_service._analyze_tokens_for_patterns(normalized.roman_tokens)

    assert analysis["surname_type"] == "korean_overlapping"
    assert "min" in analysis["korean_ambiguous_tokens"]

    score = detector._ethnicity_service._calculate_korean_score_from_analysis(
        analysis,
        normalized.roman_tokens,
        list({token.lower() for token in normalized.roman_tokens}),
        normalized.norm_map,
    )
    assert score == 0.0


def test_compound_surname_formatter_uses_token_linked_metadata(detector):
    metadata = {
        "Sima": CompoundMetadata(is_compound=True, format_type="compact", compound_target="si ma"),
        "Au": CompoundMetadata(is_compound=True, format_type="spaced", compound_target="ou yang"),
        "Yeung": CompoundMetadata(is_compound=True, format_type="spaced", compound_target="ou yang"),
        "Ka": CompoundMetadata(is_compound=True, format_type="spaced", compound_target="ka ming"),
        "Ming": CompoundMetadata(is_compound=True, format_type="spaced", compound_target="ka ming"),
    }
    formatted = detector._formatting_service.format_name_output(
        ["Au", "Yeung"],
        ["Ka", "Ming"],
        {},
        metadata,
    )
    assert formatted == "Ka-Ming Au Yeung"

    # With only unrelated metadata, the surname falls back to default behavior.
    metadata_without_surname = {
        "Sima": CompoundMetadata(is_compound=True, format_type="compact", compound_target="si ma"),
        "Ka": CompoundMetadata(is_compound=True, format_type="spaced", compound_target="ka ming"),
        "Ming": CompoundMetadata(is_compound=True, format_type="spaced", compound_target="ka ming"),
    }
    fallback = detector._formatting_service.format_name_output(
        ["Au", "Yeung"],
        ["Ka", "Ming"],
        {},
        metadata_without_surname,
    )
    assert fallback == "Ka-Ming Au-Yeung"


def test_three_token_order_preservation_regression(detector):
    ambiguous = detector.normalize_name("Shao Kuan Wei")
    assert ambiguous.success
    assert ambiguous.result == "Shao-Kuan Wei"

    clear = detector.normalize_name("Zhang Wei Ming")
    assert clear.success
    assert clear.result == "Wei-Ming Zhang"


def test_parsed_original_order_keeps_semantic_component_labels(detector):
    result = detector.normalize_name("Li Wei")

    assert result.success
    assert result.result == "Wei Li"
    assert result.parsed_original_order.given_name == "Wei"
    assert result.parsed_original_order.surname == "Li"
    assert result.parsed_original_order.order == ["surname", "given"]


def test_guarded_low_frequency_surname_ratio_preserves_given_first_order(detector):
    cases = {
        "Bei Yu": "Bei Yu",
        "Lecheng Zheng": "Lecheng Zheng",
        "Yuxuan Dong": "Yuxuan Dong",
        "Xun Zhou": "Xun Zhou",
        "mi zhang": "Mi Zhang",
    }

    for raw_name, expected in cases.items():
        result = detector.normalize_name(raw_name)
        assert result.success
        assert result.result == expected


def test_batch_format_detection_ignores_isolated_low_frequency_surname_ratio(detector):
    names = ["Diao Wang", "Bian Li", "Cen Zhang", "Luan Wang", "Rao Li"]
    expected_results = ["Wang Diao", "Li Bian", "Zhang Cen", "Wang Luan", "Li Rao"]

    pattern = detector.detect_batch_format(names)
    batch = detector.analyze_name_batch(names)

    assert pattern.dominant_format == NameFormat.SURNAME_FIRST
    assert pattern.threshold_met
    assert batch.format_pattern.dominant_format == NameFormat.SURNAME_FIRST
    assert [result.result for result in batch.results] == expected_results
    assert all(result.parsed_original_order.order == ["surname", "given"] for result in batch.results)


def test_batch_preserves_homogeneous_guarded_given_first_ratio_names(detector):
    names = ["Bei Yu", "Lecheng Zheng", "Yuxuan Dong", "Xun Zhou"]
    expected_results = ["Bei Yu", "Lecheng Zheng", "Yuxuan Dong", "Xun Zhou"]

    batch = detector.analyze_name_batch(names)

    assert [result.result for result in batch.results] == expected_results
    assert all(result.parsed_original_order.order == ["given", "surname"] for result in batch.results)


def test_guarded_low_frequency_surname_ratio_keeps_compound_and_common_surname_boundaries(detector):
    cases = {
        "Men Hao": "Hao Men",
        "Ouyang Xiu": "Xiu Ouyang",
        "Murong Xue": "Xue Murong",
    }

    for raw_name, expected in cases.items():
        result = detector.normalize_name(raw_name)
        assert result.success
        assert result.result == expected


def test_given_context_gold_split_bypasses_surname_guard(detector):
    cases = {
        "Junjie Fang": "Jun-Jie Fang",
        "Junjie Peng": "Jun-Jie Peng",
        "Junjie Ye": "Jun-Jie Ye",
    }

    for raw_name, expected in cases.items():
        result = detector.normalize_name(raw_name)
        assert result.success
        assert result.result == expected


def test_batch_preserves_homogeneous_given_context_gold_splits(detector):
    names = ["Junjie Fang", "Junjie Peng", "Junjie Ye"]
    expected_results = ["Jun-Jie Fang", "Jun-Jie Peng", "Jun-Jie Ye"]

    batch = detector.analyze_name_batch(names)

    assert [result.result for result in batch.results] == expected_results
    assert all(result.parsed_original_order.order == ["given", "surname"] for result in batch.results)


def test_given_context_gold_split_is_used_by_string_formatter(detector):
    normalized = detector._normalizer.apply("Junjie Fang")

    formatted = detector._formatting_service.format_name_output(
        ["Fang"],
        ["Junjie"],
        normalized.norm_map,
        {},
    )

    assert formatted == "Jun-Jie Fang"


def test_given_context_gold_split_keeps_ambiguous_non_gold_tokens_unsplit(detector):
    splitter = StringManipulationUtils.split_concatenated_name
    data = detector._data
    normalizer = detector._normalizer
    config = detector._config

    junjie = normalizer.apply("Junjie")
    assert splitter("Junjie", junjie.norm_map, data, normalizer, config) is None
    assert StringManipulationUtils.split_surname_like_given_name(
        "Junjie",
        junjie.norm_map,
        data,
        normalizer,
        config,
    ) == ["Jun", "jie"]

    for token in ["Yuxuan", "Lecheng"]:
        normalized = normalizer.apply(token)
        assert (
            StringManipulationUtils.split_surname_like_given_name(
                token,
                normalized.norm_map,
                data,
                normalizer,
                config,
            )
            is None
        )


def test_two_token_format_alignment_tie_break_is_directional(detector):
    parsing = detector._parsing_service

    aligned_bonus = parsing._calculate_format_alignment_bonus(["li"], ["wei"], ["wei", "li"])
    opposite_bonus = parsing._calculate_format_alignment_bonus(["wei"], ["li"], ["wei", "li"])

    assert aligned_bonus == 0.001
    assert opposite_bonus == 0.0


def test_documented_performance_target_matches_test_constant():
    readme_text = Path("README.md").read_text(encoding="utf-8")
    match = re.search(r"over\s+([0-9,]+)\s+diverse names per second", readme_text, flags=re.IGNORECASE)
    assert match is not None

    documented_target = int(match.group(1).replace(",", ""))
    assert documented_target == test_performance.MIN_DIVERSE_NAMES_PER_SECOND


def test_resource_path_repeated_calls_remain_usable():
    paths = [resource_path("familyname_orcid.csv") for _ in range(8)]

    for path in paths:
        assert path.exists()
        with path.open(encoding="utf-8") as handle:
            assert handle.readline().startswith("surname")


def test_fast_detector_fixture_uses_dependency_injection():
    signature = inspect.signature(tests_conftest.fast_detector)
    assert tuple(signature.parameters) == ("detector",)

    source = inspect.getsource(tests_conftest.fast_detector)
    assert "detector()" not in source


def test_total_given_frequency_denominator_matches_finalized_map(detector):
    given_frequencies = _build_given_frequencies_from_csv()
    total_given = sum(given_frequencies.values())

    checked = 0
    for key, freq in given_frequencies.items():
        if not key.isascii():
            continue
        if key not in detector._data.given_log_probabilities:
            continue

        inferred_total = freq / math.exp(detector._data.given_log_probabilities[key])
        assert math.isclose(inferred_total, total_given, rel_tol=1e-9, abs_tol=1e-6)

        checked += 1
        if checked >= 20:
            break

    assert checked >= 5


def test_compound_surname_log_probabilities_use_compound_inclusive_total(detector):
    data_service = detector._data_service
    surnames_raw, surname_frequencies = data_service._build_surname_data()
    base_total = sum(surname_frequencies.values())

    compound_surnames = _build_compound_surnames(surnames_raw)
    compound_hyphen_map = data_service._build_compound_hyphen_map(compound_surnames)
    surname_log_probabilities = data_service._build_surname_log_probabilities(
        surname_frequencies,
        compound_surnames,
        compound_hyphen_map,
    )

    # Replay compound updates to derive the expected denominator.
    _, replay_frequencies = data_service._build_surname_data()
    compound_delta = 0.0
    for compound_surname in compound_surnames:
        parts = compound_surname.split()
        if len(parts) != 2:
            continue

        freq1 = replay_frequencies.get(parts[0], 1.0)
        freq2 = replay_frequencies.get(parts[1], 1.0)
        compound_freq = math.sqrt(freq1 * freq2) * detector._config.compound_penalty
        compound_freq = max(compound_freq, 0.1)

        existing_freq = replay_frequencies.get(compound_surname, 0.0)
        replay_frequencies[compound_surname] = compound_freq
        compound_delta += max(compound_freq - existing_freq, 0.0)

    expected_total = base_total + compound_delta

    checked = 0
    for compound_surname in sorted(compound_surnames):
        if compound_surname not in surname_frequencies:
            continue
        if compound_surname not in surname_log_probabilities:
            continue

        freq = surname_frequencies[compound_surname]
        if freq <= 0:
            continue

        inferred_total = freq / math.exp(surname_log_probabilities[compound_surname])
        assert math.isclose(inferred_total, expected_total, rel_tol=1e-9, abs_tol=1e-6)

        checked += 1
        if checked >= 10:
            break

    assert checked >= 3


def test_split_compact_format_respects_source_structure():
    assert StringManipulationUtils.split_compact_format("szeto", ["si", "tu"]) == ["sze", "to"]
    assert StringManipulationUtils.split_compact_format("auyeung", ["ou", "yang"]) == ["au", "yeung"]


def test_batch_threshold_state_is_thread_safe(detector):
    names = ["Li Wei", "Wei Li", "Zhang Ming"]
    low_threshold = 0.55
    high_threshold = 0.8

    expected_low = _batch_signature(detector, names, low_threshold)
    expected_high = _batch_signature(detector, names, high_threshold)
    assert expected_low != expected_high

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = []
        for _ in range(20):
            futures.append(pool.submit(_batch_signature, detector, names, low_threshold))
            futures.append(pool.submit(_batch_signature, detector, names, high_threshold))

        for index, future in enumerate(futures):
            expected = expected_low if index % 2 == 0 else expected_high
            assert future.result() == expected


def test_low_confidence_batch_preserves_detected_format_metadata(detector):
    names = ["Li Wei", "Wei Li", "Zhang Ming"]
    high_threshold = 0.9

    detected = detector.detect_batch_format(names, format_threshold=high_threshold)
    batch = detector.analyze_name_batch(names, format_threshold=high_threshold)

    assert detected.total_count > 0
    assert not detected.threshold_met
    assert batch.format_pattern == detected


def test_exact_vote_tie_does_not_meet_batch_application_threshold(detector):
    names = ["Mai Liao", "Xin Liu"]

    detected = detector.detect_batch_format(names)
    batch = detector.analyze_name_batch(names)

    assert detected.surname_first_count == 1
    assert detected.given_first_count == 1
    assert detected.total_count == 2
    assert not detected.threshold_met
    assert batch.format_pattern == detected
    assert [result.result for result in batch.results] == [detector.normalize_name(name).result for name in names]


def test_batch_format_votes_downweight_weak_candidate_gaps_and_count_duplicate_rows(detector):
    weak_surname_first = ParseCandidate(
        surname_tokens=["li"],
        given_tokens=["wei"],
        score=1.0,
        format=NameFormat.SURNAME_FIRST,
        original_compound_format=None,
    )
    weak_given_first = ParseCandidate(
        surname_tokens=["wei"],
        given_tokens=["li"],
        score=0.9,
        format=NameFormat.GIVEN_FIRST,
        original_compound_format=None,
    )
    strong_given_first = ParseCandidate(
        surname_tokens=["zhang"],
        given_tokens=["ming"],
        score=1.0,
        format=NameFormat.GIVEN_FIRST,
        original_compound_format=None,
    )
    strong_surname_first = ParseCandidate(
        surname_tokens=["ming"],
        given_tokens=["zhang"],
        score=0.6,
        format=NameFormat.SURNAME_FIRST,
        original_compound_format=None,
    )
    entries = [
        BatchCandidateEntry(
            "Li Wei",
            [weak_surname_first, weak_given_first],
            weak_surname_first,
            None,
            LATIN_ONLY_REPRESENTATION,
        ),
        BatchCandidateEntry(
            "Ming Zhang",
            [strong_given_first, strong_surname_first],
            strong_given_first,
            None,
            LATIN_ONLY_REPRESENTATION,
        ),
        BatchCandidateEntry(
            "ming zhang",
            [strong_given_first, strong_surname_first],
            strong_given_first,
            None,
            LATIN_ONLY_REPRESENTATION,
        ),
        BatchCandidateEntry(
            "Yan Wang",
            [strong_given_first, strong_surname_first],
            strong_given_first,
            None,
            LATIN_ONLY_REPRESENTATION,
        ),
    ]

    stats = detector._batch_analysis_service._collect_batch_vote_stats(entries)

    assert stats.surname_first_preferences == 1
    assert stats.given_first_preferences == 3
    assert stats.names_with_candidates == 4
    assert stats.surname_first_weight < stats.given_first_weight


def test_batch_analysis_service_is_detector_owned(detector):
    with pytest.raises(TypeError):
        BatchAnalysisService(detector._parsing_service, detector._ethnicity_service)


def test_single_chinese_participant_does_not_meet_batch_application_threshold(detector):
    names = ["Mai Liao", "John Smith", "Mary Johnson"]

    detected = detector.detect_batch_format(names)
    batch = detector.analyze_name_batch(names)

    assert detected.total_count == 1
    assert not detected.threshold_met
    assert batch.format_pattern == detected


def test_strong_given_first_batch_still_overrides_ambiguous_name(detector):
    names = ["Mai Liao", "Xin Liu", "Yang Li", "Wei Li", "Yan Mo"]

    batch = detector.analyze_name_batch(names)

    assert batch.format_pattern.dominant_format == NameFormat.GIVEN_FIRST
    assert batch.format_pattern.threshold_met
    assert batch.results[0].result == "Mai Liao"


@pytest.mark.parametrize(
    ("names", "expected_results"),
    [
        (["Li Yang Hsu", "Hanwei Cao"], ["Li-Yang Hsu", "Han-Wei Cao"]),
        (["Yunbo Hu", "Fei Yu"], ["Yun-Bo Hu", "Fei Yu"]),
        (["J. Liu", "Jing Wan"], ["J Liu", "Jing Wan"]),
        (["Qinggong Ping", "Hao Fei"], ["Qing-Gong Ping", "Hao Fei"]),
    ],
)
def test_batch_format_uses_weighted_tie_break_with_weak_participants(detector, names, expected_results):
    batch = detector.analyze_name_batch(names)

    assert batch.format_pattern.total_count == 2
    assert batch.format_pattern.dominant_format == NameFormat.GIVEN_FIRST
    assert batch.format_pattern.confidence == pytest.approx(0.5)
    assert batch.format_pattern.decision_confidence >= 0.55
    assert batch.format_pattern.decision_confidence >= batch.format_pattern.confidence
    assert batch.format_pattern.threshold_met
    assert [result.result for result in batch.results] == expected_results


def test_batch_does_not_apply_latin_format_to_spaced_han_name(detector):
    han_name = "\u67f3 \u71d5\u6885"
    names = [han_name, "Xin Liu", "Yang Li", "Wei Zhang", "Ming Wang"]

    individual = detector.normalize_name(han_name)
    batch = detector.analyze_name_batch(names)

    assert individual.success
    assert individual.result == "Yan-Mei Liu"
    assert batch.format_pattern.total_count == 4
    assert batch.results[0].result == individual.result
    assert batch.results[0].parsed_original_order.order == ["surname", "given"]
    assert batch.individual_analyses[0].candidates == []
    assert batch.individual_analyses[0].best_candidate is None


def test_batch_does_not_flip_compact_han_from_latin_votes(detector):
    han_name = "\u738b\u674e"
    names = [han_name, "Ming Li", "Wei Zhang"]

    individual = detector.normalize_name(han_name)
    batch = detector.analyze_name_batch(names)

    assert individual.success
    assert individual.result == "Li Wang"
    assert batch.format_pattern.total_count == 2
    assert batch.results[0].result == individual.result
    assert batch.results[0].parsed_original_order.order == ["surname", "given"]
    assert batch.individual_analyses[0].candidates == []
    assert batch.individual_analyses[0].best_candidate is None


def test_batch_does_not_treat_compact_mixed_han_roman_as_latin_only(detector):
    mixed_name = "\u5f20Wei"
    names = [mixed_name, "Xin Liu", "Yang Li", "Wei Li", "Yan Mo"]

    individual = detector.normalize_name(mixed_name)
    batch = detector.analyze_name_batch(names)
    evidence = batch.name_order_evidence[0]

    assert individual.success
    assert individual.result == "Wei Zhang"
    assert batch.results[0].result == individual.result
    assert evidence.script_representation == "mixed_script"
    assert evidence.batch_participant is False
    assert evidence.batch_applied is False
    assert batch.individual_analyses[0].candidates == []
    assert batch.individual_analyses[0].best_candidate is None


def test_aligned_bilingual_pairs_use_han_identity(detector):
    given_first = detector.normalize_name("Mi \u5bc6 Jiang \u848b")
    surname_first = detector.normalize_name("\u9ad8 Gao \u9759 Jing")
    han_only_given_first = detector.normalize_name("\u7acb Li \u9a6c Ma")
    roman_first_given_first = detector.normalize_name("Wang \u65fa Tian \u7530")

    assert given_first.success
    assert given_first.result == "Mi Jiang"
    assert given_first.parsed_original_order.order == ["given", "surname"]

    assert surname_first.success
    assert surname_first.result == "Jing Gao"
    assert surname_first.parsed_original_order.order == ["surname", "given"]

    assert han_only_given_first.success
    assert han_only_given_first.result == "Li Ma"
    assert han_only_given_first.parsed_original_order.order == ["given", "surname"]

    assert roman_first_given_first.success
    assert roman_first_given_first.result == "Wang Tian"
    assert roman_first_given_first.parsed_original_order.order == ["given", "surname"]


@pytest.mark.parametrize("lu_token", ["Lu", "L\u00fc"])
def test_aligned_bilingual_lu_alias_matches_lv_pinyin(detector, lu_token):
    result = detector.normalize_name(f"{lu_token} \u5415 Wei \u4f1f")

    assert result.success
    assert result.result == "Wei Lu"
    assert result.parsed.surname == "Lu"
    assert result.parsed.given_name == "Wei"
    assert result.parsed_original_order.order == ["surname", "given"]


def test_hyphenated_aligned_bilingual_normalizes_han_pinyin_parts(detector):
    raw_name = "Wei-chi \u5c09\u8fdf Wei \u4f1f"
    normalized = detector._normalizer.apply(raw_name)
    pairs = detector._normalizer.aligned_bilingual_pairs(normalized)

    assert detector._normalizer.classify_script_representation(normalized) == "bilingual_aligned"
    assert pairs is not None
    assert [pair.roman_token for pair in pairs] == ["Wei-chi", "Wei"]

    result = detector.normalize_name(raw_name)

    assert result.success
    assert result.result == "Wei Wei-Chi"
    assert result.parsed_original_order.order == ["surname", "given"]


def test_aligned_bilingual_middle_initial_preserves_original_order(detector):
    result = detector.normalize_name("Zhang \u5f20 Wei \u4f1f A \u963f")

    assert result.success
    assert result.result == "Wei A Zhang"
    assert result.parsed.middle_tokens == ["A"]
    assert result.parsed_original_order.middle_tokens == ["A"]
    assert result.parsed_original_order.order == ["surname", "given", "middle"]


def test_ambiguous_aligned_single_char_pairs_do_not_force_frequency_flip(detector):
    raw_name = "Lin \u6797 Gu \u8c37"

    individual = detector.normalize_name(raw_name)
    batch = detector.analyze_name_batch([raw_name, "Ming Li", "Wei Zhang"])

    assert individual.success
    assert individual.result == "Lin Gu"
    assert individual.parsed.surname == "Gu"
    assert individual.parsed.given_name == "Lin"
    assert individual.parsed_original_order.order == ["given", "surname"]
    assert batch.results[0].result == individual.result


def test_weak_roman_han_pair_uses_source_order_for_reported_single_char_flip(detector):
    result = detector.normalize_name("Zhong \u949f Shi \u65f6")

    assert result.success
    assert result.result == "Zhong Shi"
    assert result.parsed.surname == "Shi"
    assert result.parsed.given_name == "Zhong"
    assert result.parsed_original_order.order == ["given", "surname"]


def test_strong_first_surname_roman_han_pair_keeps_han_identity(detector):
    result = detector.normalize_name("Zhang \u5f20 Wei \u4f1f")

    assert result.success
    assert result.result == "Wei Zhang"
    assert result.parsed.surname == "Zhang"
    assert result.parsed.given_name == "Wei"
    assert result.parsed_original_order.order == ["surname", "given"]


def test_zero_last_surname_signal_roman_han_pair_keeps_han_identity(detector):
    result = detector.normalize_name("Hong \u6d2a Yan \u8273")

    assert result.success
    assert result.result == "Yan Hong"
    assert result.parsed.surname == "Hong"
    assert result.parsed.given_name == "Yan"
    assert result.parsed_original_order.order == ["surname", "given"]


def test_han_roman_aligned_single_char_pairs_keep_han_order(detector):
    result = detector.normalize_name("\u5b89 An \u9759 Jing")

    assert result.success
    assert result.result == "Jing An"
    assert result.parsed.surname == "An"
    assert result.parsed.given_name == "Jing"
    assert result.parsed_original_order.order == ["surname", "given"]


def test_aligned_bilingual_compound_surname_preserves_compact_roman_token(detector):
    result = detector.normalize_name("\u5efa Jian \u6b27\u9633 Ouyang")

    assert result.success
    assert result.result == "Jian Ouyang"
    assert result.parsed.surname == "Ouyang"
    assert result.parsed.surname_tokens == ["Ouyang"]
    assert result.parsed.given_name == "Jian"
    assert result.parsed_original_order.order == ["given", "surname"]


@pytest.mark.parametrize(
    ("raw_name", "expected", "expected_order"),
    [
        ("\u5434\u65ed WU Xu", "Xu Wu", ["surname", "given"]),
        ("\u90ed\u4e3d GUO Li", "Li Guo", ["surname", "given"]),
        ("\u5f20\u6d0b ZHANG Yang", "Yang Zhang", ["surname", "given"]),
        ("\u5f90\u601d\u5fd7 XU Sizhi", "Si-Zhi Xu", ["surname", "given"]),
        ("\u7acb\u9a6c Li Ma", "Li Ma", ["given", "surname"]),
        ("\u65fa\u7530 Wang Tian", "Wang Tian", ["given", "surname"]),
        ("\u5bc6\u848b Mi Jiang", "Mi Jiang", ["given", "surname"]),
    ],
)
def test_compact_han_roman_transliteration_uses_han_order(detector, raw_name, expected, expected_order):
    result = detector.normalize_name(raw_name)

    assert result.success
    assert result.result == expected
    assert result.parsed_original_order.order == expected_order


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("Haoran wang \u6d69\u7136\u738b", "Haoran Wang"),
        ("\u6d69\u7136\u738b Haoran Wang", "Haoran Wang"),
        ("\u738b\u6d69\u7136 Wang Haoran", "Haoran Wang"),
    ],
)
def test_compact_han_roman_transliteration_uses_stronger_endpoint_surname(detector, raw_name, expected):
    result = detector.normalize_name(raw_name)

    assert result.success
    assert result.result == expected
    assert result.parsed.surname == "Wang"
    assert result.parsed.given_name == "Haoran"


@pytest.mark.parametrize(
    ("raw_name", "expected", "expected_surname"),
    [
        ("\u53f8\u9a6c\u534e Sima Hua", "Hua Sima", "Sima"),
        ("\u8bf8\u845b\u65b9 Zhuge Fang", "Fang Zhuge", "Zhuge"),
        ("\u674e\u6b27\u9633 Li Ouyang", "Li Ouyang", "Ouyang"),
    ],
)
def test_compact_han_roman_transliteration_preserves_curated_compound_surnames(
    detector,
    raw_name,
    expected,
    expected_surname,
):
    result = detector.normalize_name(raw_name)

    assert result.success
    assert result.result == expected
    assert result.parsed.surname == expected_surname


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("\u5218\u5fb7 \u534e", "De-Hua Liu"),
        ("\u738b\u660e \u534e", "Ming-Hua Wang"),
        ("\u5f20\u4f1f \u6797", "Wei-Lin Zhang"),
    ],
)
def test_noisy_internal_space_in_three_character_han_name_preserves_surname_first_order(detector, raw_name, expected):
    result = detector.normalize_name(raw_name)

    assert result.success, f"{raw_name!r}: expected accepted name, got {result.error_message!r}"
    assert result.result == expected
    assert result.parsed_original_order.order == ["surname", "given"]


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("\u4e0a\u5b98 \u5a49\u513f", "Wan-Er Shang Guan"),
        ("\u4e1c\u65b9 \u4e0d\u8d25", "Bu-Bai Dong Fang"),
        ("\u6b27\u9633 \u4fee \u6587", "Xiu-Wen Ou Yang"),
        ("\u4e0a\u5b98\u5a49\u513f", "Wan-Er Shang Guan"),
        ("\u4e1c\u65b9\u4e0d\u8d25", "Bu-Bai Dong Fang"),
        ("\u6b27\u9633\u4fee\u6587", "Xiu-Wen Ou Yang"),
    ],
)
def test_han_compound_surname_suppresses_japanese_ml_rejection(detector, raw_name, expected):
    result = detector.normalize_name(raw_name)

    assert result.success, f"{raw_name!r}: expected accepted name, got {result.error_message!r}"
    assert result.result == expected
    assert result.parsed_original_order.order == ["surname", "given"]


class BrokenJapaneseProbabilityModel:
    """Model stub whose probability API fails after reporting availability."""

    classes_: ClassVar[list[str]] = ["cn", "jp"]

    def predict_proba(self, names):
        message = "boom"
        raise RuntimeError(message)

    def predict(self, names):
        return ["jp"]


def test_japanese_probability_degrades_when_loaded_model_errors(caplog):
    classifier = ethnicity._MLJapaneseClassifier(confidence_threshold=0.8)
    classifier._available = True
    classifier._model = BrokenJapaneseProbabilityModel()

    with caplog.at_level(logging.WARNING, logger=ethnicity.__name__):
        assert classifier.japanese_probability("\u5c71\u7530") == 0.0

    assert "ML Japanese classifier probability error" in caplog.text
    assert any(record.exc_info for record in caplog.records)


def test_non_person_inputs_are_rejected_before_parsing(detector):
    cases = [
        "\u5ef6\u5b89\u5927\u5b66 \u7269\u7406\u4e0e\u7535\u5b50\u4fe1\u606f\u5b66\u9662",
        "Tian Qiang Zhou Hui Zhu Rui",
    ]

    for raw_name in cases:
        result = detector.normalize_name(raw_name)
        assert not result.success
        assert result.error_message == NON_PERSON_FAILURE_REASON


@pytest.mark.parametrize(
    "raw_name",
    [
        "\u7269\u7406\u7cfb",
        "\u5b9e\u9a8c\u5ba4",
        "\u7814\u7a76\u6240",
        "\u5b66\u9662",
        "\u5927\u5b66",
        "\u516c\u53f8",
        "\u5317\u4eac\u5927\u5b66",
        "\u6e05\u534e\u5927\u5b66",
        "\u5f20\u4f1f\u5927\u5b66",
        "\u5f20\u4f1f(\u7269\u7406\u7cfb)",
    ],
)
def test_short_standalone_cjk_non_person_markers_are_rejected(detector, raw_name):
    result = detector.normalize_name(raw_name)

    assert not result.success
    assert result.error_message == NON_PERSON_FAILURE_REASON


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("\u5f20\u4f1f", "Wei Zhang"),
        ("\u738b\u529b", "Li Wang"),
        ("\u6b27\u9633\u4fee", "Xiu Ou Yang"),
        ("\u5f20 \u4f1f", "Wei Zhang"),
        ("\u5f20\u4f1f Zhang Wei", "Wei Zhang"),
    ],
)
def test_short_cjk_non_person_marker_gate_preserves_person_names(detector, raw_name, expected):
    result = detector.normalize_name(raw_name)

    assert result.success, f"{raw_name!r}: expected accepted name, got {result.error_message!r}"
    assert result.result == expected


def test_cjk_non_person_markers_do_not_match_internal_substrings(detector):
    result = detector.normalize_name("\u738b\u5927\u5b66\u95ee\u5bb6")

    if not result.success:
        assert result.error_message != NON_PERSON_FAILURE_REASON


def test_non_person_batch_rows_do_not_vote(detector):
    raw_name = "Tian Qiang Zhou Hui Zhu Rui"
    names = [raw_name, "Xin Liu", "Yang Li", "Wei Li"]

    detected = detector.detect_batch_format(names)
    batch = detector.analyze_name_batch(names)

    assert detected.total_count == 3
    assert batch.format_pattern.total_count == 3
    assert not batch.results[0].success
    assert batch.results[0].error_message == NON_PERSON_FAILURE_REASON


def test_rejected_latin_batch_rows_are_not_marked_batch_participants(detector):
    names = ["John Smith", "Wang An", "Yan Li", "Wu Gang"]

    batch = detector.analyze_name_batch(names)
    evidence = batch.name_order_evidence[0]

    assert batch.format_pattern.threshold_met
    assert not batch.results[0].success
    assert evidence.script_representation == "latin_only"
    assert evidence.batch_participant is False
    assert evidence.batch_applied is False
    assert evidence.batch_changed_format is False


@pytest.mark.parametrize("raw_name", ["", "   ", "!!!", "A" * 201])
def test_batch_initial_failures_match_individual_failures(detector, raw_name):
    individual = detector.normalize_name(raw_name)
    batch = detector.analyze_name_batch([raw_name, "Li Wei", "Zhang Ming", "Wang Hao"])

    result = batch.results[0]
    assert not individual.success
    assert not result.success
    assert result.error_message == individual.error_message


def test_non_person_author_list_gate_does_not_split_accented_person_name(detector):
    result = detector.normalize_name("Y\u00ec Xi\u00e1ng J. W\u00e1ng")

    if not result.success:
        assert result.error_message != NON_PERSON_FAILURE_REASON


def test_korean_specific_token_signal_is_capped(detector):
    ethnicity_service = detector._ethnicity_service
    tokens = ("Ha", "Young", "Lee")
    expanded_keys = ["ha", "young", "lee"]
    normalized_cache = {"Ha": "ha", "Young": "young", "Lee": "lee"}

    base_analysis = {
        "surname_type": "korean_overlapping",
        "hyphenated_tokens": [],
        "korean_specific_tokens": ["young"],
        "korean_ambiguous_tokens": [],
        "vietnamese_tokens": [],
        "korean_given_pairs": [],
        "has_thi_pattern": False,
    }
    one_token_score = ethnicity_service._calculate_korean_score_from_analysis(
        base_analysis,
        tokens,
        expanded_keys,
        normalized_cache,
    )

    two_token_analysis = dict(base_analysis)
    two_token_analysis["korean_specific_tokens"] = ["young", "seok"]
    two_token_score = ethnicity_service._calculate_korean_score_from_analysis(
        two_token_analysis,
        tokens,
        expanded_keys,
        normalized_cache,
    )

    three_token_analysis = dict(base_analysis)
    three_token_analysis["korean_specific_tokens"] = ["young", "seok", "jin"]
    three_token_score = ethnicity_service._calculate_korean_score_from_analysis(
        three_token_analysis,
        tokens,
        expanded_keys,
        normalized_cache,
    )

    assert one_token_score == 1.0
    assert two_token_score == 2.0
    assert three_token_score == 2.0


def test_batch_tie_break_heuristics_use_normalized_tokenization(detector):
    dummy_candidate = ParseCandidate(
        surname_tokens=["li"],
        given_tokens=["wei"],
        score=0.0,
        format=NameFormat.SURNAME_FIRST,
        original_compound_format=None,
    )
    names = ["XinLiu", "YangLi", "WeiLi"]
    name_candidates = [
        BatchCandidateEntry(name, [dummy_candidate], dummy_candidate, None, LATIN_ONLY_REPRESENTATION) for name in names
    ]

    assert all(len(name.split()) == 1 for name in names)
    dominant = detector._batch_analysis_service._apply_tie_breaking_heuristics(
        name_candidates,
        detector._normalizer,
        detector._data,
    )
    assert dominant == NameFormat.GIVEN_FIRST


def test_plausible_syllable_filter_stays_strict_but_non_breaking(detector):
    checker = detector._data_service._is_plausible_chinese_syllable

    assert checker("wei")
    assert checker("xiao")
    assert checker("ai")

    assert not checker("bcdfg")
    assert not checker("thrp")
    assert not checker("qwrty")


def test_alias_sensitive_regression_names_remain_stable(detector):
    cases = {
        "Cheng Wen": "Cheng Wen",
        "Mo Yan": "Yan Mo",
    }
    for raw_name, expected in cases.items():
        result = detector.normalize_name(raw_name)
        assert result.success
        assert result.result == expected


def test_kai_default_mapping_regression(detector):
    cases = {
        "Kai Tian": "Kai Tian",
        "Kai Li": "Kai Li",
    }
    for raw_name, expected in cases.items():
        result = detector.normalize_name(raw_name)
        assert result.success
        assert result.result == expected


def test_han_conversion_parity_across_all_chinese_flag(detector):
    normalizer = detector._normalizer
    token_groups = [
        ["李明"],
        ["王小红"],
        ["张Wei"],
    ]

    for tokens in token_groups:
        all_chinese_path = normalizer._process_mixed_tokens(tokens, is_all_chinese=True)
        mixed_path = normalizer._process_mixed_tokens(tokens, is_all_chinese=False)
        assert all_chinese_path == mixed_path


def test_spaced_all_chinese_given_first_preserves_group_boundary(detector):
    result = detector.normalize_name("\u589e\u53cb  \u53f6")

    assert result.success
    assert result.result == "Zeng-You Ye"
    assert result.parsed.surname == "Ye"
    assert result.parsed.given_name == "Zeng-You"
    assert result.parsed_original_order.order == ["given", "surname"]


def test_spaced_all_chinese_regular_surname_first_preserves_group_boundary(detector):
    result = detector.normalize_name("\u674e \u660e\u534e")

    assert result.success
    assert result.result == "Ming-Hua Li"
    assert result.parsed.surname == "Li"
    assert result.parsed.given_name == "Ming-Hua"
    assert result.parsed_original_order.order == ["surname", "given"]


@pytest.mark.parametrize(
    ("raw_name", "expected", "expected_surname"),
    [
        ("\u674e \u6d69\u7136", "Hao-Ran Li", "Li"),
        ("\u738b \u4fca\u6770", "Jun-Jie Wang", "Wang"),
        ("\u5f20 \u5b87\u8f69", "Yu-Xuan Zhang", "Zhang"),
    ],
)
def test_spaced_all_chinese_uses_stronger_prefix_surname_against_false_compound(
    detector,
    raw_name,
    expected,
    expected_surname,
):
    result = detector.normalize_name(raw_name)

    assert result.success
    assert result.result == expected
    assert result.parsed.surname == expected_surname
    assert result.parsed_original_order.order == ["surname", "given"]


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("\u738b \u6b27\u9633", "Wang Ou Yang"),
        ("\u674e \u6b27\u9633", "Li Ou Yang"),
        ("\u6797 \u8bf8\u845b", "Lin Zhu Ge"),
    ],
)
def test_spaced_all_chinese_prefers_trailing_compound_surname(detector, raw_name, expected):
    result = detector.normalize_name(raw_name)

    assert result.success
    assert result.result == expected
    assert result.parsed_original_order.order == ["given", "surname"]


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("\u53f8 \u9a6c\u534e", "Hua Si Ma"),
        ("\u6b27 \u9633\u660e", "Ming Ou Yang"),
        ("\u8bf8 \u845b\u4eae", "Liang Zhu Ge"),
    ],
)
def test_spaced_all_chinese_preserves_compound_surname_boundary(detector, raw_name, expected):
    result = detector.normalize_name(raw_name)

    assert result.success
    assert result.result == expected
    assert result.parsed_original_order.order == ["surname", "given"]


def test_name_data_structures_mapping_fields_are_immutable(detector):
    data = detector._data
    mapping_fields = [
        "surname_frequencies",
        "surname_log_probabilities",
        "given_log_probabilities",
        "surname_percentile_ranks",
        "compound_hyphen_map",
        "compound_original_format_map",
    ]

    for field_name in mapping_fields:
        field_value = getattr(data, field_name)
        assert isinstance(field_value, Mapping)

        with pytest.raises(TypeError):
            field_value["mutation_probe"] = 1.0
