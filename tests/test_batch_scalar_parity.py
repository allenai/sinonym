"""Regression coverage for detector-owned policy shared by scalar and batch paths."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from sinonym.coretypes import NameFormat

if TYPE_CHECKING:
    from sinonym import ChineseNameDetector


@pytest.mark.parametrize(
    ("raw_name", "expected", "classification_tokens"),
    [
        ("Et al. Wang Wei", "Wei Wang", ["Wang", "Wei"]),
        ("Et al. Zhang Wei", "Wei Zhang", ["Zhang", "Wei"]),
        ("Et al. DNP Li", "Dnp Li", ["DNP", "Li"]),
        ("Et al. Li MS", "M.-S. Li", ["Li", "MS"]),
    ],
)
def test_leading_et_al_batch_uses_scalar_classification_surface(
    detector: ChineseNameDetector,
    raw_name: str,
    expected: str,
    classification_tokens: list[str],
) -> None:
    """Batch preparation drops the audited marker without losing raw-row identity."""
    scalar = detector.normalize_name(raw_name)
    batch = detector.analyze_name_batch([raw_name])
    selected = batch.results[0]
    evidence = batch.name_order_evidence[0]

    assert scalar.success
    assert selected.success
    assert scalar.result == selected.result == expected
    assert scalar.canonical_name is not None
    assert selected.canonical_name is not None
    assert scalar.canonical_name.normalized == selected.canonical_name.normalized
    assert batch.names == [raw_name]
    assert evidence.raw_name == raw_name
    assert evidence.raw_tokens == classification_tokens


@pytest.mark.parametrize(
    ("raw_name", "expected", "expected_format"),
    [
        ("Ou Yang Ouyang", "Ou-Yang Ouyang", NameFormat.GIVEN_FIRST),
        ("Ouyang Ou Yang", "Ou-Yang Ouyang", NameFormat.SURNAME_FIRST),
        ("Zhu Ge Zhuge", "Zhu-Ge Zhuge", NameFormat.GIVEN_FIRST),
        ("Zhuge Zhu Ge", "Zhu-Ge Zhuge", NameFormat.SURNAME_FIRST),
    ],
)
def test_compound_singleton_batch_uses_scalar_candidate_tie_break(
    detector: ChineseNameDetector,
    raw_name: str,
    expected: str,
    expected_format: NameFormat,
) -> None:
    """Equal-score compound parses resolve identically without batch context."""
    scalar = detector.normalize_name(raw_name)
    batch = detector.analyze_name_batch([raw_name])
    selected = batch.results[0]
    analysis = batch.individual_analyses[0]
    evidence = batch.name_order_evidence[0]

    assert not batch.format_pattern.threshold_met
    assert batch.format_pattern.dominant_format is expected_format
    assert scalar.success
    assert selected.success
    assert scalar.result == selected.result == expected
    assert scalar.parsed is not None
    assert scalar.parsed == selected.parsed
    assert scalar.parsed_original_order is not None
    assert scalar.parsed_original_order == selected.parsed_original_order
    assert scalar.canonical_name is not None
    assert selected.canonical_name is not None
    assert scalar.canonical_name.normalized == selected.canonical_name.normalized
    assert analysis.best_candidate is not None
    assert analysis.best_candidate.format is expected_format
    assert evidence.individual_format is expected_format
    assert evidence.selected_format is expected_format
    assert evidence.batch_applied is False
    assert evidence.batch_changed_format is False


def test_leading_et_al_batch_vote_uses_the_cleaned_name(detector: ChineseNameDetector) -> None:
    """Citation debris cannot cast an order vote opposite to the cleaned name."""
    batch = detector.analyze_name_batch(["Et al. Zhang Wei", "Li Ming"])

    assert batch.format_pattern.dominant_format is NameFormat.SURNAME_FIRST
    assert batch.format_pattern.surname_first_count == 2
    assert batch.format_pattern.given_first_count == 0
    assert batch.format_pattern.threshold_met
    assert [result.result for result in batch.results] == ["Wei Zhang", "Ming Li"]


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("Jungting Chen", "Jung-Ting Chen"),
        ("Tsung-Jr Wu", "Tsung-Jr Wu"),
    ],
)
def test_contextual_taiwan_name_keeps_scalar_policy_under_batch_context(
    detector: ChineseNameDetector,
    raw_name: str,
    expected: str,
) -> None:
    """Exact Taiwan evidence stays given-first under opposing peer order."""
    scalar = detector.normalize_name(raw_name)
    batch = detector.analyze_name_batch([raw_name, "Zhang Wei", "Li Ming", "Wang Hao"])
    selected = batch.results[0]
    evidence = batch.name_order_evidence[0]

    assert batch.format_pattern.dominant_format is NameFormat.SURNAME_FIRST
    assert batch.format_pattern.threshold_met
    assert scalar.success
    assert selected.success
    assert scalar.result == selected.result == expected
    assert scalar.canonical_name is not None
    assert selected.canonical_name is not None
    assert scalar.canonical_name.normalized == selected.canonical_name.normalized
    assert evidence.batch_participant
    assert not evidence.batch_applied
    assert evidence.selected_format is NameFormat.GIVEN_FIRST


@pytest.mark.parametrize("raw_name", ["Jungting Kim", "Tsung-Jr Kim"])
def test_contextual_taiwan_batch_keeps_korean_collision_controls(
    detector: ChineseNameDetector,
    raw_name: str,
) -> None:
    """Reusing Taiwan policy must retain its directional-Korean exclusion."""
    scalar = detector.normalize_name(raw_name)
    batch = detector.analyze_name_batch([raw_name]).results[0]

    assert not scalar.success
    assert not batch.success
    assert batch.error_message == scalar.error_message
