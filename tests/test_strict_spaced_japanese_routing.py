"""Hard-routing contracts for strict spaced native Japanese names."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from sinonym.coretypes import CanonicalName
from sinonym.coretypes.routing_resolution import (
    EAST_ASIAN_EVIDENCE_RESOLUTION_REASONS,
    ApplyAssignment,
    EastAsianEvidenceReason,
    HardScalarMaterializationFailure,
    PreserveBaseline,
    ResolutionAction,
    ResolutionProvenance,
    ResolutionReason,
    east_asian_evidence_resolution_reason,
    resolution_decision_spec,
)
from sinonym.services.east_asian_name_order import (
    EastAsianNameOrderDecision,
    EastAsianNameOrderPreservation,
    EastAsianNameOrderService,
)
from sinonym.timo.interface import Instance, Predictor, SourceAuthorFields

if TYPE_CHECKING:
    from sinonym.detector import ChineseNameDetector

SPACED_CJK_GOLD = Path(__file__).parent / "data" / "spaced_cjk_name_order_gold.json"
TERMINAL_EAST_ASIAN_EVIDENCE_REASONS = (
    EastAsianEvidenceReason.JAPANESE_ITERATION_MARK_ONE_SIDED_EXCLUSIVE,
    EastAsianEvidenceReason.JAPANESE_ITERATION_MARK_DUAL_EXCLUSIVE,
    EastAsianEvidenceReason.JAPANESE_NATIVE_SPACED_STRICT_FAMILY_FIRST,
    EastAsianEvidenceReason.JAPANESE_NATIVE_SPACED_STRICT_GIVEN_FIRST,
    EastAsianEvidenceReason.IDENTITY_BACKED_EXACT_FULL_SURFACE,
    EastAsianEvidenceReason.KOREAN_WESTERN_SUFFIX_CONFLICT,
)


def test_strict_spaced_native_directions_return_typed_hard_evidence() -> None:
    service = EastAsianNameOrderService()

    family_first = service.infer_resolution(
        "佐藤 優",
        japanese_probability=lambda _surface: 1.0,
    )
    given_first = service.infer_resolution(
        "優 佐藤",
        japanese_probability=lambda _surface: 1.0,
    )

    assert isinstance(family_first, EastAsianNameOrderDecision)
    assert family_first.reason is EastAsianEvidenceReason.JAPANESE_NATIVE_SPACED_STRICT_FAMILY_FIRST
    assert (family_first.first_name, family_first.last_name) == ("優", "佐藤")
    assert isinstance(given_first, EastAsianNameOrderPreservation)
    assert given_first.reason is EastAsianEvidenceReason.JAPANESE_NATIVE_SPACED_STRICT_GIVEN_FIRST


def test_spaced_japanese_classifier_input_preserves_the_deployed_spacing_contract() -> None:
    classifier_inputs: list[str] = []

    resolution = EastAsianNameOrderService().infer_resolution(
        "佐藤 優",
        japanese_probability=lambda surface: classifier_inputs.append(surface) or 1.0,
    )

    assert resolution is not None
    assert classifier_inputs == ["佐藤 優"]


@pytest.mark.parametrize(
    "surface",
    [
        "上田 官治",  # one-sided surname evidence
        "光 彩乃",  # both endpoints are given-name plausible
        "後藤 丹治",  # both endpoints are surname plausible
    ],
)
def test_broad_spaced_native_routes_remain_soft(
    detector: ChineseNameDetector,
    surface: str,
) -> None:
    evidence = detector._east_asian_name_order.infer_resolution(  # noqa: SLF001
        surface,
        japanese_probability=detector._ethnicity_service.japanese_probability,  # noqa: SLF001
    )
    scalar = detector.routing_scalar_resolution(surface)

    assert isinstance(evidence, EastAsianNameOrderDecision)
    assert evidence.reason is EastAsianEvidenceReason.JAPANESE_NATIVE_SPACED_DICTIONARY
    assert isinstance(scalar, CanonicalName)


def test_strict_hard_slices_remain_perfect_on_frozen_blind_gold(
    detector: ChineseNameDetector,
) -> None:
    payload = json.loads(SPACED_CJK_GOLD.read_text(encoding="utf-8"))
    counts: Counter[EastAsianEvidenceReason] = Counter()
    errors: list[str] = []
    expected_family = {
        EastAsianEvidenceReason.JAPANESE_NATIVE_SPACED_STRICT_FAMILY_FIRST: "first",
        EastAsianEvidenceReason.JAPANESE_NATIVE_SPACED_STRICT_GIVEN_FIRST: "second",
    }

    for item in payload["items"]:
        if item["family"] not in {"first", "second"}:
            continue
        resolution = detector._east_asian_name_order.infer_resolution(  # noqa: SLF001
            item["name"],
            japanese_probability=detector._ethnicity_service.japanese_probability,  # noqa: SLF001
        )
        if resolution is None or resolution.reason not in expected_family:
            continue
        counts[resolution.reason] += 1
        if item["family"] != expected_family[resolution.reason]:
            errors.append(item["name"])

    assert counts == {
        EastAsianEvidenceReason.JAPANESE_NATIVE_SPACED_STRICT_FAMILY_FIRST: 134,
        EastAsianEvidenceReason.JAPANESE_NATIVE_SPACED_STRICT_GIVEN_FIRST: 60,
    }
    assert errors == []


@pytest.mark.parametrize("vys_other_names", [[], ["Jane Doe"]], ids=["pp-only", "pp-vys"])
@pytest.mark.parametrize(
    "case",
    [
        (
            SourceAuthorFields(first_name="植松", last_name="康"),
            ("康", "", "植松", None),
            ResolutionReason.JAPANESE_NATIVE_SPACED_STRICT_ASSIGNMENT,
            ResolutionAction.ASSIGN,
        ),
        (
            SourceAuthorFields(first_name="裕", last_name="吉川"),
            ("裕", "", "吉川", None),
            ResolutionReason.JAPANESE_NATIVE_SPACED_STRICT_GIVEN_FIRST_PRESERVE_INPUT,
            ResolutionAction.PRESERVE_INPUT,
        ),
    ],
    ids=["family-first", "given-first"],
)
def test_timo_pp_cannot_overwrite_strict_spaced_native_evidence(
    predictor: Predictor,
    vys_other_names: list[str],
    case: tuple[
        SourceAuthorFields,
        tuple[str, str, str, None],
        ResolutionReason,
        ResolutionAction,
    ],
) -> None:
    source, expected, reason, action = case
    (prediction,) = predictor.predict_batch(
        [Instance(pp_authors=[source], vys_other_names=vys_other_names)],
    )
    (resolved,) = prediction.authors

    assert (resolved.first_name, resolved.middle_names, resolved.last_name, resolved.suffix) == expected
    assert resolved.resolution_provenance is ResolutionProvenance.SCALAR
    assert resolved.resolution_action is action
    assert resolved.resolution_reason is reason


def test_detector_materializes_the_strict_evidence_as_hard_constraints(
    detector: ChineseNameDetector,
) -> None:
    family_first = detector.routing_scalar_resolution("植松 康")
    given_first = detector.routing_scalar_resolution("裕 吉川")

    assert isinstance(family_first, ApplyAssignment)
    assert family_first.reason is ResolutionReason.JAPANESE_NATIVE_SPACED_STRICT_ASSIGNMENT
    assert isinstance(given_first, PreserveBaseline)
    assert given_first.reason is ResolutionReason.JAPANESE_NATIVE_SPACED_STRICT_GIVEN_FIRST_PRESERVE_INPUT


@pytest.mark.parametrize(
    ("raw_name", "expected_text", "expected_given", "expected_surname"),
    [
        ("黒澤 明", "明 黒澤", "明", "黒澤"),
        ("植松 康", "康 植松", "康", "植松"),
        ("浩 吉田", "浩 吉田", "浩", "吉田"),
        ("晶 首藤", "晶 首藤", "晶", "首藤"),
    ],
)
def test_terminal_japanese_evidence_precedes_chinese_across_public_ingress(
    detector: ChineseNameDetector,
    raw_name: str,
    expected_text: str,
    expected_given: str,
    expected_surname: str,
) -> None:
    first_name, last_name = raw_name.split()
    direct = detector.normalize_person_name(raw_name)
    sidecar = detector.normalize_name(raw_name).canonical_name
    structured = detector.normalize_person_name_components(first_name=first_name, last_name=last_name)

    for canonical in (direct, sidecar, structured):
        assert canonical is not None
        assert canonical.text == expected_text
        assert canonical.normalized.given_name == expected_given
        assert canonical.normalized.surname == expected_surname


@pytest.mark.parametrize(
    "evidence_reason",
    TERMINAL_EAST_ASIAN_EVIDENCE_REASONS,
    ids=lambda reason: reason.value,
)
def test_every_mapped_terminal_evidence_reason_uses_the_shared_materializer(
    detector: ChineseNameDetector,
    evidence_reason: EastAsianEvidenceReason,
) -> None:
    assert set(TERMINAL_EAST_ASIAN_EVIDENCE_REASONS) == {
        reason for reason, resolution_reason in EAST_ASIAN_EVIDENCE_RESOLUTION_REASONS.items() if resolution_reason is not None
    }
    baseline = detector._person_name_baseline("Family Given")  # noqa: SLF001
    assert baseline is not None
    resolution_reason = east_asian_evidence_resolution_reason(evidence_reason)
    assert resolution_reason is not None

    if resolution_decision_spec(resolution_reason).action is ResolutionAction.ASSIGN:
        resolution = EastAsianNameOrderDecision(
            surface="Family Given",
            given_tokens=("Given",),
            middle_tokens=(),
            surname_tokens=("Family",),
            source_order=("surname", "given"),
            reason=evidence_reason,
        )
        expected_type = ApplyAssignment
    else:
        resolution = EastAsianNameOrderPreservation(
            surface="Given Family",
            reason=evidence_reason,
        )
        expected_type = PreserveBaseline

    materialized = detector._materialize_terminal_east_asian_resolution(  # noqa: SLF001
        baseline,
        resolution,
    )

    assert isinstance(materialized, expected_type)
    assert materialized.evidence_reason is evidence_reason


def test_failed_terminal_assignment_preserves_public_baselines_and_remains_typed_for_timo(
    detector: ChineseNameDetector,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_name = "黒澤 明"
    baseline = detector._person_name_baseline(raw_name)  # noqa: SLF001
    assert baseline is not None
    resolution = EastAsianNameOrderDecision(
        surface=raw_name,
        given_tokens=("明",),
        middle_tokens=(),
        surname_tokens=("黒澤",),
        source_order=("surname", "given"),
        reason=EastAsianEvidenceReason.JAPANESE_NATIVE_SPACED_STRICT_FAMILY_FIRST,
    )
    monkeypatch.setattr(
        detector,
        "_infer_east_asian_name_order_resolution",
        lambda _surface, *, legacy_raw_name: resolution,
    )
    monkeypatch.setattr(detector, "_canonical_name_from_order_decision", lambda _baseline, _resolution: None)
    monkeypatch.setattr(
        detector,
        "_canonical_chinese_name_with_source",
        lambda *_args, **_kwargs: pytest.fail("direct canonicalization fell through to Chinese"),
    )
    monkeypatch.setattr(
        detector,
        "_canonical_name_from_chinese_result",
        lambda *_args, **_kwargs: pytest.fail("canonical sidecar fell through to Chinese"),
    )

    assert detector.normalize_person_name(raw_name) == baseline
    assert detector.normalize_name(raw_name).canonical_name == baseline
    with pytest.raises(HardScalarMaterializationFailure):
        detector.routing_scalar_resolution(raw_name)
