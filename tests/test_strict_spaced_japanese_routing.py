"""Hard-routing contracts for strict spaced native Japanese names."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from sinonym.coretypes import CanonicalName
from sinonym.coretypes.routing_resolution import (
    ApplyAssignment,
    EastAsianEvidenceReason,
    PreserveBaseline,
    ResolutionAction,
    ResolutionProvenance,
    ResolutionReason,
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
