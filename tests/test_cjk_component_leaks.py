"""CJK letters leaking into routed component fields.

The Chinese path romanizes what it parses, so a `success=true` row whose components still
carry a CJK letter is a mis-segmentation by construction: Japanese names read as Chinese with
the unmapped U+FA11 `﨑` left as a literal middle token (`田﨑 修 (Osamu Tasaki)` -> given `Xiu`,
middle `﨑`, surname `Tian`), and dual-name rows whose Han surname char lands in the middle
field (`劉美慧 Wen-Hua Chen` -> middle `劉`). The full corpus set is 211 rows, frozen in
`tests/data/cjk_trio_leaks.json`.

The tests assert the desired behaviour — such rows must be rejected, and a mixed-script
canonical must not be exposed — and fail today; the failures are allowlisted in the
scripts/check_test_status.py baseline. All-CJK output from all-CJK input (`김효진` -> given
`효진`, surname `김`) is NOT a leak — those are correct segmentations and must keep working.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sinonym.timo.interface import PredictorConfig, RoutingInstance, RoutingPredictorV2
from tests._case_assertions import (
    assert_no_canonical_name,
    assert_person_normalized_name,
    assert_rejected,
    assert_routed_rejection,
)


@pytest.fixture(scope="module")
def routing_predictor() -> RoutingPredictorV2:
    return RoutingPredictorV2(config=PredictorConfig(parallel="never"), artifacts_dir=".")


def _cases():
    path = Path(__file__).resolve().parent / "data" / "cjk_trio_leaks.json"
    return json.loads(path.read_text(encoding="utf-8"))["cases"]


CASES = _cases()


def _route_one(predictor, raw):
    return predictor.predict_batch([RoutingInstance(pp_names=[raw])])[0].authors[0]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["raw"])
def test_a_row_whose_components_would_carry_a_cjk_letter_is_rejected(routing_predictor, case):
    author = _route_one(routing_predictor, case["raw"])

    assert_routed_rejection(author, case["raw"])


@pytest.mark.parametrize(
    ("raw", "given", "surname"),
    [
        # All-CJK segmentations of all-CJK input are correct output, not leaks: the corpus
        # holds these as one blob in a single field, and the split is the improvement.
        ("김효진", "효진", "김"),
        ("吉田 隆", "隆", "吉田"),
    ],
)
def test_all_cjk_segmentation_of_all_cjk_input_is_kept(routing_predictor, raw, given, surname):
    author = _route_one(routing_predictor, raw)

    assert not author.success
    canonical = author.canonical_name
    assert canonical is not None
    assert canonical.normalized.given_name == given
    assert canonical.normalized.surname == surname


@pytest.mark.parametrize(
    "raw",
    [
        # U+FA11 﨑 is a compatibility ideograph with no Unicode decomposition, folded to 崎 in
        # fix_ocr_artifacts; the classification gate applies the same fold, so these reach the
        # ML classifier as the 崎 forms it knows and are rejected as Japanese at the library
        # level, not merely by the serving-interface guard.
        ("田﨑 修"),
        ("野﨑 涼太朗"),
        ("山﨑 洋輔"),
        ("汐﨑 綾子"),
        ("田崎 修"),
    ],
)
def test_compatibility_ideograph_saki_names_classify_as_japanese(detector, raw):
    assert_rejected(detector, raw)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # The family-first rule for spaced kanji names folds compatibility ideographs for its
        # surname/given lookups, so 田﨑 is recognised as the surname 田崎; the emitted tokens
        # keep the glyph the author wrote.
        ("田﨑 修", "修 田﨑"),
        ("岩﨑 一郎", "一郎 岩﨑"),
        ("岡﨑 惠美子", "惠美子 岡﨑"),
        ("田崎 修", "修 田崎"),
        # Given-first input needs no flip; the default reading is already right.
        ("涼太朗 野﨑", "涼太朗 野﨑"),
    ],
)
def test_spaced_kanji_family_first_recognises_compatibility_ideographs(detector, raw, expected):
    person = detector.normalize_person_name(raw)

    assert_person_normalized_name(person, raw, expected)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Expected failures, in the check_test_status baseline: 濱崎/間崎 are not in the
        # Japanese surname asset (濱 is itself a variant of 浜, a second-order fold), so the
        # conservative family-first rule declines and the default reading keeps the surname in
        # the given field.
        ("濱﨑 将臣", "将臣 濱﨑"),
        ("間﨑 光", "光 間﨑"),
    ],
)
def test_spaced_kanji_family_first_surnames_missing_from_the_asset(detector, raw, expected):
    person = detector.normalize_person_name(raw)

    assert_person_normalized_name(person, raw, expected)


@pytest.mark.parametrize(
    "raw",
    [
        # Mixed-script canonical leaks: the person path re-segments across fields, moving the
        # Latin surname out of the surname field (`A Ra 아라 Cho 조` -> given `A`, middle
        # `Ra 아라 Cho`, surname `조`). The consumer overwrites its own fields with that, so no
        # canonical at all is the desired output.
        "A Ra 아라 Cho 조",
        "Yuriko 由利子 Doi 土井",
    ],
)
def test_a_mixed_script_canonical_is_not_exposed(routing_predictor, raw):
    author = _route_one(routing_predictor, raw)

    assert not author.success
    assert_no_canonical_name(author, raw)
