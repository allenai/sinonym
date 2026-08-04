"""Versioning and canonical-name coverage for TIMO v2."""

import hashlib
import json
from pathlib import Path

import pytest

from sinonym.timo.interface import (
    BatchPrediction,
    BatchSummary,
    Instance,
    PPRoutedPrediction,
    Prediction,
    PredictionV2,
    Predictor,
    PredictorConfig,
    PredictorV2,
    RoutedPaperPrediction,
    RoutedPaperPredictionV2,
    RoutedPrediction,
    RoutingInstance,
    RoutingPredictorV2,
    TimoModel,
)

V1_SCHEMA_FINGERPRINTS = {
    "Prediction": "32ec52f7cae59d443bd8c221db6a7fc8f06dd5658978e54d264f919820807775",
    "BatchPrediction": "3f3ccaa1fc9185296f6bb0f68acbe087da4aa900326eedc325e4505dfc3b15c2",
    "BatchSummary": "01640c80b212f7f58dc3f5c0d83c876c60864dad88ec77c0c4ac326c0fee0f97",
    "RoutedPrediction": "859a0b5c5f57061ee6850cebbc623667ae3a6b6e4e0d5b952f8d4774da7f5f22",
    "PPRoutedPrediction": "ed2febe81a511784ece36c52a17f57dbbbe4b57ff64de9c68c205e47a348debf",
    "RoutedPaperPrediction": "b45280f7abb5af523f30d129907e4ff09e0c2932074a23de5d4deca73bcf8738",
}


@pytest.fixture(scope="module")
def predictor_v1() -> Predictor:
    return Predictor(config=PredictorConfig(parallel="never"), artifacts_dir=".")


@pytest.fixture(scope="module")
def predictor_v2() -> PredictorV2:
    return PredictorV2(config=PredictorConfig(parallel="never"), artifacts_dir=".")


def _schema_fingerprint(model: type[TimoModel]) -> str:
    payload = json.dumps(model.schema(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _without_canonical(value):
    if isinstance(value, list):
        return [_without_canonical(item) for item in value]
    if isinstance(value, dict):
        return {key: _without_canonical(item) for key, item in value.items() if key != "canonical_name"}
    return value


def _assert_steve_marsh(prediction: PredictionV2) -> None:
    assert not prediction.success
    assert prediction.given_name is None
    assert prediction.surname is None
    assert prediction.canonical_name is not None
    assert prediction.canonical_name.text == "Steve Marsh"
    assert prediction.canonical_name.normalized.given_name == "Steve"
    assert prediction.canonical_name.normalized.middle_name == ""
    assert prediction.canonical_name.normalized.surname == "Marsh"
    assert prediction.canonical_name.normalized.suffix == ""


def _apply_routed_fields(result, raw_name: str) -> tuple[str, str, str]:
    """Mirror Scholar's routed -> canonical -> source apply precedence."""
    if result.success and result.surname:
        return result.given_name or "", result.middle_name or "", result.surname
    if result.canonical_name is not None and result.canonical_name.normalized.surname:
        normalized = result.canonical_name.normalized
        return normalized.given_name or "", normalized.middle_name or "", normalized.surname
    tokens = raw_name.split()
    return (" ".join(tokens[:-1]), "", tokens[-1]) if tokens else ("", "", "")


def test_v1_schema_fingerprints_match_main_before_v2() -> None:
    """Adding v2 types must not mutate any existing TIMO response schema."""
    models = (Prediction, BatchPrediction, BatchSummary, RoutedPrediction, PPRoutedPrediction, RoutedPaperPrediction)

    assert {model.__name__: _schema_fingerprint(model) for model in models} == V1_SCHEMA_FINGERPRINTS


def test_v1_dict_excludes_canonical_name(predictor_v1: Predictor) -> None:
    (prediction,) = predictor_v1.predict_batch([Instance(name="Dr. Steve Marsh PhD")])

    assert prediction.dict() == {
        "success": False,
        "error_message": "no Chinese evidence found",
        "given_name": None,
        "surname": None,
        "middle_name": None,
        "confidence": 0.0,
        "format_pattern": {
            "dominant_format": "mixed",
            "confidence": 0.0,
            "decision_confidence": 0.0,
            "surname_first_count": 0,
            "given_first_count": 0,
            "total_count": 0,
            "voting_count": 0,
            "vote_margin_count": 0,
            "vote_margin": 0.0,
            "threshold_met": False,
        },
    }


def test_v2_flat_prediction_surfaces_non_chinese_components_and_suffix(predictor_v2: PredictorV2) -> None:
    steve_marsh, steve_blando = predictor_v2.predict_batch(
        [Instance(name="Dr. Steve Marsh PhD"), Instance(name="Steve Blando IV")],
    )

    _assert_steve_marsh(steve_marsh)
    assert steve_blando.canonical_name is not None
    assert steve_blando.canonical_name.text == "Steve Blando IV"
    assert steve_blando.canonical_name.normalized.given_name == "Steve"
    assert steve_blando.canonical_name.normalized.surname == "Blando"
    assert steve_blando.canonical_name.normalized.suffix == "IV"
    assert steve_blando.canonical_name.normalized.suffix_tokens == ["IV"]
    assert steve_blando.canonical_name.normalized.order == ["given", "surname", "suffix"]


def test_v2_batch_and_process_helpers_retain_canonical_name(predictor_v2: PredictorV2) -> None:
    name = "Dr. Steve Marsh PhD"

    _assert_steve_marsh(predictor_v2.process_name_batch([name])[0])
    _assert_steve_marsh(predictor_v2.process_name_batches([[name]], parallel="never")[0][0])
    _assert_steve_marsh(predictor_v2.analyze_name_batch([name]).results[0])
    _assert_steve_marsh(predictor_v2.score_name_batch([name]).results[0])


def test_v2_multiprocess_helper_retains_canonical_name(predictor_v2: PredictorV2) -> None:
    (prediction,) = predictor_v2.process_name_batch_multiprocess(
        ["Dr. Steve Marsh PhD"],
        max_workers=1,
        chunk_size=1,
    )

    _assert_steve_marsh(prediction)


def test_v2_pp_only_routing_retains_canonical_without_changing_decision(
    predictor_v1: Predictor,
    predictor_v2: PredictorV2,
) -> None:
    names = ["Steve Blando IV", "Yue Lin", "Wei Wang"]
    v1 = predictor_v1.route_pp(names)
    v2 = predictor_v2.route_pp(names)

    assert [_without_canonical(result.dict()) for result in v2] == [result.dict() for result in v1]
    assert v2[0].canonical_name is not None
    assert v2[0].canonical_name.normalized.suffix == "IV"
    assert v2[0].pp.canonical_name == v2[0].canonical_name


def test_v2_pp_vys_routing_retains_canonical_without_changing_decision(
    predictor_v1: Predictor,
    predictor_v2: PredictorV2,
) -> None:
    pp_names = ["Steve Blando IV", "Yue Lin"]
    pool = [*pp_names, "Wei Wang", "Jun Zhao", "Hui Li", "Tao Sun"]
    v1 = predictor_v1.route_pp_vys(pp_names, pool)
    v2 = predictor_v2.route_pp_vys(pp_names, pool)

    assert [_without_canonical(result.dict()) for result in v2] == [result.dict() for result in v1]
    assert v2[0].canonical_name is not None
    assert v2[0].canonical_name.normalized.suffix == "IV"
    assert v2[0].pp.canonical_name is not None
    assert v2[0].vys is not None
    assert v2[0].vys.canonical_name is not None


@pytest.mark.parametrize(
    ("raw_name", "expected", "reason"),
    [
        ("佐々木克典", ("克典", "", "佐々木"), "japanese_iteration_mark_canonical_override"),
        ("克典 佐々木", ("克典", "", "佐々木"), "japanese_iteration_mark_canonical_override"),
        ("純 野々崎", ("純", "", "野々崎"), "japanese_iteration_mark_canonical_override"),
        ("Kim Stene-Larsen", ("Kim", "", "Stene-Larsen"), "korean_western_suffix_conflict"),
    ],
)
def test_v2_canonical_override_controls_final_pp_and_pp_vys_output(
    raw_name: str,
    expected: tuple[str, str, str],
    reason: str,
) -> None:
    predictor = RoutingPredictorV2(config=PredictorConfig(parallel="never"), artifacts_dir=".")
    pool = [raw_name, "Li Xiaoming", "Chen Jianguo", "Zhao Yuting", "Sun Haoran"]
    pp_only_paper, pooled_paper = predictor.predict_batch(
        [
            RoutingInstance(pp_names=[raw_name]),
            RoutingInstance(pp_names=[raw_name], vys_pool_names=pool),
        ],
    )
    pp_only = pp_only_paper.authors[0]
    pooled = pooled_paper.authors[0]

    for result in (pp_only, pooled):
        assert result.router_prediction.value == "abstain"
        assert result.router_reason == reason
        assert not result.success
        assert (result.given_name, result.middle_name, result.surname) == (None, None, None)
        assert result.canonical_name is not None
        normalized = result.canonical_name.normalized
        assert (normalized.given_name, normalized.middle_name, normalized.surname) == expected
        assert _apply_routed_fields(result, raw_name) == expected
    assert pooled.input_order_candidate.value == "unknown"


@pytest.mark.parametrize("raw_name", ["三谷 奈々", "Kim Hŭisŏn", "Hồ Anderson", "Cho Arison"])
def test_v2_canonical_override_leaves_counterexamples_on_existing_route(raw_name: str) -> None:
    predictor = RoutingPredictorV2(config=PredictorConfig(parallel="never"), artifacts_dir=".")

    result = predictor.route_pp([raw_name])[0]

    assert result.router_reason not in {
        "japanese_iteration_mark_canonical_override",
        "korean_western_suffix_conflict",
    }


def test_v2_pp_only_abstain_without_input_order_drops_canonical(predictor_v2: PredictorV2) -> None:
    # These end in a middle initial, so the as-typed reading cannot be materialized and the
    # abstain surfaces as a failure. The PP parse pins the surname to the first token either
    # way ("Wei Zhang Q." -> surname Wei), so the declined reorder must not reach canonical.
    for result in predictor_v2.route_pp(["Zhang Wei Q.", "Wei Zhang Q."]):
        assert result.router_prediction.value == "abstain"
        assert not result.success
        assert (result.given_name, result.middle_name, result.surname) == (None, None, None)
        assert result.pp.success
        assert result.canonical_name is None
        # The candidate object keeps its own canonical: `pp` is not the routed answer.
        assert result.pp.canonical_name is not None
        assert result.pp.canonical_name.normalized.middle_name == "Q."


def test_routing_predictor_v2_drops_canonical_for_declined_abstain_but_keeps_pooled_answer() -> None:
    predictor = RoutingPredictorV2(config=PredictorConfig(parallel="never"), artifacts_dir=".")
    pp_names = ["Zhang Wei Q."]
    pool = [*pp_names, "Li Xiaoming", "Chen Jianguo", "Zhao Yuting", "Sun Haoran", "Zhou Mengqi"]

    pp_only, pooled = predictor.predict_batch(
        [RoutingInstance(pp_names=pp_names), RoutingInstance(pp_names=pp_names, vys_pool_names=pool)],
    )

    (declined,) = pp_only.authors
    assert declined.router_prediction.value == "abstain"
    assert not declined.success
    assert declined.canonical_name is None

    # A venue pool supplies the evidence PP-only lacks, so the same name still normalizes.
    (routed,) = pooled.authors
    assert routed.success
    assert (routed.given_name, routed.middle_name, routed.surname) == ("Wei", "Q.", "Zhang")
    assert routed.canonical_name is not None
    assert routed.canonical_name.normalized.given_name == "Wei"
    assert routed.canonical_name.normalized.middle_name == "Q."
    assert routed.canonical_name.normalized.surname == "Zhang"


def test_v2_routing_keeps_generic_canonical_for_non_chinese_rows(predictor_v2: PredictorV2) -> None:
    results = predictor_v2.route_pp(["Michael Johnson", "Dr. Ana-Maria O'Neill PhD", "UW University"])

    assert not any(result.success for result in results)
    assert results[0].canonical_name is not None
    assert results[0].canonical_name.text == "Michael Johnson"
    assert results[1].canonical_name is not None
    assert results[1].canonical_name.text == "Ana-Maria O'Neill"
    assert results[2].canonical_name is None


def test_v2_routing_suppresses_unproven_canonical_for_a_declined_parse(predictor_v2: PredictorV2) -> None:
    pp_names = ["Zhang Wei Q.", "Wang Lin A.", "Zhang Wei", "Michael Johnson", "UW University"]
    pool = [*pp_names, "Li Xiaoming", "Chen Jianguo", "Zhao Yuting", "Sun Haoran"]

    for results in (predictor_v2.route_pp(pp_names), predictor_v2.route_pp_vys(pp_names, pool)):
        for result in results:
            if not result.success and result.pp.success:
                assert result.canonical_name is None


def test_routing_predictor_v2_is_one_to_one_and_round_trips() -> None:
    predictor = RoutingPredictorV2(config=PredictorConfig(parallel="never"), artifacts_dir=".")
    instances = [
        RoutingInstance(pp_names=["Steve Blando IV"]),
        RoutingInstance(pp_names=["Yue Lin"], vys_pool_names=["Yue Lin", "Wei Wang", "Jun Zhao"]),
        RoutingInstance(pp_names=[]),
    ]

    results = predictor.predict_batch(instances)

    assert len(results) == len(instances)
    assert [len(result.authors) for result in results] == [1, 1, 0]
    assert results[0].authors[0].canonical_name is not None
    assert results[0].authors[0].canonical_name.normalized.suffix == "IV"
    assert [RoutedPaperPredictionV2(**result.dict()) for result in results] == results


def test_timo_config_exposes_separate_v2_variants() -> None:
    config = Path("sinonym/timo/config.yaml").read_text(encoding="utf-8")

    assert "sinonym_v2:" in config
    assert "prediction: sinonym.timo.interface.PredictionV2" in config
    assert "predictor: sinonym.timo.interface.PredictorV2" in config
    assert "sinonym_routing_v2:" in config
    assert "prediction: sinonym.timo.interface.RoutedPaperPredictionV2" in config
    assert "predictor: sinonym.timo.interface.RoutingPredictorV2" in config
