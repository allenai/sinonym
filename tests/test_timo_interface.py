import pytest
from pydantic import ValidationError

from sinonym.coretypes import BatchParseResult
from sinonym.detector import ChineseNameDetector
from sinonym.pipeline.name_order_routing import (
    build_pp_vys_abstain_rows,
    route_pp_vys_abstain_rows,
)
from sinonym.services.non_person import NON_PERSON_FAILURE_REASON
from sinonym.timo.interface import (
    FormatPattern,
    Instance,
    NameFormatValue,
    NameOrderEvidence,
    Prediction,
    Predictor,
    PredictorConfig,
    RoutedPaperPrediction,
    RoutedPrediction,
    RoutingInstance,
    RoutingPredictor,
    ScriptRepresentationValue,
)

EXPECTED_BATCH_CONTEXT_RESULT_COUNT = 3
TIE_CONFIDENCE = 0.5
LEGACY_PATTERN_CONFIDENCE = 0.75
LEGACY_SURNAME_FIRST_COUNT = 3
LEGACY_GIVEN_FIRST_COUNT = 1
LEGACY_TOTAL_COUNT = 4
LEGACY_VOTE_MARGIN_COUNT = 2
TIMO_TEST_MAX_WORKERS = 3
TIMO_TEST_CHUNK_SIZE = 11
TIMO_TEST_MIN_PARALLEL_BATCHES = 7
TIMO_TEST_FORMAT_THRESHOLD = 0.7
TIMO_TEST_MINIMUM_BATCH_SIZE = 3


@pytest.fixture(scope="module")
def predictor() -> Predictor:
    return Predictor(config=PredictorConfig(), artifacts_dir=".")


def test_single_chinese_name(predictor: Predictor):
    results = predictor.predict_batch([Instance(name="Li Wei")])
    assert len(results) == 1
    result = results[0]
    assert result.success
    assert result.surname == "Li"
    assert result.given_name == "Wei"
    assert result.confidence is None
    assert result.format_pattern is None
    assert result.error_message is None


def test_chinese_name_characters(predictor: Predictor):
    results = predictor.predict_batch([Instance(name="巩俐")])
    assert results[0].success
    assert results[0].surname is not None


def test_chinese_name_comma_format(predictor: Predictor):
    results = predictor.predict_batch([Instance(name="Zhang, Ming")])
    result = results[0]
    assert result.success
    assert result.surname is not None
    assert result.given_name is not None


def test_non_chinese_name(predictor: Predictor):
    results = predictor.predict_batch([Instance(name="John Smith")])
    result = results[0]
    assert not result.success
    assert result.error_message is not None
    assert result.given_name is None
    assert result.surname is None


def test_empty_string(predictor: Predictor):
    results = predictor.predict_batch([Instance(name="")])
    assert not results[0].success


def test_whitespace_only(predictor: Predictor):
    results = predictor.predict_batch([Instance(name="   ")])
    assert not results[0].success


def test_batch_with_context(predictor: Predictor):
    instances = [
        Instance(name="Li Wei"),
        Instance(name="Wang Weiming"),
        Instance(name="John Smith"),
    ]
    results = predictor.predict_batch(instances)
    assert len(results) == EXPECTED_BATCH_CONTEXT_RESULT_COUNT
    assert results[0].success
    assert results[1].success
    assert not results[2].success


def test_prediction_is_pydantic_model(predictor: Predictor):
    results = predictor.predict_batch([Instance(name="Li Wei")])
    result = results[0]
    assert isinstance(result, Prediction)
    as_dict = result.dict()
    assert "success" in as_dict
    assert "error_message" in as_dict
    assert "given_name" in as_dict
    assert "surname" in as_dict
    assert "middle_name" in as_dict
    assert "confidence" in as_dict
    assert "format_pattern" in as_dict


def test_prediction_json_serialization(predictor: Predictor):
    results = predictor.predict_batch([Instance(name="Li Wei")])
    json_str = results[0].json()
    assert '"success": true' in json_str
    assert '"confidence":' in json_str


def test_format_pattern_exposes_batch_decision_confidence(predictor: Predictor):
    summary = predictor.score_name_batch(["J. Liu", "Jing Wan"])
    pattern = summary.format_pattern

    assert pattern.confidence == TIE_CONFIDENCE
    assert pattern.decision_confidence >= pattern.confidence
    assert pattern.voting_count == pattern.surname_first_count + pattern.given_first_count
    assert pattern.vote_margin_count == abs(pattern.surname_first_count - pattern.given_first_count)
    assert pattern.vote_margin == pattern.vote_margin_count / pattern.total_count
    assert pattern.threshold_met


def test_compound_surname(predictor: Predictor):
    results = predictor.predict_batch([Instance(name="Ouyang Ming")])
    assert results[0].success
    assert results[0].given_name == "Ming"
    assert results[0].surname == "Ouyang"


def test_prediction_models_keep_mutable_backwards_compatible_shapes_and_enum_typed(predictor: Predictor):
    results = predictor.predict_batch([Instance(name="Li Wei"), Instance(name="Zhang Ming")])

    first_pattern = results[0].format_pattern
    second_pattern = results[1].format_pattern
    assert first_pattern is None
    assert second_pattern is None

    first_pattern = FormatPattern(
        dominant_format="surname_first",
        confidence=1.0,
        decision_confidence=1.0,
        surname_first_count=2,
        given_first_count=0,
        total_count=2,
        voting_count=2,
        vote_margin_count=2,
        vote_margin=1.0,
        threshold_met=True,
    )
    second_pattern = first_pattern.copy(deep=True)

    second_threshold = second_pattern.threshold_met
    first_threshold = not second_threshold
    first_pattern.threshold_met = first_threshold
    assert first_pattern.threshold_met is first_threshold
    assert second_pattern.threshold_met is second_threshold
    assert isinstance(second_pattern.dominant_format, NameFormatValue)
    assert second_pattern.dominant_format == NameFormatValue.SURNAME_FIRST

    with pytest.raises(ValidationError):
        FormatPattern(
            dominant_format="sideways",
            confidence=0.0,
            decision_confidence=0.0,
            surname_first_count=0,
            given_first_count=0,
            total_count=0,
            voting_count=0,
            vote_margin_count=0,
            vote_margin=0.0,
            threshold_met=False,
        )


def test_format_pattern_accepts_legacy_payload_and_dict_serializes_enum_values():
    pattern = FormatPattern(
        dominant_format="surname_first",
        confidence=LEGACY_PATTERN_CONFIDENCE,
        surname_first_count=LEGACY_SURNAME_FIRST_COUNT,
        given_first_count=LEGACY_GIVEN_FIRST_COUNT,
        total_count=LEGACY_TOTAL_COUNT,
        threshold_met=True,
    )

    assert pattern.decision_confidence == LEGACY_PATTERN_CONFIDENCE
    assert pattern.voting_count == LEGACY_TOTAL_COUNT
    assert pattern.vote_margin_count == LEGACY_VOTE_MARGIN_COUNT
    assert pattern.vote_margin == TIE_CONFIDENCE
    assert isinstance(pattern.dominant_format, NameFormatValue)
    assert pattern.dict()["dominant_format"] == "surname_first"


def test_prediction_dict_serializes_nested_format_pattern_enum_values(predictor: Predictor):
    result = Prediction(
        success=True,
        format_pattern=FormatPattern(
            dominant_format="surname_first",
            confidence=1.0,
            decision_confidence=1.0,
            surname_first_count=1,
            given_first_count=0,
            total_count=1,
            voting_count=1,
            vote_margin_count=1,
            vote_margin=1.0,
            threshold_met=True,
        ),
    )

    assert result.dict()["format_pattern"]["dominant_format"] == "surname_first"


def test_batch_models_keep_list_shaped_python_api(predictor: Predictor):
    names = ["Li Wei", "Wang Weiming"]

    summary = predictor.score_name_batch(names)
    batch = predictor.analyze_name_batch(names)

    assert summary.names == names
    assert isinstance(summary.names, list)
    assert isinstance(summary.results, list)
    assert isinstance(summary.confidences, list)
    assert isinstance(batch.names, list)
    assert isinstance(batch.results, list)
    assert isinstance(batch.individual_analyses, list)
    assert isinstance(batch.improvements, list)
    assert isinstance(batch.name_order_evidence, list)
    assert isinstance(batch.name_order_evidence[0].raw_tokens, list)
    assert isinstance(batch.name_order_evidence[0].all_caps_tokens, list)
    assert isinstance(batch.individual_analyses[0].candidates, list)


def test_name_order_evidence_required_fields_precede_defaulted_fields():
    fields = list(NameOrderEvidence.__fields__)

    assert fields.index("has_all_caps_token") < fields.index("first_token_surname_frequency")
    assert fields.index("all_caps_tokens") < fields.index("first_token_surname_frequency")


def test_empty_batch(predictor: Predictor):
    results = predictor.predict_batch([])
    assert len(results) == 0


def test_process_name_batch_multiprocess_preserves_batch_context(predictor: Predictor):
    names = ["Wang An", "Yan Li", "Wu Gang", "Li Bao"]

    expected = predictor.process_name_batch(names)
    actual = predictor.process_name_batch_multiprocess(names, max_workers=2, chunk_size=1)

    assert [(result.given_name, result.surname) for result in actual] == [
        (result.given_name, result.surname) for result in expected
    ]


def test_process_name_batches_forwards_auto_parallel_options(predictor: Predictor, monkeypatch):
    captured = {}
    detector = object.__getattribute__(predictor, "_detector")
    parse_result = detector.normalize_name("Li Wei")

    def fake_process_name_batches(batches, **kwargs):
        captured["batches"] = batches
        captured.update(kwargs)
        return [[parse_result]]

    monkeypatch.setattr(detector, "process_name_batches", fake_process_name_batches)

    actual = predictor.process_name_batches(
        [["Li Wei"]],
        parallel="never",
        max_workers=TIMO_TEST_MAX_WORKERS,
        chunk_size=TIMO_TEST_CHUNK_SIZE,
        min_parallel_batches=TIMO_TEST_MIN_PARALLEL_BATCHES,
        format_threshold=TIMO_TEST_FORMAT_THRESHOLD,
        minimum_batch_size=TIMO_TEST_MINIMUM_BATCH_SIZE,
    )

    assert captured["batches"] == [["Li Wei"]]
    assert captured["parallel"] == "never"
    assert captured["max_workers"] == TIMO_TEST_MAX_WORKERS
    assert captured["chunk_size"] == TIMO_TEST_CHUNK_SIZE
    assert captured["min_parallel_batches"] == TIMO_TEST_MIN_PARALLEL_BATCHES
    assert captured["format_threshold"] == TIMO_TEST_FORMAT_THRESHOLD
    assert captured["minimum_batch_size"] == TIMO_TEST_MINIMUM_BATCH_SIZE
    assert actual[0][0].surname == "Li"


def test_analyze_name_batch_handles_detector_fallback_evidence(predictor: Predictor):
    detector = object.__getattribute__(predictor, "_detector")
    original_service = object.__getattribute__(detector, "_batch_analysis_service")
    object.__setattr__(detector, "_batch_analysis_service", None)
    try:
        result = predictor.analyze_name_batch(["Li Wei"])
    finally:
        object.__setattr__(detector, "_batch_analysis_service", original_service)

    assert len(result.name_order_evidence) == 1
    assert result.name_order_evidence[0].script_representation == ScriptRepresentationValue.UNKNOWN


def test_route_pp_vys_requires_paper_authors_first(predictor: Predictor):
    # paper authors not the leading slice of the pool -> raise
    with pytest.raises(ValueError, match="must start with the paper"):
        predictor.route_pp_vys(["Yue Lin", "Wei Wang"], ["Wei Wang", "Yue Lin", "Tao Sun"])


def test_route_pp_vys_empty(predictor: Predictor):
    assert predictor.route_pp_vys([], []) == []


def test_route_pp_vys_matches_manual_pp_then_vys_then_router(predictor: Predictor):
    """Documents the underlying flow: run PP batch, run VYS batch, feed BOTH to the router.

    `route_pp_vys` is a thin wrapper over exactly these steps; this test rebuilds them by hand
    and asserts the routed answers match.
    """
    pp_names = ["Yue Lin", "Wei Wang", "Chuang Yang"]
    # paper authors first, then the other venue authors
    vys_pool = ["Yue Lin", "Wei Wang", "Chuang Yang", "Jun Zhao", "Hui Li", "Tao Sun", "Min Guo"]
    n = len(pp_names)

    detector = ChineseNameDetector()
    # 1) PP batch: the paper's authors, parsed jointly
    pp_batch = detector.analyze_name_batch(pp_names)
    # 2) VYS batch: the venue-source-year pool, parsed jointly, then aligned to the paper's authors
    pool = detector.analyze_name_batch(vys_pool)
    vys_batch = BatchParseResult(
        names=list(pp_names),
        results=list(pool.results[:n]),
        format_pattern=pool.format_pattern,
        individual_analyses=[],
        improvements=set(),
        name_order_evidence=list(pool.name_order_evidence[:n]),
    )
    # 3) feed both batch results to the router
    routed_rows = route_pp_vys_abstain_rows(build_pp_vys_abstain_rows(pp_batch, vys_batch))

    def manual_surname(i):
        pred = routed_rows[i]["router_prediction"]
        ioc = routed_rows[i].get("input_order_candidate", "unknown")
        if pred == "pp":
            res = pp_batch.results[i]
        elif pred == "vys":
            res = vys_batch.results[i]
        elif pred == "abstain":
            res = vys_batch.results[i] if ioc == "vys" else pp_batch.results[i]
        else:
            return None
        return res.parsed.surname if (res.success and res.parsed) else None

    # the wrapper endpoint reproduces the manual PP->VYS->router result
    endpoint = predictor.route_pp_vys(pp_names, vys_pool)
    assert [manual_surname(i) for i in range(len(pp_names))] == [r.surname for r in endpoint]
    assert [routed_rows[i]["router_prediction"] for i in range(len(pp_names))] == [r.router_prediction for r in endpoint]


# Fixed-scenario regression: exact routed values (batch-composition dependent).
ROUTE_SCENARIO_PP = ["Yue Lin", "Wei Wang", "Chuang Yang"]
ROUTE_SCENARIO_POOL = ["Yue Lin", "Wei Wang", "Chuang Yang", "Jun Zhao", "Hui Li", "Tao Sun", "Min Guo"]


def test_route_pp_vys_exact_values(predictor: Predictor):
    out = predictor.route_pp_vys(ROUTE_SCENARIO_PP, ROUTE_SCENARIO_POOL)
    got = [(r.router_prediction.value, r.given_name, r.surname) for r in out]
    assert got == [
        ("pp", "Lin", "Yue"),
        ("vys", "Wei", "Wang"),
        ("vys", "Chuang", "Yang"),
    ]
    # candidate parses carried through exactly
    assert (out[1].pp.given_name, out[1].pp.surname) == ("Wang", "Wei")  # PP (paper-batch) reading
    assert (out[1].vys.given_name, out[1].vys.surname) == ("Wei", "Wang")  # VYS (venue-pool) reading


def test_route_pp_exact_values(predictor: Predictor):
    out = predictor.route_pp(ROUTE_SCENARIO_PP)
    got = [(r.router_prediction.value, r.given_name, r.surname) for r in out]
    # NB: differs from route_pp_vys on "Wei Wang" — PP-only keeps the PP-batch reading
    # (given=Wang/surname=Wei) whereas route_pp_vys routed it to VYS (given=Wei/surname=Wang).
    assert got == [
        ("pp", "Lin", "Yue"),
        ("pp", "Wang", "Wei"),
        ("pp", "Yang", "Chuang"),
    ]
    assert predictor.route_pp([]) == []


def test_routed_candidate_format_patterns_are_independent(predictor: Predictor):
    pp_only = predictor.route_pp(["Yue Lin", "Wei Wang"])
    before = pp_only[1].pp.format_pattern.threshold_met
    pp_only[0].pp.format_pattern.threshold_met = not before
    assert pp_only[1].pp.format_pattern.threshold_met is before

    pp_names = ["Yue Lin", "Wei Wang"]
    pool = ["Yue Lin", "Wei Wang", "Jun Zhao", "Hui Li"]
    routed = predictor.route_pp_vys(pp_names, pool)
    pp_before = routed[1].pp.format_pattern.threshold_met
    vys_before = routed[1].vys.format_pattern.threshold_met
    routed[0].pp.format_pattern.threshold_met = not pp_before
    routed[0].vys.format_pattern.threshold_met = not vys_before
    assert routed[1].pp.format_pattern.threshold_met is pp_before
    assert routed[1].vys.format_pattern.threshold_met is vys_before


# --- abstain coverage -------------------------------------------------------
# The exact-value fixtures above only route to pp/vys. These lock the abstain
# branches, which the routers do exercise on real batches.


def test_route_pp_abstain_is_script_aware(predictor: Predictor):
    """Latin abstain keeps trailing-token surname; spaced Han abstain keeps its source boundary."""
    # Spaced-Han zero-batch input: the source space marks surname-first order,
    # so abstain preserves the PP parse instead of flipping to trailing surname.
    (reordered,) = predictor.route_pp(["\u738b \u4f1f"])
    assert reordered.router_prediction.value == "abstain"
    assert reordered.success
    assert (reordered.pp.given_name, reordered.pp.surname) == ("Wei", "Wang")
    assert (reordered.given_name, reordered.middle_name, reordered.surname) == ("Wei", None, "Wang")
    assert (reordered.given_name, reordered.surname) == (reordered.pp.given_name, reordered.pp.surname)

    liu, zheng = predictor.route_pp(["\u5289 \u6587\u69ae", "\u912d \u4fe1\u529b"])
    assert (liu.router_prediction.value, liu.given_name, liu.surname) == ("abstain", "Wen-Rong", "Liu")
    assert (zheng.router_prediction.value, zheng.given_name, zheng.surname) == ("abstain", "Xin-Li", "Zheng")

    # Latin pair: the given-first read abstains and keeps its (input-order) reading;
    # the surname-first read is accepted as pp and keeps the batch reorder.
    lin, wang = predictor.route_pp(["Lin Yue", "Wang Wei"])
    assert lin.router_prediction.value == "abstain"
    assert (lin.given_name, lin.middle_name, lin.surname) == ("Lin", None, "Yue")
    assert (lin.given_name, lin.surname) == (lin.pp.given_name, lin.pp.surname)
    assert wang.router_prediction.value == "pp"
    assert (wang.given_name, wang.middle_name, wang.surname) == ("Wei", None, "Wang")


def test_route_pp_rejects_mixed_initial_cjk_transliterations(predictor: Predictor):
    for raw_name in ("G.\u970d\u5f17", "H\u00b7\u7eb3\u683c\u5c14\u65af"):
        (routed,) = predictor.route_pp([raw_name])

        assert not routed.success
        assert routed.router_prediction.value == "not_person"
        assert routed.pp.error_message == NON_PERSON_FAILURE_REASON


def test_route_pp_vys_abstain_picks_input_order_candidate(predictor: Predictor):
    """PP-vs-VYS abstain resolves to the input_order_candidate side (pp/vys), per README
    semantics + the routing fixture. Covers both ioc=pp and ioc=vys."""
    # ioc=pp: both authors abstain and resolve to the PP reading (which differs from VYS)
    pp = ["Na Li", "Jie Chen"]
    pool = ["Na Li", "Jie Chen", "Yue Lin", "Zhang Wei", "Kai Yang", "Wang Fang", "Sima Qian"]
    out = predictor.route_pp_vys(pp, pool)
    assert [r.router_prediction.value for r in out] == ["abstain", "abstain"]
    for r in out:
        assert r.input_order_candidate.value == "pp"
        # the pick is meaningful: pp and vys candidates genuinely disagree here
        assert (r.pp.given_name, r.pp.middle_name, r.pp.surname) != (r.vys.given_name, r.vys.middle_name, r.vys.surname)
        # routed answer equals the chosen (pp) candidate, component by component
        assert (r.given_name, r.middle_name, r.surname) == (r.pp.given_name, r.pp.middle_name, r.pp.surname)
    na, jie = out
    assert (na.given_name, na.middle_name, na.surname) == ("Na", None, "Li")
    assert (jie.given_name, jie.middle_name, jie.surname) == ("Jie", None, "Chen")

    # ioc=vys: the abstaining author resolves to the VYS reading
    pp2 = ["Lei Sun", "Sima Qian"]
    pool2 = ["Lei Sun", "Sima Qian", "Jun Zhao", "Liu Yang", "Tao Sun", "Ming Li", "Hui Guo", "Kai Yang"]
    out2 = predictor.route_pp_vys(pp2, pool2)
    sima = out2[1]
    assert sima.router_prediction.value == "abstain"
    assert sima.input_order_candidate.value == "vys"
    assert (sima.pp.given_name, sima.pp.middle_name, sima.pp.surname) != (
        sima.vys.given_name,
        sima.vys.middle_name,
        sima.vys.surname,
    )
    # routed answer equals the chosen (vys) candidate, component by component
    assert (sima.given_name, sima.middle_name, sima.surname) == (sima.vys.given_name, sima.vys.middle_name, sima.vys.surname)
    assert (sima.given_name, sima.middle_name, sima.surname) == ("Sima", None, "Qian")


def test_route_unified_falls_back_to_pp_when_no_pool(predictor: Predictor):
    pp_names = ["Yue Lin", "Wei Wang", "Chuang Yang"]
    pool = ["Yue Lin", "Wei Wang", "Chuang Yang", "Jun Zhao", "Hui Li", "Tao Sun", "Min Guo"]

    # no pool (None or []) -> PP-only fallback: vys/input_order_candidate are None, matches route_pp
    for empty in (None, []):
        out = predictor.route(pp_names, empty)
        assert all(isinstance(r, RoutedPrediction) for r in out)
        assert all(r.vys is None and r.input_order_candidate is None for r in out)
        assert all(r.router_prediction in {"pp", "abstain", "not_person"} for r in out)
        pp_only = predictor.route_pp(pp_names)
        assert [(r.given_name, r.surname) for r in out] == [(r.given_name, r.surname) for r in pp_only]

    # with pool -> delegates to route_pp_vys (vys candidate present)
    out = predictor.route(pp_names, pool)
    assert [(r.given_name, r.surname) for r in out] == [(r.given_name, r.surname) for r in predictor.route_pp_vys(pp_names, pool)]
    assert all(r.vys is not None for r in out)
    assert isinstance(out[0].dict(), dict)


# --- RoutingPredictor: timo 1:1 instance->prediction contract ----------------


def test_routing_predictor_one_prediction_per_instance():
    rp = RoutingPredictor(config=PredictorConfig(), artifacts_dir=".")
    instances = [
        RoutingInstance(pp_names=["Yue Lin", "Wei Wang"], vys_pool_names=["Yue Lin", "Wei Wang", "Jun Zhao"]),
        RoutingInstance(pp_names=["Zhang San"]),  # PP-only
        RoutingInstance(pp_names=[]),  # empty paper must still emit one prediction
    ]
    results = rp.predict_batch(instances)

    # the guard timo's InferenceServerContainer asserts: len(predictions) == len(instances)
    assert len(results) == len(instances)
    assert all(isinstance(p, RoutedPaperPrediction) for p in results)
    assert [len(p.authors) for p in results] == [2, 1, 0]

    # timo reconstructs each prediction via prediction_class(**raw_pred); round-trip must hold
    rebuilt = [RoutedPaperPrediction(**p.dict()) for p in results]
    assert rebuilt == results


def test_routing_predictor_batches_instance_analysis_through_auto_wrapper(monkeypatch):
    rp = RoutingPredictor(
        config=PredictorConfig(
            parallel="never",
            mp_max_workers=TIMO_TEST_MAX_WORKERS,
            mp_chunk_size=TIMO_TEST_CHUNK_SIZE,
            mp_min_parallel_batches=TIMO_TEST_MIN_PARALLEL_BATCHES,
        ),
        artifacts_dir=".",
    )
    detector = object.__getattribute__(rp, "_detector")
    original_analyze_name_batches = detector.analyze_name_batches
    captured = {}

    def wrapped_analyze_name_batches(batches, **kwargs):
        captured["batches"] = batches
        captured.update(kwargs)
        return original_analyze_name_batches(batches, **kwargs)

    monkeypatch.setattr(detector, "analyze_name_batches", wrapped_analyze_name_batches)

    good_vys = RoutingInstance(pp_names=["Yue Lin", "Wei Wang"], vys_pool_names=["Yue Lin", "Wei Wang", "Jun Zhao"])
    pp_only = RoutingInstance(pp_names=["Zhang San"])
    empty = RoutingInstance(pp_names=[])

    results = rp.predict_batch([good_vys, pp_only, empty])

    assert len(results) == TIMO_TEST_MAX_WORKERS
    assert [len(result.authors) for result in results] == [2, 1, 0]
    assert captured["batches"] == [
        good_vys.pp_names,
        good_vys.vys_pool_names,
        pp_only.pp_names,
    ]
    assert captured["parallel"] == "never"
    assert captured["max_workers"] == TIMO_TEST_MAX_WORKERS
    assert captured["chunk_size"] == TIMO_TEST_CHUNK_SIZE
    assert captured["min_parallel_batches"] == TIMO_TEST_MIN_PARALLEL_BATCHES


def test_routing_predictor_rejects_malformed_instance_in_batch():
    """A malformed pool in one co-batched instance should fail loudly."""
    rp = RoutingPredictor(config=PredictorConfig(), artifacts_dir=".")
    good_a = RoutingInstance(pp_names=["Yue Lin", "Wei Wang"], vys_pool_names=["Yue Lin", "Wei Wang", "Jun Zhao"])
    # middle: paper authors are NOT the leading slice of the pool -> pool precondition violated
    malformed = RoutingInstance(pp_names=["Yue Lin", "Wei Wang"], vys_pool_names=["Wei Wang", "Yue Lin", "Tao Sun"])
    good_b = RoutingInstance(pp_names=["Chuang Yang"], vys_pool_names=["Chuang Yang", "Hui Li", "Tao Sun"])

    with pytest.raises(ValueError, match="must start with the paper"):
        rp.predict_batch([good_a, malformed, good_b])

    assert len(rp.predict_batch([good_a])[0].authors) == len(good_a.pp_names)
    assert len(rp.predict_batch([good_b])[0].authors) == len(good_b.pp_names)


def test_routing_predictor_all_malformed_instances_raise():
    """Malformed pools should not be serialized as valid empty-paper predictions."""
    rp = RoutingPredictor(config=PredictorConfig(), artifacts_dir=".")
    instances = [
        RoutingInstance(pp_names=["Yue Lin", "Wei Wang"], vys_pool_names=["Wei Wang", "Yue Lin", "Tao Sun"]),
        # pool smaller than the paper -> also a pool precondition violation
        RoutingInstance(pp_names=["Chuang Yang", "Hui Li"], vys_pool_names=["Chuang Yang"]),
    ]

    with pytest.raises(ValueError, match="must start with the paper"):
        rp.predict_batch(instances)
