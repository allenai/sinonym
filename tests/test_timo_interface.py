import pytest
from pydantic import ValidationError

from sinonym.timo.interface import (
    FormatPattern,
    Instance,
    NameFormatValue,
    NameOrderEvidence,
    Prediction,
    Predictor,
    PredictorConfig,
    RoutedPaperPrediction,
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
    assert result.confidence is not None
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
    assert first_pattern is not None
    assert second_pattern is not None

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
    result = predictor.predict_batch([Instance(name="Li Wei")])[0]

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
    from sinonym.coretypes import BatchParseResult
    from sinonym.detector import ChineseNameDetector
    from sinonym.pipeline.name_order_routing import (
        build_pp_vys_abstain_rows,
        route_pp_vys_abstain_rows,
    )

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
    assert [routed_rows[i]["router_prediction"] for i in range(len(pp_names))] == [
        r.router_prediction for r in endpoint
    ]


# Fixed-scenario regression: exact routed values for v0.2.9 (batch-composition dependent).
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
    assert (out[1].pp.given_name, out[1].pp.surname) == ("Wang", "Wei")   # PP (paper-batch) reading
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


# --- abstain coverage -------------------------------------------------------
# The exact-value fixtures above only route to pp/vys. These lock the abstain
# branches, which the routers do exercise on real batches.

def test_route_pp_abstain_emits_input_order_parse(predictor: Predictor):
    """PP-only abstain = "keep the input-order parse": emit the name's normalized standalone
    reading, NOT the (possibly reordered) PP-batch reading."""
    pp = ["Lin Yue", "Wang Wei", "Yang Chuang", "Zhao Jun", "Li Hui"]
    out = predictor.route_pp(pp)

    assert [r.router_prediction.value for r in out] == ["abstain"] * len(pp)
    assert all(r.success for r in out)
    # each abstain row is the name's input-order (standalone) reading; assert every component
    expected = {
        "Lin Yue": ("Lin", None, "Yue"),
        "Wang Wei": ("Wei", None, "Wang"),      # input order kept (batch would flip to Wang/Wei)
        "Yang Chuang": ("Yang", None, "Chuang"),
        "Zhao Jun": ("Jun", None, "Zhao"),      # input order kept (batch would flip to Zhao/Jun)
        "Li Hui": ("Li", None, "Hui"),
    }
    for name, r in zip(pp, out):
        assert (r.given_name, r.middle_name, r.surname) == expected[name]
    # the batch reordered "Wang Wei"/"Zhao Jun"; abstain reverts to input order, so the routed
    # answer differs from the PP candidate there (proving abstain is not a no-op)
    for name in ("Wang Wei", "Zhao Jun"):
        r = next(r for n, r in zip(pp, out) if n == name)
        assert (r.given_name, r.surname) != (r.pp.given_name, r.pp.surname)


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
    assert (sima.pp.given_name, sima.pp.middle_name, sima.pp.surname) != (sima.vys.given_name, sima.vys.middle_name, sima.vys.surname)
    # routed answer equals the chosen (vys) candidate, component by component
    assert (sima.given_name, sima.middle_name, sima.surname) == (sima.vys.given_name, sima.vys.middle_name, sima.vys.surname)
    assert (sima.given_name, sima.middle_name, sima.surname) == ("Sima", None, "Qian")



def test_route_unified_falls_back_to_pp_when_no_pool(predictor: Predictor):
    from sinonym.timo.interface import RoutedPrediction

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
    assert [(r.given_name, r.surname) for r in out] == [
        (r.given_name, r.surname) for r in predictor.route_pp_vys(pp_names, pool)
    ]
    assert all(r.vys is not None for r in out)
    assert isinstance(out[0].dict(), dict)


# --- RoutingPredictor: timo 1:1 instance->prediction contract ----------------

def test_routing_predictor_one_prediction_per_instance():
    rp = RoutingPredictor(config=PredictorConfig(), artifacts_dir=".")
    instances = [
        RoutingInstance(pp_names=["Yue Lin", "Wei Wang"], vys_pool_names=["Yue Lin", "Wei Wang", "Jun Zhao"]),
        RoutingInstance(pp_names=["Zhang San"]),  # PP-only
        RoutingInstance(pp_names=[]),              # empty paper must still emit one prediction
    ]
    results = rp.predict_batch(instances)

    # the guard timo's InferenceServerContainer asserts: len(predictions) == len(instances)
    assert len(results) == len(instances)
    assert all(isinstance(p, RoutedPaperPrediction) for p in results)
    assert [len(p.authors) for p in results] == [2, 1, 0]

    # timo reconstructs each prediction via prediction_class(**raw_pred); round-trip must hold
    rebuilt = [RoutedPaperPrediction(**p.dict()) for p in results]
    assert rebuilt == results
