# ruff: noqa: RUF012, EM101, PLC0415, SLF001, SIM117, TRY003
import logging

import numpy as np
import pytest

from sinonym import chinese_names_data
from sinonym.coretypes import BatchFormatPattern, NameFormat, ParseCandidate, ParseResult
from sinonym.pipeline import name_order_routing
from sinonym.pipeline.name_order_routing import (
    _input_order_display,
    _required_bool,
    route_pp_abstain_rows,
    route_pp_vys_abstain_rows,
)
from sinonym.services.batch_analysis import LATIN_ONLY_REPRESENTATION, BatchCandidateEntry
from sinonym.timo.interface import Instance, Predictor, PredictorConfig, PredictorV2


def _pp_abstain_row(**overrides):
    row = {
        "pp_success": True,
        "pp_result_token_count": 2,
        "selected_format": "given_first",
        "selected_surname_position": "last",
        "selected_surname_token_count": 1,
        "batch_total_count": 1,
        "selected_surname_frequency": 100.0,
        "has_cjk": False,
        "has_latin": True,
        "cjk_has_space": False,
        "raw_tokens": 2,
    }
    row.update(overrides)
    return row


def _pp_vys_row(**overrides):
    row = {
        "name": "Zhang Wei",
        "pp_success": True,
        "pp_result": "Wei Zhang",
        "pp_selected_format": "surname_first",
        "pp_selected_surname_position": "first",
        "pp_selected_surname_frequency": 100.0,
        "pp_batch_dominant_format": "surname_first",
        "pp_batch_threshold_met": True,
        "pp_batch_total_count": 3,
        "pp_batch_confidence": 0.69,
        "pp_batch_vote_margin": 0.9,
        "pp_selected_surname_frequency_ratio": 2.0,
        "vys_success": True,
        "vys_result": "Zhang Wei",
        "vys_selected_format": "given_first",
        "vys_selected_surname_position": "first",
        "vys_selected_surname_frequency": 100.0,
        "vys_batch_dominant_format": "given_first",
        "vys_batch_threshold_met": True,
        "vys_batch_total_count": 3,
        "vys_batch_confidence": 0.85,
        "vys_batch_vote_margin": 0.0,
    }
    row.update(overrides)
    return row


def test_chinese_first_surname_blocks_korean_pair_rejection(detector):
    expected = {
        "Zhang Ye Jin Lee": "Zhang-Ye-Jin Lee",
        "Wang Min Soo Han": "Min-Soo-Han Wang",
    }

    for raw_name, normalized in expected.items():
        result = detector.normalize_name(raw_name)
        assert result.success, raw_name
        assert result.result == normalized


@pytest.mark.parametrize(
    ("raw_name", "expected", "surname"),
    [
        ("Nan-Gong Li", "Li Nan-Gong", "Nan-Gong"),
        ("Duan-Mu Li", "Li Duan-Mu", "Duan-Mu"),
        ("Si-Tu Chen", "Chen Si-Tu", "Si-Tu"),
    ],
)
def test_curated_hyphenated_compound_surnames_win(detector, raw_name, expected, surname):
    result = detector.normalize_name(raw_name)

    assert result.success
    assert result.result == expected
    assert result.parsed.surname == surname


@pytest.mark.parametrize(
    ("raw_name", "expected", "surname", "given"),
    [
        ("Sun Zhong-Shan", "Zhong-Shan Sun", "Sun", "Zhong-Shan"),
        ("Zhong-Shan Li", "Zhong-Shan Li", "Li", "Zhong-Shan"),
        ("Duan-Gan Mu", "Duan-Gan Mu", "Mu", "Duan-Gan"),
    ],
)
def test_non_curated_hyphenated_given_names_keep_endpoint_surname(detector, raw_name, expected, surname, given):
    result = detector.normalize_name(raw_name)

    assert result.success
    assert result.result == expected
    assert result.parsed.surname == surname
    assert result.parsed.given_name == given


def test_attested_remapped_wade_giles_surname_preserves_source_order(detector):
    result = detector.normalize_name("Li Kuang")

    assert result.success
    assert result.result == "Li Kuang"
    assert result.parsed.surname == "Kuang"


def test_mixed_cjk_trailing_initial_is_not_rejected_as_non_person(detector):
    accepted = detector.normalize_name("\u738b\u5c0f\u660e\u00b7J")
    rejected = detector.normalize_name("G.\u970d\u5f17")
    embedded = detector.normalize_name("\u7f57\u4f2f\u7279\u00b7M\u00b7\u5a01\u6069\u65af\u5766")

    assert accepted.success
    assert accepted.result == "Xiao-Ming J Wang"
    assert not rejected.success
    assert rejected.error_message == "not a personal name"
    assert not embedded.success
    assert embedded.error_message == "not a personal name"


def test_mixed_initial_keeps_strong_chinese_cjk_name_shapes(detector):
    accepted = [
        "Y.\u5f20\u4f1f",
        "G.\u674e\u660e",
        "A.\u738b\u660e",
        "O.\u6b27\u9633",
    ]

    for raw_name in accepted:
        result = detector.normalize_name(raw_name)
        assert result.success, raw_name

    western_transliteration = detector.normalize_name("G.\u970d\u5f17")
    assert not western_transliteration.success
    assert western_transliteration.error_message == "not a personal name"


def test_parenthetical_surname_first_hint_suppresses_guarded_given_first_bonus(detector):
    result = detector.normalize_name("Diao (Alice) Wang")

    assert result.success
    assert result.parsed.surname == "Diao"
    assert result.parsed.given_name == "Wang"


def test_flat_timo_predict_batch_uses_batch_context():
    predictor = Predictor(PredictorConfig(parallel="never"), "")
    solo = predictor.predict_batch([Instance(name="Yan Li")])[0]
    cobatched = predictor.predict_batch(
        [
            Instance(name=name)
            for name in [
                "Wang An",
                "Yan Li",
                "Wu Gang",
                "Li Bao",
            ]
        ],
    )[1]

    assert (solo.given_name, solo.surname) == ("Yan", "Li")
    assert (cobatched.given_name, cobatched.surname) == ("Li", "Yan")
    assert cobatched.confidence is not None
    assert cobatched.format_pattern is not None


@pytest.mark.parametrize(
    ("raw_name", "expected_result"),
    [
        ("Qin Shi", "Qin Shi"),
        ("Wen Jing", "Wen Jing"),
        ("Jing Wen", "Wen Jing"),
        ("xu feng", "Xu Feng"),
    ],
)
def test_reclassified_ambiguous_baseline_cases_are_explicit(detector, raw_name, expected_result):
    result = detector.normalize_name(raw_name)

    assert result.success
    assert result.result == expected_result


def test_predictor_config_uses_sinonym_env_prefix(monkeypatch):
    monkeypatch.setenv("PARALLEL", "1")
    monkeypatch.setenv("MP_CHUNK_SIZE", "999")
    config = PredictorConfig()
    assert config.parallel == "auto"
    assert config.mp_chunk_size == 64

    monkeypatch.setenv("SINONYM_MP_CHUNK_SIZE", "17")
    assert PredictorConfig().mp_chunk_size == 17


def test_pp_abstain_treats_single_author_count_as_unreliable_batch():
    weak = route_pp_abstain_rows([_pp_abstain_row()])[0]
    mixed_long = route_pp_abstain_rows(
        [
            _pp_abstain_row(
                selected_surname_frequency=1000.0,
                has_cjk=True,
                has_latin=True,
                raw_tokens=3,
            ),
        ],
    )[0]

    assert weak["router_reason"] == "weak_zero_batch"
    assert mixed_long["router_reason"] == "zero_batch_mixed_long"


def test_pp_abstain_accepts_compound_surname_first_at_input_start():
    ordinary_three_token = route_pp_abstain_rows(
        [
            _pp_abstain_row(
                pp_result_token_count=3,
                selected_format="surname_first",
                selected_surname_position="first",
                selected_surname_token_count=1,
                raw_tokens=3,
            ),
        ],
    )[0]
    compound = route_pp_abstain_rows(
        [
            _pp_abstain_row(
                pp_result_token_count=3,
                selected_format="surname_first",
                selected_surname_position="first",
                selected_surname_token_count=2,
                raw_tokens=3,
            ),
        ],
    )[0]

    assert ordinary_three_token["router_prediction"] == "abstain"
    assert ordinary_three_token["router_reason"] == "weak_zero_batch"
    assert compound["router_prediction"] == "pp"
    assert compound["router_reason"] == "compound_surname_first_input_start"


def test_route_pp_preserves_latin_compound_surname_first_parse():
    predictor = Predictor(PredictorConfig(parallel="never"), "")

    au, ou, given_first = predictor.route_pp(["Au Yeung Ming", "Ou Yang Ming", "Ming Au Yeung"])

    assert (au.router_prediction.value, au.given_name, au.surname) == ("pp", "Ming", "Au Yeung")
    assert (ou.router_prediction.value, ou.given_name, ou.surname) == ("pp", "Ming", "Ou Yang")
    assert (given_first.router_prediction.value, given_first.given_name, given_first.surname) == (
        "abstain",
        "Ming",
        "Au Yeung",
    )


def test_internal_compound_surname_span_reports_real_width(detector):
    batch = detector.analyze_name_batch(["Wei Zhu Ge Ming"])
    result = batch.results[0]
    evidence = batch.name_order_evidence[0]

    assert result.success
    assert result.result == "Wei-Ming Zhu Ge"
    assert result.parsed is not None
    assert result.parsed.surname_tokens == ["Zhu", "Ge"]
    assert result.parsed.given_tokens == ["Wei", "Ming"]
    assert evidence.selected_surname_position == "internal"
    assert evidence.selected_surname_token_count == 2


def test_internal_compound_surname_fix_propagates_through_pp_only_routing():
    predictor = Predictor(PredictorConfig(parallel="never"), "")

    routed = predictor.route_pp(["Wei Zhu Ge Ming"])[0]

    assert routed.router_prediction.value == "abstain"
    assert routed.router_reason == "weak_zero_batch"
    assert (routed.given_name, routed.middle_name, routed.surname) == ("Wei-Ming", None, "Zhu Ge")
    assert (routed.pp.given_name, routed.pp.middle_name, routed.pp.surname) == ("Wei-Ming", None, "Zhu Ge")


def test_internal_compound_surname_fix_propagates_to_v2_canonical_name():
    predictor = PredictorV2(PredictorConfig(parallel="never"), "")

    routed = predictor.route_pp(["Wei Zhu Ge Ming"])[0]

    assert routed.router_prediction.value == "abstain"
    assert routed.router_reason == "weak_zero_batch"
    assert (routed.given_name, routed.middle_name, routed.surname) == ("Wei-Ming", None, "Zhu Ge")
    assert routed.canonical_name is not None
    assert routed.canonical_name.text == "Wei-Ming Zhu Ge"
    assert (
        routed.canonical_name.normalized.given_name,
        routed.canonical_name.normalized.middle_name,
        routed.canonical_name.normalized.surname,
    ) == ("Wei-Ming", "", "Zhu Ge")


@pytest.mark.parametrize(
    ("raw_name", "expected_result", "expected_surname", "expected_given"),
    [
        ("AuyeongWei", "Auyeong Wei", "Wei", "Auyeong"),
        ("ZhaWei", "Zha Wei", "Wei", "Zha"),
        ("ZhaWang", "Zha Wang", "Wang", "Zha"),
        ("BuLi", "Bu Li", "Li", "Bu"),
    ],
)
def test_camel_case_pair_lets_zero_frequency_first_surname_lose_to_attested_last_surname(
    detector,
    raw_name,
    expected_result,
    expected_surname,
    expected_given,
):
    result = detector.normalize_name(raw_name)

    assert result.success
    assert result.result == expected_result
    assert result.parsed.surname == expected_surname
    assert result.parsed.given_name == expected_given


def test_camel_case_pair_uses_as_written_frequency_before_zero_frequency_guard(detector):
    result = detector.normalize_name("GunWei")
    spaced = detector.normalize_name("Gun Wei")

    assert result.success
    assert result.result == spaced.result == "Gun Wei"
    assert result.parsed.surname == spaced.parsed.surname == "Wei"
    assert result.parsed.given_name == spaced.parsed.given_name == "Gun"


def test_route_pp_abstain_preserves_compact_compound_token_when_flipping_to_input_order():
    predictor = Predictor(PredictorConfig(parallel="never"), "")

    (routed,) = predictor.route_pp(["Ouyang K. Wei"])

    assert routed.router_prediction.value == "abstain"
    assert routed.router_reason == "weak_zero_batch"
    assert (routed.given_name, routed.middle_name, routed.surname) == ("Ouyang", "K", "Wei")
    assert (routed.pp.given_name, routed.pp.middle_name, routed.pp.surname) == ("Wei", "K", "Ouyang")


def test_pp_vys_accepts_mixed_selected_format_as_unknown_input_order():
    routed = route_pp_vys_abstain_rows([_pp_vys_row(pp_selected_format="mixed", vys_selected_format="mixed")])[0]

    assert routed["input_order_candidate"] == "unknown"
    assert routed["router_prediction"] in {"pp", "vys", "abstain"}


def test_required_bool_accepts_numpy_integer_booleans():
    assert _required_bool({"value": np.int64(0)}, "value") is False
    assert _required_bool({"value": np.int64(1)}, "value") is True
    with pytest.raises(ValueError, match="value must be a boolean"):
        _required_bool({"value": np.int64(2)}, "value")


def test_strong_vys_context_requires_real_batch_count():
    low_count = route_pp_vys_abstain_rows([_pp_vys_row(vys_batch_total_count=1)])[0]
    enough_count = route_pp_vys_abstain_rows([_pp_vys_row(vys_batch_total_count=3)])[0]

    assert low_count["new_reason"] == "weak_or_conflicting_evidence"
    assert enough_count["new_reason"] == "strong_vys_batch_context"
    assert enough_count["router_prediction"] == "vys"


def test_input_order_display_preserves_repeated_middle_initials(detector):
    result = detector.normalize_name("J. K. Ming L. Zhang")

    assert result.success
    assert result.parsed_original_order.order == ["middle", "middle", "given", "middle", "surname"]
    assert " ".join(token for _role, token in _input_order_display(result.parsed_original_order)) == "J K Ming L Zhang"


class BrokenJapaneseProbabilityModel:
    classes_ = ["cn", "jp"]

    def predict_proba(self, names):
        raise RuntimeError("boom")

    def predict(self, names):
        return ["jp"]


def test_japanese_probability_raises_when_loaded_model_errors(caplog):
    from sinonym.services import ethnicity

    classifier = ethnicity._MLJapaneseClassifier(confidence_threshold=0.8)
    classifier._available = True
    classifier._model = BrokenJapaneseProbabilityModel()

    with caplog.at_level(logging.WARNING, logger=ethnicity.__name__):
        with pytest.raises(RuntimeError, match="probability failed"):
            classifier.japanese_probability("\u5c71\u7530")

    assert "ML Japanese classifier probability error" in caplog.text
    assert any(record.exc_info for record in caplog.records)


def test_ml_classifier_runtime_failure_surfaces_in_detector():
    detector = Predictor(PredictorConfig(parallel="never"), "")._detector
    service = detector._ethnicity_service
    service._ml_classifier._available = True
    service._ml_classifier._model = object()
    service._ml_classifier.is_available = lambda: True
    service._ml_classifier.classify_all_chinese_name = lambda _name: ParseResult.failure(
        "ML Japanese classifier failed",
    )

    result = detector.normalize_name("\u738b\u4f1f")

    assert not result.success
    assert result.error_message == "ML Japanese classifier failed"


class FlakyJapaneseClassifierModel:
    classes_ = ["cn", "jp"]

    def __init__(self):
        self.predict_calls = 0

    def predict(self, _names):
        self.predict_calls += 1
        if self.predict_calls == 1:
            raise RuntimeError("transient model error")
        return ["cn"]

    def predict_proba(self, _names):
        return [[0.99, 0.01]]


def test_ml_classifier_runtime_failure_is_not_cached():
    from sinonym.services import ethnicity

    classifier = ethnicity._MLJapaneseClassifier(confidence_threshold=0.8)
    model = FlakyJapaneseClassifierModel()
    classifier._available = True
    classifier._model = model

    first = classifier.classify_all_chinese_name("\u738b\u4f1f")
    second = classifier.classify_all_chinese_name("\u738b\u4f1f")

    assert not first.success
    assert first.error_message == "ML Japanese classifier failed"
    assert second.success
    assert model.predict_calls == 2


def test_batch_format_requires_real_voter_share():
    predictor = Predictor(PredictorConfig(parallel="never"), "")
    service = predictor._detector._batch_analysis_service
    surname_first = ParseCandidate(["Li"], ["Wei"], 1.0, NameFormat.SURNAME_FIRST)
    mixed = ParseCandidate(["Unknown"], ["Name"], 1.0, NameFormat.MIXED)
    entries = [
        *[
            BatchCandidateEntry(f"sf-{index}", [surname_first], surname_first, {}, LATIN_ONLY_REPRESENTATION)
            for index in range(2)
        ],
        *[BatchCandidateEntry(f"mixed-{index}", [mixed], mixed, {}, LATIN_ONLY_REPRESENTATION) for index in range(8)],
    ]

    pattern = service._detect_format_pattern(entries, predictor._detector._normalizer, 0.55)

    assert pattern.surname_first_count == 2
    assert pattern.voting_count == 2
    assert pattern.total_count == 10
    assert pattern.decision_confidence == 1.0
    assert not pattern.threshold_met


def test_timo_han_only_success_reports_full_structural_confidence():
    predictor = Predictor(PredictorConfig(parallel="never"), "")

    result = predictor.predict_batch([Instance(name="\u5de9\u4fd0")])[0]

    assert result.success
    assert result.surname == "Gong"
    assert result.given_name == "Li"
    assert result.confidence == 1.0


def test_hyphenated_korean_given_name_uses_name_prior():
    routed = route_pp_vys_abstain_rows(
        [
            _pp_vys_row(
                name="Hyun-Woo Kim",
                pp_result="Kim Hyun-Woo",
                pp_selected_format="surname_first",
                pp_selected_surname_position="first",
                pp_batch_threshold_met=False,
                pp_batch_total_count=1,
                pp_batch_confidence=0.6,
                pp_batch_vote_margin=0.0,
                vys_result="Hyun-Woo Kim",
                vys_selected_format="given_first",
                vys_selected_surname_position="last",
                vys_batch_threshold_met=False,
                vys_batch_total_count=1,
                vys_batch_confidence=0.6,
                vys_batch_vote_margin=0.0,
            ),
        ],
    )[0]

    assert routed["router_prediction"] == "vys"
    assert routed["router_reason"] == "name_prior_korean_given_first_three_token"


def test_route_pp_and_pp_vys_delegate_to_single_materialization_helpers(monkeypatch):
    predictor = Predictor(PredictorConfig(), "")
    called = []

    def fake_route_pp_batch(pp_batch):
        called.append(("pp", list(pp_batch.names)))
        return []

    def fake_route_pp_vys_batches(pp_batch, pool, n):
        called.append(("pp_vys", list(pp_batch.names), list(pool.names), n))
        return []

    monkeypatch.setattr(predictor, "_route_pp_batch", fake_route_pp_batch)
    monkeypatch.setattr(predictor, "_route_pp_vys_batches", fake_route_pp_vys_batches)

    assert predictor.route_pp(["Li Wei"]) == []
    assert predictor.route_pp_vys(["Li Wei"], ["Li Wei", "Zhang Ming"]) == []
    assert called[0] == ("pp", ["Li Wei"])
    assert called[1] == ("pp_vys", ["Li Wei"], ["Li Wei", "Zhang Ming"], 1)


def test_name_order_evidence_uses_cached_raw_tokens(detector):
    batch = detector.analyze_name_batch(["Li Wei"])
    result = batch.results[0]
    entry = BatchCandidateEntry(
        "Li Wei",
        [],
        None,
        {},
        LATIN_ONLY_REPRESENTATION,
        raw_tokens=("Li", "Wei"),
    )

    class NoApplyNormalizer:
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def apply(self, name):
            raise AssertionError("normalizer.apply should not run when raw_tokens are cached")

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    evidence = detector._batch_analysis_service._build_name_order_evidence(
        [entry],
        [result],
        NoApplyNormalizer(detector._normalizer),
        BatchFormatPattern(NameFormat.SURNAME_FIRST, 1.0, 1, 0, 1, False, 1.0),
    )

    assert evidence[0].raw_tokens == ["Li", "Wei"]


def test_name_order_routing_priors_are_imported_from_canonical_data():
    assert name_order_routing.NAME_PRIOR_COMMON_CHINESE_SURNAMES is chinese_names_data.NAME_ORDER_ROUTING_COMMON_CHINESE_SURNAMES
    assert name_order_routing.NAME_PRIOR_KOREAN_SURNAMES is chinese_names_data.NAME_ORDER_ROUTING_KOREAN_SURNAMES


def test_capitalize_name_part_all_combining_marks_does_not_crash():
    """Regression: a name part consisting solely of combining marks / variation selectors
    (all Unicode category Mn) strips to '' after NFD + Mn removal. capitalize_name_part then
    indexed [0] into the empty string and raised IndexError. It must now return '' instead."""
    from sinonym.utils.string_manipulation import StringManipulationUtils

    assert StringManipulationUtils.capitalize_name_part("́") == ""           # lone combining acute
    assert StringManipulationUtils.capitalize_name_part("\U000E0100") == ""       # ideographic variation selector
    # real names are unaffected (still stripped + capitalized)
    assert StringManipulationUtils.capitalize_name_part("josé") == "Jose"
    assert StringManipulationUtils.capitalize_name_part("ou-yang") == "Ou-Yang"


def test_normalize_name_with_embedded_variation_selector_does_not_crash(detector):
    """Integration regression: a corpus name with an embedded ideographic variation selector
    (U+E0100, seen on paper_id 274957019) crashed normalize_name via the capitalize step."""
    result = detector.normalize_name("彬人 樽\U000E0100井")  # 彬人 樽󠄀井
    assert result is not None  # no exception raised; the classification value is out of scope here


@pytest.mark.parametrize(
    "raw_name",
    [
        "Shin -Ichi Hara",   # space + leading-hyphen token (117 of 124 prod failures)
        "O -P Sairanen",
        "Yang -Gyu Jei",
        "KU 'TSAO-CHUEN",    # space + leading-apostrophe token
        "O --Sl",            # double hyphen
    ],
)
def test_leading_hyphen_or_apostrophe_token_does_not_crash(detector, raw_name):
    """Regression: a stray leading-hyphen/apostrophe token made the East Asian order route
    produce invalid components, and _canonical_name_from_order_decision raised RuntimeError,
    crashing normalize_name (prod chinese_detected backfill 20260714, 124 rows). The route must
    now abstain (fall back to the baseline canonical name), not raise."""
    result = detector.normalize_name(raw_name)  # must not raise
    assert result is not None
