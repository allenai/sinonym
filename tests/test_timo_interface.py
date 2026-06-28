import pytest
from pydantic import ValidationError

from sinonym.timo.interface import (
    FormatPattern,
    Instance,
    NameFormatValue,
    Prediction,
    Predictor,
    PredictorConfig,
    ScriptRepresentationValue,
)

EXPECTED_BATCH_CONTEXT_RESULT_COUNT = 3
TIE_CONFIDENCE = 0.5


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

    results[0].format_pattern.threshold_met = False
    assert results[0].format_pattern.threshold_met is False
    assert isinstance(results[1].format_pattern.dominant_format, NameFormatValue)
    assert results[1].format_pattern.dominant_format == NameFormatValue.SURNAME_FIRST

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
