"""Tests for surname-position-specific Han romanization readings."""

import pytest

from sinonym.timo.interface import RoutingPredictorV3
from sinonym.timo.routing_v3 import RoutingInstanceV3, SourceAuthorFields


@pytest.fixture
def routing_predictor(routing_predictor_v3: RoutingPredictorV3) -> RoutingPredictorV3:
    """Alias the shared session predictor under this module's name."""
    return routing_predictor_v3


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("曾义权", "Yi-Quan Zeng"),
        ("曾显斌", "Xian-Bin Zeng"),
        ("曾向红", "Xiang-Hong Zeng"),
        ("明曾", "Ming Zeng"),
    ],
)
def test_zeng_reading_is_used_for_assigned_han_surname(detector, raw_name, expected):
    result = detector.normalize_name(raw_name)

    assert result.success
    assert result.result == expected
    assert result.parsed.surname == "Zeng"
    assert result.canonical_name.normalized.surname == "Zeng"
    assert result.result.isascii()


def test_zeng_reading_does_not_change_han_given_name(detector):
    result = detector.normalize_name("陈曾明")

    assert result.success
    assert result.result == "Ceng-Ming Chen"
    assert result.parsed.given_name == "Ceng-Ming"


def test_zeng_surname_reading_reaches_batch_results(detector):
    batch = detector.analyze_name_batch(["曾义权", "李小明"])

    assert batch.results[0].result == "Yi-Quan Zeng"
    assert batch.results[0].parsed.surname == "Zeng"
    assert batch.name_order_evidence[0].selected_surname_position == "first"
    assert batch.name_order_evidence[0].selected_surname_frequency > 0


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (SourceAuthorFields(last_name="\u66fe\u4e49\u6743"), ("Yi-Quan", "", "Zeng")),
        (SourceAuthorFields(last_name="\u66fe\u663e\u658c"), ("Xian-Bin", "", "Zeng")),
        (SourceAuthorFields(last_name="\u660e\u66fe"), ("Ming", "", "Zeng")),
        (SourceAuthorFields(last_name="\u9648\u66fe\u660e"), ("Ceng-Ming", "", "Chen")),
        (SourceAuthorFields(first_name="\u66fd", last_name="\u5c1a\u6587"), ("Shang-Wen", "", "Ceng")),
    ],
)
def test_zeng_reading_preserves_v3_name_boundary(
    routing_predictor: RoutingPredictorV3,
    source: SourceAuthorFields,
    expected: tuple[str, str, str],
) -> None:
    """A contextual reading must not change V3's surname endpoint."""
    (paper,) = routing_predictor.predict_batch([RoutingInstanceV3(pp_authors=[source])])
    resolved = paper.authors[0].resolved_fields

    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == expected


@pytest.mark.parametrize("raw_name", ["Ceng Ming", "Ceng 曾 Yi 义"])
def test_zeng_reading_does_not_rewrite_explicit_roman_source(detector, raw_name):
    result = detector.normalize_name(raw_name)

    assert result.success
    assert result.parsed.surname == "Ceng"


def test_zeng_reading_does_not_change_japanese_rejection(detector):
    result = detector.normalize_name("曾根綾子")

    assert not result.success
    assert result.error_message == "Japanese name detected by ML classifier"
