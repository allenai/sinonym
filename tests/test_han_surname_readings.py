"""Tests for surname-position-specific Han romanization readings."""

import pytest

from sinonym.timo.interface import Instance, Predictor, SourceAuthorFields


@pytest.mark.parametrize(
    ("raw_name", "expected", "surname"),
    [
        ("曾义权", "Yi-Quan Zeng", "Zeng"),
        ("曾显斌", "Xian-Bin Zeng", "Zeng"),
        ("曾向红", "Xiang-Hong Zeng", "Zeng"),
        ("明曾", "Ming Zeng", "Zeng"),
        ("仇洪冰", "Hong-Bing Qiu", "Qiu"),
    ],
)
def test_contextual_reading_is_used_for_assigned_han_surname(detector, raw_name, expected, surname):
    result = detector.normalize_name(raw_name)

    assert result.success
    assert result.result == expected
    assert result.parsed.surname == surname
    assert result.canonical_name.normalized.surname == surname
    assert result.result.isascii()


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [("陈曾明", "Ceng-Ming Chen"), ("陈仇明", "Chou-Ming Chen")],
)
def test_contextual_surname_reading_does_not_change_han_given_name(detector, raw_name, expected):
    result = detector.normalize_name(raw_name)

    assert result.success
    assert result.result == expected
    assert result.parsed.given_name == expected.split()[0]


@pytest.mark.parametrize(
    ("raw_name", "expected", "surname", "check_frequency"),
    [
        ("曾义权", "Yi-Quan Zeng", "Zeng", True),
        ("仇洪冰", "Hong-Bing Qiu", "Qiu", False),
    ],
)
def test_contextual_surname_reading_reaches_batch_results(detector, raw_name, expected, surname, check_frequency):
    batch = detector.analyze_name_batch([raw_name, "李小明"])
    evidence = batch.name_order_evidence[0]
    assert batch.results[0].result == expected
    assert batch.results[0].parsed.surname == surname
    assert evidence.selected_surname_position == "first"
    assert not check_frequency or evidence.selected_surname_frequency > 0


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (SourceAuthorFields(last_name="\u66fe\u4e49\u6743"), ("Yi-Quan", "", "Zeng")),
        (SourceAuthorFields(last_name="\u66fe\u663e\u658c"), ("Xian-Bin", "", "Zeng")),
        (SourceAuthorFields(last_name="\u660e\u66fe"), ("Ming", "", "Zeng")),
        (SourceAuthorFields(last_name="\u9648\u66fe\u660e"), ("Ceng-Ming", "", "Chen")),
        (SourceAuthorFields(first_name="\u66fd", last_name="\u5c1a\u6587"), ("Shang-Wen", "", "Ceng")),
        (SourceAuthorFields(last_name="\u4ec7\u6d2a\u51b0"), ("Hong-Bing", "", "Qiu")),
    ],
)
def test_contextual_reading_preserves_v3_name_boundary(
    predictor: Predictor,
    source: SourceAuthorFields,
    expected: tuple[str, str, str],
) -> None:
    (paper,) = predictor.predict_batch([Instance(pp_authors=[source])])
    resolved = paper.authors[0]

    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == expected


@pytest.mark.parametrize(
    ("raw_name", "expected_result", "surname"),
    [
        ("Ceng Ming", None, "Ceng"),
        ("Ceng 曾 Yi 义", None, "Ceng"),
        ("Chou Hongbing", "Hong-Bing Chou", "Chou"),
        ("Hongbing Chou", "Hong-Bing Chou", "Chou"),
    ],
)
def test_contextual_reading_does_not_rewrite_explicit_roman_source(
    detector,
    raw_name,
    expected_result,
    surname,
):
    result = detector.normalize_name(raw_name)

    assert result.success
    if expected_result is not None:
        assert result.result == expected_result
    assert result.parsed.surname == surname


def test_zeng_reading_does_not_change_japanese_rejection(detector):
    result = detector.normalize_name("曾根綾子")

    assert not result.success
    assert result.error_message == "Japanese name detected by ML classifier"
