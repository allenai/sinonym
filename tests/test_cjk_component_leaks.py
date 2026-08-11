"""Compatibility-ideograph behavior in raw diagnostic parsing."""

from __future__ import annotations

import pytest

from tests._case_assertions import assert_person_normalized_name, assert_rejected


@pytest.mark.parametrize(
    ("raw", "given", "surname"),
    [
        ("\uae40\ud6a8\uc9c4", "\ud6a8\uc9c4", "\uae40"),
        ("\u5409\u7530 \u9686", "\u9686", "\u5409\u7530"),
    ],
)
def test_all_cjk_segmentation_of_all_cjk_input_is_kept(detector, raw, given, surname):
    result = detector.normalize_name(raw)

    assert not result.success
    canonical = result.canonical_name
    assert canonical is not None
    assert canonical.normalized.given_name == given
    assert canonical.normalized.surname == surname


@pytest.mark.parametrize(
    "raw",
    [
        "\u7530\ufa11 \u4fee",
        "\u91ce\ufa11 \u6dbc\u592a\u90ce",
        "\u5c71\ufa11 \u6d0b\u8f14",
        "\u6c50\ufa11 \u7dbe\u5b50",
        "\u7530\u5d0e \u4fee",
    ],
)
def test_compatibility_ideograph_saki_names_classify_as_japanese(detector, raw):
    assert_rejected(detector, raw)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("\u7530\ufa11 \u4fee", "\u4fee \u7530\ufa11"),
        ("\u5ca9\ufa11 \u4e00\u90ce", "\u4e00\u90ce \u5ca9\ufa11"),
        ("\u5ca1\ufa11 \u6075\u7f8e\u5b50", "\u6075\u7f8e\u5b50 \u5ca1\ufa11"),
        ("\u7530\u5d0e \u4fee", "\u4fee \u7530\u5d0e"),
        ("\u6dbc\u592a\u90ce \u91ce\ufa11", "\u6dbc\u592a\u90ce \u91ce\ufa11"),
        ("\u6ff1\ufa11 \u5c06\u81e3", "\u5c06\u81e3 \u6ff1\ufa11"),
        ("\u9593\ufa11 \u5149", "\u5149 \u9593\ufa11"),
    ],
)
def test_spaced_kanji_family_first_recognises_compatibility_ideographs(detector, raw, expected):
    person = detector.normalize_person_name(raw)

    assert_person_normalized_name(person, raw, expected)
