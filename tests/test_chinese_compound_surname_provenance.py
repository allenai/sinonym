"""Regression coverage for native-script Chinese compound-surname evidence."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize("raw_name", ["裕 吉川", "中山 和貴"])
def test_generated_pinyin_pair_does_not_override_confident_japanese(detector, raw_name: str) -> None:
    """Accidental Pinyin lexicon matches are not authored Chinese evidence."""
    result = detector.normalize_name(raw_name)

    assert not result.success
    assert result.error_message == "Japanese name detected by ML classifier"


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("上官 婉儿", "Wan-Er Shang Guan"),
        ("欧阳 伟", "Wei Ou Yang"),
        ("司马 光", "Guang Si Ma"),
    ],
)
def test_authored_han_compound_surname_retains_chinese_rescue(
    detector,
    raw_name: str,
    expected: str,
) -> None:
    """Curated Han compound surnames remain affirmative Chinese evidence."""
    result = detector.normalize_name(raw_name)

    assert result.success
    assert result.result == expected
