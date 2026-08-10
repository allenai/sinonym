"""Runtime and performance tests for the fast Japanese-classifier scorer.

``FastJapaneseScorer`` re-implements the fitted sklearn pipeline's decision
function with plain dict lookups. These tests verify the production artifact
reader and guard the performance win that motivated it.
"""

from __future__ import annotations

import itertools
import time

import pytest

from sinonym import ChineseNameDetector
from sinonym.ml_fast_scorer import FastJapaneseScorer
from sinonym.resources import read_bytes

# Han-only names driven through the full detector for the end-to-end guard.
HAN_SURNAME_CHARS = "王李张刘陈杨黄赵吴周徐孙马朱胡郭何高林罗"
HAN_GIVEN_CHARS = "伟芳娜秀敏静丽强磊军洋勇艳杰娟涛明超霞平"

# A reintroduced per-name sklearn call costs ~0.9ms alone, so these bounds are
# generous for slow machines yet impossible to meet with the old code path.
MAX_SCORER_MICROSECONDS_PER_NAME = 200
MIN_UNIQUE_HAN_NAMES_PER_SECOND = 900


@pytest.fixture(scope="module")
def scorer():
    return FastJapaneseScorer.from_skops_bytes(read_bytes("chinese_japanese_classifier.skops"))


def test_detector_ml_classifier_is_loaded():
    """Load failures are downgraded to a warning, so pin availability explicitly."""
    detector = ChineseNameDetector()
    assert detector._ethnicity_service._ml_classifier.is_available()  # noqa: SLF001


def test_scorer_is_fast(scorer):
    names = [s + a + b for s, a, b in itertools.product(HAN_SURNAME_CHARS, HAN_GIVEN_CHARS, HAN_GIVEN_CHARS)][:2000]
    for name in names[:200]:  # warm up
        scorer.japanese_probability(name)

    start = time.perf_counter()
    for name in names:
        scorer.japanese_probability(name)
    elapsed = time.perf_counter() - start

    microseconds_per_name = elapsed / len(names) * 1e6
    assert microseconds_per_name < MAX_SCORER_MICROSECONDS_PER_NAME, (
        f"scorer took {microseconds_per_name:.1f}us/name, expected < {MAX_SCORER_MICROSECONDS_PER_NAME}us"
    )


def test_unique_han_name_throughput():
    """End-to-end guard: unique Han-character names no longer pay 1-row sklearn calls."""
    detector = ChineseNameDetector()
    names = [s + a + b for s, a, b in itertools.product(HAN_SURNAME_CHARS, HAN_GIVEN_CHARS, HAN_GIVEN_CHARS)][:2500]
    for name in names[:500]:  # warm up code paths and per-character caches
        detector.normalize_name(name)

    start = time.perf_counter()
    for name in names[500:]:
        detector.normalize_name(name)
    elapsed = time.perf_counter() - start

    names_per_second = (len(names) - 500) / elapsed
    assert names_per_second > MIN_UNIQUE_HAN_NAMES_PER_SECOND, (
        f"unique Han-name throughput {names_per_second:.0f} names/sec is below {MIN_UNIQUE_HAN_NAMES_PER_SECOND} names/sec"
    )
