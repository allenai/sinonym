"""Parity and performance tests for the fast Japanese-classifier scorer.

``FastJapaneseScorer`` re-implements the fitted sklearn pipeline's decision
function with plain dict lookups. These tests pin numerical parity against the
actual pipeline artifact and guard the performance win that motivated it.
"""

from __future__ import annotations

import itertools
import math
import time

import pytest

import sinonym.ml_model_components  # noqa: F401 - required to deserialize the model
from sinonym import ChineseNameDetector
from sinonym.ml_fast_scorer import FastJapaneseScorer
from sinonym.resources import load_skops, read_bytes

# Diverse inputs: Japanese, Chinese, edge cases, non-CJK, and analyzer quirks
# (lowercasing, whitespace collapsing, out-of-vocabulary characters).
PARITY_NAMES = [
    "山田太郎",
    "佐藤健一",
    "田中花子",
    "鈴木一郎",
    "高橋愛",
    "佐々木希",
    "王伟",
    "陈志强",
    "刘德华",
    "欧阳修",
    "司马光",
    "张伟",
    "王",
    "",
    "々",
    "山田  太郎",
    "a\t\t山田b",
    "John Smith",
    "JOHN SMITH",
    "MIXED山田",
    "김민준",
    "1234!?",
    "王伟" * 40,
]

# Han-only names driven through the full detector for the end-to-end guard.
HAN_SURNAME_CHARS = "王李张刘陈杨黄赵吴周徐孙马朱胡郭何高林罗"
HAN_GIVEN_CHARS = "伟芳娜秀敏静丽强磊军洋勇艳杰娟涛明超霞平"

# A reintroduced per-name sklearn call costs ~0.9ms alone, so these bounds are
# generous for slow machines yet impossible to meet with the old code path.
MAX_SCORER_MICROSECONDS_PER_NAME = 200
MIN_UNIQUE_HAN_NAMES_PER_SECOND = 900


@pytest.fixture(scope="module")
def pipeline():
    return load_skops("chinese_japanese_classifier.skops")


@pytest.fixture(scope="module")
def scorer(pipeline):
    return FastJapaneseScorer.from_pipeline(pipeline)


def test_from_skops_bytes_is_bit_identical_to_from_pipeline(scorer):
    """The minimal artifact reader must extract exactly what skops deserializes."""
    zip_scorer = FastJapaneseScorer.from_skops_bytes(read_bytes("chinese_japanese_classifier.skops"))
    assert zip_scorer == scorer


def test_detector_ml_classifier_is_loaded():
    """Load failures are downgraded to a warning, so pin availability explicitly."""
    detector = ChineseNameDetector()
    assert detector._ethnicity_service._ml_classifier.is_available()  # noqa: SLF001


def test_scorer_matches_pipeline_probabilities(pipeline, scorer):
    jp_column = list(pipeline.classes_).index("jp")
    expected = pipeline.predict_proba(PARITY_NAMES)[:, jp_column]

    for name, expected_probability in zip(PARITY_NAMES, expected, strict=True):
        actual = scorer.japanese_probability(name)
        assert math.isclose(actual, expected_probability, rel_tol=1e-9, abs_tol=1e-12), (
            f"{name!r}: scorer={actual!r} pipeline={expected_probability!r}"
        )


def test_scorer_matches_pipeline_rejection_decisions(pipeline, scorer):
    """The classifier's reject rule must be unchanged: predict()=='jp' with confidence >= 0.8."""
    threshold = 0.8
    predictions = pipeline.predict(PARITY_NAMES)
    probabilities = pipeline.predict_proba(PARITY_NAMES)

    for name, prediction, probability_row in zip(PARITY_NAMES, predictions, probabilities, strict=True):
        old_rule = prediction == "jp" and max(probability_row) >= threshold
        jp_probability = scorer.japanese_probability(name)
        new_rule = jp_probability > 0.5 and jp_probability >= threshold
        assert new_rule == old_rule, f"{name!r}: rejection decision diverged"


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
