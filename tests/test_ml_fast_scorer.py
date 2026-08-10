"""Runtime and performance tests for the fast Japanese-classifier scorer.

``FastJapaneseScorer`` re-implements the fitted sklearn pipeline's decision
function with plain dict lookups. These tests verify the production artifact
reader and guard the performance win that motivated it.
"""

from __future__ import annotations

import io
import itertools
import json
import time
import zipfile
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import skops.io as skops_io
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from sinonym import ChineseNameDetector
from sinonym.ml_fast_scorer import FastJapaneseScorer
from sinonym.resources import read_bytes

if TYPE_CHECKING:
    from collections.abc import Callable

# Han-only names driven through the full detector for the end-to-end guard.
HAN_SURNAME_CHARS = "王李张刘陈杨黄赵吴周徐孙马朱胡郭何高林罗"
HAN_GIVEN_CHARS = "伟芳娜秀敏静丽强磊军洋勇艳杰娟涛明超霞平"

# A reintroduced per-name sklearn call costs ~0.9ms alone, so these bounds are
# generous for slow machines yet impossible to meet with the old code path.
MAX_SCORER_MICROSECONDS_PER_NAME = 200
MIN_UNIQUE_HAN_NAMES_PER_SECOND = 900


@pytest.fixture(scope="module")
def artifact_bytes():
    return read_bytes("chinese_japanese_classifier.skops")


@pytest.fixture(scope="module")
def sklearn_pipeline(artifact_bytes):
    trusted = skops_io.get_untrusted_types(data=artifact_bytes)
    return skops_io.loads(artifact_bytes, trusted=trusted)


@pytest.fixture(scope="module")
def scorer(artifact_bytes):
    return FastJapaneseScorer.from_skops_bytes(artifact_bytes)


def _rewrite_schema(data: bytes, update: Callable[[dict[str, Any]], None]) -> bytes:
    """Return an in-memory artifact with one deliberate schema mutation."""
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as source:
        schema = json.loads(source.read("schema.json"))
        update(schema)
        with zipfile.ZipFile(output, "w") as target:
            for entry in source.infolist():
                payload = json.dumps(schema).encode() if entry.filename == "schema.json" else source.read(entry.filename)
                target.writestr(entry, payload)
    return output.getvalue()


def _schema_feature_data(schema: dict[str, Any]) -> dict[str, Any]:
    """Return the FeatureUnion state from a skops schema."""
    steps = schema["content"]["content"]["steps"]["content"]
    for step in steps:
        if json.loads(step["content"][0]["content"]) == "features":
            return step["content"][1]["content"]["content"]
    message = "test artifact has no features step"
    raise AssertionError(message)


def _assert_both_loaders_reject(pipeline, message: str) -> None:
    """Assert development and runtime loaders reject the same corrupt model."""
    with pytest.raises(ValueError, match=message):
        FastJapaneseScorer.from_pipeline(pipeline)
    with pytest.raises(ValueError, match=message):
        FastJapaneseScorer.from_skops_bytes(skops_io.dumps(pipeline))


@pytest.mark.parametrize(
    "name",
    [
        "",
        "A",
        "\u738b\u4f1f",
        "\u5c71\u7530\u592a\u90ce",
        "\u4f50\u3005\u6728\u514b\u5178",
        " \u4f50\u3005\u6728 \u514b\u5178 ",
        "\u4f50\u3005\u6728\ufe0f\u514b\u5178",
    ],
)
def test_supported_artifact_loaders_match_sklearn(name, artifact_bytes, sklearn_pipeline, scorer):
    expected = float(sklearn_pipeline.predict_proba([name])[0][1])
    assert scorer.japanese_probability(name) == pytest.approx(expected, abs=1e-12)
    assert FastJapaneseScorer.from_pipeline(sklearn_pipeline).japanese_probability(name) == pytest.approx(
        expected,
        abs=1e-12,
    )


def test_pipeline_loader_rejects_reordered_feature_blocks(sklearn_pipeline):
    reordered = deepcopy(sklearn_pipeline)
    features = reordered.named_steps["features"]
    original_coef = reordered.named_steps["clf"].coef_.copy()
    n_chars = len(dict(features.transformer_list)["chars"].vocabulary_)
    features.transformer_list = list(reversed(features.transformer_list))
    reordered.named_steps["clf"].coef_ = np.concatenate(
        (original_coef[:, n_chars:], original_coef[:, :n_chars]),
        axis=1,
    )

    names = ["\u738b\u4f1f", "\u5c71\u7530\u592a\u90ce", "\u4f50\u3005\u6728\u514b\u5178"]
    assert reordered.predict_proba(names)[:, 1] == pytest.approx(sklearn_pipeline.predict_proba(names)[:, 1])
    with pytest.raises(ValueError, match="unsupported FeatureUnion order"):
        FastJapaneseScorer.from_pipeline(reordered)


def test_skops_loader_rejects_reordered_feature_blocks(artifact_bytes):
    def reverse_transformers(schema):
        _schema_feature_data(schema)["transformer_list"]["content"].reverse()

    reordered = _rewrite_schema(artifact_bytes, reverse_transformers)
    with pytest.raises(ValueError, match="unsupported FeatureUnion order"):
        FastJapaneseScorer.from_skops_bytes(reordered)


def test_loaders_reject_extra_pipeline_steps(sklearn_pipeline):
    extended = Pipeline(
        [
            ("identity", FunctionTransformer()),
            *deepcopy(sklearn_pipeline).steps,
        ],
    )
    names = ["\u738b\u4f1f", "\u5c71\u7530\u592a\u90ce"]

    assert extended.predict_proba(names) == pytest.approx(sklearn_pipeline.predict_proba(names))
    _assert_both_loaders_reject(extended, "unsupported Pipeline order")


def test_loaders_reject_reordered_heuristic_flags(artifact_bytes, sklearn_pipeline):
    pipeline = deepcopy(sklearn_pipeline)
    flags = dict(pipeline.named_steps["features"].transformer_list)["flags"]
    flags.flag_names[0], flags.flag_names[1] = flags.flag_names[1], flags.flag_names[0]
    with pytest.raises(ValueError, match="unsupported heuristic flag order"):
        FastJapaneseScorer.from_pipeline(pipeline)

    def reverse_flags(schema):
        transformers = _schema_feature_data(schema)["transformer_list"]["content"]
        flags_node = next(item["content"][1] for item in transformers if json.loads(item["content"][0]["content"]) == "flags")
        flags_node["content"]["content"]["flag_names"]["content"].reverse()

    reordered = _rewrite_schema(artifact_bytes, reverse_flags)
    with pytest.raises(ValueError, match="unsupported heuristic flag order"):
        FastJapaneseScorer.from_skops_bytes(reordered)


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("coefficient_rows", "expected coefficient shape"),
        ("intercept_dimensions", "expected intercept shape"),
        ("class_dimensions", "unexpected classes"),
        ("idf_dimensions", "expected one-dimensional idf"),
    ],
)
def test_loaders_reject_parameter_shape_drift(sklearn_pipeline, corruption: str, message: str):
    pipeline = deepcopy(sklearn_pipeline)
    tfidf = dict(pipeline.named_steps["features"].transformer_list)["chars"]
    clf = pipeline.named_steps["clf"]
    if corruption == "coefficient_rows":
        clf.coef_ = np.vstack((clf.coef_, -clf.coef_))
    elif corruption == "intercept_dimensions":
        clf.intercept_ = clf.intercept_.reshape(1, 1)
    elif corruption == "class_dimensions":
        clf.classes_ = clf.classes_.reshape(1, 2)
    else:
        tfidf._tfidf.__dict__["idf_"] = tfidf.idf_.reshape(1, -1)  # noqa: SLF001 - deliberate fitted-state corruption

    _assert_both_loaders_reject(pipeline, message)


@pytest.mark.parametrize("non_finite", [np.nan, np.inf, -np.inf])
def test_loaders_reject_non_finite_parameters(sklearn_pipeline, non_finite: float):
    pipeline = deepcopy(sklearn_pipeline)
    pipeline.named_steps["clf"].coef_[0, 0] = non_finite

    _assert_both_loaders_reject(pipeline, "coefficient values must be finite real numbers")


def test_loaders_reject_complex_parameters(sklearn_pipeline):
    pipeline = deepcopy(sklearn_pipeline)
    pipeline.named_steps["clf"].coef_ = pipeline.named_steps["clf"].coef_.astype(np.complex128) + 1j

    _assert_both_loaders_reject(pipeline, "coefficient values must be finite real numbers")


@pytest.mark.parametrize("extra_entry", [False, True], ids=["duplicate-column", "extra-token"])
def test_loaders_require_vocabulary_columns_to_be_a_bijection(sklearn_pipeline, extra_entry: bool):
    pipeline = deepcopy(sklearn_pipeline)
    vocabulary = dict(pipeline.named_steps["features"].transformer_list)["chars"].vocabulary_
    first, second = itertools.islice(vocabulary, 2)
    if extra_entry:
        vocabulary["__unsupported_extra_token__"] = vocabulary[first]
    else:
        vocabulary[second] = vocabulary[first]

    _assert_both_loaders_reject(pipeline, "vocabulary columns must be a bijection")


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
