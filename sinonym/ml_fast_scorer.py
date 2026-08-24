"""Fast scalar scoring for the Chinese vs Japanese name classifier.

The trained artifact is a sklearn ``Pipeline`` of ``FeatureUnion([char tf-idf,
EnhancedHeuristicFlags])`` into a binary ``LogisticRegression``. Calling that
pipeline per name on 1-row matrices costs ~0.9ms/call in sklearn dispatch and
sparse-matrix overhead, dwarfing the actual math. ``FastJapaneseScorer``
extracts the fitted parameters once and evaluates the identical decision
function with plain dict lookups, which is ~2 orders of magnitude faster.

``from_skops_bytes`` reads those parameters straight out of the ``.skops``
zip (``schema.json`` plus ``.npy`` payloads) with zipfile/json/numpy, so the
runtime never imports skops, sklearn, or scipy — the skops import alone costs
~2s of cold start. ``from_pipeline`` remains available as a model-development
helper for comparison with a deserialized sklearn pipeline.
"""

from __future__ import annotations

import io
import json
import math
import re
import zipfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from sinonym.ml_flag_data import (
    CN_FREQUENT_CHARS,
    CN_NAME_ENDINGS,
    CN_SIMPLIFIED_CHARS,
    CN_SURNAME_CHARS,
    HEURISTIC_FLAG_NAMES,
    ITERATION_MARK,
    JP_FREQUENT_CHARS,
    JP_NAME_ENDINGS,
    JP_SURNAME_CHARS,
    JP_UNIQUE_CHARS,
)

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

# sklearn's char analyzer collapses runs of 2+ whitespace characters.
_WHITESPACE_RUNS = re.compile(r"\s\s+")

_CJK_START = 0x4E00
_CJK_END = 0x9FFF

_EXPECTED_TFIDF_SETTINGS = {
    "analyzer": "char",
    "sublinear_tf": True,
    "norm": "l2",
    "lowercase": True,
    "use_idf": True,
    "binary": False,
    "strip_accents": None,
    "preprocessor": None,
}

_EXPECTED_FEATURE_TRANSFORMER_ORDER = ("chars", "flags")
_EXPECTED_PIPELINE_STEP_ORDER = ("features", "clf")
_N_FLAGS = len(HEURISTIC_FLAG_NAMES)


@dataclass(frozen=True)
class FastJapaneseScorer:
    """Scores P(Japanese) with the exact decision function of the fitted pipeline.

    Attributes:
        _ngram_weights: n-gram -> (idf, idf * coefficient) for every vocabulary entry.
        _ngram_sizes: n-gram lengths extracted by the char analyzer, ascending.
        _flag_weights: coefficients for the EnhancedHeuristicFlags block.
        _intercept: logistic-regression intercept.
    """

    _ngram_weights: dict[str, tuple[float, float]]
    _ngram_sizes: tuple[int, ...]
    _flag_weights: tuple[float, ...]
    _intercept: float

    @classmethod
    def from_skops_bytes(cls, data: bytes) -> FastJapaneseScorer:
        """Build a scorer from the raw ``.skops`` artifact without importing skops.

        The artifact is a zip whose ``schema.json`` describes the object tree
        and points at ``.npy`` payload entries for arrays. Only the handful of
        fitted parameters the scorer needs are read.

        Raises:
            ValueError | KeyError: if the artifact does not match the pipeline
                structure this scorer replicates.
        """
        archive = zipfile.ZipFile(io.BytesIO(data))
        schema = json.loads(archive.read("schema.json"))
        step_items = [
            (_json_value(step["content"][0]), step["content"][1]) for step in schema["content"]["content"]["steps"]["content"]
        ]
        steps = _validated_named_items(
            step_items,
            expected=_EXPECTED_PIPELINE_STEP_ORDER,
            label="Pipeline",
        )
        feature_data = steps["features"]["content"]["content"]
        transformer_items = [
            (_json_value(item["content"][0]), item["content"][1]) for item in feature_data["transformer_list"]["content"]
        ]
        transformers = _validated_feature_transformers(transformer_items)
        tfidf = transformers["chars"]["content"]["content"]
        flags = transformers["flags"]["content"]["content"]
        clf = steps["clf"]["content"]["content"]

        if _json_value(feature_data["transformer_weights"]) is not None:
            msg = "FeatureUnion transformer_weights are not supported"
            raise ValueError(msg)
        _validate_flag_names(tuple(_json_value(node) for node in flags["flag_names"]["content"]))

        return cls._from_params(
            settings={key: _json_value(tfidf[key]) for key in _EXPECTED_TFIDF_SETTINGS},
            ngram_range=tuple(_json_value(n) for n in tfidf["ngram_range"]["content"]),
            vocabulary={ngram: int(_ndarray(archive, node)) for ngram, node in tfidf["vocabulary_"]["content"].items()},
            idf=_ndarray(archive, tfidf["_tfidf"]["content"]["content"]["idf_"]),
            coef=_ndarray(archive, clf["coef_"]),
            intercept=_ndarray(archive, clf["intercept_"]),
            classes=_ndarray(archive, clf["classes_"]),
        )

    @classmethod
    def from_pipeline(cls, pipeline: Pipeline) -> FastJapaneseScorer:
        """Build a scorer from a deserialized sklearn pipeline.

        Raises:
            ValueError | TypeError: if the pipeline does not match the structure
                this scorer replicates (feature layout, tf-idf settings, classes).
        """
        from sinonym.ml_model_components import EnhancedHeuristicFlags  # noqa: PLC0415 - keeps sklearn off the runtime path

        steps = _validated_named_items(
            pipeline.steps,
            expected=_EXPECTED_PIPELINE_STEP_ORDER,
            label="Pipeline",
        )
        features = steps["features"]
        clf = steps["clf"]
        transformers = _validated_feature_transformers(features.transformer_list)
        tfidf = transformers["chars"]
        flags = transformers["flags"]

        if features.transformer_weights is not None:
            msg = "FeatureUnion transformer_weights are not supported"
            raise ValueError(msg)
        if not isinstance(flags, EnhancedHeuristicFlags):
            msg = f"unexpected flags transformer: {type(flags).__name__}"
            raise TypeError(msg)
        _validate_flag_names(tuple(flags.flag_names))

        return cls._from_params(
            settings={key: getattr(tfidf, key) for key in _EXPECTED_TFIDF_SETTINGS},
            ngram_range=tuple(tfidf.ngram_range),
            vocabulary={ngram: int(col) for ngram, col in tfidf.vocabulary_.items()},
            idf=tfidf.idf_,
            coef=clf.coef_,
            intercept=clf.intercept_,
            classes=clf.classes_,
        )

    @classmethod
    def _from_params(
        cls,
        *,
        settings: dict[str, Any],
        ngram_range: tuple[int, int],
        vocabulary: dict[str, int],
        idf: np.ndarray,
        coef: np.ndarray,
        intercept: np.ndarray,
        classes: np.ndarray,
    ) -> FastJapaneseScorer:
        """Validate extracted parameters and construct the scorer."""
        if settings != _EXPECTED_TFIDF_SETTINGS:
            msg = f"unsupported tf-idf settings: {settings}"
            raise ValueError(msg)
        if idf.ndim != 1:
            msg = f"expected one-dimensional idf values, got shape {idf.shape}"
            raise ValueError(msg)
        n_tfidf = len(idf)
        expected_coef_shape = (1, n_tfidf + _N_FLAGS)
        if coef.shape != expected_coef_shape:
            msg = f"expected coefficient shape {expected_coef_shape}, got {coef.shape}"
            raise ValueError(msg)
        if intercept.shape != (1,):
            msg = f"expected intercept shape (1,), got {intercept.shape}"
            raise ValueError(msg)
        if classes.shape != (2,) or tuple(str(value) for value in classes) != ("cn", "jp"):
            msg = f"unexpected classes: shape={classes.shape}, values={classes.tolist()!r}"
            raise ValueError(msg)
        if len(vocabulary) != n_tfidf or set(vocabulary.values()) != set(range(n_tfidf)):
            msg = f"vocabulary columns must be a bijection over [0, {n_tfidf})"
            raise ValueError(msg)
        for label, values in (("idf", idf), ("coefficient", coef), ("intercept", intercept)):
            real_numeric = np.issubdtype(values.dtype, np.integer) or np.issubdtype(values.dtype, np.floating)
            if not real_numeric or not np.isfinite(values).all():
                msg = f"{label} values must be finite real numbers"
                raise ValueError(msg)

        min_n, max_n = ngram_range
        coef_row = coef[0]
        return cls(
            _ngram_weights={ngram: (float(idf[col]), float(idf[col] * coef_row[col])) for ngram, col in vocabulary.items()},
            _ngram_sizes=tuple(range(min_n, max_n + 1)),
            _flag_weights=tuple(float(weight) for weight in coef_row[n_tfidf:]),
            _intercept=float(intercept[0]),
        )

    def japanese_probability(self, name: str) -> float:
        """Return P(jp), identical to ``pipeline.predict_proba([name])[0][1]``."""
        return 1.0 / (1.0 + math.exp(-self._decision(name)))

    def _decision(self, name: str) -> float:
        decision = self._intercept + self._tfidf_term(name)
        if len(name) >= 2:
            for value, weight in zip(_flag_values(name), self._flag_weights, strict=True):
                if value:
                    decision += value * weight
        return decision

    def _tfidf_term(self, name: str) -> float:
        text = _WHITESPACE_RUNS.sub(" ", name.lower())
        length = len(text)
        counts: dict[str, int] = {}
        for n in self._ngram_sizes:
            for i in range(length - n + 1):
                gram = text[i : i + n]
                counts[gram] = counts.get(gram, 0) + 1

        weighted_sum = 0.0
        squared_norm = 0.0
        lookup = self._ngram_weights.get
        for gram, count in counts.items():
            weights = lookup(gram)
            if weights is None:
                continue
            idf, idf_coef = weights
            tf = 1.0 + math.log(count) if count > 1 else 1.0
            weighted_sum += tf * idf_coef
            squared_norm += (tf * idf) ** 2
        if squared_norm == 0.0:
            return 0.0
        return weighted_sum / math.sqrt(squared_norm)


def _json_value(node: dict[str, Any]) -> Any:
    """Decode a skops JsonNode."""
    if node.get("__loader__") != "JsonNode":
        msg = f"expected JsonNode, got {node.get('__loader__')!r}"
        raise ValueError(msg)
    return json.loads(node["content"])


def _validated_named_items(
    items: list[tuple[str, Any]],
    *,
    expected: tuple[str, ...],
    label: str,
) -> dict[str, Any]:
    """Return named objects only when their order exactly matches a supported layout."""
    actual = tuple(name for name, _value in items)
    if actual != expected:
        msg = f"unsupported {label} order: expected {expected!r}, got {actual!r}"
        raise ValueError(msg)
    return dict(items)


def _validated_feature_transformers(items: list[tuple[str, Any]]) -> dict[str, Any]:
    """Return transformers only when their serialized feature order is supported."""
    return _validated_named_items(
        items,
        expected=_EXPECTED_FEATURE_TRANSFORMER_ORDER,
        label="FeatureUnion",
    )


def _validate_flag_names(names: tuple[str, ...]) -> None:
    """Reject a heuristic block whose columns do not match the scalar implementation."""
    if names != HEURISTIC_FLAG_NAMES:
        msg = f"unsupported heuristic flag order: expected {HEURISTIC_FLAG_NAMES!r}, got {names!r}"
        raise ValueError(msg)


def _ndarray(archive: zipfile.ZipFile, node: dict[str, Any]) -> np.ndarray:
    """Load a skops NdArrayNode payload from the artifact zip."""
    if node.get("__loader__") != "NdArrayNode":
        msg = f"expected NdArrayNode, got {node.get('__loader__')!r}"
        raise ValueError(msg)
    return np.load(io.BytesIO(archive.read(node["file"])), allow_pickle=False)


def _flag_values(name: str) -> tuple[float, ...]:
    """Replicate ``EnhancedHeuristicFlags.transform`` for one name of length >= 2."""
    chars = list(name)
    first_char = chars[0]
    last_char = chars[-1]
    return (
        float(ITERATION_MARK in name),
        float(any(c in JP_SURNAME_CHARS for c in chars[:2])),
        float(any(c in CN_SURNAME_CHARS for c in chars[:2])),
        float(last_char in JP_NAME_ENDINGS),
        float(last_char in CN_NAME_ENDINGS),
        float(any(c in JP_UNIQUE_CHARS for c in chars)),
        float(any(c in CN_SIMPLIFIED_CHARS for c in chars)),
        float(len(name) == 2),
        float(len(name) == 3),
        float(len(name) >= 4),
        float(any(c in JP_FREQUENT_CHARS for c in chars)),
        float(any(c in CN_FREQUENT_CHARS for c in chars)),
        float(first_char in JP_SURNAME_CHARS),
        float(first_char in CN_SURNAME_CHARS),
        float(any(c in JP_NAME_ENDINGS for c in chars[1:])),
        float(any(c in CN_NAME_ENDINGS for c in chars[1:])),
        sum(1 for c in chars if c in JP_NAME_ENDINGS) / len(chars),
        sum(1 for c in chars if c in CN_NAME_ENDINGS) / len(chars),
        len(set(chars)) / len(chars),
        _stroke_complexity(chars),
    )


def _stroke_complexity(chars: list[str]) -> float:
    """Replicate ``EnhancedHeuristicFlags._estimate_stroke_complexity``."""
    scores = [(code - _CJK_START) / (_CJK_END - _CJK_START) for code in map(ord, chars) if _CJK_START <= code <= _CJK_END]
    return sum(scores) / len(scores) if scores else 0.0
