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
~2s of cold start. ``from_pipeline`` builds the same scorer from a
deserialized sklearn pipeline and anchors the parity tests in
``tests/test_ml_fast_scorer.py``.
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

# Width of the EnhancedHeuristicFlags feature block; must match _flag_values().
_N_FLAGS = 20


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
        steps = {_json_value(step["content"][0]): step["content"][1] for step in schema["content"]["content"]["steps"]["content"]}
        transformers = {
            _json_value(item["content"][0]): item["content"][1]
            for item in steps["features"]["content"]["content"]["transformer_list"]["content"]
        }
        tfidf = transformers["chars"]["content"]["content"]
        clf = steps["clf"]["content"]["content"]

        return cls._from_params(
            settings={key: _json_value(tfidf[key]) for key in _EXPECTED_TFIDF_SETTINGS},
            ngram_range=tuple(_json_value(n) for n in tfidf["ngram_range"]["content"]),
            vocabulary={ngram: int(_ndarray(archive, node)) for ngram, node in tfidf["vocabulary_"]["content"].items()},
            idf=_ndarray(archive, tfidf["_tfidf"]["content"]["content"]["idf_"]),
            coef=_ndarray(archive, clf["coef_"])[0],
            intercept=float(_ndarray(archive, clf["intercept_"])[0]),
            classes=[str(c) for c in _ndarray(archive, clf["classes_"])],
        )

    @classmethod
    def from_pipeline(cls, pipeline: Pipeline) -> FastJapaneseScorer:
        """Build a scorer from a deserialized sklearn pipeline.

        Raises:
            ValueError | TypeError: if the pipeline does not match the structure
                this scorer replicates (feature layout, tf-idf settings, classes).
        """
        from sinonym.ml_model_components import EnhancedHeuristicFlags  # noqa: PLC0415 - keeps sklearn off the runtime path

        features = pipeline.named_steps["features"]
        clf = pipeline.named_steps["clf"]
        transformers = dict(features.transformer_list)
        tfidf = transformers["chars"]
        flags = transformers["flags"]

        if features.transformer_weights is not None:
            msg = "FeatureUnion transformer_weights are not supported"
            raise ValueError(msg)
        if not isinstance(flags, EnhancedHeuristicFlags):
            msg = f"unexpected flags transformer: {type(flags).__name__}"
            raise TypeError(msg)
        if len(flags.flag_names) != _N_FLAGS:
            msg = f"expected {_N_FLAGS} flags, model has {len(flags.flag_names)}"
            raise ValueError(msg)

        return cls._from_params(
            settings={key: getattr(tfidf, key) for key in _EXPECTED_TFIDF_SETTINGS},
            ngram_range=tuple(tfidf.ngram_range),
            vocabulary={ngram: int(col) for ngram, col in tfidf.vocabulary_.items()},
            idf=tfidf.idf_,
            coef=clf.coef_[0],
            intercept=float(clf.intercept_[0]),
            classes=[str(c) for c in clf.classes_],
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
        intercept: float,
        classes: list[str],
    ) -> FastJapaneseScorer:
        """Validate extracted parameters and construct the scorer."""
        if settings != _EXPECTED_TFIDF_SETTINGS:
            msg = f"unsupported tf-idf settings: {settings}"
            raise ValueError(msg)
        if classes != ["cn", "jp"]:
            msg = f"unexpected classes: {classes!r}"
            raise ValueError(msg)
        n_tfidf = len(idf)
        if len(vocabulary) != n_tfidf or len(coef) != n_tfidf + _N_FLAGS:
            msg = f"layout mismatch: {len(coef)} coefficients, {n_tfidf} tf-idf features, {len(vocabulary)} vocabulary entries"
            raise ValueError(msg)

        min_n, max_n = ngram_range
        return cls(
            _ngram_weights={ngram: (float(idf[col]), float(idf[col] * coef[col])) for ngram, col in vocabulary.items()},
            _ngram_sizes=tuple(range(min_n, max_n + 1)),
            _flag_weights=tuple(float(w) for w in coef[n_tfidf:]),
            _intercept=float(intercept),
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
