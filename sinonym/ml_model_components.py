"""
ML Model Components for Chinese vs Japanese Name Classification

This module contains the custom transformer classes needed to deserialize the
pre-trained ML model with skops (training and parity tests). Runtime inference
uses ``sinonym.ml_fast_scorer`` instead and never imports this module.
"""

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin

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

__all__ = [
    "CN_FREQUENT_CHARS",
    "CN_NAME_ENDINGS",
    "CN_SIMPLIFIED_CHARS",
    "CN_SURNAME_CHARS",
    "HEURISTIC_FLAG_NAMES",
    "ITERATION_MARK",
    "JP_FREQUENT_CHARS",
    "JP_NAME_ENDINGS",
    "JP_SURNAME_CHARS",
    "JP_UNIQUE_CHARS",
    "EnhancedHeuristicFlags",
]


class EnhancedHeuristicFlags(BaseEstimator, TransformerMixin):
    """Enhanced transformer with improved linguistic features for Chinese vs Japanese classification."""

    def __init__(self):
        self.flag_names = list(HEURISTIC_FLAG_NAMES)

    def fit(self, X, y=None):
        """Fit method (no-op for this transformer)."""
        return self

    def transform(self, X):
        """Transform names into heuristic feature vectors."""
        rows, cols, data = [], [], []

        for i, name in enumerate(X):
            if len(name) < 2:
                continue

            # Basic character analysis
            chars = list(name)
            first_char = chars[0]
            last_char = chars[-1]

            # Calculate features
            features = [
                # Original features
                ITERATION_MARK in name,
                any(c in JP_SURNAME_CHARS for c in chars[:2]),  # First 2 chars
                any(c in CN_SURNAME_CHARS for c in chars[:2]),
                last_char in JP_NAME_ENDINGS,
                last_char in CN_NAME_ENDINGS,
                any(c in JP_UNIQUE_CHARS for c in chars),
                any(c in CN_SIMPLIFIED_CHARS for c in chars),
                len(name) == 2,
                len(name) == 3,
                len(name) >= 4,
                # Enhanced features
                sum(1 for c in chars if c in JP_FREQUENT_CHARS) > 0,
                sum(1 for c in chars if c in CN_FREQUENT_CHARS) > 0,
                first_char in JP_SURNAME_CHARS,
                first_char in CN_SURNAME_CHARS,
                any(c in JP_NAME_ENDINGS for c in chars[1:]),  # Given name area
                any(c in CN_NAME_ENDINGS for c in chars[1:]),
                sum(1 for c in chars if c in JP_NAME_ENDINGS) / len(chars),  # Ratio
                sum(1 for c in chars if c in CN_NAME_ENDINGS) / len(chars),
                len(set(chars)) / len(chars),  # Character diversity
                self._estimate_stroke_complexity(chars),
            ]

            for j, val in enumerate(features):
                if isinstance(val, bool) and val:
                    rows.append(i)
                    cols.append(j)
                    data.append(1)
                elif isinstance(val, (int, float)) and val > 0:
                    rows.append(i)
                    cols.append(j)
                    data.append(float(val))

        n_samples = len(X)
        n_features = len(self.flag_names)
        return sparse.csr_matrix((data, (rows, cols)), shape=(n_samples, n_features))

    def _estimate_stroke_complexity(self, chars):
        """Rough estimate of average stroke complexity."""
        complexity_scores = []
        for char in chars:
            char_code = ord(char)
            if 0x4E00 <= char_code <= 0x9FFF:  # CJK Unified Ideographs
                # Simple heuristic: higher unicode values tend to be more complex
                complexity = (char_code - 0x4E00) / (0x9FFF - 0x4E00)
                complexity_scores.append(complexity)

        return np.mean(complexity_scores) if complexity_scores else 0.0
