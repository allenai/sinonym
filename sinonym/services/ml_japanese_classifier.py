"""
ML-based Japanese Name Classification Service

This service provides machine learning-based classification to distinguish
Chinese names from Japanese names using all-Chinese characters, complementing
the existing rule-based ethnicity classification.

Architecture:
- Integrates with existing EthnicityClassificationService
- Uses pre-trained scikit-learn model with 99.5% accuracy
- Focuses on all-Chinese character names that pass initial Chinese surname checks
- Provides confidence scores for uncertain classifications
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from sinonym.types import ParseResult

# Optional imports - ML classifier only works if dependencies are available
try:
    import joblib
    import numpy as np
    from sklearn.pipeline import Pipeline

    # Import custom model components needed for loading the trained model
    import sinonym.ml_model_components  # This makes the classes available for joblib.load

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    joblib = None
    np = None
    Pipeline = None


class MLJapaneseClassifier:
    """Machine learning-based Japanese name classifier for all-Chinese character names."""

    def __init__(self, model_path: str | Path | None = None, confidence_threshold: float = 0.8):
        """
        Initialize ML Japanese classifier.
        
        Args:
            model_path: Path to trained model file. If None, uses default path in data/ folder.
            confidence_threshold: Minimum confidence for classification (0.0-1.0).
                                 High threshold (0.8) ensures we only reject when very confident.
        
        Note:
            If scikit-learn dependencies are not available, classifier will be disabled
            and all classifications will return "uncertain" status.
        """
        self.confidence_threshold = confidence_threshold
        self.model: Pipeline | None = None
        self.enabled = False

        if not ML_AVAILABLE:
            logging.warning(
                "ML Japanese classifier disabled: scikit-learn dependencies not available. "
                "Install with: uv add scikit-learn numpy scipy joblib",
            )
            return

        # Default model path - look in data/ folder
        if model_path is None:
            # Get path relative to this file: sinonym/services/ml_japanese_classifier.py
            project_root = Path(__file__).parent.parent.parent
            model_path = project_root / "data" / "chinese_japanese_classifier.joblib"

        self.model_path = Path(model_path)
        self._load_model()

    def _load_model(self) -> None:
        """Load the pre-trained ML model."""
        if not self.model_path.exists():
            logging.warning(
                f"ML model not found at {self.model_path}. "
                f"Japanese classifier will be disabled. "
                f"Run the training script to generate the model.",
            )
            return

        try:
            self.model = joblib.load(self.model_path)
            self.enabled = True
            logging.info(f"ML Japanese classifier loaded successfully (confidence_threshold={self.confidence_threshold})")
        except Exception as e:
            logging.exception(f"Failed to load ML model: {e}")
            self.model = None

    def is_available(self) -> bool:
        """Check if ML classifier is available and enabled."""
        return self.enabled and self.model is not None

    def classify_all_chinese_name(self, chinese_text: str) -> ParseResult:
        """
        Classify an all-Chinese character name as Chinese or Japanese.
        
        This is the main integration point for the ethnicity classification service.
        
        Args:
            chinese_text: The name to classify (should be all-Chinese characters like "山田太郎")
            
        Returns:
            ParseResult with:
            - success=False, error_message="japanese" if classified as Japanese with high confidence
            - success=True, result="chinese" if classified as Chinese or confidence below threshold
            - success=False, error_message="ml_unavailable" if ML model not available
        """
        if not self.is_available():
            # Graceful fallback - don't block processing if ML unavailable
            return ParseResult.success_with_name("chinese")

        if not chinese_text or len(chinese_text) < 2:
            return ParseResult.success_with_name("chinese")

        try:
            # Get prediction and confidence
            prediction = self.model.predict([chinese_text])[0]  # 'cn' or 'jp'
            probabilities = self.model.predict_proba([chinese_text])[0]
            confidence = max(probabilities)

            # Only reject as Japanese if we're very confident
            if prediction == "jp" and confidence >= self.confidence_threshold:
                return ParseResult.failure("japanese")

            # Default to Chinese (existing behavior preserved)
            return ParseResult.success_with_name("chinese")

        except Exception as e:
            logging.exception(f"ML classification error for '{chinese_text}': {e}")
            # Graceful fallback - don't block processing on ML errors
            return ParseResult.success_with_name("chinese")

    def get_classification_details(self, chinese_text: str) -> dict[str, Any]:
        """
        Get detailed classification information including confidence scores.
        
        Args:
            chinese_text: The name to analyze (all-Chinese characters)
            
        Returns:
            Dictionary with classification details or error information
        """
        if not self.is_available():
            return {
                "error": "ml_unavailable",
                "available": False,
                "fallback": "chinese",
            }

        if not chinese_text or len(chinese_text) < 2:
            return {
                "error": "invalid_name_length",
                "fallback": "chinese",
            }

        try:
            prediction = self.model.predict([chinese_text])[0]
            probabilities = self.model.predict_proba([chinese_text])[0]

            # Assuming model classes are ['cn', 'jp']
            class_names = self.model.classes_
            confidence_scores = dict(zip(class_names, probabilities, strict=False))

            max_confidence = max(probabilities)
            would_reject = prediction == "jp" and max_confidence >= self.confidence_threshold

            return {
                "prediction": "chinese" if prediction == "cn" else "japanese",
                "confidence": max_confidence,
                "confidence_scores": {
                    "chinese": confidence_scores.get("cn", 0.0),
                    "japanese": confidence_scores.get("jp", 0.0),
                },
                "would_reject_as_japanese": would_reject,
                "threshold": self.confidence_threshold,
                "available": True,
            }

        except Exception as e:
            logging.exception(f"ML detailed classification error for '{chinese_text}': {e}")
            return {
                "error": "classification_error",
                "details": str(e),
                "fallback": "chinese",
            }


# Factory function for easy integration
def create_ml_japanese_classifier(
    model_path: str | Path | None = None,
    confidence_threshold: float = 0.8,
) -> MLJapaneseClassifier:
    """
    Factory function to create ML Japanese classifier.
    
    Args:
        model_path: Path to model file (None for default in data/ folder)
        confidence_threshold: Confidence threshold (0.0-1.0, default 0.8 for high precision)
        
    Returns:
        MLJapaneseClassifier instance
    """
    return MLJapaneseClassifier(model_path=model_path, confidence_threshold=confidence_threshold)
