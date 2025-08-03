"""
Services package for Chinese name processing.

This package contains all service classes used by the Chinese name
detection system, organized by domain responsibility.
"""

from sinonym.services.batch_analysis import BatchAnalysisService
from sinonym.services.cache import PinyinCacheService
from sinonym.services.ethnicity import EthnicityClassificationService
from sinonym.services.formatting import NameFormattingService
from sinonym.services.initialization import DataInitializationService, NameDataStructures
from sinonym.services.ml_japanese_classifier import MLJapaneseClassifier, create_ml_japanese_classifier
from sinonym.services.normalization import LazyNormalizationMap, NormalizationService, NormalizedInput
from sinonym.services.parsing import NameParsingService
from sinonym.types import CacheInfo, ChineseNameConfig, ParseResult

__all__ = [
    # Batch Services
    "BatchAnalysisService",
    # Types (re-exported for compatibility)
    "CacheInfo",
    "ChineseNameConfig",
    "DataInitializationService",
    "EthnicityClassificationService",
    "LazyNormalizationMap",
    # Data structures
    "NameDataStructures",
    "NameFormattingService",
    "NameParsingService",
    "NormalizationService",
    "NormalizedInput",
    "ParseResult",
    # Services
    "PinyinCacheService",
    # ML Services
    "MLJapaneseClassifier",
    "create_ml_japanese_classifier",
]
