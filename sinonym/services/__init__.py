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
from sinonym.services.normalization import LazyNormalizationMap, NormalizationService, NormalizedInput
from sinonym.services.parsing import NameParsingService
from sinonym.coretypes import CacheInfo, ChineseNameConfig, ParseResult


class ServiceContext:
    """Lightweight dependency context for services to reduce injection complexity."""

    __slots__ = ["config", "data", "normalizer"]

    def __init__(self, config, normalizer, data):
        self.config = config
        self.normalizer = normalizer
        self.data = data

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
    "ServiceContext",
]
