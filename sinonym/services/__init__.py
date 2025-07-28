"""
Services package for Chinese name processing.

This package contains all service classes used by the Chinese name
detection system, organized by domain responsibility.
"""

from sinonym.services.cache import PinyinCacheService
from sinonym.services.ethnicity import EthnicityClassificationService
from sinonym.services.formatting import NameFormattingService
from sinonym.services.initialization import DataInitializationService, NameDataStructures
from sinonym.services.normalization import LazyNormalizationMap, NormalizationService, NormalizedInput
from sinonym.services.parsing import NameParsingService
from sinonym.types import CacheInfo, ChineseNameConfig, ParseResult

__all__ = [
    # Services
    "PinyinCacheService",
    "DataInitializationService",
    "EthnicityClassificationService",
    "NameFormattingService",
    "NameParsingService",
    "NormalizationService",
    # Data structures
    "NameDataStructures",
    "NormalizedInput",
    "LazyNormalizationMap",
    # Types (re-exported for compatibility)
    "CacheInfo",
    "ChineseNameConfig",
    "ParseResult",
]
