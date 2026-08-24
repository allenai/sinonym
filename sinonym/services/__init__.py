"""
Services package for Chinese name processing.

This package contains all service classes used by the Chinese name
detection system, organized by domain responsibility.
"""

from sinonym.coretypes import CacheInfo, ChineseNameConfig, ParseResult
from sinonym.services.batch_analysis import BatchAnalysisDependencies, BatchAnalysisOptions, BatchAnalysisService
from sinonym.services.cache import PinyinCacheService
from sinonym.services.ethnicity import EthnicityClassificationService
from sinonym.services.formatting import NameFormattingService
from sinonym.services.initialization import DataInitializationService, NameDataStructures
from sinonym.services.name_lookup import SurnameResolver
from sinonym.services.non_person import NonPersonInputDetectionService
from sinonym.services.normalization import LazyNormalizationMap, NormalizationService, NormalizedInput
from sinonym.services.parsing import NameParsingService
from sinonym.services.person_name_normalization import (
    DroppedNameToken,
    DropReason,
    PersonNameNormalizationResult,
    PersonNameNormalizationService,
    PersonNameOutcome,
)


class ServiceContext:
    """Lightweight dependency context for services to reduce injection complexity."""

    __slots__ = ["config", "data", "normalizer"]

    def __init__(self, config, normalizer, data):
        self.config = config
        self.normalizer = normalizer
        self.data = data


__all__ = [
    # Batch Services
    "BatchAnalysisDependencies",
    "BatchAnalysisOptions",
    "BatchAnalysisService",
    # Types (re-exported for compatibility)
    "CacheInfo",
    "ChineseNameConfig",
    "DataInitializationService",
    "DropReason",
    "DroppedNameToken",
    "EthnicityClassificationService",
    "LazyNormalizationMap",
    # Data structures
    "NameDataStructures",
    "NameFormattingService",
    "NameParsingService",
    "NonPersonInputDetectionService",
    "NormalizationService",
    "NormalizedInput",
    "ParseResult",
    "PersonNameNormalizationResult",
    "PersonNameNormalizationService",
    "PersonNameOutcome",
    # Services
    "PinyinCacheService",
    "ServiceContext",
    "SurnameResolver",
]
