"""
Core types for Chinese name processing.

This package contains result types, configuration classes, and other
data structures used throughout the Chinese name detection system.
"""

from sinonym.coretypes.config import ChineseNameConfig
from sinonym.coretypes.results import (
    BatchFormatPattern,
    BatchParseResult,
    CacheInfo,
    IndividualAnalysis,
    NameFormat,
    ParseCandidate,
    ParsedName,
    ParseResult,
)

__all__ = [
    "BatchFormatPattern",
    "BatchParseResult",
    "CacheInfo",
    "ChineseNameConfig",
    "IndividualAnalysis",
    "NameFormat",
    "ParseCandidate",
    "ParseResult",
    "ParsedName",
]
