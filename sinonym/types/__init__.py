"""
Types package for Chinese name processing.

This package contains result types, configuration classes, and other
data structures used throughout the Chinese name detection system.
"""

from sinonym.types.config import ChineseNameConfig
from sinonym.types.results import (
    BatchFormatPattern,
    BatchParseResult,
    CacheInfo,
    IndividualAnalysis,
    NameFormat,
    ParseCandidate,
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
]
