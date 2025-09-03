"""
Result types for Chinese name processing.

This module contains result classes that provide Scala-friendly error handling
and immutable data structures.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class ParsedName:
    """Parsed name with surname and given name components."""
    surname: str
    given_name: str
    surname_tokens: list[str]
    given_tokens: list[str]


@dataclass(frozen=True)
class ParseResult:
    """Result of name parsing operation - Scala Either-like structure."""

    success: bool
    result: str | tuple[list[str], list[str]]
    error_message: str | None = None
    # Original compound surname format (preserves input format like "Duanmu" vs "Duan-Mu")
    original_compound_surname: str | None = None
    # Structured parsed components when available
    parsed: ParsedName | None = None

    @classmethod
    def success_with_name(
        cls,
        formatted_name: str,
        original_compound_surname: str | None = None,
        parsed: ParsedName | None = None,
    ) -> ParseResult:
        """Create a successful result with final formatted name.

        The optional 'parsed' provides access to individual tokens and
        component strings (surname/given) for downstream consumers.
        """
        return cls(
            success=True,
            result=formatted_name,
            error_message=None,
            original_compound_surname=original_compound_surname,
            parsed=parsed,
        )

    @classmethod
    def success_with_parse(
        cls,
        surname_tokens: list[str],
        given_tokens: list[str],
        original_compound_surname: str | None = None,
    ) -> ParseResult:
        """Create a successful intermediate parse with raw tokens.

        Note: 'parsed' will be populated with token lists, while the
        component strings are simple space-joined placeholders. Final
        capitalization and hyphenation are determined by formatting.
        """
        parsed = ParsedName(
            surname=" ".join(surname_tokens),
            given_name=" ".join(given_tokens),
            surname_tokens=list(surname_tokens),
            given_tokens=list(given_tokens),
        )
        return cls(
            success=True,
            result=(surname_tokens, given_tokens),
            error_message=None,
            original_compound_surname=original_compound_surname,
            parsed=parsed,
        )

    @classmethod
    def failure(cls, error_message: str) -> ParseResult:
        return cls(success=False, result="", error_message=error_message, original_compound_surname=None)

    def map(self, f) -> ParseResult:
        """Functor map operation - Scala-like transformation"""
        if self.success:
            try:
                return ParseResult.success_with_name(
                    f(self.result),
                    self.original_compound_surname,
                    self.parsed,
                )
            except Exception as e:
                return ParseResult.failure(str(e))
        return self

    def flat_map(self, f) -> ParseResult:
        """Monadic flatMap operation - Scala-like chaining"""
        if self.success:
            try:
                result = f(self.result)
                # Preserve the original compound surname if the result doesn't already have one
                if result.success and result.original_compound_surname is None:
                    return ParseResult(
                        result.success,
                        result.result,
                        result.error_message,
                        self.original_compound_surname,
                        result.parsed,
                    )
                return result
            except Exception as e:
                return ParseResult.failure(str(e))
        return self


@dataclass(frozen=True)
class CacheInfo:
    """Immutable cache information structure."""

    cache_built: bool
    cache_size: int
    pickle_file_exists: bool
    pickle_file_size: int | None = None
    pickle_file_mtime: float | None = None


class NameFormat(Enum):
    """Name format enumeration for batch processing."""
    SURNAME_FIRST = "surname_first"  # Chinese style: "Zhang Wei"
    GIVEN_FIRST = "given_first"      # Western style: "Wei Zhang"
    MIXED = "mixed"                  # No clear pattern


@dataclass(frozen=True)
class ParseCandidate:
    """Individual parse candidate with scoring details."""
    surname_tokens: list[str]
    given_tokens: list[str]
    score: float
    format: NameFormat
    original_compound_format: str | None = None


@dataclass(frozen=True)
class IndividualAnalysis:
    """Detailed analysis of a single name with all candidates."""
    raw_name: str
    candidates: list[ParseCandidate]
    best_candidate: ParseCandidate | None
    confidence: float  # Confidence in best candidate


@dataclass(frozen=True)
class BatchFormatPattern:
    """Detected formatting pattern for a batch of names."""
    dominant_format: NameFormat
    confidence: float  # Percentage of names following dominant format
    surname_first_count: int
    given_first_count: int
    total_count: int
    threshold_met: bool  # Whether confidence >= threshold (e.g., 67%)


@dataclass(frozen=True)
class BatchParseResult:
    """Complete batch processing result."""
    names: list[str]
    results: list[ParseResult]
    format_pattern: BatchFormatPattern
    individual_analyses: list[IndividualAnalysis]
    improvements: list[int]  # Indices of names improved by batch processing
