"""
Result types for Chinese name processing.

This module contains result classes that provide Scala-friendly error handling
and immutable data structures.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParseResult:
    """Result of name parsing operation - Scala Either-like structure."""

    success: bool
    result: str | tuple[list[str], list[str]]
    error_message: str | None = None
    # Original compound surname format (preserves input format like "Duanmu" vs "Duan-Mu")
    original_compound_surname: str | None = None

    @classmethod
    def success_with_name(cls, formatted_name: str, original_compound_surname: str | None = None) -> ParseResult:
        return cls(success=True, result=formatted_name, error_message=None, original_compound_surname=original_compound_surname)

    @classmethod
    def success_with_parse(cls, surname_tokens: list[str], given_tokens: list[str], original_compound_surname: str | None = None) -> ParseResult:
        return cls(success=True, result=(surname_tokens, given_tokens), error_message=None, original_compound_surname=original_compound_surname)

    @classmethod
    def failure(cls, error_message: str) -> ParseResult:
        return cls(success=False, result="", error_message=error_message, original_compound_surname=None)

    def map(self, f) -> ParseResult:
        """Functor map operation - Scala-like transformation"""
        if self.success:
            try:
                return ParseResult.success_with_name(f(self.result), self.original_compound_surname)
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
                    return ParseResult(result.success, result.result, result.error_message, self.original_compound_surname)
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
