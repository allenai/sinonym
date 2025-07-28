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

    @classmethod
    def success_with_name(cls, formatted_name: str) -> ParseResult:
        return cls(success=True, result=formatted_name, error_message=None)

    @classmethod
    def success_with_parse(cls, surname_tokens: list[str], given_tokens: list[str]) -> ParseResult:
        return cls(success=True, result=(surname_tokens, given_tokens), error_message=None)

    @classmethod
    def failure(cls, error_message: str) -> ParseResult:
        return cls(success=False, result="", error_message=error_message)

    def map(self, f) -> ParseResult:
        """Functor map operation - Scala-like transformation"""
        if self.success:
            try:
                return ParseResult.success_with_name(f(self.result))
            except Exception as e:
                return ParseResult.failure(str(e))
        return self

    def flat_map(self, f) -> ParseResult:
        """Monadic flatMap operation - Scala-like chaining"""
        if self.success:
            try:
                return f(self.result)
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
