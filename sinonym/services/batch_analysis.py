"""
Batch analysis service for detecting format patterns in name lists.

This service analyzes multiple names together to detect consistent formatting
patterns (surname-first vs given-first) and applies the dominant pattern to
improve parsing accuracy for ambiguous individual names.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sinonym.types import (
    BatchFormatPattern,
    BatchParseResult,
    NameFormat,
    ParseCandidate,
    ParseResult,
)

if TYPE_CHECKING:
    from sinonym.services.parsing import NameParsingService


class BatchAnalysisService:
    """Service for analyzing batches of names to detect format patterns."""

    def __init__(self, parsing_service: NameParsingService, format_threshold: float = 0.55):
        self._parsing_service = parsing_service
        self._format_threshold = format_threshold  # Minimum threshold for format detection

    def analyze_name_batch(
        self,
        names: list[str],
        normalizer,
        data,
        formatting_service,
        minimum_batch_size: int = 2,
    ) -> BatchParseResult:
        """
        Analyze a batch of names and apply consistent formatting.

        Args:
            names: List of raw name strings to analyze
            normalizer: Normalization service instance
            data: Name data structures
            minimum_batch_size: Minimum batch size for format detection

        Returns:
            BatchParseResult with individual analyses and batch-corrected results
        """
        if len(names) < minimum_batch_size:
            # Too small for batch analysis - fall back to individual processing
            return self._process_individually(names, normalizer, data, formatting_service)

        # Phase 1: Analyze each name individually and collect all parse candidates
        name_candidates = []  # List of (name, candidates, best_candidate, compound_metadata)

        for name in names:
            candidates, best_candidate = self._analyze_individual_name(name, normalizer, data)

            # Get compound metadata for this name
            normalized_input = normalizer.apply(name)
            name_candidates.append((name, candidates, best_candidate, normalized_input.compound_metadata))

        # Phase 2: Detect the dominant format pattern
        format_pattern = self._detect_format_pattern(name_candidates)

        # Phase 3: Apply batch formatting if strong pattern detected
        if format_pattern.threshold_met:
            results = self._apply_batch_format(
                name_candidates,
                format_pattern.dominant_format,
                formatting_service,
            )
            improvements = self._find_improvements(name_candidates, results)
        else:
            # No strong pattern - raise error instead of falling back to individual processing
            raise ValueError(
                f"Batch format detection confidence too low: {format_pattern.confidence:.1%} "
                f"(threshold: {self._format_threshold:.1%}). Detected format: {format_pattern.dominant_format.name} "
                f"with {format_pattern.given_first_count} given-first and {format_pattern.surname_first_count} "
                f"surname-first preferences. Consider processing names individually or adjusting the threshold.",
            )

        return BatchParseResult(
            names=names,
            results=results,
            format_pattern=format_pattern,
            individual_analyses=[],  # Simplified - no longer track this
            improvements=improvements,
        )

    def detect_batch_format(self, names: list[str], normalizer, data) -> BatchFormatPattern:
        """
        Detect the format pattern of a batch without full processing.

        Returns:
            BatchFormatPattern indicating the dominant format and confidence
        """
        name_candidates = []
        for name in names:
            candidates, best_candidate = self._analyze_individual_name(name, normalizer, data)
            name_candidates.append((name, candidates, best_candidate, None))  # No compound_metadata needed for format detection

        return self._detect_format_pattern(name_candidates)

    def _process_individually(self, names: list[str], normalizer, data, formatting_service) -> BatchParseResult:
        """Process names individually when batch is too small."""
        results = []

        for name in names:
            candidates, best_candidate = self._analyze_individual_name(name, normalizer, data)

            # Get compound metadata for this name
            normalized_input = normalizer.apply(name)
            results.append(
                self._format_best_candidate(
                    best_candidate, formatting_service, normalized_input.compound_metadata,
                ),
            )

        # Create a dummy format pattern for small batches
        format_pattern = BatchFormatPattern(
            dominant_format=NameFormat.MIXED,
            confidence=0.0,
            surname_first_count=0,
            given_first_count=0,
            total_count=len(names),
            threshold_met=False,
        )

        return BatchParseResult(
            names=names,
            results=results,
            format_pattern=format_pattern,
            individual_analyses=[],  # Simplified
            improvements=[],
        )

    def _analyze_individual_name(self, name: str, normalizer, _data) -> tuple[list[ParseCandidate], ParseCandidate | None]:
        """Analyze a single name and return all parse candidates with scores."""
        # Normalize the input
        normalized_input = normalizer.apply(name)
        tokens = list(normalized_input.roman_tokens)

        if len(tokens) < self._parsing_service._config.min_tokens_required:
            return [], None

        # Generate all possible parses
        parses_with_format = self._parsing_service._generate_all_parses_with_format(
            tokens,
            normalized_input.norm_map,
            normalized_input.compound_metadata,
        )

        if not parses_with_format:
            return [], None

        # Score all parses and determine their formats
        candidates = []
        for surname_tokens, given_tokens, original_compound_format in parses_with_format:
            score = self._parsing_service.calculate_parse_score(
                surname_tokens,
                given_tokens,
                tokens,
                normalized_input.norm_map,
                False,
                original_compound_format,
            )

            # Determine the format of this parse
            parse_format = self._determine_parse_format(surname_tokens, given_tokens, tokens)

            candidate = ParseCandidate(
                surname_tokens=surname_tokens,
                given_tokens=given_tokens,
                score=score,
                format=parse_format,
                original_compound_format=original_compound_format,
            )
            candidates.append(candidate)

        # Sort by score (highest first)
        candidates.sort(key=lambda x: x.score, reverse=True)

        best_candidate = candidates[0] if candidates else None
        return candidates, best_candidate

    def _determine_parse_format(
        self, surname_tokens: list[str], _given_tokens: list[str], original_tokens: list[str],
    ) -> NameFormat:
        """Determine if a parse follows surname-first or given-first format."""
        if not surname_tokens or not original_tokens:
            return NameFormat.SURNAME_FIRST

        # Check if surname is at the beginning (surname-first) or end (given-first)
        if surname_tokens[0] == original_tokens[0]:
            return NameFormat.SURNAME_FIRST
        if surname_tokens[-1] == original_tokens[-1]:
            return NameFormat.GIVEN_FIRST

        # Default to surname-first for unclear cases
        return NameFormat.SURNAME_FIRST

    def _detect_format_pattern(self, name_candidates: list[tuple[str, list[ParseCandidate], ParseCandidate | None, dict | None]]) -> BatchFormatPattern:
        """Detect the dominant format pattern by counting individual format preferences."""
        surname_first_preferences = 0
        given_first_preferences = 0
        names_with_candidates = 0

        # Count format preferences: which format wins for each individual name
        for _name, candidates, best_candidate, _ in name_candidates:
            if not candidates or not best_candidate:
                continue

            names_with_candidates += 1

            # Count the preference of the best (winning) candidate for this name
            if best_candidate.format == NameFormat.SURNAME_FIRST:
                surname_first_preferences += 1
            elif best_candidate.format == NameFormat.GIVEN_FIRST:
                given_first_preferences += 1

        if names_with_candidates == 0:
            return BatchFormatPattern(
                dominant_format=NameFormat.MIXED,
                confidence=0.0,
                surname_first_count=0,
                given_first_count=0,
                total_count=0,
                threshold_met=False,
            )

        # Determine dominant format based on preference count
        total_preferences = surname_first_preferences + given_first_preferences
        if surname_first_preferences > given_first_preferences:
            dominant_format = NameFormat.SURNAME_FIRST
            confidence = surname_first_preferences / total_preferences
        elif given_first_preferences > surname_first_preferences:
            dominant_format = NameFormat.GIVEN_FIRST
            confidence = given_first_preferences / total_preferences
        else:
            dominant_format = NameFormat.MIXED
            confidence = 0.5

        threshold_met = confidence >= self._format_threshold

        return BatchFormatPattern(
            dominant_format=dominant_format,
            confidence=confidence,
            surname_first_count=surname_first_preferences,
            given_first_count=given_first_preferences,
            total_count=names_with_candidates,
            threshold_met=threshold_met,
        )

    def _apply_batch_format(
        self,
        name_candidates: list[tuple[str, list[ParseCandidate], ParseCandidate | None, dict | None]],
        target_format: NameFormat,
        formatting_service,
    ) -> list[ParseResult]:
        """Apply the detected batch format by selecting best candidate matching the format."""
        results = []
        unambiguous_names = []

        # Process all names in one pass - check for unambiguous names and apply format
        for name, candidates, best_candidate, compound_metadata in name_candidates:
            # Find candidates that match the target format
            matching_candidates = [c for c in candidates if c.format == target_format]

            if candidates and not matching_candidates:
                # This name has no candidates for the target format - it's unambiguous
                unambiguous_names.append(name)
                # Use the best available candidate
                selected_candidate = best_candidate
            elif matching_candidates:
                # Use the best candidate that matches the batch format
                selected_candidate = max(matching_candidates, key=lambda x: x.score)
            else:
                # No candidates at all
                selected_candidate = None

            result = self._candidate_to_parse_result(
                selected_candidate, formatting_service, compound_metadata,
            )
            results.append(result)

        # If we have unambiguous names, raise an error with results
        if unambiguous_names:
            name_list = "', '".join(unambiguous_names[:5])  # Show first 5
            if len(unambiguous_names) > 5:
                name_list += f"' and {len(unambiguous_names) - 5} more"
            else:
                name_list += "'"

            raise ValueError(
                f"Cannot apply batch format {target_format.name} to {len(unambiguous_names)} "
                f"unambiguous names: '{name_list}'. These names have only one possible parse format "
                f"and cannot be forced into the detected batch format. Consider processing these "
                f"names individually or using a different batch.",
            )

        return results

    def _format_best_candidate(
        self, best_candidate: ParseCandidate | None, formatting_service, compound_metadata,
    ) -> ParseResult:
        """Format the best candidate from an individual analysis."""
        return self._candidate_to_parse_result(
            best_candidate, formatting_service, compound_metadata,
        )

    def _candidate_to_parse_result(
        self, candidate: ParseCandidate | None, formatting_service, compound_metadata,
    ) -> ParseResult:
        """Convert a ParseCandidate to a ParseResult using the real formatting service."""
        if not candidate:
            return ParseResult.failure("no valid parse found")

        try:
            # Use the EXACT same formatting pipeline as individual processing
            formatted_name = formatting_service.format_name_output(
                candidate.surname_tokens,
                candidate.given_tokens,
                {},  # norm_map - not needed for this step since tokens are already normalized
                candidate.original_compound_format,
                compound_metadata,
            )
            return ParseResult.success_with_name(formatted_name)
        except ValueError as e:
            return ParseResult.failure(str(e))

    def _find_improvements(
        self, name_candidates: list[tuple[str, list[ParseCandidate], ParseCandidate | None, dict | None]], batch_results: list[ParseResult],
    ) -> list[int]:
        """Find indices of names that were improved by batch processing."""
        improvements = []

        for i, ((name, candidates, best_candidate, _compound_metadata), batch_result) in enumerate(zip(name_candidates, batch_results, strict=False)):
            if not best_candidate or not batch_result.success:
                continue

            # Check if the batch result is different from the individual best result
            # For now, we'll consider any change in format as an improvement
            # A more sophisticated approach would compare the actual format changes

            # Simple heuristic: if the batch applied a different format than what was individually preferred
            if len(candidates) >= 2:
                # Try to determine the format from the batch result
                # This is a simplification - in a full implementation we'd track this better
                tokens = name.split()
                if len(tokens) == 2:
                    # Simple heuristic: check if the order was changed
                    expected_individual = f"{best_candidate.given_tokens[0].capitalize()} {best_candidate.surname_tokens[0].capitalize()}"
                    if expected_individual != batch_result.result:
                        improvements.append(i)

        return improvements
