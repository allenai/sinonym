"""
Batch analysis service for detecting format patterns in name lists.

This service analyzes multiple names together to detect consistent formatting
patterns (surname-first vs given-first) and applies the dominant pattern to
improve parsing accuracy for ambiguous individual names.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sinonym.coretypes import (
    BatchFormatPattern,
    BatchParseResult,
    IndividualAnalysis,
    NameFormat,
    ParseCandidate,
    ParseResult,
)
from sinonym.coretypes.results import ParsedName

if TYPE_CHECKING:
    from sinonym.services.ethnicity import EthnicityClassificationService
    from sinonym.services.parsing import NameParsingService


class BatchAnalysisService:
    """Service for analyzing batches of names to detect format patterns."""

    def __init__(
        self,
        parsing_service: NameParsingService,
        ethnicity_service: EthnicityClassificationService | None = None,
        format_threshold: float = 0.55,
    ):
        self._parsing_service = parsing_service
        self._ethnicity_service = ethnicity_service
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

        # Phase 3: Apply batch formatting without gating on confidence
        # If we have at least one Chinese participant (names that produced candidates),
        # apply the detected dominant format. Otherwise, fall back to individual processing
        # to surface per-name outcomes (including non-Chinese reasons).
        if format_pattern.total_count > 0:
            results = self._apply_batch_format(
                name_candidates,
                format_pattern.dominant_format,
                formatting_service,
            )
            improvements = self._find_improvements(name_candidates, results)
        else:
            return self._process_individually(names, normalizer, data, formatting_service)

        # Build per-name analysis details
        individual_analyses = self._build_individual_analyses(name_candidates)

        return BatchParseResult(
            names=names,
            results=results,
            format_pattern=format_pattern,
            individual_analyses=individual_analyses,
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
        name_candidates: list[tuple[str, list[ParseCandidate], ParseCandidate | None, dict | None]] = []

        for name in names:
            candidates, best_candidate = self._analyze_individual_name(name, normalizer, data)

            # Get compound metadata for this name
            normalized_input = normalizer.apply(name)
            if best_candidate is None and self._ethnicity_service is not None:
                eth = self._ethnicity_service.classify_ethnicity(
                    normalized_input.roman_tokens,
                    normalized_input.norm_map,
                    name,
                )
                if eth.success is False:
                    results.append(eth)
                else:
                    results.append(
                        self._format_best_candidate(
                            best_candidate, formatting_service, normalized_input.compound_metadata,
                        ),
                    )
            else:
                results.append(
                    self._format_best_candidate(
                        best_candidate, formatting_service, normalized_input.compound_metadata,
                    ),
                )
            name_candidates.append((name, candidates, best_candidate, normalized_input.compound_metadata))

        # Create a dummy format pattern for small batches or non-participant fallbacks
        format_pattern = BatchFormatPattern(
            dominant_format=NameFormat.MIXED,
            confidence=0.0,
            surname_first_count=0,
            given_first_count=0,
            # total_count counts only Chinese-participant names; in individual fallback,
            # treat as zero participants to match detect_batch_format semantics.
            total_count=0,
            threshold_met=False,
        )

        individual_analyses = self._build_individual_analyses(name_candidates)

        return BatchParseResult(
            names=names,
            results=results,
            format_pattern=format_pattern,
            individual_analyses=individual_analyses,
            improvements=[],
        )

    def _analyze_individual_name(self, name: str, normalizer, _data) -> tuple[list[ParseCandidate], ParseCandidate | None]:
        """Analyze a single name and return all parse candidates with scores."""
        # Normalize the input
        normalized_input = normalizer.apply(name)
        tokens = list(normalized_input.roman_tokens)

        if len(tokens) < self._parsing_service._config.min_tokens_required:
            return [], None

        # Ethnicity pre-filter: mirror individual pipeline to avoid false positives
        if self._ethnicity_service is not None:
            eth = self._ethnicity_service.classify_ethnicity(
                normalized_input.roman_tokens,
                normalized_input.norm_map,
                name,
            )
            if eth.success is False:
                # Treat as non-Chinese for batch purposes; no candidates
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
        """Detect the dominant format pattern with simple vote counting and confidence-weighted tie-breaking."""
        surname_first_preferences = 0
        given_first_preferences = 0
        surname_first_weight = 0.0
        given_first_weight = 0.0
        total_weight = 0.0
        names_with_candidates = 0

        # Count format preferences and collect confidence data for tie-breaking
        for _name, candidates, best_candidate, _ in name_candidates:
            if not candidates or not best_candidate:
                continue

            names_with_candidates += 1

            # Calculate confidence gap between top two candidates
            if len(candidates) >= 2:
                # Sort by score (highest first)
                sorted_candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
                confidence_gap = sorted_candidates[0].score - sorted_candidates[1].score

                # Weight by confidence gap - higher gap = more influence
                # Minimum weight of 0.1 to ensure all names have some influence
                weight = max(0.1, confidence_gap * 2)
            else:
                # Single candidate gets maximum weight
                weight = 1.0

            total_weight += weight

            # Add weighted vote for the best candidate's format
            if best_candidate.format == NameFormat.SURNAME_FIRST:
                surname_first_preferences += 1
                surname_first_weight += weight
            elif best_candidate.format == NameFormat.GIVEN_FIRST:
                given_first_preferences += 1
                given_first_weight += weight

        if names_with_candidates == 0:
            return BatchFormatPattern(
                dominant_format=NameFormat.MIXED,
                confidence=0.0,
                surname_first_count=0,
                given_first_count=0,
                total_count=0,
                threshold_met=False,
            )

        # Primary decision: simple vote counting (original behavior)
        total_preferences = surname_first_preferences + given_first_preferences

        if surname_first_preferences > given_first_preferences:
            # Clear winner by vote count
            dominant_format = NameFormat.SURNAME_FIRST
            confidence = surname_first_preferences / total_preferences
        elif given_first_preferences > surname_first_preferences:
            # Clear winner by vote count
            dominant_format = NameFormat.GIVEN_FIRST
            confidence = given_first_preferences / total_preferences
        # Exact tie in vote count - use confidence-weighted tie-breaking
        elif total_weight > 0:
            surname_first_confidence = surname_first_weight / total_weight
            given_first_confidence = given_first_weight / total_weight

            if surname_first_confidence > given_first_confidence:
                dominant_format = NameFormat.SURNAME_FIRST
                confidence = surname_first_confidence
            elif given_first_confidence > surname_first_confidence:
                dominant_format = NameFormat.GIVEN_FIRST
                confidence = given_first_confidence
            else:
                # Still tied even with confidence weighting - use heuristics
                dominant_format = self._apply_tie_breaking_heuristics(name_candidates)
                confidence = 0.5
        else:
            # Fallback to heuristics if confidence calculation fails
            dominant_format = self._apply_tie_breaking_heuristics(name_candidates)
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

            # If no candidate could be selected (likely non-Chinese), try to return
            # a specific ethnicity-based failure to mirror single-name behavior.
            if selected_candidate is None and self._ethnicity_service is not None:
                try:
                    normalizer = getattr(self._parsing_service, "_normalizer", None)
                    if normalizer is not None:
                        normalized_input = normalizer.apply(name)
                        eth = self._ethnicity_service.classify_ethnicity(
                            normalized_input.roman_tokens,
                            normalized_input.norm_map,
                            name,
                        )
                        if eth.success is False:
                            results.append(eth)
                            continue
                except Exception:
                    # Fall back to generic failure formatting below
                    pass

            result = self._candidate_to_parse_result(
                selected_candidate, formatting_service, compound_metadata,
            )
            results.append(result)

        return results

    def _apply_tie_breaking_heuristics(self, name_candidates: list[tuple[str, list[ParseCandidate], ParseCandidate | None, dict | None]]) -> NameFormat:
        """Apply secondary heuristics for tie-breaking when confidence-weighted voting fails."""
        # Heuristic 1: Surname strength analysis
        surname_first_strength = 0
        given_first_strength = 0

        for name, candidates, best_candidate, _ in name_candidates:
            if not candidates or not best_candidate:
                continue

            tokens = name.split()
            if len(tokens) != 2:
                continue

            first_token, second_token = tokens

            # Get surname frequencies for both positions
            first_as_surname = self._parsing_service._data.get_surname_freq(first_token, 0)
            second_as_surname = self._parsing_service._data.get_surname_freq(second_token, 0)

            # Evaluate surname-first hypothesis: first token should be strong surname
            if first_as_surname > 1000:  # Strong surname threshold
                surname_first_strength += 2
            elif first_as_surname > 100:  # Medium surname
                surname_first_strength += 1

            # Evaluate given-first hypothesis: second token should be strong surname
            if second_as_surname > 1000:  # Strong surname threshold
                given_first_strength += 2
            elif second_as_surname > 100:  # Medium surname
                given_first_strength += 1

        # Heuristic 2: Cultural default (surname-first is traditional Chinese ordering)
        # Add small bias toward surname-first as tiebreaker
        surname_first_strength += 0.5

        # Decision based on heuristic strength
        if surname_first_strength > given_first_strength:
            return NameFormat.SURNAME_FIRST
        if given_first_strength > surname_first_strength:
            return NameFormat.GIVEN_FIRST
        # Final fallback: use cultural default
        return NameFormat.SURNAME_FIRST

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
            # Use the EXACT same formatting pipeline as individual processing, with tokens
            formatted_name, given_final, surname_final, surname_str, given_str, middle_tokens = (
                formatting_service.format_name_output_with_tokens(
                    candidate.surname_tokens,
                    candidate.given_tokens,
                    {},  # norm_map - not needed for this step since tokens are already normalized
                    compound_metadata,
                )
            )
            parsed = ParsedName(
                surname=surname_str,
                given_name=given_str,
                surname_tokens=surname_final,
                given_tokens=given_final,
                middle_name=" ".join(middle_tokens) if middle_tokens else "",
                middle_tokens=middle_tokens,
            )
            return ParseResult.success_with_name(formatted_name, parsed=parsed)
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

    def _build_individual_analyses(
        self, name_candidates: list[tuple[str, list[ParseCandidate], ParseCandidate | None, dict | None]],
    ) -> list[IndividualAnalysis]:
        """Build IndividualAnalysis entries with a simple confidence per name.

        Confidence is computed via a softmax over candidate scores.
        - No candidates: confidence = 0.0
        - One candidate: confidence = 1.0
        - Multiple: exp(score_i - max)/sum(exp(score_j - max)) for best candidate
        """
        import math

        analyses: list[IndividualAnalysis] = []
        for name, candidates, best_candidate, _ in name_candidates:
            if not candidates or best_candidate is None:
                analyses.append(
                    IndividualAnalysis(
                        raw_name=name,
                        candidates=[],
                        best_candidate=None,
                        confidence=0.0,
                    ),
                )
                continue

            if len(candidates) == 1:
                confidence = 1.0
            else:
                max_score = max(c.score for c in candidates)
                exps = [math.exp(c.score - max_score) for c in candidates]
                denom = sum(exps) if exps else 1.0
                # Locate index of best_candidate (fall back to top-1 if not found)
                try:
                    idx = candidates.index(best_candidate)
                except ValueError:
                    idx = 0
                confidence = exps[idx] / denom if denom > 0 else 0.0

            analyses.append(
                IndividualAnalysis(
                    raw_name=name,
                    candidates=candidates,
                    best_candidate=best_candidate,
                    confidence=float(confidence),
                ),
            )

        return analyses
