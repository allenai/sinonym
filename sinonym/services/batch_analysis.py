"""
Batch analysis service for detecting format patterns in name lists.

This service analyzes multiple names together to detect consistent formatting
patterns (surname-first vs given-first) and applies the dominant pattern to
improve parsing accuracy for ambiguous individual names.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
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

GUARDED_GIVEN_FIRST_BATCH_MIN_SHARE = 0.75
GIVEN_NAME_SHAPE_MIN_SHARE = 0.5
ULTRA_LOW_FIRST_SURNAME_MEDIAN_MAX = 130.0
LONG_GIVEN_NAME_TOKEN_MIN_LENGTH = 6
TWO_TOKEN_NAME_LENGTH = 2
BATCH_PARTICIPANT_MIN = 2
LATIN_ONLY_REPRESENTATION = "latin_only"
REJECTED_INPUT_REPRESENTATION = "rejected_input"

NameCandidateEntry = tuple[str, list[ParseCandidate], ParseCandidate | None, dict | None, str]


@dataclass
class BatchVoteStats:
    """Vote totals used to detect a batch-wide name order."""

    surname_first_preferences: int = 0
    given_first_preferences: int = 0
    surname_first_weight: float = 0.0
    given_first_weight: float = 0.0
    total_weight: float = 0.0
    names_with_candidates: int = 0

    @property
    def total_preferences(self) -> int:
        """Return total format votes."""
        return self.surname_first_preferences + self.given_first_preferences

if TYPE_CHECKING:
    from collections.abc import Callable

    from sinonym.services.ethnicity import EthnicityClassificationService
    from sinonym.services.parsing import NameParsingService


class BatchAnalysisService:
    """Service for analyzing batches of names to detect format patterns."""

    def __init__(
        self,
        parsing_service: NameParsingService,
        individual_parser: Callable[[str], ParseResult],
        ethnicity_service: EthnicityClassificationService | None = None,
        format_threshold: float = 0.55,
        input_failure_reason: Callable[[str], str | None] | None = None,
    ):
        self._parsing_service = parsing_service
        self._ethnicity_service = ethnicity_service
        self._format_threshold = format_threshold  # Minimum threshold for format detection
        self._individual_parser = individual_parser
        self._input_failure_reason = input_failure_reason

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
        name_candidates: list[NameCandidateEntry] = []

        for name in names:
            input_failure = self._input_failure(name)
            if input_failure is not None:
                name_candidates.append((name, [], None, {}, REJECTED_INPUT_REPRESENTATION))
                continue

            # Normalize once and reuse
            normalized_input = normalizer.apply(name)
            representation = self._script_representation(normalizer, normalized_input)
            candidates, best_candidate = self._analyze_individual_name_with_normalized(
                name,
                normalized_input,
                data,
                allow_guarded_given_first_bonus=False,
            )
            name_candidates.append((name, candidates, best_candidate, normalized_input.compound_metadata, representation))

        name_candidates = self._promote_guarded_given_first_batch_votes(name_candidates, normalizer, data)

        # Phase 2: Detect the dominant format pattern
        format_pattern = self._detect_format_pattern(name_candidates)

        # Phase 3: Apply batch formatting only when the dominant pattern confidence
        # clears the configured threshold. Otherwise, fall back to individual
        # processing to avoid over-applying a weak batch signal.
        if format_pattern.total_count > 0 and format_pattern.threshold_met:
            results = self._apply_batch_format(
                name_candidates,
                format_pattern.dominant_format,
                formatting_service,
            )
            improvements = self._find_improvements(name_candidates, results)
        else:
            individual_fallback = self._process_individually(names, normalizer, data, formatting_service)
            return BatchParseResult(
                names=individual_fallback.names,
                results=individual_fallback.results,
                format_pattern=format_pattern,
                individual_analyses=individual_fallback.individual_analyses,
                improvements=[],
            )

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
        name_candidates: list[NameCandidateEntry] = []
        for name in names:
            input_failure = self._input_failure(name)
            if input_failure is not None:
                name_candidates.append((name, [], None, None, REJECTED_INPUT_REPRESENTATION))
                continue

            # Normalize once for format detection (compound_metadata not needed)
            normalized_input = normalizer.apply(name)
            representation = self._script_representation(normalizer, normalized_input)
            candidates, best_candidate = self._analyze_individual_name_with_normalized(
                name,
                normalized_input,
                data,
                allow_guarded_given_first_bonus=False,
            )
            name_candidates.append((name, candidates, best_candidate, None, representation))

        name_candidates = self._promote_guarded_given_first_batch_votes(name_candidates, normalizer, data)

        return self._detect_format_pattern(name_candidates)

    def _process_individually(self, names: list[str], normalizer, data, formatting_service) -> BatchParseResult:
        """Process names individually when batch is too small."""
        results = []
        name_candidates: list[NameCandidateEntry] = []

        for name in names:
            input_failure = self._input_failure(name)
            if input_failure is not None:
                results.append(input_failure)
                name_candidates.append((name, [], None, {}, REJECTED_INPUT_REPRESENTATION))
                continue

            # Normalize once and reuse
            normalized_input = normalizer.apply(name)
            representation = self._script_representation(normalizer, normalized_input)
            candidates, best_candidate = self._analyze_individual_name_with_normalized(
                name, normalized_input, data,
            )
            if not self._is_batch_format_participant(representation):
                results.append(self._locked_representation_result(name))
            elif best_candidate is None and self._ethnicity_service is not None:
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
            name_candidates.append((name, candidates, best_candidate, normalized_input.compound_metadata, representation))

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

    def _analyze_individual_name_with_normalized(
        self,
        name: str,
        normalized_input,
        _data,
        *,
        allow_guarded_given_first_bonus: bool = True,
    ) -> tuple[list[ParseCandidate], ParseCandidate | None]:
        """Analyze a single name using pre-computed normalized input."""
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
                is_all_chinese=False,
                original_compound_format=original_compound_format,
                allow_guarded_given_first_bonus=allow_guarded_given_first_bonus,
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

    def _promote_guarded_given_first_batch_votes(
        self,
        name_candidates: list[NameCandidateEntry],
        normalizer,
        data,
    ) -> list[NameCandidateEntry]:
        """Promote contested given-first votes when batch-level shape evidence supports them."""
        participant_count = sum(
            1
            for entry in name_candidates
            if self._candidate_entry_participates(entry)
        )
        if participant_count == 0:
            return name_candidates

        promoted, first_surname_freqs, given_shape_count = self._collect_guarded_given_first_promotions(
            name_candidates,
            normalizer,
            data,
        )

        if not self._should_promote_guarded_given_first_votes(
            participant_count,
            promoted,
            first_surname_freqs,
            given_shape_count,
        ):
            return name_candidates

        adjusted = list(name_candidates)
        for index, promoted_candidate in promoted:
            name, candidates, _best_candidate, compound_metadata, representation = adjusted[index]
            adjusted[index] = (name, candidates, promoted_candidate, compound_metadata, representation)
        return adjusted

    def _collect_guarded_given_first_promotions(
        self,
        name_candidates: list[NameCandidateEntry],
        normalizer,
        data,
    ) -> tuple[list[tuple[int, ParseCandidate]], list[float], int]:
        """Collect given-first candidates that only become best under individual guarded scoring."""
        promoted: list[tuple[int, ParseCandidate]] = []
        first_surname_freqs: list[float] = []
        given_shape_count = 0

        for index, entry in enumerate(name_candidates):
            name, candidates, best_candidate, _compound_metadata, representation = entry
            if not self._is_batch_format_participant(representation):
                continue

            promotion = self._guarded_given_first_promotion(name, candidates, best_candidate, normalizer, data)
            if promotion is None:
                continue

            promoted_candidate, first_token, first_norm = promotion
            promoted.append((index, promoted_candidate))
            first_surname_freqs.append(data.get_surname_freq(first_norm))
            if self._has_given_name_shape(first_token):
                given_shape_count += 1

        return promoted, first_surname_freqs, given_shape_count

    def _guarded_given_first_promotion(
        self,
        name: str,
        candidates: list[ParseCandidate],
        best_candidate: ParseCandidate | None,
        normalizer,
        data,
    ) -> tuple[ParseCandidate, str, str] | None:
        """Return the promoted given-first candidate and first-token evidence, if any."""
        if not candidates or best_candidate is None or best_candidate.format != NameFormat.SURNAME_FIRST:
            return None

        normalized_input = normalizer.apply(name)
        tokens = list(normalized_input.roman_tokens)
        if len(tokens) != TWO_TOKEN_NAME_LENGTH:
            return None

        _individual_candidates, individual_best = self._analyze_individual_name_with_normalized(
            name,
            normalized_input,
            data,
            allow_guarded_given_first_bonus=True,
        )
        if individual_best is None or individual_best.format != NameFormat.GIVEN_FIRST:
            return None

        promoted_candidate = self._matching_given_first_candidate(candidates, individual_best)
        if promoted_candidate is None:
            return None

        first_token = tokens[0]
        first_norm = normalizer.get_normalized(first_token, normalized_input.norm_map).replace(" ", "")
        return promoted_candidate, first_token, first_norm

    @staticmethod
    def _matching_given_first_candidate(
        candidates: list[ParseCandidate],
        individual_best: ParseCandidate,
    ) -> ParseCandidate | None:
        """Find the no-bonus candidate that matches the individual guarded winner."""
        return next(
            (
                candidate
                for candidate in candidates
                if candidate.format == NameFormat.GIVEN_FIRST
                and candidate.surname_tokens == individual_best.surname_tokens
                and candidate.given_tokens == individual_best.given_tokens
            ),
            None,
        )

    @staticmethod
    def _should_promote_guarded_given_first_votes(
        participant_count: int,
        promoted: list[tuple[int, ParseCandidate]],
        first_surname_freqs: list[float],
        given_shape_count: int,
    ) -> bool:
        """Return whether guarded given-first votes have enough batch-level support."""
        if not promoted or len(promoted) / participant_count < GUARDED_GIVEN_FIRST_BATCH_MIN_SHARE:
            return False

        has_shape_evidence = given_shape_count / len(promoted) >= GIVEN_NAME_SHAPE_MIN_SHARE
        has_ultra_low_first_surname_evidence = (
            bool(first_surname_freqs)
            and median(first_surname_freqs) < ULTRA_LOW_FIRST_SURNAME_MEDIAN_MAX
        )
        return has_shape_evidence or has_ultra_low_first_surname_evidence

    @staticmethod
    def _has_given_name_shape(token: str) -> bool:
        """Return whether a token has independent shape evidence for given-name position."""
        raw = token.replace("-", "").replace("'", "")
        return len(raw) >= LONG_GIVEN_NAME_TOKEN_MIN_LENGTH

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

        # Compound surname: surname_tokens may be sub-tokens of a single
        # original token (e.g. ['Ou','yang'] from 'Ouyang').
        # Reconstruct and check against original tokens.
        joined_surname = "".join(surname_tokens).lower()
        if original_tokens[0].lower() == joined_surname:
            return NameFormat.SURNAME_FIRST
        if original_tokens[-1].lower() == joined_surname:
            return NameFormat.GIVEN_FIRST

        # Default to surname-first for unclear cases
        return NameFormat.SURNAME_FIRST

    def _detect_format_pattern(self, name_candidates: list[NameCandidateEntry]) -> BatchFormatPattern:
        """Detect the dominant format pattern with simple vote counting and confidence-weighted tie-breaking."""
        stats = self._collect_batch_vote_stats(name_candidates)
        if stats.names_with_candidates == 0:
            return BatchFormatPattern(
                dominant_format=NameFormat.MIXED,
                confidence=0.0,
                surname_first_count=0,
                given_first_count=0,
                total_count=0,
                threshold_met=False,
            )

        dominant_format, confidence = self._dominant_format_and_confidence(stats, name_candidates)
        has_decisive_vote = stats.surname_first_preferences != stats.given_first_preferences
        has_multiple_participants = stats.names_with_candidates >= BATCH_PARTICIPANT_MIN
        threshold_met = confidence >= self._format_threshold and has_decisive_vote and has_multiple_participants

        return BatchFormatPattern(
            dominant_format=dominant_format,
            confidence=confidence,
            surname_first_count=stats.surname_first_preferences,
            given_first_count=stats.given_first_preferences,
            total_count=stats.names_with_candidates,
            threshold_met=threshold_met,
        )

    def _collect_batch_vote_stats(self, name_candidates: list[NameCandidateEntry]) -> BatchVoteStats:
        """Count candidate format votes and confidence weights."""
        stats = BatchVoteStats()

        for entry in name_candidates:
            _name, candidates, best_candidate, _compound_metadata, representation = entry
            if not self._is_batch_format_participant(representation) or not candidates or not best_candidate:
                continue

            weight = self._candidate_vote_weight(candidates)
            stats.names_with_candidates += 1
            stats.total_weight += weight

            if best_candidate.format == NameFormat.SURNAME_FIRST:
                stats.surname_first_preferences += 1
                stats.surname_first_weight += weight
            elif best_candidate.format == NameFormat.GIVEN_FIRST:
                stats.given_first_preferences += 1
                stats.given_first_weight += weight

        return stats

    @staticmethod
    def _candidate_vote_weight(candidates: list[ParseCandidate]) -> float:
        """Return vote weight from the confidence gap between top candidates."""
        if len(candidates) < 2:
            return 1.0

        sorted_candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
        confidence_gap = sorted_candidates[0].score - sorted_candidates[1].score
        return max(0.1, confidence_gap * 2)

    def _dominant_format_and_confidence(
        self,
        stats: BatchVoteStats,
        name_candidates: list[NameCandidateEntry],
    ) -> tuple[NameFormat, float]:
        """Return dominant batch format and confidence from collected votes."""
        if stats.surname_first_preferences > stats.given_first_preferences:
            return NameFormat.SURNAME_FIRST, stats.surname_first_preferences / stats.total_preferences
        if stats.given_first_preferences > stats.surname_first_preferences:
            return NameFormat.GIVEN_FIRST, stats.given_first_preferences / stats.total_preferences
        if stats.total_weight <= 0:
            return self._apply_tie_breaking_heuristics(name_candidates), 0.5

        surname_first_confidence = stats.surname_first_weight / stats.total_weight
        given_first_confidence = stats.given_first_weight / stats.total_weight
        if surname_first_confidence > given_first_confidence:
            return NameFormat.SURNAME_FIRST, surname_first_confidence
        if given_first_confidence > surname_first_confidence:
            return NameFormat.GIVEN_FIRST, given_first_confidence
        return self._apply_tie_breaking_heuristics(name_candidates), 0.5

    def _apply_batch_format(
        self,
        name_candidates: list[NameCandidateEntry],
        target_format: NameFormat,
        formatting_service,
    ) -> list[ParseResult]:
        """Apply the detected batch format by selecting best candidate matching the format."""
        results = []
        unambiguous_names = []

        # Process all names in one pass - check for unambiguous names and apply format
        for entry in name_candidates:
            name, candidates, best_candidate, compound_metadata, representation = entry
            input_failure = self._input_failure(name)
            if input_failure is not None:
                results.append(input_failure)
                continue

            if not self._is_batch_format_participant(representation):
                results.append(self._locked_representation_result(name))
                continue

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
                normalized_input = self._parsing_service._normalizer.apply(name)
                eth = self._ethnicity_service.classify_ethnicity(
                    normalized_input.roman_tokens,
                    normalized_input.norm_map,
                    name,
                )
                if eth.success is False:
                    results.append(eth)
                    continue

            result = self._candidate_to_parse_result(
                selected_candidate, formatting_service, compound_metadata,
            )
            results.append(result)

        return results

    def _apply_tie_breaking_heuristics(self, name_candidates: list[NameCandidateEntry]) -> NameFormat:
        """Apply secondary heuristics for tie-breaking when confidence-weighted voting fails."""
        surname_first_strength = 0
        given_first_strength = 0

        for entry in name_candidates:
            name, candidates, best_candidate, _compound_metadata, representation = entry
            if not self._candidate_entry_participates((name, candidates, best_candidate, _compound_metadata, representation)):
                continue

            tokens = self._two_token_name_tokens(name)
            if tokens is None:
                continue

            first_token, second_token = tokens
            surname_first_strength += self._surname_position_strength(first_token)
            given_first_strength += self._surname_position_strength(second_token)

        surname_first_strength += 0.5

        if surname_first_strength > given_first_strength:
            return NameFormat.SURNAME_FIRST
        if given_first_strength > surname_first_strength:
            return NameFormat.GIVEN_FIRST
        return NameFormat.SURNAME_FIRST

    def _two_token_name_tokens(self, name: str) -> list[str] | None:
        """Return normalized roman tokens for two-token tie-break analysis."""
        normalized_input = self._parsing_service._normalizer.apply(name)
        tokens = list(normalized_input.roman_tokens)
        if len(tokens) != TWO_TOKEN_NAME_LENGTH:
            return None
        return tokens

    def _surname_position_strength(self, token: str) -> int:
        """Return coarse surname-frequency strength for a token position."""
        key = self._parsing_service._normalizer.norm(token)
        surname_freq = self._parsing_service._data.get_surname_freq(key, 0)
        if surname_freq > 1000:
            return 2
        if surname_freq > 100:
            return 1
        return 0

    @staticmethod
    def _script_representation(normalizer, normalized_input) -> str:
        """Return the script cohort used for batch convention voting."""
        return normalizer.classify_script_representation(normalized_input)

    @staticmethod
    def _is_batch_format_participant(representation: str) -> bool:
        """Return whether the record should vote in and receive Latin batch format."""
        return representation == LATIN_ONLY_REPRESENTATION

    def _input_failure(self, name: str) -> ParseResult | None:
        """Return an early input failure, if one is configured and matches."""
        if self._input_failure_reason is None:
            return None

        reason = self._input_failure_reason(name)
        if reason is None:
            return None
        return ParseResult.failure(reason)

    def _candidate_entry_participates(self, entry) -> bool:
        """Return whether a candidate entry contributes to batch format detection."""
        _name, candidates, best_candidate, _compound_metadata, representation = entry
        return bool(candidates and best_candidate and self._is_batch_format_participant(representation))

    def _locked_representation_result(self, name: str) -> ParseResult:
        """Return a structural individual parse for records outside the Latin batch cohort."""
        return self._individual_parser(name)

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
                order=["given", "middle", "surname"],
            )
            # Determine original order from candidate format and assign fields
            if candidate.format == NameFormat.GIVEN_FIRST:
                order_list = ["given"] + (["middle"] if middle_tokens else []) + ["surname"]
                parsed_original_order = ParsedName(
                    surname=surname_str,
                    given_name=given_str,
                    surname_tokens=surname_final,
                    given_tokens=given_final,
                    middle_name=" ".join(middle_tokens) if middle_tokens else "",
                    middle_tokens=middle_tokens,
                    order=order_list,
                )
            else:
                order_list = ["surname"] + (["middle"] if middle_tokens else []) + ["given"]
                # Original: surname-first → preserve component labels and annotate order only
                parsed_original_order = ParsedName(
                    surname=surname_str,
                    given_name=given_str,
                    surname_tokens=surname_final,
                    given_tokens=given_final,
                    middle_name=" ".join(middle_tokens) if middle_tokens else "",
                    middle_tokens=middle_tokens,
                    order=order_list,
                )
            return ParseResult.success_with_name(
                formatted_name,
                parsed=parsed,
                parsed_original_order=parsed_original_order,
            )
        except ValueError as e:
            return ParseResult.failure(str(e))

    def _find_improvements(
        self, name_candidates: list[NameCandidateEntry], batch_results: list[ParseResult],
    ) -> list[int]:
        """Find indices of names that were improved by batch processing."""
        improvements = []

        for i, (entry, batch_result) in enumerate(
            zip(name_candidates, batch_results, strict=False),
        ):
            name, candidates, best_candidate, _compound_metadata, representation = entry
            if not self._is_batch_format_participant(representation):
                continue

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
                    expected_individual = (
                        f"{best_candidate.given_tokens[0].capitalize()} "
                        f"{best_candidate.surname_tokens[0].capitalize()}"
                    )
                    if expected_individual != batch_result.result:
                        improvements.append(i)

        return improvements

    def _build_individual_analyses(
        self, name_candidates: list[NameCandidateEntry],
    ) -> list[IndividualAnalysis]:
        """Build IndividualAnalysis entries with a simple confidence per name.

        Confidence is computed via a softmax over candidate scores.
        - No candidates: confidence = 0.0
        - One candidate: confidence = 1.0
        - Multiple: exp(score_i - max)/sum(exp(score_j - max)) for best candidate
        """
        import math

        analyses: list[IndividualAnalysis] = []
        for entry in name_candidates:
            name, candidates, best_candidate, _compound_metadata, _representation = entry
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
