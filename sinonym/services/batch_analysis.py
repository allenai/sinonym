"""
Batch analysis service for detecting format patterns in name lists.

This service analyzes multiple names together to detect consistent formatting
patterns (surname-first vs given-first) and applies the dominant pattern to
improve parsing accuracy for ambiguous individual names.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import TYPE_CHECKING

from sinonym.coretypes import (
    BatchFormatPattern,
    BatchParseResult,
    IndividualAnalysis,
    NameFormat,
    NameOrderEvidence,
    ParseCandidate,
    ParseResult,
)
from sinonym.coretypes.results import ParsedName
from sinonym.services.order_metadata import original_component_order

GUARDED_GIVEN_FIRST_BATCH_MIN_SHARE = 0.75
GIVEN_NAME_SHAPE_MIN_SHARE = 0.5
ULTRA_LOW_FIRST_SURNAME_MEDIAN_MAX = 130.0
LONG_GIVEN_NAME_TOKEN_MIN_LENGTH = 6
TWO_TOKEN_NAME_LENGTH = 2
MIN_CANDIDATES_FOR_CONFIDENCE_GAP = 2
BATCH_PARTICIPANT_MIN = 2
BATCH_FORMAT_MIN_VOTE_WEIGHT = 0.5
BATCH_FORMAT_DIRECTION_MIN_CONFIDENCE = 0.5
HIGH_SURNAME_FREQUENCY_MIN = 1000
MEDIUM_SURNAME_FREQUENCY_MIN = 100
HIGH_SURNAME_POSITION_STRENGTH = 2
MEDIUM_SURNAME_POSITION_STRENGTH = 1
LOW_SURNAME_POSITION_STRENGTH = 0
LATIN_ONLY_REPRESENTATION = "latin_only"
REJECTED_INPUT_REPRESENTATION = "rejected_input"


@dataclass(frozen=True)
class BatchAnalysisDependencies:
    """Detector-owned callbacks and constants needed by batch analysis."""

    min_tokens_required: int
    individual_parser: Callable[[str], ParseResult]
    input_failure: Callable[[str], ParseResult | None]


@dataclass(frozen=True)
class BatchAnalysisOptions:
    """Per-call batch analysis options."""

    minimum_batch_size: int = 2
    format_threshold: float | None = None


@dataclass(frozen=True)
class BatchCandidateEntry:
    """Internal batch candidate record with explicit participation state."""

    name: str
    candidates: list[ParseCandidate]
    best_candidate: ParseCandidate | None
    compound_metadata: dict | None
    representation: str

    @property
    def participates(self) -> bool:
        """Return whether this entry votes in Latin batch format detection."""
        return bool(
            self.candidates and self.best_candidate and self.representation == LATIN_ONLY_REPRESENTATION,
        )


@dataclass(frozen=True)
class SurnameEndpointSpan:
    """Matched raw-token span for surname endpoint evidence."""

    position: str
    start: int
    end: int
    lookup_key: str

    @property
    def width(self) -> int:
        """Return the number of raw tokens covered by the span."""
        return self.end - self.start


@dataclass
class BatchVoteStats:
    """Vote totals used to detect a batch-wide name order."""

    surname_first_preferences: int = 0
    given_first_preferences: int = 0
    surname_first_weight: float = 0.0
    given_first_weight: float = 0.0
    total_weight: float = 0.0
    names_with_candidates: int = 0
    unopposed_surname_first_preferences: int = 0
    unopposed_given_first_preferences: int = 0

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
        ethnicity_service: EthnicityClassificationService | None = None,
        format_threshold: float = 0.55,
        *,
        dependencies: BatchAnalysisDependencies,
    ):
        self._parsing_service = parsing_service
        self._ethnicity_service = ethnicity_service
        self._default_format_threshold = format_threshold
        self._min_tokens_required = dependencies.min_tokens_required
        self._individual_parser = dependencies.individual_parser
        self._input_failure_callback = dependencies.input_failure

    def analyze_name_batch(
        self,
        names: list[str],
        normalizer,
        data,
        formatting_service,
        options: BatchAnalysisOptions | None = None,
    ) -> BatchParseResult:
        """
        Analyze a batch of names and apply consistent formatting.

        Args:
            names: List of raw name strings to analyze
            normalizer: Normalization service instance
            data: Name data structures
            options: Per-call threshold and minimum batch size

        Returns:
            BatchParseResult with individual analyses and batch-corrected results
        """
        options = options or BatchAnalysisOptions()
        if len(names) < options.minimum_batch_size:
            # Too small for batch analysis - fall back to individual processing
            return self._process_individually(names, normalizer, data, formatting_service)

        # Phase 1: Analyze each name individually and collect all parse candidates
        name_candidates: list[BatchCandidateEntry] = []

        for name in names:
            input_failure = self._input_failure(name)
            if input_failure is not None:
                name_candidates.append(BatchCandidateEntry(name, [], None, {}, REJECTED_INPUT_REPRESENTATION))
                continue

            # Normalize once and reuse
            normalized_input = normalizer.apply(name)
            representation = self._script_representation(normalizer, normalized_input)
            if not self._is_batch_format_participant(representation):
                name_candidates.append(BatchCandidateEntry(name, [], None, normalized_input.compound_metadata, representation))
                continue

            candidates, best_candidate = self._analyze_individual_name_with_normalized(
                name,
                normalized_input,
                allow_guarded_given_first_bonus=False,
            )
            name_candidates.append(
                BatchCandidateEntry(name, candidates, best_candidate, normalized_input.compound_metadata, representation),
            )

        name_candidates = self._promote_guarded_given_first_batch_votes(name_candidates, normalizer, data)

        # Phase 2: Detect the dominant format pattern
        resolved_threshold = self._resolved_format_threshold(options.format_threshold)
        format_pattern = self._detect_format_pattern(name_candidates, normalizer, data, resolved_threshold)

        # Phase 3: Apply batch formatting only when the dominant pattern decision confidence
        # clears the configured threshold. Otherwise, fall back to individual
        # processing to avoid over-applying a weak batch signal.
        if format_pattern.total_count > 0 and format_pattern.threshold_met:
            results = self._apply_batch_format(
                name_candidates,
                format_pattern.dominant_format,
                normalizer,
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
                name_order_evidence=individual_fallback.name_order_evidence,
            )

        # Build per-name analysis details
        individual_analyses = self._build_individual_analyses(name_candidates)
        name_order_evidence = self._build_name_order_evidence(
            name_candidates,
            results,
            normalizer,
            data,
            format_pattern,
        )

        return BatchParseResult(
            names=names,
            results=results,
            format_pattern=format_pattern,
            individual_analyses=individual_analyses,
            improvements=improvements,
            name_order_evidence=name_order_evidence,
        )

    def detect_batch_format(
        self,
        names: list[str],
        normalizer,
        data,
        *,
        format_threshold: float | None = None,
    ) -> BatchFormatPattern:
        """
        Detect the format pattern of a batch without full processing.

        Returns:
            BatchFormatPattern indicating the dominant format and confidence
        """
        name_candidates: list[BatchCandidateEntry] = []
        for name in names:
            input_failure = self._input_failure(name)
            if input_failure is not None:
                name_candidates.append(BatchCandidateEntry(name, [], None, None, REJECTED_INPUT_REPRESENTATION))
                continue

            # Normalize once for format detection (compound_metadata not needed)
            normalized_input = normalizer.apply(name)
            representation = self._script_representation(normalizer, normalized_input)
            if not self._is_batch_format_participant(representation):
                name_candidates.append(BatchCandidateEntry(name, [], None, None, representation))
                continue

            candidates, best_candidate = self._analyze_individual_name_with_normalized(
                name,
                normalized_input,
                allow_guarded_given_first_bonus=False,
            )
            name_candidates.append(BatchCandidateEntry(name, candidates, best_candidate, None, representation))

        name_candidates = self._promote_guarded_given_first_batch_votes(name_candidates, normalizer, data)

        resolved_threshold = self._resolved_format_threshold(format_threshold)
        return self._detect_format_pattern(name_candidates, normalizer, data, resolved_threshold)

    def _process_individually(self, names: list[str], normalizer, data, formatting_service) -> BatchParseResult:
        """Process names individually when batch is too small."""
        results = []
        name_candidates: list[BatchCandidateEntry] = []

        for name in names:
            input_failure = self._input_failure(name)
            if input_failure is not None:
                results.append(input_failure)
                name_candidates.append(BatchCandidateEntry(name, [], None, {}, REJECTED_INPUT_REPRESENTATION))
                continue

            # Normalize once and reuse
            normalized_input = normalizer.apply(name)
            representation = self._script_representation(normalizer, normalized_input)
            if not self._is_batch_format_participant(representation):
                results.append(self._locked_representation_result(name))
                name_candidates.append(BatchCandidateEntry(name, [], None, normalized_input.compound_metadata, representation))
                continue

            candidates, best_candidate = self._analyze_individual_name_with_normalized(
                name,
                normalized_input,
            )
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
                            best_candidate,
                            formatting_service,
                            normalized_input.compound_metadata,
                        ),
                    )
            else:
                results.append(
                    self._format_best_candidate(
                        best_candidate,
                        formatting_service,
                        normalized_input.compound_metadata,
                    ),
                )
            name_candidates.append(
                BatchCandidateEntry(name, candidates, best_candidate, normalized_input.compound_metadata, representation),
            )

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
        name_order_evidence = self._build_name_order_evidence(
            name_candidates,
            results,
            normalizer,
            data,
            format_pattern,
        )

        return BatchParseResult(
            names=names,
            results=results,
            format_pattern=format_pattern,
            individual_analyses=individual_analyses,
            improvements=[],
            name_order_evidence=name_order_evidence,
        )

    def _analyze_individual_name_with_normalized(
        self,
        name: str,
        normalized_input,
        *,
        allow_guarded_given_first_bonus: bool = True,
    ) -> tuple[list[ParseCandidate], ParseCandidate | None]:
        """Analyze a single name using pre-computed normalized input."""
        tokens = list(normalized_input.roman_tokens)

        if len(tokens) < self._min_tokens_required:
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
        parses_with_format = self._parsing_service.generate_parse_options(
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
        name_candidates: list[BatchCandidateEntry],
        normalizer,
        data,
    ) -> list[BatchCandidateEntry]:
        """Promote contested given-first votes when batch-level shape evidence supports them."""
        participant_count = sum(1 for entry in name_candidates if self._candidate_entry_participates(entry))
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
            entry = adjusted[index]
            adjusted[index] = BatchCandidateEntry(
                entry.name,
                entry.candidates,
                promoted_candidate,
                entry.compound_metadata,
                entry.representation,
            )
        return adjusted

    def _collect_guarded_given_first_promotions(
        self,
        name_candidates: list[BatchCandidateEntry],
        normalizer,
        data,
    ) -> tuple[list[tuple[int, ParseCandidate]], list[float], int]:
        """Collect given-first candidates that only become best under individual guarded scoring."""
        promoted: list[tuple[int, ParseCandidate]] = []
        first_surname_freqs: list[float] = []
        given_shape_count = 0

        for index, entry in enumerate(name_candidates):
            if not self._is_batch_format_participant(entry.representation):
                continue

            promotion = self._guarded_given_first_promotion(
                entry.name,
                entry.candidates,
                entry.best_candidate,
                normalizer,
            )
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
            bool(first_surname_freqs) and median(first_surname_freqs) < ULTRA_LOW_FIRST_SURNAME_MEDIAN_MAX
        )
        return has_shape_evidence or has_ultra_low_first_surname_evidence

    @staticmethod
    def _has_given_name_shape(token: str) -> bool:
        """Return whether a token has independent shape evidence for given-name position."""
        raw = token.replace("-", "").replace("'", "")
        return len(raw) >= LONG_GIVEN_NAME_TOKEN_MIN_LENGTH

    def _determine_parse_format(
        self,
        surname_tokens: list[str],
        _given_tokens: list[str],
        original_tokens: list[str],
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

    def _detect_format_pattern(
        self,
        name_candidates: list[BatchCandidateEntry],
        normalizer,
        data,
        format_threshold: float,
    ) -> BatchFormatPattern:
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

        dominant_format, decision_confidence = self._dominant_format_and_confidence(
            stats,
            name_candidates,
            normalizer,
            data,
        )
        confidence = self._count_confidence(stats, dominant_format)
        has_decisive_vote = stats.surname_first_preferences != stats.given_first_preferences
        has_unopposed_dominant_vote = (
            dominant_format == NameFormat.SURNAME_FIRST and stats.unopposed_surname_first_preferences > 0
        ) or (dominant_format == NameFormat.GIVEN_FIRST and stats.unopposed_given_first_preferences > 0)
        has_confident_direction = decision_confidence > BATCH_FORMAT_DIRECTION_MIN_CONFIDENCE and (
            has_decisive_vote or has_unopposed_dominant_vote
        )
        has_multiple_participants = stats.names_with_candidates >= BATCH_PARTICIPANT_MIN
        threshold_met = decision_confidence >= format_threshold and has_confident_direction and has_multiple_participants

        return BatchFormatPattern(
            dominant_format=dominant_format,
            confidence=confidence,
            surname_first_count=stats.surname_first_preferences,
            given_first_count=stats.given_first_preferences,
            total_count=stats.names_with_candidates,
            threshold_met=threshold_met,
            decision_confidence=decision_confidence,
        )

    def _collect_batch_vote_stats(self, name_candidates: list[BatchCandidateEntry]) -> BatchVoteStats:
        """Count candidate format votes and confidence weights."""
        stats = BatchVoteStats()
        seen_vote_names: set[str] = set()

        for entry in name_candidates:
            if not entry.participates:
                continue
            vote_name_key = entry.name.casefold().strip()
            if vote_name_key in seen_vote_names:
                continue
            seen_vote_names.add(vote_name_key)
            stats.names_with_candidates += 1

            weight = self._candidate_vote_weight(entry.candidates)
            if weight < BATCH_FORMAT_MIN_VOTE_WEIGHT:
                continue
            stats.total_weight += weight

            if entry.best_candidate.format == NameFormat.SURNAME_FIRST:
                stats.surname_first_preferences += 1
                stats.surname_first_weight += weight
                if len(entry.candidates) == 1:
                    stats.unopposed_surname_first_preferences += 1
            elif entry.best_candidate.format == NameFormat.GIVEN_FIRST:
                stats.given_first_preferences += 1
                stats.given_first_weight += weight
                if len(entry.candidates) == 1:
                    stats.unopposed_given_first_preferences += 1

        return stats

    @staticmethod
    def _count_confidence(stats: BatchVoteStats, dominant_format: NameFormat) -> float:
        """Return count-based dominant confidence over all candidate participants."""
        if stats.names_with_candidates <= 0:
            return 0.0
        if dominant_format == NameFormat.SURNAME_FIRST:
            return stats.surname_first_preferences / stats.names_with_candidates
        if dominant_format == NameFormat.GIVEN_FIRST:
            return stats.given_first_preferences / stats.names_with_candidates
        return 0.0

    @staticmethod
    def _candidate_vote_weight(candidates: list[ParseCandidate]) -> float:
        """Return vote weight from the confidence gap between top candidates."""
        if len(candidates) < MIN_CANDIDATES_FOR_CONFIDENCE_GAP:
            return 1.0

        sorted_candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
        confidence_gap = sorted_candidates[0].score - sorted_candidates[1].score
        return max(0.1, confidence_gap * 2)

    def _dominant_format_and_confidence(
        self,
        stats: BatchVoteStats,
        name_candidates: list[BatchCandidateEntry],
        normalizer,
        data,
    ) -> tuple[NameFormat, float]:
        """Return dominant batch format and confidence from collected votes."""
        if stats.surname_first_preferences > stats.given_first_preferences:
            return NameFormat.SURNAME_FIRST, stats.surname_first_preferences / stats.total_preferences
        if stats.given_first_preferences > stats.surname_first_preferences:
            return NameFormat.GIVEN_FIRST, stats.given_first_preferences / stats.total_preferences
        if stats.total_weight <= 0:
            return self._apply_tie_breaking_heuristics(name_candidates, normalizer, data), 0.5

        surname_first_confidence = stats.surname_first_weight / stats.total_weight
        given_first_confidence = stats.given_first_weight / stats.total_weight
        if surname_first_confidence > given_first_confidence:
            return NameFormat.SURNAME_FIRST, surname_first_confidence
        if given_first_confidence > surname_first_confidence:
            return NameFormat.GIVEN_FIRST, given_first_confidence
        return self._apply_tie_breaking_heuristics(name_candidates, normalizer, data), 0.5

    def _apply_batch_format(
        self,
        name_candidates: list[BatchCandidateEntry],
        target_format: NameFormat,
        normalizer,
        formatting_service,
    ) -> list[ParseResult]:
        """Apply the detected batch format by selecting best candidate matching the format."""
        results = []
        unambiguous_names = []

        # Process all names in one pass - check for unambiguous names and apply format
        for entry in name_candidates:
            input_failure = self._input_failure(entry.name)
            if input_failure is not None:
                results.append(input_failure)
                continue

            if not self._is_batch_format_participant(entry.representation):
                results.append(self._locked_representation_result(entry.name))
                continue

            # Find candidates that match the target format
            matching_candidates = [c for c in entry.candidates if c.format == target_format]

            if entry.candidates and not matching_candidates:
                # This name has no candidates for the target format - it's unambiguous
                unambiguous_names.append(entry.name)
                # Use the best available candidate
                selected_candidate = entry.best_candidate
            elif matching_candidates:
                # Use the best candidate that matches the batch format
                selected_candidate = max(matching_candidates, key=lambda x: x.score)
            else:
                # No candidates at all
                selected_candidate = None

            # If no candidate could be selected (likely non-Chinese), try to return
            # a specific ethnicity-based failure to mirror single-name behavior.
            if selected_candidate is None and self._ethnicity_service is not None:
                normalized_input = normalizer.apply(entry.name)
                eth = self._ethnicity_service.classify_ethnicity(
                    normalized_input.roman_tokens,
                    normalized_input.norm_map,
                    entry.name,
                )
                if eth.success is False:
                    results.append(eth)
                    continue

            result = self._candidate_to_parse_result(
                selected_candidate,
                formatting_service,
                entry.compound_metadata,
            )
            results.append(result)

        return results

    def _apply_tie_breaking_heuristics(self, name_candidates: list[BatchCandidateEntry], normalizer, data) -> NameFormat:
        """Apply secondary heuristics for tie-breaking when confidence-weighted voting fails."""
        surname_first_strength = 0
        given_first_strength = 0

        for entry in name_candidates:
            if not entry.participates:
                continue

            tokens = self._two_token_name_tokens(entry.name, normalizer)
            if tokens is None:
                continue

            first_token, second_token = tokens
            surname_first_strength += self._surname_position_strength(first_token, normalizer, data)
            given_first_strength += self._surname_position_strength(second_token, normalizer, data)

        surname_first_strength += 0.5

        if surname_first_strength > given_first_strength:
            return NameFormat.SURNAME_FIRST
        if given_first_strength > surname_first_strength:
            return NameFormat.GIVEN_FIRST
        return NameFormat.SURNAME_FIRST

    @staticmethod
    def _two_token_name_tokens(name: str, normalizer) -> list[str] | None:
        """Return normalized roman tokens for two-token tie-break analysis."""
        normalized_input = normalizer.apply(name)
        tokens = list(normalized_input.roman_tokens)
        if len(tokens) != TWO_TOKEN_NAME_LENGTH:
            return None
        return tokens

    @staticmethod
    def _surname_position_strength(token: str, normalizer, data) -> int:
        """Return coarse surname-frequency strength for a token position."""
        key = normalizer.norm(token)
        surname_freq = data.get_surname_freq(key, 0)
        if surname_freq > HIGH_SURNAME_FREQUENCY_MIN:
            return HIGH_SURNAME_POSITION_STRENGTH
        if surname_freq > MEDIUM_SURNAME_FREQUENCY_MIN:
            return MEDIUM_SURNAME_POSITION_STRENGTH
        return LOW_SURNAME_POSITION_STRENGTH

    def _resolved_format_threshold(self, format_threshold: float | None) -> float:
        """Return the per-call threshold or the service default."""
        if format_threshold is None:
            return self._default_format_threshold
        return format_threshold

    @staticmethod
    def _script_representation(normalizer, normalized_input) -> str:
        """Return the script cohort used for batch convention voting."""
        return normalizer.classify_script_representation(normalized_input)

    @staticmethod
    def _is_batch_format_participant(representation: str) -> bool:
        """Return whether the record should vote in and receive Latin batch format."""
        return representation == LATIN_ONLY_REPRESENTATION

    def _input_failure(self, name: str) -> ParseResult | None:
        """Return an early detector-owned input failure, if one matches."""
        return self._input_failure_callback(name)

    @staticmethod
    def _candidate_entry_participates(entry: BatchCandidateEntry) -> bool:
        """Return whether a candidate entry contributes to batch format detection."""
        return entry.participates

    def _locked_representation_result(
        self,
        name: str,
    ) -> ParseResult:
        """Return a structural individual parse for records outside the Latin batch cohort."""
        return self._individual_parser(name)

    def _format_best_candidate(
        self,
        best_candidate: ParseCandidate | None,
        formatting_service,
        compound_metadata,
    ) -> ParseResult:
        """Format the best candidate from an individual analysis."""
        return self._candidate_to_parse_result(
            best_candidate,
            formatting_service,
            compound_metadata,
        )

    def _candidate_to_parse_result(
        self,
        candidate: ParseCandidate | None,
        formatting_service,
        compound_metadata,
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
            order_list = original_component_order(candidate.format, candidate.given_tokens, middle_tokens)
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
        self,
        name_candidates: list[BatchCandidateEntry],
        batch_results: list[ParseResult],
    ) -> list[int]:
        """Find indices of names that were improved by batch processing."""
        improvements = []

        for i, (entry, batch_result) in enumerate(
            zip(name_candidates, batch_results, strict=False),
        ):
            if not self._is_batch_format_participant(entry.representation):
                continue

            if not entry.best_candidate or not batch_result.success:
                continue

            # Check if the batch result is different from the individual best result
            # For now, we'll consider any change in format as an improvement
            # A more sophisticated approach would compare the actual format changes

            # Simple heuristic: if the batch applied a different format than what was individually preferred
            if len(entry.candidates) >= MIN_CANDIDATES_FOR_CONFIDENCE_GAP:
                # Try to determine the format from the batch result
                # This is a simplification - in a full implementation we'd track this better
                tokens = entry.name.split()
                if len(tokens) == TWO_TOKEN_NAME_LENGTH:
                    # Simple heuristic: check if the order was changed
                    expected_individual = (
                        f"{entry.best_candidate.given_tokens[0].capitalize()} "
                        f"{entry.best_candidate.surname_tokens[0].capitalize()}"
                    )
                    if expected_individual != batch_result.result:
                        improvements.append(i)

        return improvements

    def _build_name_order_evidence(
        self,
        name_candidates: list[BatchCandidateEntry],
        results: list[ParseResult],
        normalizer,
        data,
        format_pattern: BatchFormatPattern,
    ) -> list[NameOrderEvidence]:
        """Build aligned per-name evidence for external batch-context routing."""
        evidence: list[NameOrderEvidence] = []
        batch_format_applied = format_pattern.threshold_met and format_pattern.total_count > 0

        for entry, result in zip(name_candidates, results, strict=False):
            normalized_input = normalizer.apply(entry.name)
            raw_tokens = list(normalized_input.roman_tokens)
            normalized_raw_tokens = [normalizer.norm(token) for token in raw_tokens]
            first_freq, last_freq = self._endpoint_surname_frequencies(raw_tokens, normalizer, data)
            selected_format = self._format_from_parse_result(result)
            individual_format = entry.best_candidate.format if entry.best_candidate else NameFormat.MIXED
            selected_span = self._selected_surname_span(
                result,
                raw_tokens,
                normalized_raw_tokens,
                normalizer,
                normalized_input.compound_metadata,
            )
            selected_position = selected_span.position if selected_span is not None else "unknown"
            selected_freq, alternate_freq, selected_ratio = self._selected_endpoint_frequency_evidence(
                selected_span,
                raw_tokens,
                normalized_raw_tokens,
                normalized_input.compound_metadata,
                data,
            )
            all_caps_tokens = self._all_caps_tokens(raw_tokens)
            batch_participant = self._candidate_entry_participates(entry)
            name_batch_applied = batch_format_applied and batch_participant and result.success
            batch_changed_format = (
                name_batch_applied
                and NameFormat.MIXED not in {individual_format, selected_format}
                and individual_format != selected_format
            )

            evidence.append(
                NameOrderEvidence(
                    raw_name=entry.name,
                    raw_tokens=raw_tokens,
                    raw_token_count=len(raw_tokens),
                    script_representation=entry.representation,
                    batch_participant=batch_participant,
                    batch_applied=name_batch_applied,
                    batch_changed_format=batch_changed_format,
                    individual_format=individual_format,
                    selected_format=selected_format,
                    selected_surname_position=selected_position,
                    first_token_surname_frequency=first_freq,
                    last_token_surname_frequency=last_freq,
                    selected_surname_frequency=selected_freq,
                    alternate_endpoint_surname_frequency=alternate_freq,
                    selected_over_alternate_surname_frequency_ratio=selected_ratio,
                    has_all_caps_token=bool(all_caps_tokens),
                    all_caps_tokens=all_caps_tokens,
                ),
            )

        return evidence

    @staticmethod
    def _format_from_parse_result(result: ParseResult) -> NameFormat:
        """Return the selected order encoded on a successful parse result."""
        if not result.success or result.parsed_original_order is None:
            return NameFormat.MIXED

        order = result.parsed_original_order.order
        if order and order[0] == "surname":
            return NameFormat.SURNAME_FIRST
        if order and order[-1] == "surname":
            return NameFormat.GIVEN_FIRST
        return NameFormat.MIXED

    def _selected_surname_span(
        self,
        result: ParseResult,
        raw_tokens: list[str],
        normalized_raw_tokens: list[str],
        normalizer,
        compound_metadata,
    ) -> SurnameEndpointSpan | None:
        """Return the selected surname's matched span in the normalized input."""
        if not result.success or result.parsed_original_order is None:
            return None

        surname_tokens = result.parsed_original_order.surname_tokens
        if not surname_tokens or not raw_tokens:
            return None

        normalized_surname_tokens = [normalizer.norm(token) for token in surname_tokens]
        selected_surname = "".join(normalized_surname_tokens)
        selected_format = self._format_from_parse_result(result)
        if selected_surname and selected_format == NameFormat.SURNAME_FIRST:
            for end in range(1, len(normalized_raw_tokens) + 1):
                if "".join(normalized_raw_tokens[:end]) == selected_surname:
                    lookup_key = self._selected_surname_lookup_key(
                        normalized_surname_tokens,
                        raw_tokens,
                        0,
                        end,
                        compound_metadata,
                    )
                    return SurnameEndpointSpan("first", 0, end, lookup_key)

        if selected_surname and selected_format == NameFormat.GIVEN_FIRST:
            for start in range(len(normalized_raw_tokens)):
                if "".join(normalized_raw_tokens[start:]) == selected_surname:
                    lookup_key = self._selected_surname_lookup_key(
                        normalized_surname_tokens,
                        raw_tokens,
                        start,
                        len(raw_tokens),
                        compound_metadata,
                    )
                    return SurnameEndpointSpan(
                        "last",
                        start,
                        len(raw_tokens),
                        lookup_key,
                    )

        if selected_surname and self._internal_window_joins_to(normalized_raw_tokens, selected_surname):
            return SurnameEndpointSpan(
                position="internal",
                start=-1,
                end=-1,
                lookup_key=self._selected_surname_lookup_key(
                    normalized_surname_tokens,
                    raw_tokens,
                    0,
                    None,
                    compound_metadata,
                ),
            )
        return None

    @staticmethod
    def _selected_surname_lookup_key(
        normalized_surname_tokens: list[str],
        raw_tokens: list[str],
        start: int,
        end: int | None,
        compound_metadata,
    ) -> str:
        """Return a surname lookup key for the selected parsed surname."""
        if end is not None:
            compound_target = BatchAnalysisService._compound_target_for_span(
                raw_tokens,
                start,
                end,
                compound_metadata,
            )
            if compound_target is not None:
                return compound_target
        if len(normalized_surname_tokens) > 1:
            return " ".join(normalized_surname_tokens)
        return normalized_surname_tokens[0]

    @staticmethod
    def _internal_window_joins_to(tokens: list[str], target: str) -> bool:
        """Return whether any non-endpoint token window joins to target."""
        for start in range(1, len(tokens) - 1):
            for end in range(start + 1, len(tokens)):
                if "".join(tokens[start:end]) == target:
                    return True
        return False

    @staticmethod
    def _compound_target_for_span(raw_tokens: list[str], start: int, end: int, compound_metadata) -> str | None:
        """Return the shared compound target for a raw span when metadata identifies one."""
        if compound_metadata is None or not (0 <= start < end <= len(raw_tokens)):
            return None

        target: str | None = None
        for token in raw_tokens[start:end]:
            meta = compound_metadata.get(token)
            if not meta or not meta.is_compound or not meta.compound_target:
                return None
            if target is None:
                target = meta.compound_target
            elif target != meta.compound_target:
                return None
        return target

    @staticmethod
    def _endpoint_surname_frequencies(raw_tokens: list[str], normalizer, data) -> tuple[float | None, float | None]:
        """Return surname frequencies for the first and last normalized tokens."""
        if not raw_tokens:
            return None, None

        first_freq = float(data.get_surname_freq(normalizer.norm(raw_tokens[0]), 0))
        last_freq = float(data.get_surname_freq(normalizer.norm(raw_tokens[-1]), 0))
        return first_freq, last_freq

    def _selected_endpoint_frequency_evidence(
        self,
        selected_span: SurnameEndpointSpan | None,
        raw_tokens: list[str],
        normalized_raw_tokens: list[str],
        compound_metadata,
        data,
    ) -> tuple[float | None, float | None, float | None]:
        """Return selected endpoint frequency, alternate frequency, and selected/alternate ratio."""
        if selected_span is None or selected_span.position not in {"first", "last"}:
            return None, None, None

        selected_freq = float(data.get_surname_freq(selected_span.lookup_key, 0))
        alternate_span = self._alternate_endpoint_span(
            selected_span,
            raw_tokens,
            normalized_raw_tokens,
            compound_metadata,
        )
        alternate_freq = None
        if alternate_span is not None:
            alternate_freq = float(data.get_surname_freq(alternate_span.lookup_key, 0))

        if selected_freq is None or alternate_freq is None or alternate_freq <= 0:
            return selected_freq, alternate_freq, None
        return selected_freq, alternate_freq, selected_freq / alternate_freq

    def _alternate_endpoint_span(
        self,
        selected_span: SurnameEndpointSpan,
        raw_tokens: list[str],
        normalized_raw_tokens: list[str],
        compound_metadata,
    ) -> SurnameEndpointSpan | None:
        """Return the opposite endpoint span using the selected raw span width."""
        if selected_span.width <= 0 or selected_span.width > len(raw_tokens):
            return None

        if selected_span.position == "first":
            start = len(raw_tokens) - selected_span.width
            end = len(raw_tokens)
            position = "last"
        elif selected_span.position == "last":
            start = 0
            end = selected_span.width
            position = "first"
        else:
            return None

        lookup_key = self._raw_surname_span_lookup_key(
            raw_tokens,
            normalized_raw_tokens,
            start,
            end,
            compound_metadata,
        )
        return SurnameEndpointSpan(position, start, end, lookup_key)

    @staticmethod
    def _raw_surname_span_lookup_key(
        raw_tokens: list[str],
        normalized_raw_tokens: list[str],
        start: int,
        end: int,
        compound_metadata,
    ) -> str:
        """Return a surname lookup key for a raw endpoint span."""
        compound_target = BatchAnalysisService._compound_target_for_span(
            raw_tokens,
            start,
            end,
            compound_metadata,
        )
        if compound_target is not None:
            return compound_target

        normalized_span = normalized_raw_tokens[start:end]
        if len(normalized_span) > 1:
            return " ".join(normalized_span)
        return normalized_span[0]

    @staticmethod
    def _all_caps_tokens(raw_tokens: list[str]) -> list[str]:
        """Return source tokens that carry an all-caps cue."""
        all_caps = []
        for token in raw_tokens:
            letters_only = "".join(char for char in token if char.isalpha())
            if len(letters_only) > 1 and letters_only.isupper():
                all_caps.append(token)
        return all_caps

    def _build_individual_analyses(
        self,
        name_candidates: list[BatchCandidateEntry],
    ) -> list[IndividualAnalysis]:
        """Build IndividualAnalysis entries with a simple confidence per name.

        Confidence is computed via a softmax over candidate scores.
        - No candidates: confidence = 0.0
        - One candidate: confidence = 1.0
        - Multiple: exp(score_i - max)/sum(exp(score_j - max)) for best candidate
        """
        analyses: list[IndividualAnalysis] = []
        for entry in name_candidates:
            if not entry.candidates or entry.best_candidate is None:
                analyses.append(
                    IndividualAnalysis(
                        raw_name=entry.name,
                        candidates=[],
                        best_candidate=None,
                        confidence=0.0,
                    ),
                )
                continue

            if len(entry.candidates) == 1:
                confidence = 1.0
            else:
                max_score = max(c.score for c in entry.candidates)
                exps = [math.exp(c.score - max_score) for c in entry.candidates]
                denom = sum(exps) if exps else 1.0
                # Locate index of best_candidate (fall back to top-1 if not found)
                try:
                    idx = entry.candidates.index(entry.best_candidate)
                except ValueError:
                    idx = 0
                confidence = exps[idx] / denom if denom > 0 else 0.0

            analyses.append(
                IndividualAnalysis(
                    raw_name=entry.name,
                    candidates=entry.candidates,
                    best_candidate=entry.best_candidate,
                    confidence=float(confidence),
                ),
            )

        return analyses
