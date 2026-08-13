"""
Batch analysis service for detecting format patterns in name lists.

This service analyzes multiple names together to detect consistent formatting
patterns (surname-first vs given-first) and applies the dominant pattern to
improve parsing accuracy for ambiguous individual names.
"""

from __future__ import annotations

import collections.abc  # noqa: TC003
import math
from dataclasses import dataclass, field, replace
from statistics import median
from typing import TYPE_CHECKING

from sinonym.chinese_names_data import HAN_SURNAME_POSITION_READINGS
from sinonym.coretypes import (
    BatchFormatPattern,
    BatchParseResult,
    IndividualAnalysis,
    NameFormat,
    NameOrderEvidence,
    ParseCandidate,
    ParseResult,
)
from sinonym.utils.string_manipulation import StringManipulationUtils

GUARDED_GIVEN_FIRST_BATCH_MIN_SHARE = 0.75
GIVEN_NAME_SHAPE_MIN_SHARE = 0.5
ULTRA_LOW_FIRST_SURNAME_MEDIAN_MAX = 130.0
LONG_GIVEN_NAME_TOKEN_MIN_LENGTH = 6
TWO_TOKEN_NAME_LENGTH = 2
MIN_CANDIDATES_FOR_CONFIDENCE_GAP = 2
BATCH_PARTICIPANT_MIN = 2
BATCH_FORMAT_DIRECTION_MIN_CONFIDENCE = 0.5
HIGH_SURNAME_FREQUENCY_MIN = 1000
MEDIUM_SURNAME_FREQUENCY_MIN = 100
HIGH_SURNAME_POSITION_STRENGTH = 2
MEDIUM_SURNAME_POSITION_STRENGTH = 1
LOW_SURNAME_POSITION_STRENGTH = 0
LATIN_ONLY_REPRESENTATION = "latin_only"
HAN_ONLY_REPRESENTATION = "han_only"
REJECTED_INPUT_REPRESENTATION = "rejected_input"
_HAN_SURNAME_READING_SOURCE_CHARACTERS = frozenset(character for character, _reading in HAN_SURNAME_POSITION_READINGS)


def _identity_classification_input(name: str) -> str:
    """Return an unchanged classification surface."""
    return name


@dataclass(frozen=True)
class BatchAnalysisDependencies:
    """Detector-owned callbacks and constants needed by batch analysis."""

    min_tokens_required: int
    individual_parser: collections.abc.Callable[[str], ParseResult]
    input_failure: collections.abc.Callable[[str], ParseResult | None]
    surname_resolver: SurnameResolver | None = None
    classification_input: collections.abc.Callable[[str], str] = field(
        default=_identity_classification_input,
        kw_only=True,
    )


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
    vote_eligible: bool = True
    raw_tokens: tuple[str, ...] = field(kw_only=True)
    spaced_compound_spans: tuple[SpacedCompoundSpan, ...] = field(default=(), kw_only=True)
    batch_format_locked: bool = field(default=False, kw_only=True)
    individual_result: ParseResult | None = field(default=None, kw_only=True)
    input_failure: ParseResult | None = None
    individual_failure: ParseResult | None = None

    @property
    def participates(self) -> bool:
        """Return whether this entry votes in Latin batch format detection."""
        return bool(
            self.vote_eligible
            and self.candidates
            and self.best_candidate
            and self.best_candidate.format is not NameFormat.MIXED
            and self.representation == LATIN_ONLY_REPRESENTATION,
        )


@dataclass(frozen=True)
class _PreparedCandidate:
    """Deeply immutable parse candidate shared only inside one request."""

    surname_tokens: tuple[str, ...]
    given_tokens: tuple[str, ...]
    score: float
    format: NameFormat
    original_compound_format: str | None

    def materialize(self) -> ParseCandidate:
        """Return a fresh public candidate with unshared token lists."""
        return ParseCandidate(
            surname_tokens=list(self.surname_tokens),
            given_tokens=list(self.given_tokens),
            score=self.score,
            format=self.format,
            original_compound_format=self.original_compound_format,
        )


@dataclass(frozen=True)
class _PreparedName:
    """Name-local work that is independent of a PP or VYS batch policy."""

    name: str
    representation: str
    vote_eligible: bool
    raw_tokens: tuple[str, ...]
    compound_metadata: tuple[tuple[str, object], ...]
    spaced_compound_spans: tuple[SpacedCompoundSpan, ...]
    format_candidates: tuple[_PreparedCandidate, ...]
    individual_candidates: tuple[_PreparedCandidate, ...]
    batch_format_locked: bool = False
    individual_result: ParseResult | None = None
    input_failure: ParseResult | None = None
    individual_failure: ParseResult | None = None


@dataclass(frozen=True)
class RelatedBatchParseResult:
    """PP result plus an optional focal VYS result derived from a full context."""

    pp_batch: BatchParseResult
    vys_batch: BatchParseResult | None
    vys_context_names: tuple[str, ...] | None


_PreparedCache = dict[tuple[str, bool], _PreparedName]


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
    from sinonym.services.ethnicity import EthnicityClassificationService
    from sinonym.services.name_lookup import SurnameResolver
    from sinonym.services.normalization import SpacedCompoundSpan
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
        self._classification_input_callback = dependencies.classification_input
        self._surname_resolver = dependencies.surname_resolver

    def _require_surname_resolver(self) -> SurnameResolver:
        """Return the shared surname resolver for batch lookup policy."""
        if self._surname_resolver is not None:
            return self._surname_resolver
        message = "batch analysis surname resolver is not initialized"
        raise RuntimeError(message)

    def analyze_name_batch(
        self,
        names: list[str],
        normalizer,
        formatting_service,
        options: BatchAnalysisOptions | None = None,
    ) -> BatchParseResult:
        """
        Analyze a batch of names and apply consistent formatting.

        Args:
            names: List of raw name strings to analyze
            normalizer: Normalization service instance
            options: Per-call threshold and minimum batch size

        Returns:
            BatchParseResult with individual analyses and batch-corrected results
        """
        options = options or BatchAnalysisOptions()
        prepared = [self._prepare_name(name, normalizer) for name in names]
        return self._materialize_prepared_batch(
            names,
            prepared,
            normalizer,
            formatting_service,
            options=options,
        )

    def analyze_related_name_batches(  # noqa: PLR0913
        self,
        pp_names: list[str],
        vys_pool_names: list[str] | None,
        normalizer,
        formatting_service,
        options: BatchAnalysisOptions | None = None,
        prepared_cache: _PreparedCache | None = None,
    ) -> RelatedBatchParseResult:
        """Analyze one PP/VYS pair while preparing shared focal names once."""
        options = options or BatchAnalysisOptions()
        if vys_pool_names is None:
            prepared = self._prepare_names_deduplicated(pp_names, normalizer, prepared_cache)
            pp_batch = self._materialize_prepared_batch(
                pp_names,
                prepared,
                normalizer,
                formatting_service,
                options=options,
            )
            return RelatedBatchParseResult(pp_batch, None, None)

        if vys_pool_names[: len(pp_names)] != pp_names:
            message = "VYS pool must start with the PP names"
            raise ValueError(message)

        pool_prepared = self._prepare_names_deduplicated(
            vys_pool_names,
            normalizer,
            prepared_cache,
            individual_count=len(pp_names),
        )
        pp_batch = self._materialize_prepared_batch(
            pp_names,
            pool_prepared[: len(pp_names)],
            normalizer,
            formatting_service,
            options=options,
        )
        vys_batch = self._materialize_prepared_batch(
            vys_pool_names,
            pool_prepared,
            normalizer,
            formatting_service,
            options=options,
            output_count=len(pp_names),
        )
        return RelatedBatchParseResult(pp_batch, vys_batch, tuple(vys_pool_names))

    def detect_batch_format(
        self,
        names: list[str],
        normalizer,
        *,
        format_threshold: float | None = None,
    ) -> BatchFormatPattern:
        """
        Detect the format pattern of a batch without full processing.

        Returns:
            BatchFormatPattern indicating the dominant format and confidence
        """
        prepared = [self._prepare_name(name, normalizer, need_individual=False) for name in names]
        name_candidates, individual_candidates = self._materialize_candidate_entries(prepared)
        name_candidates = self._promote_guarded_given_first_batch_votes(name_candidates, individual_candidates)

        resolved_threshold = self._resolved_format_threshold(format_threshold)
        return self._detect_format_pattern(name_candidates, resolved_threshold)

    def _prepare_names_deduplicated(
        self,
        names: list[str],
        normalizer,
        cache: _PreparedCache | None = None,
        *,
        individual_count: int | None = None,
    ) -> list[_PreparedName]:
        """Prepare each distinct raw name once and preserve occurrence order."""
        cache = {} if cache is None else cache
        prepared: list[_PreparedName] = []
        individual_count = len(names) if individual_count is None else individual_count
        for index, name in enumerate(names):
            need_individual = index < individual_count
            record = cache.get((name, True)) if not need_individual else None
            if record is None:
                record = cache.get((name, need_individual))
            if record is None:
                record = self._prepare_name(name, normalizer, need_individual=need_individual)
                cache[(name, need_individual)] = record
            prepared.append(record)
        return prepared

    def _prepare_name(self, name: str, normalizer, *, need_individual: bool = True) -> _PreparedName:
        """Compute the name-local inputs shared by all batch contexts."""
        input_failure = self._input_failure(name)
        if input_failure is not None:
            return _PreparedName(
                name=name,
                representation=REJECTED_INPUT_REPRESENTATION,
                vote_eligible=False,
                raw_tokens=(),
                compound_metadata=(),
                spaced_compound_spans=(),
                format_candidates=(),
                individual_candidates=(),
                input_failure=input_failure,
            )

        classification_input = self._classification_input_callback(name)
        normalized_input = normalizer.apply(classification_input)
        contextual_taiwan = self._is_contextual_taiwan_name(normalized_input.roman_tokens)
        representation = self._script_representation(normalizer, normalized_input)
        has_authoritative_structure = bool(
            normalized_input.authoritative_source_format is not None or normalized_input.from_camel_case_pair,
        )
        parsed_individual = self._individual_parser(name) if has_authoritative_structure else None
        individual_result = parsed_individual if parsed_individual is not None and parsed_individual.success else None
        authoritative_candidate = self._authoritative_structural_candidate(normalized_input, individual_result)
        common = {
            "name": name,
            "representation": representation,
            "vote_eligible": self._batch_vote_eligible(normalized_input),
            "batch_format_locked": (
                normalized_input.surname_first_parenthetical_hint or contextual_taiwan or authoritative_candidate is not None
            ),
            "individual_result": individual_result,
            "raw_tokens": normalized_input.authored_roman_tokens or tuple(normalized_input.roman_tokens),
            "compound_metadata": tuple(normalized_input.compound_metadata.items()),
            "spaced_compound_spans": normalized_input.spaced_compound_spans,
        }
        if authoritative_candidate is not None:
            candidates = (authoritative_candidate,)
            return _PreparedName(
                format_candidates=candidates,
                individual_candidates=candidates,
                **common,
            )
        if not self._is_batch_format_participant(representation):
            return _PreparedName(format_candidates=(), individual_candidates=(), **common)

        format_candidates, individual_candidates, failure = self._prepare_candidate_views(
            classification_input,
            normalized_input,
            need_individual=need_individual,
        )
        return _PreparedName(
            format_candidates=format_candidates,
            individual_candidates=individual_candidates,
            individual_failure=failure,
            **common,
        )

    def _authoritative_structural_candidate(
        self,
        normalized_input,
        individual_result: ParseResult | None,
    ) -> _PreparedCandidate | None:
        """Return one locked candidate when source structure decides the scalar parse."""
        if not (normalized_input.authoritative_source_format is not None or normalized_input.from_camel_case_pair):
            return None
        if individual_result is None or not individual_result.success or individual_result.parsed is None:
            return None

        selected_format = normalized_input.authoritative_source_format or self._format_from_parse_result(
            individual_result,
        )
        if selected_format is NameFormat.MIXED:
            return None
        parsed = individual_result.parsed
        return _PreparedCandidate(
            surname_tokens=tuple(parsed.surname_tokens),
            given_tokens=tuple(parsed.given_tokens),
            score=0.0,
            format=selected_format,
            original_compound_format=individual_result.original_compound_surname,
        )

    def _prepare_candidate_views(
        self,
        name: str,
        normalized_input,
        *,
        need_individual: bool,
    ) -> tuple[tuple[_PreparedCandidate, ...], tuple[_PreparedCandidate, ...], ParseResult | None]:
        """Build exact no-bonus and guarded score views from one parse-option pass."""
        tokens = list(normalized_input.roman_tokens)
        if len(tokens) < self._min_tokens_required:
            return (), (), None

        if self._ethnicity_service is not None:
            ethnicity = self._ethnicity_service.classify_ethnicity(
                normalized_input.roman_tokens,
                normalized_input.norm_map,
                name,
            )
            if ethnicity.success is False:
                return (), (), ethnicity

        parses = self._parsing_service.generate_parse_options(
            tokens,
            normalized_input.norm_map,
            normalized_input.compound_metadata,
            normalized_input.spaced_compound_spans,
        )
        format_candidates: list[_PreparedCandidate] = []
        scored_parses: list[tuple[list[str], list[str], _PreparedCandidate]] = []
        score_cache: dict[str, dict] = {}

        def candidate_rank_key(candidate: _PreparedCandidate) -> tuple[float, float, str]:
            return self._parsing_service.candidate_rank_key(
                candidate.surname_tokens,
                candidate.given_tokens,
                candidate.score,
                tokens,
            )

        for surname_tokens, given_tokens, original_compound_format in parses:
            common = {
                "surname_tokens": tuple(surname_tokens),
                "given_tokens": tuple(given_tokens),
                "format": self._determine_parse_format(surname_tokens, given_tokens, tokens),
                "original_compound_format": original_compound_format,
            }
            no_bonus_score = self._parsing_service.calculate_parse_score(
                surname_tokens,
                given_tokens,
                tokens,
                normalized_input.norm_map,
                is_all_chinese=False,
                original_compound_format=original_compound_format,
                score_cache=score_cache,
                allow_guarded_given_first_bonus=False,
                surname_first_parenthetical_hint=normalized_input.surname_first_parenthetical_hint,
            )
            candidate = _PreparedCandidate(score=no_bonus_score, **common)
            format_candidates.append(candidate)
            scored_parses.append((surname_tokens, given_tokens, candidate))

        format_candidates.sort(key=candidate_rank_key, reverse=True)
        if not need_individual and (not format_candidates or format_candidates[0].format != NameFormat.SURNAME_FIRST):
            return tuple(format_candidates), (), None

        individual_candidates: list[_PreparedCandidate] = []
        for surname_tokens, given_tokens, candidate in scored_parses:
            guarded_score = candidate.score
            if self._guarded_score_may_differ(
                surname_tokens,
                given_tokens,
                tokens,
                surname_first_parenthetical_hint=normalized_input.surname_first_parenthetical_hint,
            ):
                guarded_score = self._parsing_service.calculate_parse_score(
                    surname_tokens,
                    given_tokens,
                    tokens,
                    normalized_input.norm_map,
                    is_all_chinese=False,
                    original_compound_format=candidate.original_compound_format,
                    score_cache=score_cache,
                    allow_guarded_given_first_bonus=True,
                    surname_first_parenthetical_hint=False,
                )
            individual_candidates.append(replace(candidate, score=guarded_score))

        individual_candidates.sort(key=candidate_rank_key, reverse=True)
        return tuple(format_candidates), tuple(individual_candidates), None

    @staticmethod
    def _guarded_score_may_differ(
        surname_tokens: list[str],
        given_tokens: list[str],
        tokens: list[str],
        *,
        surname_first_parenthetical_hint: bool,
    ) -> bool:
        """Return whether guarded scoring can enter its only flag-dependent branch."""
        return bool(
            not surname_first_parenthetical_hint
            and len(tokens) == TWO_TOKEN_NAME_LENGTH
            and len(surname_tokens) == 1
            and len(given_tokens) == 1
            and given_tokens[0] == tokens[0]
            and surname_tokens[0] == tokens[1],
        )

    @staticmethod
    def _materialize_candidate_entries(
        prepared_names: list[_PreparedName],
    ) -> tuple[list[BatchCandidateEntry], list[BatchCandidateEntry]]:
        """Create unshared slot-local candidate DTOs from immutable preparations."""
        format_entries: list[BatchCandidateEntry] = []
        individual_entries: list[BatchCandidateEntry] = []
        for prepared in prepared_names:
            compound_metadata = dict(prepared.compound_metadata)
            format_candidates = [candidate.materialize() for candidate in prepared.format_candidates]
            individual_candidates = [candidate.materialize() for candidate in prepared.individual_candidates]
            common = {
                "name": prepared.name,
                "compound_metadata": compound_metadata,
                "representation": prepared.representation,
                "vote_eligible": prepared.vote_eligible,
                "batch_format_locked": prepared.batch_format_locked,
                "individual_result": prepared.individual_result,
                "raw_tokens": prepared.raw_tokens,
                "spaced_compound_spans": prepared.spaced_compound_spans,
            }
            format_entries.append(
                BatchCandidateEntry(
                    candidates=format_candidates,
                    best_candidate=format_candidates[0] if format_candidates else None,
                    input_failure=prepared.input_failure,
                    individual_failure=prepared.individual_failure,
                    **common,
                ),
            )
            individual_entries.append(
                BatchCandidateEntry(
                    candidates=individual_candidates,
                    best_candidate=individual_candidates[0] if individual_candidates else None,
                    input_failure=prepared.input_failure,
                    individual_failure=prepared.individual_failure,
                    **common,
                ),
            )
        return format_entries, individual_entries

    def _materialize_prepared_batch(  # noqa: PLR0913
        self,
        names: list[str],
        prepared_names: list[_PreparedName],
        normalizer,
        formatting_service,
        *,
        options: BatchAnalysisOptions,
        output_count: int | None = None,
    ) -> BatchParseResult:
        """Apply one batch policy and materialize only its requested leading rows."""
        materialized_count = len(names) if output_count is None else output_count
        if not 0 <= materialized_count <= len(names) or len(prepared_names) != len(names):
            message = "prepared names and requested output count must align with the submitted batch"
            raise ValueError(message)

        format_entries, individual_entries = self._materialize_candidate_entries(prepared_names)
        format_entries = self._promote_guarded_given_first_batch_votes(format_entries, individual_entries)
        format_pattern = self._detect_format_pattern(
            format_entries,
            self._resolved_format_threshold(options.format_threshold),
        )
        if len(names) < options.minimum_batch_size:
            format_pattern = replace(format_pattern, threshold_met=False)

        focal_format_entries = format_entries[:materialized_count]
        focal_individual_entries = individual_entries[:materialized_count]
        if format_pattern.total_count > 0 and format_pattern.threshold_met:
            results = self._apply_batch_format(
                focal_format_entries,
                focal_individual_entries,
                format_pattern.dominant_format,
                formatting_service,
            )
            improvements = self._find_improvements(focal_individual_entries, results)
        else:
            results = self._materialize_individual_results(focal_individual_entries)
            improvements = []

        return BatchParseResult(
            names=list(names[:materialized_count]),
            results=results,
            format_pattern=format_pattern,
            individual_analyses=self._build_individual_analyses(focal_individual_entries, results),
            improvements=improvements,
            name_order_evidence=self._build_name_order_evidence(
                focal_individual_entries,
                results,
                normalizer,
                format_pattern,
            ),
        )

    def _materialize_individual_results(
        self,
        entries: list[BatchCandidateEntry],
    ) -> list[ParseResult]:
        """Materialize standalone results through the authoritative scalar policy."""
        return [self._materialize_individual_result(entry) for entry in entries]

    def _materialize_individual_result(
        self,
        entry: BatchCandidateEntry,
    ) -> ParseResult:
        """Materialize one row without applying a peer-derived format."""
        if entry.input_failure is not None:
            return entry.input_failure
        if entry.individual_result is not None:
            return entry.individual_result
        if not self._is_batch_format_participant(entry.representation):
            return self._locked_representation_result(entry.name)
        if self._is_contextual_taiwan_name(entry.raw_tokens):
            return self._locked_representation_result(entry.name)
        if entry.best_candidate is None:
            return entry.individual_failure or ParseResult.failure("no valid parse found")
        return self._individual_parser(entry.name)

    def _promote_guarded_given_first_batch_votes(
        self,
        name_candidates: list[BatchCandidateEntry],
        individual_candidates: list[BatchCandidateEntry],
    ) -> list[BatchCandidateEntry]:
        """Promote contested given-first votes when batch-level shape evidence supports them."""
        participant_count = sum(1 for entry in name_candidates if self._candidate_entry_participates(entry))
        if participant_count == 0:
            return name_candidates

        promoted, first_surname_freqs, given_shape_count = self._collect_guarded_given_first_promotions(
            name_candidates,
            individual_candidates,
        )
        given_first_support = sum(
            1
            for entry in name_candidates
            if entry.participates and entry.best_candidate is not None and entry.best_candidate.format == NameFormat.GIVEN_FIRST
        )

        if not self._should_promote_guarded_given_first_votes(
            participant_count,
            promoted,
            given_first_support,
            first_surname_freqs,
            given_shape_count,
        ):
            return name_candidates

        adjusted = list(name_candidates)
        for index, promoted_candidate in promoted:
            adjusted[index] = replace(adjusted[index], best_candidate=promoted_candidate)
        return adjusted

    def _collect_guarded_given_first_promotions(
        self,
        name_candidates: list[BatchCandidateEntry],
        individual_candidates: list[BatchCandidateEntry],
    ) -> tuple[list[tuple[int, ParseCandidate]], list[float], int]:
        """Collect given-first candidates that only become best under individual guarded scoring."""
        promoted: list[tuple[int, ParseCandidate]] = []
        first_surname_freqs: list[float] = []
        given_shape_count = 0
        surname_resolver = self._require_surname_resolver()

        for index, entry in enumerate(name_candidates):
            if not self._candidate_entry_participates(entry):
                continue

            promotion = self._guarded_given_first_promotion(
                entry,
                individual_candidates[index].best_candidate,
            )
            if promotion is None:
                continue

            promoted_candidate, first_token = promotion
            promoted.append((index, promoted_candidate))
            first_surname_freqs.append(surname_resolver.evidence_frequency(first_token))
            if self._has_given_name_shape(first_token):
                given_shape_count += 1

        return promoted, first_surname_freqs, given_shape_count

    def _guarded_given_first_promotion(
        self,
        entry: BatchCandidateEntry,
        individual_best: ParseCandidate | None,
    ) -> tuple[ParseCandidate, str] | None:
        """Return the promoted given-first candidate and first-token evidence, if any."""
        if not entry.candidates or entry.best_candidate is None or entry.best_candidate.format != NameFormat.SURNAME_FIRST:
            return None

        tokens = list(entry.raw_tokens)
        if len(tokens) != TWO_TOKEN_NAME_LENGTH:
            return None

        if individual_best is None or individual_best.format != NameFormat.GIVEN_FIRST:
            return None

        promoted_candidate = self._matching_given_first_candidate(entry.candidates, individual_best)
        if promoted_candidate is None:
            return None

        first_token = tokens[0]
        return promoted_candidate, first_token

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
        given_first_support: int,
        first_surname_freqs: list[float],
        given_shape_count: int,
    ) -> bool:
        """Return whether guarded given-first votes have enough batch-level support."""
        if not promoted or (len(promoted) + given_first_support) / participant_count < GUARDED_GIVEN_FIRST_BATCH_MIN_SHARE:
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

        # Rebuild the whole token stream so equal endpoint text cannot transfer
        # lineage from a different occurrence (for example, ``Li Wei Li``).
        # Expanded compact/hyphenated surnames deliberately fall through to
        # their narrower representation-specific handling.
        surname_first = [*surname_tokens, *_given_tokens] == original_tokens
        given_first = [*_given_tokens, *surname_tokens] == original_tokens
        if surname_first != given_first:
            return NameFormat.SURNAME_FIRST if surname_first else NameFormat.GIVEN_FIRST
        if surname_first and given_first:
            return NameFormat.MIXED

        # Compact compound surname: parsed surname tokens may be sub-tokens of a
        # single original token (e.g. ['Ou', 'yang'] from 'Ouyang').
        joined_surname = "".join(surname_tokens).lower()
        if original_tokens[0].lower() == joined_surname:
            return NameFormat.SURNAME_FIRST
        if original_tokens[-1].lower() == joined_surname:
            return NameFormat.GIVEN_FIRST

        # Preserve the existing surname-first default for genuinely ambiguous,
        # internal, or non-lineage-preserving expanded parses.
        return NameFormat.SURNAME_FIRST

    def _detect_format_pattern(
        self,
        name_candidates: list[BatchCandidateEntry],
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
        )
        confidence = self._count_confidence(stats, dominant_format)
        has_decisive_vote = stats.surname_first_preferences != stats.given_first_preferences
        has_unopposed_dominant_vote = (
            dominant_format == NameFormat.SURNAME_FIRST and stats.unopposed_surname_first_preferences > 0
        ) or (dominant_format == NameFormat.GIVEN_FIRST and stats.unopposed_given_first_preferences > 0)
        has_confident_direction = decision_confidence > BATCH_FORMAT_DIRECTION_MIN_CONFIDENCE and (
            has_decisive_vote or has_unopposed_dominant_vote
        )
        has_enough_voters = stats.total_preferences >= BATCH_PARTICIPANT_MIN
        threshold_met = decision_confidence >= format_threshold and has_confident_direction and has_enough_voters

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

        for entry in name_candidates:
            if not entry.participates:
                continue
            stats.names_with_candidates += 1

            weight = self._candidate_vote_weight(entry.candidates)
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
        format_entries: list[BatchCandidateEntry],
        individual_entries: list[BatchCandidateEntry],
        target_format: NameFormat,
        formatting_service,
    ) -> list[ParseResult]:
        """Apply the detected batch format by selecting best candidate matching the format."""
        results = []

        # Process all names in one pass and apply the target format.
        for format_entry, individual_entry in zip(format_entries, individual_entries, strict=True):
            if not self._batch_format_applies_to_entry(format_entry):
                results.append(self._materialize_individual_result(individual_entry))
                continue

            # Participation guarantees a non-empty candidate list and a best_candidate,
            # so a candidate is always selected below.
            matching_candidates = [c for c in format_entry.candidates if c.format == target_format]
            selected_candidate = matching_candidates[0] if matching_candidates else format_entry.best_candidate

            result = self._candidate_to_parse_result(
                selected_candidate,
                formatting_service,
                format_entry.compound_metadata,
            )
            results.append(result)

        return results

    def _apply_tie_breaking_heuristics(self, name_candidates: list[BatchCandidateEntry]) -> NameFormat:
        """Apply secondary heuristics for tie-breaking when confidence-weighted voting fails."""
        surname_first_strength = 0
        given_first_strength = 0
        surname_resolver = self._require_surname_resolver()

        for entry in name_candidates:
            if not entry.participates:
                continue

            if len(entry.raw_tokens) != TWO_TOKEN_NAME_LENGTH:
                continue

            tokens = list(entry.raw_tokens)
            first_token, second_token = tokens
            surname_first_strength += self._surname_position_strength(first_token, surname_resolver)
            given_first_strength += self._surname_position_strength(second_token, surname_resolver)

        surname_first_strength += 0.5

        if surname_first_strength > given_first_strength:
            return NameFormat.SURNAME_FIRST
        if given_first_strength > surname_first_strength:
            return NameFormat.GIVEN_FIRST
        return NameFormat.SURNAME_FIRST

    @staticmethod
    def _surname_position_strength(token: str, surname_resolver: SurnameResolver) -> int:
        """Return parser-policy surname-frequency strength for a decision position."""
        surname_freq = surname_resolver.parser_frequency((token,))
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

    @classmethod
    def _batch_vote_eligible(cls, normalized_input) -> bool:
        """Return whether a normalized Latin row should vote in batch order detection."""
        return not cls._all_caps_tokens(list(normalized_input.roman_tokens))

    def _input_failure(self, name: str) -> ParseResult | None:
        """Return an early detector-owned input failure, if one matches."""
        return self._input_failure_callback(name)

    def _is_contextual_taiwan_name(self, tokens: tuple[str, ...]) -> bool:
        """Return whether existing Taiwan evidence makes source order terminal."""
        return bool(
            self._ethnicity_service is not None and self._ethnicity_service.contextual_taiwan_given_parts(tokens) is not None,
        )

    @staticmethod
    def _candidate_entry_participates(entry: BatchCandidateEntry) -> bool:
        """Return whether a candidate entry contributes to batch format detection."""
        return entry.participates

    @staticmethod
    def _batch_format_applies_to_entry(entry: BatchCandidateEntry) -> bool:
        """Return whether peers may override this row's selected format."""
        return entry.participates and not entry.batch_format_locked

    def _locked_representation_result(
        self,
        name: str,
    ) -> ParseResult:
        """Return a structural individual parse for records outside the Latin batch cohort."""
        return self._individual_parser(name)

    def _candidate_to_parse_result(
        self,
        candidate: ParseCandidate | None,
        formatting_service,
        compound_metadata,
    ) -> ParseResult:
        """Convert a ParseCandidate to a ParseResult using the real formatting service."""
        if not candidate:
            return ParseResult.failure("no valid parse found")

        return formatting_service.materialize_parse_result(
            candidate.surname_tokens,
            candidate.given_tokens,
            candidate.format,
            {},  # norm_map - not needed here because candidate tokens are already normalized
            compound_metadata,
            original_compound_format=candidate.original_compound_format,
        )

    def _find_improvements(
        self,
        name_candidates: list[BatchCandidateEntry],
        batch_results: list[ParseResult],
    ) -> list[int]:
        """Find indices whose selected parse format changed under batch context."""
        improvements = []

        for i, (entry, batch_result) in enumerate(
            zip(name_candidates, batch_results, strict=True),
        ):
            if not self._batch_format_applies_to_entry(entry):
                continue

            if not entry.best_candidate or not batch_result.success:
                continue

            batch_format = self._format_from_parse_result(batch_result)
            if (
                NameFormat.MIXED not in {entry.best_candidate.format, batch_format}
                and entry.best_candidate.format != batch_format
            ):
                improvements.append(i)

        return improvements

    def _build_name_order_evidence(
        self,
        name_candidates: list[BatchCandidateEntry],
        results: list[ParseResult],
        normalizer,
        format_pattern: BatchFormatPattern,
    ) -> list[NameOrderEvidence]:
        """Build aligned per-name evidence for external batch-context routing."""
        evidence: list[NameOrderEvidence] = []
        batch_format_applied = format_pattern.threshold_met and format_pattern.total_count > 0
        surname_resolver: SurnameResolver | None = None

        for entry, result in zip(name_candidates, results, strict=True):
            if entry.representation == REJECTED_INPUT_REPRESENTATION:
                evidence.append(
                    NameOrderEvidence(
                        raw_name=entry.name,
                        script_representation=entry.representation,
                        selected_format=self._format_from_parse_result(result),
                    ),
                )
                continue

            if entry.raw_tokens:
                raw_tokens = list(entry.raw_tokens)
                compound_metadata = entry.compound_metadata or {}
                spaced_compound_spans = entry.spaced_compound_spans
            else:
                normalized_input = normalizer.apply(entry.name)
                raw_tokens = list(normalized_input.roman_tokens)
                compound_metadata = normalized_input.compound_metadata
                spaced_compound_spans = normalized_input.spaced_compound_spans

            if surname_resolver is None:
                surname_resolver = self._require_surname_resolver()
            normalized_raw_tokens = [self._surname_lookup_key_for_token(token, surname_resolver) for token in raw_tokens]
            normalized_span_tokens = normalized_raw_tokens
            if entry.representation == HAN_ONLY_REPRESENTATION and any(
                character in _HAN_SURNAME_READING_SOURCE_CHARACTERS for character in entry.name
            ):
                source_characters = normalizer.han_roman_source_characters(normalizer.apply(entry.name))
                normalized_span_tokens = self._han_surname_reading_alignment_keys(
                    normalized_raw_tokens,
                    source_characters,
                )
            first_freq, last_freq = self._endpoint_surname_frequencies(raw_tokens, surname_resolver)
            selected_format = self._format_from_parse_result(result)
            individual_format = entry.best_candidate.format if entry.best_candidate else self._format_from_parse_result(result)
            selected_span = self._selected_surname_span(
                result,
                raw_tokens,
                normalized_span_tokens,
                surname_resolver,
                compound_metadata,
                spaced_compound_spans,
            )
            selected_position = selected_span.position if selected_span is not None else "unknown"
            selected_token_count = selected_span.width if selected_span is not None else 0
            selected_freq, alternate_freq, selected_ratio = self._selected_endpoint_frequency_evidence(
                selected_span,
                raw_tokens,
                normalized_span_tokens,
                compound_metadata,
                surname_resolver,
                spaced_compound_spans,
            )
            all_caps_tokens = self._all_caps_tokens(raw_tokens)
            batch_participant = self._candidate_entry_participates(entry)
            name_batch_applied = (
                batch_format_applied
                and self._batch_format_applies_to_entry(entry)
                and result.success
                and selected_format is format_pattern.dominant_format
            )
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
                    selected_surname_token_count=selected_token_count,
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

    def _selected_surname_span(  # noqa: PLR0913 - raw and normalized span evidence are independent inputs
        self,
        result: ParseResult,
        raw_tokens: list[str],
        normalized_raw_tokens: list[str],
        surname_resolver: SurnameResolver,
        compound_metadata,
        spaced_compound_spans: tuple[SpacedCompoundSpan, ...],
    ) -> SurnameEndpointSpan | None:
        """Return the selected surname's matched span in the normalized input."""
        if not result.success or result.parsed_original_order is None:
            return None

        surname_tokens = result.parsed_original_order.surname_tokens
        if not surname_tokens or not raw_tokens:
            return None

        normalized_surname_tokens = [self._surname_lookup_key_for_token(token, surname_resolver) for token in surname_tokens]
        selected_surname = "".join(normalized_surname_tokens)
        selected_surname_key = self._selected_compound_surname_lookup_key(
            surname_tokens,
            raw_tokens,
            compound_metadata,
            spaced_compound_spans,
        ) or " ".join(normalized_surname_tokens)
        selected_format = self._format_from_parse_result(result)
        if selected_surname and selected_format == NameFormat.SURNAME_FIRST:
            for end in range(1, len(normalized_raw_tokens) + 1):
                lookup_key = self._raw_surname_span_lookup_key(
                    raw_tokens,
                    normalized_raw_tokens,
                    0,
                    end,
                    compound_metadata,
                    spaced_compound_spans,
                )
                if "".join(normalized_raw_tokens[:end]) == selected_surname or lookup_key == selected_surname_key:
                    return SurnameEndpointSpan("first", 0, end, lookup_key)

        if selected_surname and selected_format == NameFormat.GIVEN_FIRST:
            for start in range(len(normalized_raw_tokens)):
                lookup_key = self._raw_surname_span_lookup_key(
                    raw_tokens,
                    normalized_raw_tokens,
                    start,
                    len(raw_tokens),
                    compound_metadata,
                    spaced_compound_spans,
                )
                if "".join(normalized_raw_tokens[start:]) == selected_surname or lookup_key == selected_surname_key:
                    return SurnameEndpointSpan(
                        "last",
                        start,
                        len(raw_tokens),
                        lookup_key,
                    )

        internal_window = self._internal_window_span(normalized_raw_tokens, selected_surname)
        if selected_surname and internal_window is not None:
            start, end = internal_window
            lookup_key = self._raw_surname_span_lookup_key(
                raw_tokens,
                normalized_raw_tokens,
                start,
                end,
                compound_metadata,
                spaced_compound_spans,
            )
            return SurnameEndpointSpan(
                position="internal",
                start=start,
                end=end,
                lookup_key=lookup_key,
            )
        return None

    @staticmethod
    def _han_surname_reading_alignment_keys(tokens: list[str], source_characters: tuple[str, ...]) -> list[str]:
        """Align Han-derived pypinyin keys with surname-position readings."""
        return [
            HAN_SURNAME_POSITION_READINGS.get((character, token), token)
            for character, token in zip(source_characters, tokens, strict=True)
        ]

    @staticmethod
    def _selected_compound_surname_lookup_key(
        surname_tokens: list[str],
        raw_tokens: list[str],
        compound_metadata,
        spaced_compound_spans: tuple[SpacedCompoundSpan, ...],
    ) -> str | None:
        """Return the source compound key when formatter-preserved tokens match it."""
        if compound_metadata is None:
            return None

        selected_key = BatchAnalysisService._display_parts_key(surname_tokens)
        for start in range(len(raw_tokens)):
            for end in range(start + 1, len(raw_tokens) + 1):
                compound_target = BatchAnalysisService._compound_target_for_span(
                    raw_tokens,
                    start,
                    end,
                    compound_metadata,
                    spaced_compound_spans,
                )
                if compound_target is None:
                    continue
                span_parts = BatchAnalysisService._compound_display_parts_for_span(
                    raw_tokens,
                    start,
                    end,
                    compound_metadata,
                )
                if BatchAnalysisService._display_parts_key(span_parts) == selected_key:
                    return compound_target
        return None

    @staticmethod
    def _compound_display_parts_for_span(raw_tokens: list[str], start: int, end: int, compound_metadata) -> list[str]:
        """Return source-preserving parts for a compound raw span."""
        if end - start > 1:
            return raw_tokens[start:end]

        token = raw_tokens[start]
        meta = compound_metadata.get(token)
        if meta is not None and meta.is_compound:
            return StringManipulationUtils.split_compound_token(token, meta)
        return [token]

    @staticmethod
    def _display_parts_key(parts: list[str]) -> str:
        """Return a separator-insensitive key for source-display name parts."""
        return "".join(char.lower() for part in parts for char in part if char.isalpha())

    @staticmethod
    def _internal_window_span(tokens: list[str], target: str) -> tuple[int, int] | None:
        """Return the non-endpoint token window whose joined text matches target."""
        for start in range(1, len(tokens) - 1):
            for end in range(start + 1, len(tokens)):
                if "".join(tokens[start:end]) == target:
                    return start, end
        return None

    @staticmethod
    def _compound_target_for_span(
        raw_tokens: list[str],
        start: int,
        end: int,
        compound_metadata,
        spaced_compound_spans: tuple[SpacedCompoundSpan, ...],
    ) -> str | None:
        """Return the compound target for one exact raw-token occurrence."""
        if compound_metadata is None or not (0 <= start < end <= len(raw_tokens)):
            return None

        if end - start > 1:
            for span in spaced_compound_spans:
                if span.start == start and span.end == end:
                    return span.compound_target
            return None

        metadata = compound_metadata.get(raw_tokens[start])
        if metadata and metadata.is_compound and metadata.format_type != "spaced" and metadata.compound_target:
            return metadata.compound_target
        return None

    @staticmethod
    def _endpoint_surname_frequencies(
        raw_tokens: list[str],
        surname_resolver: SurnameResolver,
    ) -> tuple[float | None, float | None]:
        """Return surname frequencies for the first and last normalized tokens."""
        if not raw_tokens:
            return None, None

        first_freq = float(surname_resolver.evidence_frequency(raw_tokens[0]))
        last_freq = float(surname_resolver.evidence_frequency(raw_tokens[-1]))
        return first_freq, last_freq

    @staticmethod
    def _surname_lookup_key_for_token(token: str, surname_resolver: SurnameResolver) -> str:
        """Return the evidence key for batch span matching only."""
        return surname_resolver.evidence_span_key(token)

    def _selected_endpoint_frequency_evidence(  # noqa: PLR0913 - endpoint evidence inputs remain explicit
        self,
        selected_span: SurnameEndpointSpan | None,
        raw_tokens: list[str],
        normalized_raw_tokens: list[str],
        compound_metadata,
        surname_resolver: SurnameResolver,
        spaced_compound_spans: tuple[SpacedCompoundSpan, ...] = (),
    ) -> tuple[float | None, float | None, float | None]:
        """Return selected endpoint frequency, alternate frequency, and selected/alternate ratio."""
        if selected_span is None or selected_span.position not in {"first", "last"}:
            return None, None, None

        selected_freq = float(surname_resolver.evidence_frequency_for_key(selected_span.lookup_key))
        alternate_span = self._alternate_endpoint_span(
            selected_span,
            raw_tokens,
            normalized_raw_tokens,
            compound_metadata,
            spaced_compound_spans,
        )
        alternate_freq = None
        if alternate_span is not None:
            alternate_freq = float(surname_resolver.evidence_frequency_for_key(alternate_span.lookup_key))

        if alternate_freq is None or alternate_freq <= 0:
            return selected_freq, alternate_freq, None
        return selected_freq, alternate_freq, selected_freq / alternate_freq

    def _alternate_endpoint_span(
        self,
        selected_span: SurnameEndpointSpan,
        raw_tokens: list[str],
        normalized_raw_tokens: list[str],
        compound_metadata,
        spaced_compound_spans: tuple[SpacedCompoundSpan, ...],
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
            spaced_compound_spans,
        )
        return SurnameEndpointSpan(position, start, end, lookup_key)

    @staticmethod
    def _raw_surname_span_lookup_key(  # noqa: PLR0913 - raw and normalized span evidence are independent inputs
        raw_tokens: list[str],
        normalized_raw_tokens: list[str],
        start: int,
        end: int,
        compound_metadata,
        spaced_compound_spans: tuple[SpacedCompoundSpan, ...],
    ) -> str:
        """Return a surname lookup key for a raw endpoint span."""
        compound_target = BatchAnalysisService._compound_target_for_span(
            raw_tokens,
            start,
            end,
            compound_metadata,
            spaced_compound_spans,
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
        results: list[ParseResult],
    ) -> list[IndividualAnalysis]:
        """Build IndividualAnalysis entries with a simple confidence per name.

        Confidence is computed via a softmax over candidate scores.
        - No candidates: confidence = 1.0 for successful structural parses, else 0.0
        - One candidate: confidence = 1.0
        - Multiple: exp(score_i - max)/sum(exp(score_j - max)) for best candidate
        """
        analyses: list[IndividualAnalysis] = []
        for entry, result in zip(name_candidates, results, strict=True):
            if not entry.candidates or entry.best_candidate is None:
                analyses.append(
                    IndividualAnalysis(
                        raw_name=entry.name,
                        candidates=[],
                        best_candidate=None,
                        confidence=1.0 if result.success else 0.0,
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
