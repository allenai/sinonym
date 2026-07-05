"""
Name parsing service for Chinese name processing.

This module provides sophisticated name parsing algorithms to identify
surname/given name boundaries using probabilistic scoring and cultural
validation patterns.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from sinonym.chinese_names_data import COMPOUND_VARIANTS, KOREAN_GIVEN_PATTERNS, OVERLAPPING_KOREAN_SURNAMES
from sinonym.coretypes import ParseResult
from sinonym.utils.string_manipulation import StringManipulationUtils

if TYPE_CHECKING:
    from sinonym.services.normalization import CompoundMetadata

LOW_FREQUENCY_SURNAME_MAX = 500.0
GIVEN_FIRST_SURNAME_FREQ_RATIO_MIN = 50.0
GIVEN_FIRST_ORDER_PRESERVATION_BONUS = 4.0
# Unattested-vs-attested ambiguity ceiling: an unattested token can plausibly tie
# with a modest surname, but not with a dominant one (see _is_ambiguous_case and
# the comparative-feature gating).
UNATTESTED_AMBIGUITY_SURNAME_FREQ_MAX = 5000.0
# Positional given-syllable feature for 3-token parses: the deadband keeps small
# unigram differences neutral and the cap bounds the feature magnitude. The
# deadband is calibrated to the corpus-scale positional table (given_position.csv,
# 963k pairs): differences below 1.5 log-units are dominated by benign asymmetries
# (measured floor: Zhang Yi Gong stays correct up to a wrong-way delta of 2.01
# with deadband >= ~1.3; 1.5 leaves margin while Huang Yu Chang's 3.93 delta
# still saturates the cap).
GIVEN_POSITION_DEADBAND = 1.5
GIVEN_POSITION_MAGNITUDE_CAP = 1.5
RARE_TRAILING_OVERLAPPING_SURNAME_MAX = 100.0
# Romanization-conditional surname discount: log(target_share) for a spelling
# whose surname mass is reachable only via romanization remapping and that is
# missing from surname_romanizations.csv (open-ended Wade-Giles prefix/suffix
# rewrites); listed spellings carry their own share in the table.
REMAP_ONLY_SURNAME_DEFAULT_LOG_SHARE = -4.0
CURATED_COMPOUND_TARGETS = frozenset(COMPOUND_VARIANTS.values()) | frozenset(
    variant for variant in COMPOUND_VARIANTS if " " in variant
)

DEFAULT_WEIGHTS = (
    0.465,  # surname_logp
    0.395,  # given_logp_sum
    -0.888,  # surname_rank_bonus
    1.348,  # compound_given_bonus
    1.102,  # order_preservation_bonus
    0.425,  # surname_first_bonus
    -0.042,  # surname_freq_log_ratio (log-based comparative feature)
    -0.573,  # surname_rank_difference (comparative feature)
    0.06,  # given_position_logp (comparative positional given-syllable feature)
)


class NameParsingService:
    """Service for parsing Chinese names into surname and given name components."""

    def __init__(self, context_or_config, normalizer=None, data=None, *, weights: list[float] | None = None):
        # Support both old interface (config, normalizer, data) and new context interface
        if hasattr(context_or_config, "config"):
            # New context interface
            self._config = context_or_config.config
            self._normalizer = context_or_config.normalizer
            self._data = context_or_config.data
        else:
            # Legacy interface - maintain backwards compatibility
            self._config = context_or_config
            self._normalizer = normalizer
            self._data = data

        # Weight parameters - can be overridden. Legacy shorter vectors (e.g. from
        # pickled configs or process-pool workers) get the default coefficients for
        # the newer trailing features appended, keeping index order stable.
        if weights and len(weights) in (8, 9):
            self._weights = list(weights) + list(DEFAULT_WEIGHTS[len(weights) :])
        else:
            self._weights = list(DEFAULT_WEIGHTS)

    def parse_name_order(
        self,
        order: list[str],
        normalized_cache: dict[str, str],
        compound_metadata: dict[str, CompoundMetadata],
    ) -> ParseResult:
        """Parse and return ParseResult for compatibility with external callers."""
        parsed = self.parse_name_order_tokens(order, normalized_cache, compound_metadata)
        if parsed is None:
            return ParseResult.failure("surname not recognised")

        surname_tokens, given_tokens, original_compound_surname = parsed
        return ParseResult.success_with_parse(surname_tokens, given_tokens, original_compound_surname)

    def parse_name_order_tokens(
        self,
        order: list[str],
        normalized_cache: dict[str, str],
        compound_metadata: dict[str, CompoundMetadata],
    ) -> tuple[list[str], list[str], str | None] | None:
        """Fast internal parse path that avoids ParseResult object construction."""
        # Try probabilistic parsing first
        best_parse = self._best_parse_tokens(order, normalized_cache, compound_metadata)
        if best_parse is not None:
            return best_parse

        # Fallback parsing - try different surname positions
        fallback_attempts = [
            (-1, slice(None, -1)),  # surname-last pattern
            (0, slice(1, None)),  # surname-first pattern
        ]

        for surname_pos, given_slice in fallback_attempts:
            result = self._try_fallback_parse_tokens(order, surname_pos, given_slice, normalized_cache)
            if result is not None:
                return result

        return None

    def generate_parse_options(
        self,
        tokens: list[str],
        normalized_cache: dict[str, str],
        compound_metadata: dict[str, CompoundMetadata],
    ) -> list[tuple[list[str], list[str], str | None]]:
        """Return possible surname/given parses for batch-level scoring."""
        return self._generate_all_parses_with_format(tokens, normalized_cache, compound_metadata)

    def _try_fallback_parse(
        self,
        order: list[str],
        surname_pos: int,
        given_slice: slice,
        normalized_cache: dict[str, str],
    ) -> ParseResult:
        """Try a single fallback parse configuration - pure function"""
        surname_token = order[surname_pos]
        normalized_surname = self._normalizer.get_normalized(surname_token, normalized_cache)

        if (
            len(surname_token) > 1  # Don't treat single letters as surnames
            and StringManipulationUtils.remove_spaces(normalized_surname) in self._data.surnames_normalized
        ):
            surname_tokens = [surname_token]
            given_tokens = order[given_slice]
            if given_tokens:
                # Check if this parse would have a reasonable score
                score = self.calculate_parse_score(
                    surname_tokens,
                    given_tokens,
                    order,
                    normalized_cache,
                    is_all_chinese=False,
                )

                # Western name detection pattern
                has_single_letter_given = any(len(token) == 1 for token in given_tokens)
                has_multi_syllable_tokens = any(len(token) > 3 for token in order)

                # Check if any multi-syllable token is a known Chinese surname
                has_chinese_surname_in_tokens = any(
                    len(token) > 3
                    and self._data.is_surname(
                        token,
                        StringManipulationUtils.remove_spaces(
                            self._normalizer.get_normalized(token, normalized_cache),
                        ),
                    )
                    for token in order
                )

                if (
                    has_single_letter_given
                    and has_multi_syllable_tokens
                    and score < self._config.poor_score_threshold
                    and not has_chinese_surname_in_tokens
                ):
                    # This looks like a Western name where single letters are initials
                    return ParseResult.failure("Western name pattern detected")

                return ParseResult.success_with_parse(surname_tokens, given_tokens, None)

        return ParseResult.failure("No valid surname found")

    def _try_fallback_parse_tokens(
        self,
        order: list[str],
        surname_pos: int,
        given_slice: slice,
        normalized_cache: dict[str, str],
    ) -> tuple[list[str], list[str], str | None] | None:
        """Token-only fallback parse for hot internal call paths."""
        surname_token = order[surname_pos]
        normalized_surname = self._normalizer.get_normalized(surname_token, normalized_cache)

        if len(surname_token) > 1 and StringManipulationUtils.remove_spaces(normalized_surname) in self._data.surnames_normalized:
            surname_tokens = [surname_token]
            given_tokens = order[given_slice]
            if given_tokens:
                score = self.calculate_parse_score(
                    surname_tokens,
                    given_tokens,
                    order,
                    normalized_cache,
                    is_all_chinese=False,
                )

                has_single_letter_given = any(len(token) == 1 for token in given_tokens)
                has_multi_syllable_tokens = any(len(token) > 3 for token in order)
                has_chinese_surname_in_tokens = any(
                    len(token) > 3
                    and self._data.is_surname(
                        token,
                        StringManipulationUtils.remove_spaces(
                            self._normalizer.get_normalized(token, normalized_cache),
                        ),
                    )
                    for token in order
                )

                if (
                    has_single_letter_given
                    and has_multi_syllable_tokens
                    and score < self._config.poor_score_threshold
                    and not has_chinese_surname_in_tokens
                ):
                    return None

                return (surname_tokens, given_tokens, None)

        return None

    def _best_parse(
        self,
        tokens: list[str],
        normalized_cache: dict[str, str],
        compound_metadata: dict[str, CompoundMetadata],
    ) -> ParseResult:
        """Find the best parse using probabilistic scoring."""
        if len(tokens) < self._config.min_tokens_required:
            return ParseResult.failure(f"needs at least {self._config.min_tokens_required} tokens")

        parses_with_format = self._generate_all_parses_with_format(
            tokens,
            normalized_cache,
            compound_metadata,
        )
        parses = [(surname, given) for surname, given, _ in parses_with_format]
        if not parses:
            return ParseResult.failure("surname not recognised")

        # Score all parses with early validation filtering
        scored_parses = []
        # Pre-compute expensive checks once for all parses
        has_multi_syllable_tokens = any(len(token) > 3 for token in tokens)
        has_chinese_surname_in_tokens = None  # Lazy evaluation
        score_cache = {"surname_key": {}, "given_key": {}, "ambiguous": {}}

        for surname_tokens, given_tokens, original_compound_format in parses_with_format:
            # Early validation: reject parses where single letters are used as given names
            # when there are multi-syllable alternatives available (likely Western names)
            has_single_letter_given = any(len(token) == 1 for token in given_tokens)

            if has_single_letter_given and has_multi_syllable_tokens:
                # Lazy evaluation of expensive Chinese surname check
                if has_chinese_surname_in_tokens is None:
                    has_chinese_surname_in_tokens = any(
                        len(token) > 3
                        and self._data.is_surname(
                            token,
                            StringManipulationUtils.remove_spaces(
                                self._normalizer.get_normalized(token, normalized_cache),
                            ),
                        )
                        for token in tokens
                    )

                # Quick score check before expensive full scoring
                if not has_chinese_surname_in_tokens:
                    # Do a quick surname validity check before full scoring
                    surname_key = self._surname_key(surname_tokens, normalized_cache)
                    quick_score_estimate = self._data.get_surname_logp(surname_key, self._config.default_surname_logp)
                    if quick_score_estimate < self._config.poor_score_threshold:
                        # This looks like a Western name where single letters are initials
                        continue

            # Full scoring for remaining candidates
            score = self.calculate_parse_score(
                surname_tokens,
                given_tokens,
                tokens,
                normalized_cache,
                is_all_chinese=False,
                original_compound_format=original_compound_format,
                score_cache=score_cache,
            )

            if score > float("-inf"):
                scored_parses.append((surname_tokens, given_tokens, score, original_compound_format))

        if not scored_parses:
            return ParseResult.failure("no valid parse found")

        # Find best scoring parse with deterministic tie-breaking.
        # Tie-break metadata is computed lazily only when scores tie.
        best_parse_result = None
        best_score = float("-inf")
        best_format_alignment = 0.0
        best_secondary_key = ""
        tie_break_ready = False
        for candidate in scored_parses:
            surname_tokens, given_tokens, score, _original_compound_format = candidate
            if best_parse_result is None or score > best_score:
                best_parse_result = candidate
                best_score = score
                tie_break_ready = False
                continue
            if score < best_score:
                continue

            if not tie_break_ready:
                best_surname_tokens, best_given_tokens, _best_score, _best_original_compound = best_parse_result
                best_format_alignment = self._calculate_format_alignment_bonus(
                    best_surname_tokens,
                    best_given_tokens,
                    tokens,
                )
                best_secondary_key = f"{best_surname_tokens}|{best_given_tokens}"
                tie_break_ready = True

            format_alignment = self._calculate_format_alignment_bonus(surname_tokens, given_tokens, tokens)
            if format_alignment > best_format_alignment:
                best_parse_result = candidate
                best_format_alignment = format_alignment
                best_secondary_key = f"{surname_tokens}|{given_tokens}"
                continue
            if format_alignment < best_format_alignment:
                continue

            secondary_key = f"{surname_tokens}|{given_tokens}"
            if secondary_key > best_secondary_key:
                best_parse_result = candidate
                best_secondary_key = secondary_key

        return ParseResult.success_with_parse(best_parse_result[0], best_parse_result[1], best_parse_result[3])

    def _best_parse_tokens(
        self,
        tokens: list[str],
        normalized_cache: dict[str, str],
        compound_metadata: dict[str, CompoundMetadata],
    ) -> tuple[list[str], list[str], str | None] | None:
        """Token-only best-parse path for internal hot loops."""
        if len(tokens) < self._config.min_tokens_required:
            return None

        parses_with_format = self._generate_all_parses_with_format(
            tokens,
            normalized_cache,
            compound_metadata,
        )
        parses = [(surname, given) for surname, given, _ in parses_with_format]
        if not parses:
            return None

        scored_parses = []
        has_multi_syllable_tokens = any(len(token) > 3 for token in tokens)
        has_chinese_surname_in_tokens = None
        score_cache = {"surname_key": {}, "given_key": {}, "ambiguous": {}}

        for surname_tokens, given_tokens, original_compound_format in parses_with_format:
            has_single_letter_given = any(len(token) == 1 for token in given_tokens)

            if has_single_letter_given and has_multi_syllable_tokens:
                if has_chinese_surname_in_tokens is None:
                    has_chinese_surname_in_tokens = any(
                        len(token) > 3
                        and self._data.is_surname(
                            token,
                            StringManipulationUtils.remove_spaces(
                                self._normalizer.get_normalized(token, normalized_cache),
                            ),
                        )
                        for token in tokens
                    )

                if not has_chinese_surname_in_tokens:
                    surname_key = self._surname_key(surname_tokens, normalized_cache)
                    quick_score_estimate = self._data.get_surname_logp(surname_key, self._config.default_surname_logp)
                    if quick_score_estimate < self._config.poor_score_threshold:
                        continue

            score = self.calculate_parse_score(
                surname_tokens,
                given_tokens,
                tokens,
                normalized_cache,
                is_all_chinese=False,
                original_compound_format=original_compound_format,
                score_cache=score_cache,
            )

            if score > float("-inf"):
                scored_parses.append((surname_tokens, given_tokens, score, original_compound_format))

        if not scored_parses:
            return None

        best_parse_result = None
        best_score = float("-inf")
        best_format_alignment = 0.0
        best_secondary_key = ""
        tie_break_ready = False
        for candidate in scored_parses:
            surname_tokens, given_tokens, score, _original_compound_format = candidate
            if best_parse_result is None or score > best_score:
                best_parse_result = candidate
                best_score = score
                tie_break_ready = False
                continue
            if score < best_score:
                continue

            if not tie_break_ready:
                best_surname_tokens, best_given_tokens, _best_score, _best_original_compound = best_parse_result
                best_format_alignment = self._calculate_format_alignment_bonus(
                    best_surname_tokens,
                    best_given_tokens,
                    tokens,
                )
                best_secondary_key = f"{best_surname_tokens}|{best_given_tokens}"
                tie_break_ready = True

            format_alignment = self._calculate_format_alignment_bonus(surname_tokens, given_tokens, tokens)
            if format_alignment > best_format_alignment:
                best_parse_result = candidate
                best_format_alignment = format_alignment
                best_secondary_key = f"{surname_tokens}|{given_tokens}"
                continue
            if format_alignment < best_format_alignment:
                continue

            secondary_key = f"{surname_tokens}|{given_tokens}"
            if secondary_key > best_secondary_key:
                best_parse_result = candidate
                best_secondary_key = secondary_key

        return best_parse_result[0], best_parse_result[1], best_parse_result[3]

    def _generate_all_parses_with_format(
        self,
        tokens: list[str],
        normalized_cache: dict[str, str],
        compound_metadata: dict[str, CompoundMetadata],
    ) -> list[tuple[list[str], list[str], str | None]]:
        """Generate all possible (surname, given_name) parses for the tokens."""
        if len(tokens) < 2:
            return []

        parses = []
        first_is_hyphenated_compound = self._is_hyphenated_compound_metadata(compound_metadata.get(tokens[0]))
        last_is_hyphenated_compound = self._is_hyphenated_compound_metadata(compound_metadata.get(tokens[-1]))

        # 1. Check compound surnames using centralized metadata
        if len(tokens) >= 3:
            # Check first two tokens for compound
            first_meta = compound_metadata.get(tokens[0])
            second_meta = compound_metadata.get(tokens[1])
            if (
                first_meta
                and first_meta.is_compound
                and second_meta
                and second_meta.is_compound
                and first_meta.compound_target == second_meta.compound_target
            ):
                # This is a multi-token compound at the beginning
                original_format = self._get_compound_original_format(first_meta, tokens[0:2])
                parses.append((tokens[0:2], tokens[2:], original_format))

            # Check second and third tokens for compound (surname in middle)
            if len(tokens) >= 3:
                second_meta = compound_metadata.get(tokens[1])
                third_meta = compound_metadata.get(tokens[2])
                if (
                    second_meta
                    and second_meta.is_compound
                    and third_meta
                    and third_meta.is_compound
                    and second_meta.compound_target == third_meta.compound_target
                ):
                    # This is a multi-token compound in the middle
                    original_format = self._get_compound_original_format(second_meta, tokens[1:3])
                    parses.append((tokens[1:3], tokens[0:1], original_format))

        # 2. Single-token surnames - only at beginning or end (contiguous sequences only)
        # Surname-first pattern: surname + given_names
        # Check both original and normalized forms, but exclude single letters
        first_token = tokens[0]
        first_normalized = self._normalizer.get_normalized(first_token, normalized_cache)

        # Check first token using centralized metadata
        first_meta = compound_metadata.get(first_token)
        if first_meta and first_meta.is_compound and first_meta.format_type in ["compact", "camelCase"]:
            # This is a compact/camelCase compound surname (single token representing compound)
            # Preserve original token structure instead of converting to compound_target
            compound_parts = StringManipulationUtils.split_compound_token(first_token, first_meta)
            original_format = self._get_compound_original_format(first_meta, [first_token])
            parses.append((compound_parts, tokens[1:], original_format))
        elif len(first_token) > 1 and "-" not in first_token and not last_is_hyphenated_compound:
            # Check for regular single surnames
            is_regular_surname = self._data.is_surname(
                first_token,
                StringManipulationUtils.remove_spaces(first_normalized),
            )
            if is_regular_surname:
                parses.append(([first_token], tokens[1:], None))

        # Surname-last pattern: given_names + surname
        if len(tokens) >= 2:
            last_token = tokens[-1]
            last_normalized = self._normalizer.get_normalized(last_token, normalized_cache)

            # Check last token using centralized metadata
            last_meta = compound_metadata.get(last_token)
            if last_meta and last_meta.is_compound and last_meta.format_type in ["compact", "camelCase"]:
                # This is a compact/camelCase compound surname (single token representing compound)
                # Preserve original token structure instead of converting to compound_target
                compound_parts = StringManipulationUtils.split_compound_token(last_token, last_meta)
                original_format = self._get_compound_original_format(last_meta, [last_token])
                parses.append((compound_parts, tokens[:-1], original_format))
            elif len(last_token) > 1 and "-" not in last_token and not first_is_hyphenated_compound:
                # Check for regular single surnames
                is_regular_surname = self._data.is_surname(
                    last_token,
                    StringManipulationUtils.remove_spaces(last_normalized),
                )
                if is_regular_surname:
                    parses.append(([last_token], tokens[:-1], None))

        # 3. Check for hyphenated compounds using centralized metadata
        # Beginning position
        first_meta = compound_metadata.get(tokens[0])
        if (
            first_meta
            and first_meta.is_compound
            and first_meta.format_type == "hyphenated"
            and first_meta.compound_target
        ):
            target_compound = first_meta.compound_target
            compound_parts = [StringManipulationUtils.capitalize_name_part(part) for part in target_compound.split()]
            if len(compound_parts) == 2 and len(tokens) > 1:
                original_format = tokens[0]  # Keep the hyphenated format
                parses.append((compound_parts, tokens[1:], original_format))

        # End position
        if len(tokens) >= 2:
            last_meta = compound_metadata.get(tokens[-1])
            if (
                last_meta
                and last_meta.is_compound
                and last_meta.format_type == "hyphenated"
                and last_meta.compound_target
            ):
                target_compound = last_meta.compound_target
                compound_parts = [StringManipulationUtils.capitalize_name_part(part) for part in target_compound.split()]
                if len(compound_parts) == 2:
                    original_format = tokens[-1]  # Keep the hyphenated format
                    parses.append((compound_parts, tokens[:-1], original_format))

        return parses

    @staticmethod
    def _is_hyphenated_compound_metadata(compound_meta: CompoundMetadata | None) -> bool:
        """Return whether metadata identifies a token as a hyphenated compound surname."""
        return bool(
            compound_meta
            and compound_meta.is_compound
            and compound_meta.format_type == "hyphenated"
            and compound_meta.compound_target
        )

    def _get_compound_original_format(self, compound_meta: CompoundMetadata, tokens: list[str]) -> str | None:
        """Get the original format for a compound surname from centralized metadata."""
        if not compound_meta.is_compound:
            return None

        # For single-token compounds (compact/camelCase), return the original token
        if len(tokens) == 1:
            return tokens[0].lower()

        # For multi-token compounds (spaced), return the spaced format
        if len(tokens) == 2:
            return StringManipulationUtils.lowercase_join_with_spaces(tokens)

        return None

    def calculate_parse_score(
        self,
        surname_tokens: list[str],
        given_tokens: list[str],
        tokens: list[str],
        normalized_cache: dict[str, str],
        *,
        is_all_chinese: bool = False,
        original_compound_format: str | None = None,
        score_cache: dict[str, dict] | None = None,
        allow_guarded_given_first_bonus: bool = True,
        surname_first_parenthetical_hint: bool = False,
    ) -> float:
        """Calculate unified score for a parse candidate."""
        if not given_tokens:
            return float("-inf")

        surname_key_cache = score_cache.get("surname_key") if score_cache else None
        given_key_cache = score_cache.get("given_key") if score_cache else None
        ambiguous_cache = score_cache.get("ambiguous") if score_cache else None

        surname_key = self._surname_key_cached(surname_tokens, normalized_cache, surname_key_cache)

        surname_logp = self._data.get_surname_logp(surname_key, self._config.default_surname_logp)

        # Handle compound surname mapping mismatches
        if surname_logp == self._config.default_surname_logp and len(surname_tokens) > 1:
            # First try using original compound format if available (e.g., "szeto" -> "si tu")
            if original_compound_format and original_compound_format.lower() in COMPOUND_VARIANTS:
                compound_target = COMPOUND_VARIANTS[original_compound_format.lower()]
                surname_logp = self._data.get_surname_logp(compound_target, self._config.default_surname_logp)
            else:
                # Fallback to old behavior
                original_compound = StringManipulationUtils.lowercase_join_with_spaces(surname_tokens)
                surname_logp = self._data.get_surname_logp(original_compound, self._config.default_surname_logp)

        # Romanization-conditional surname discount: a surname key reached only
        # through Wade-Giles/Cantonese/Taiwanese remapping, with no as-written
        # attestation as a surname romanization, must not inherit the full
        # frequency mass of its Mandarin target (e.g. 'fai'->hui 13,395 ppm,
        # 'kung'->gong). Attested spellings (surname tables, full-share rows in
        # surname_romanizations.csv) keep their mass untouched.
        if (
            len(surname_tokens) == 1
            and surname_logp > self._config.default_surname_logp
            and not self._config.cjk_pattern.search(surname_tokens[0])
        ):
            share_logp = self._surname_spelling_share_logp(surname_tokens[0])
            if share_logp < 0.0:
                surname_logp = max(
                    surname_logp + share_logp,
                    self._config.default_surname_logp,
                )

        given_logp_sum = 0.0
        has_decomposed_given_pair = False
        for g_token in given_tokens:
            given_key = self._given_key_cached(g_token, normalized_cache, given_key_cache)
            if given_key in self._data.given_log_probabilities:
                given_logp_sum += self._data.given_log_probabilities[given_key]
                continue

            decomposed = self._decompose_unknown_given(g_token, normalized_cache, score_cache)
            if decomposed is None:
                given_logp_sum += self._config.default_given_logp
                continue

            decomposed_logp, both_known = decomposed
            given_logp_sum += decomposed_logp
            if both_known:
                has_decomposed_given_pair = True

        compound_given_bonus = 0.0
        if len(given_tokens) == 2 and all(
            self._data.is_given_name(self._normalizer.get_normalized(t, normalized_cache)) for t in given_tokens
        ):
            compound_given_bonus = 1.0
        elif len(given_tokens) == 1 and has_decomposed_given_pair:
            compound_given_bonus = 1.0

        # Bonus for surname-first pattern in all-Chinese inputs
        surname_first_bonus = 0.0
        if is_all_chinese and len(tokens) == 2:
            # Check if this parse follows surname-first pattern (surname is the first token)
            if (
                len(surname_tokens) == 1
                and surname_tokens[0] == tokens[0]
                and len(given_tokens) == 1
                and given_tokens[0] == tokens[1]
            ):
                # This follows the traditional Chinese surname-first order
                surname_first_bonus = 1.0  # Strong bonus for correct Chinese order

        # Percentile rank-based scoring for surnames only
        surname_rank_bonus = 0.0
        if surname_tokens:
            surname_rank_bonus = self._data.get_surname_rank(surname_key)

        # Comparative features for 1+1 parses: my surname token vs the competing token
        surname_freq_log_ratio = 0.0
        surname_rank_difference = 0.0

        if len(surname_tokens) == 1 and len(given_tokens) == 1:
            given_as_surname_key = self._surname_key_cached(given_tokens, normalized_cache, surname_key_cache)

            # SURNAME comparative features. When one side is unattested, the small
            # default plus the negative weights manufacture a large bonus for the
            # unattested side; grant that benefit of the doubt only against a
            # modest surname, never against a dominant one.
            my_surname_freq = self._data.get_surname_freq(surname_key)
            alt_surname_freq = self._data.get_surname_freq(given_as_surname_key)
            both_attested = my_surname_freq > 0 and alt_surname_freq > 0
            if both_attested or max(my_surname_freq, alt_surname_freq) < UNATTESTED_AMBIGUITY_SURNAME_FREQ_MAX:
                surname_freq_log_ratio = math.log(max(my_surname_freq, 0.001) / max(alt_surname_freq, 0.001))
                # Reuse precomputed rank from the same scoring pass.
                surname_rank_difference = surname_rank_bonus - self._data.get_surname_rank(given_as_surname_key)

        # Order preservation bonus for ambiguous romanized cases
        order_preservation_bonus = 0.0
        if not is_all_chinese and len(tokens) == 2 and len(surname_tokens) == 1 and len(given_tokens) == 1:
            if surname_first_parenthetical_hint and surname_tokens[0] == tokens[0] and given_tokens[0] == tokens[1]:
                ambiguous_key = (surname_tokens[0], given_tokens[0])
                if ambiguous_cache is not None:
                    is_ambiguous = ambiguous_cache.get(ambiguous_key)
                    if is_ambiguous is None:
                        is_ambiguous = self._is_ambiguous_case(surname_tokens[0], given_tokens[0], normalized_cache)
                        ambiguous_cache[ambiguous_key] = is_ambiguous
                else:
                    is_ambiguous = self._is_ambiguous_case(surname_tokens[0], given_tokens[0], normalized_cache)

                if is_ambiguous:
                    order_preservation_bonus = 1.0
            # Check if this parse maintains the original given-surname order (given first, surname last)
            elif (
                not surname_first_parenthetical_hint
                and given_tokens[0] == tokens[0]
                and surname_tokens[0] == tokens[1]
            ):
                # This maintains the original order - check if case is ambiguous
                ambiguous_key = (surname_tokens[0], given_tokens[0])
                if ambiguous_cache is not None:
                    is_ambiguous = ambiguous_cache.get(ambiguous_key)
                    if is_ambiguous is None:
                        is_ambiguous = self._is_ambiguous_case(surname_tokens[0], given_tokens[0], normalized_cache)
                        ambiguous_cache[ambiguous_key] = is_ambiguous
                else:
                    is_ambiguous = self._is_ambiguous_case(surname_tokens[0], given_tokens[0], normalized_cache)

                if is_ambiguous or self._is_wade_giles_initial_remapped_surname_token(surname_tokens[0]):
                    order_preservation_bonus = 1.0
                elif allow_guarded_given_first_bonus and self._has_guarded_given_first_surname_ratio(
                    surname_tokens[0],
                    given_tokens[0],
                    normalized_cache,
                ):
                    order_preservation_bonus = GIVEN_FIRST_ORDER_PRESERVATION_BONUS
        if not is_all_chinese and len(tokens) == 3 and len(surname_tokens) == 1 and len(given_tokens) == 2:
            # Narrow extension for hard 3-token given-first cases:
            # only apply when surname-last is materially more plausible than surname-first.
            if given_tokens == tokens[0:2] and surname_tokens[0] == tokens[2]:
                first_norm = self._normalizer.get_normalized(tokens[0], normalized_cache)
                second_norm = self._normalizer.get_normalized(tokens[1], normalized_cache)
                last_norm = self._normalizer.get_normalized(tokens[2], normalized_cache)
                first_is_surname = self._data.is_surname(tokens[0], first_norm)
                last_is_surname = self._data.is_surname(tokens[2], last_norm)
                first_is_given = self._data.is_given_name(first_norm)
                second_is_given = self._data.is_given_name(second_norm)
                last_is_given = self._data.is_given_name(last_norm)
                first_surname_freq = self._data.get_surname_freq(first_norm)
                last_surname_freq = self._data.get_surname_freq(last_norm)
                freq_ratio = (last_surname_freq / first_surname_freq) if first_surname_freq > 0 else 0.0
                second_clean = StringManipulationUtils.remove_spaces(tokens[1]).lower()
                second_has_korean_given_signal = second_clean in KOREAN_GIVEN_PATTERNS

                if (
                    first_is_surname
                    and last_is_surname
                    and first_is_given
                    and last_is_given
                    and first_surname_freq > 0
                    and first_surname_freq < 3000
                    and last_surname_freq > first_surname_freq
                    and freq_ratio <= 3.0
                ):
                    order_preservation_bonus = max(order_preservation_bonus, 0.5)
                elif (
                    first_is_surname
                    and last_is_surname
                    and first_is_given
                    and second_is_given
                    and first_surname_freq > last_surname_freq
                    and 0 < last_surname_freq <= RARE_TRAILING_OVERLAPPING_SURNAME_MAX
                    and StringManipulationUtils.remove_spaces(last_norm) in OVERLAPPING_KOREAN_SURNAMES
                    and second_has_korean_given_signal
                ):
                    order_preservation_bonus = max(order_preservation_bonus, 1.0)

        # Positional given-syllable evidence for 3-token single-surname parses.
        # Given-name syllables are strongly position-skewed, which separates
        # "Surname G1 G2" from "G1 G2 Surname". Comparative and antisymmetric:
        # this parse's in-order given pair versus the opposite-end segmentation,
        # with a deadband so small (noisy) unigram differences stay neutral.
        # Keyed on surname position only, so permuted given-order variants of
        # the same segmentation score identically.
        given_position_logp = 0.0
        if not is_all_chinese and len(tokens) == 3 and len(surname_tokens) == 1 and len(given_tokens) == 2:
            surname_sign = 0.0
            if surname_tokens[0] == tokens[0]:
                surname_sign = 1.0
            elif surname_tokens[0] == tokens[2]:
                surname_sign = -1.0
            if surname_sign != 0.0:
                first_key = self._normalizer.norm_light(tokens[0])
                middle_key = self._normalizer.norm_light(tokens[1])
                last_key = self._normalizer.norm_light(tokens[2])
                surname_first_delta = (
                    self._data.get_given_initial_logp(middle_key)
                    + self._data.get_given_final_logp(last_key)
                    - self._data.get_given_initial_logp(first_key)
                    - self._data.get_given_final_logp(middle_key)
                )
                magnitude = min(
                    abs(surname_first_delta) - GIVEN_POSITION_DEADBAND,
                    GIVEN_POSITION_MAGNITUDE_CAP,
                )
                if magnitude > 0.0:
                    given_position_logp = surname_sign * math.copysign(magnitude, surname_first_delta)

        total_score = (
            surname_logp * self._weights[0]  # Surname frequency weight
            + given_logp_sum * self._weights[1]  # Given name frequency weight
            + surname_rank_bonus * self._weights[2]  # Surname rank bonus weight
            + compound_given_bonus * self._weights[3]  # Compound given bonus weight
            + order_preservation_bonus * self._weights[4]  # Order preservation weight
            + surname_first_bonus * self._weights[5]  # Surname first bonus weight
            + surname_freq_log_ratio * self._weights[6]  # Surname frequency log ratio weight
            + surname_rank_difference * self._weights[7]  # Surname rank difference weight
            + given_position_logp * self._weights[8]  # Positional given-syllable weight
        )

        return total_score

    def _surname_spelling_share_logp(self, token: str) -> float:
        """Return log(target_share) for scoring ``token`` as a single surname.

        0.0 when the as-written (norm_light) spelling is itself an attested
        surname romanization; otherwise the spelling's share from
        surname_romanizations.csv, falling back to the remap-only penalty
        share for spellings the table does not enumerate.
        """
        spelling = self._normalizer.norm_light(token)
        if spelling in self._data.surnames:
            return 0.0
        resolution = self._data.resolve_surname_spelling(spelling)
        if resolution is None:
            return REMAP_ONLY_SURNAME_DEFAULT_LOG_SHARE
        return math.log(resolution.target_share)

    def _decompose_unknown_given(
        self,
        given_token: str,
        normalized_cache: dict[str, str],
        score_cache: dict[str, dict] | None = None,
    ) -> tuple[float, bool] | None:
        """Score an unknown given token as a two-syllable fused given name."""
        decomp_cache = score_cache.setdefault("given_decomp", {}) if score_cache is not None else None
        if decomp_cache is not None and given_token in decomp_cache:
            return decomp_cache[given_token]

        if "-" in given_token:
            parts = StringManipulationUtils.split_and_clean_hyphens(given_token)
            candidates = [(parts[0], parts[1])] if len(parts) == 2 else []
        else:
            candidates = [
                (given_token[:index], given_token[index:])
                for index in range(1, len(given_token))
                if self._normalizer.norm_light(given_token[:index]) in self._data.plausible_components
                and self._normalizer.norm_light(given_token[index:]) in self._data.plausible_components
            ]

        best: tuple[float, bool] | None = None
        for first, second in candidates:
            first_key = self._given_name_key(first, normalized_cache)
            second_key = self._given_name_key(second, normalized_cache)
            logp = self._data.get_given_logp(first_key, self._config.default_given_logp) + self._data.get_given_logp(
                second_key,
                self._config.default_given_logp,
            )
            both_known = first_key in self._data.given_log_probabilities and second_key in self._data.given_log_probabilities
            if best is None or logp > best[0]:
                best = (logp, both_known)

        if best is not None and best[0] <= self._config.default_given_logp:
            best = None
        if decomp_cache is not None:
            decomp_cache[given_token] = best
        return best

    def _is_ambiguous_case(self, surname_token: str, given_token: str, normalized_cache: dict[str, str]) -> bool:
        """
        Determine if a case is ambiguous enough to warrant order preservation.

        A case is considered ambiguous when:
        1. Both tokens are valid as surnames AND given names
        2. Surname frequencies are similar (ratio < 3x)
        """
        # Use parser surname keys for surname checks/frequencies so curated
        # romanizations do not get flattened through broader normalization.
        surname_key = self._surname_key([surname_token], normalized_cache)
        given_surname_key = self._surname_key([given_token], normalized_cache)
        surname_norm = self._normalizer.get_normalized(surname_token, normalized_cache)
        given_norm = self._normalizer.get_normalized(given_token, normalized_cache)

        # Check if both can be surnames
        surname_is_surname = self._data.is_surname(surname_token, surname_key)
        given_is_surname = self._data.is_surname(given_token, given_surname_key)

        if not (surname_is_surname and given_is_surname):
            return False  # Not ambiguous if one clearly can't be a surname

        # Check if both can be given names (as-written form first: remapping can
        # point a valid pinyin given name at a syllable outside the table)
        surname_is_given = self._data.is_given_name(
            self._normalizer.norm_light(surname_token),
        ) or self._data.is_given_name(surname_norm)
        given_is_given = self._data.is_given_name(
            self._normalizer.norm_light(given_token),
        ) or self._data.is_given_name(given_norm)

        if not (surname_is_given and given_is_given):
            return False  # Not ambiguous if one clearly can't be a given name

        # Check frequency similarity - ambiguous if frequencies are similar
        surname_freq = self._data.get_surname_freq(surname_key)
        given_freq = self._data.get_surname_freq(given_surname_key)

        if surname_freq == 0 or given_freq == 0:
            # An unattested token can plausibly tie with a modest surname, but not
            # with a dominant one: strong evidence on one side is not ambiguity.
            return max(surname_freq, given_freq) < UNATTESTED_AMBIGUITY_SURNAME_FREQ_MAX

        freq_ratio = max(surname_freq, given_freq) / min(surname_freq, given_freq)
        return freq_ratio < 5.0  # Ambiguous if frequencies are within 5x of each other

    def _has_guarded_given_first_surname_ratio(
        self,
        surname_token: str,
        given_token: str,
        normalized_cache: dict[str, str],
    ) -> bool:
        """Return whether surname frequencies strongly support given-first order."""
        if self._is_compound_surname_token(given_token, normalized_cache):
            return False

        surname_key = self._surname_key([surname_token], normalized_cache)
        given_as_surname_key = self._surname_key([given_token], normalized_cache)

        if not (self._data.is_surname(surname_token, surname_key) and self._data.is_surname(given_token, given_as_surname_key)):
            return False

        given_surname_freq = self._data.get_surname_freq(given_as_surname_key)
        surname_freq = self._data.get_surname_freq(surname_key)
        if not (0 < given_surname_freq < LOW_FREQUENCY_SURNAME_MAX):
            return False

        return surname_freq / given_surname_freq > GIVEN_FIRST_SURNAME_FREQ_RATIO_MIN

    def _is_wade_giles_initial_remapped_surname_token(self, token: str) -> bool:
        """Return whether a direct surname was remapped by a Wade-Giles initial rule."""
        light_key = self._normalizer.norm_light(token)
        remapped_key = self._normalizer.norm(token)
        return (
            light_key != remapped_key
            and bool(light_key)
            and bool(remapped_key)
            and light_key[0] in {"k", "p", "t"}
            and remapped_key[0] in {"g", "b", "d"}
            and self._data.get_surname_freq(light_key) > 0
            and self._data.get_surname_freq(remapped_key) == 0
        )

    def _is_compound_surname_token(self, token: str, _normalized_cache: dict[str, str]) -> bool:
        """Return whether a single token is a curated compact or hyphenated compound surname."""
        token_key = token.strip().lower()
        if token_key in COMPOUND_VARIANTS:
            return True
        if "-" in token_key:
            target = self._data.compound_hyphen_map.get(token_key)
            return target in CURATED_COMPOUND_TARGETS
        return False

    def _surname_key_cached(
        self,
        surname_tokens: list[str],
        normalized_cache: dict[str, str],
        cache: dict[tuple[str, ...], str] | None,
    ) -> str:
        """Surname lookup key with optional per-call memoization."""
        if cache is None:
            return self._surname_key(surname_tokens, normalized_cache)
        surname_tuple = tuple(surname_tokens)
        key = cache.get(surname_tuple)
        if key is None:
            key = self._surname_key(surname_tokens, normalized_cache)
            cache[surname_tuple] = key
        return key

    def _given_key_cached(
        self,
        given_token: str,
        normalized_cache: dict[str, str],
        cache: dict[str, str] | None,
    ) -> str:
        """Given-name lookup key with optional per-call memoization."""
        if cache is None:
            return self._given_name_key(given_token, normalized_cache)
        key = cache.get(given_token)
        if key is None:
            key = self._given_name_key(given_token, normalized_cache)
            cache[given_token] = key
        return key

    def _surname_key(self, surname_tokens: list[str], normalized_cache: dict[str, str]) -> str:
        """Convert surname tokens to lookup key, preferring Chinese characters when available.

        Parser scoring uses a stricter romanization-conditional key than the
        shared ``NameDataStructures.surname_lookup_key`` resolver (used by
        batch/routing evidence): a spelling only wins its as-written key here
        when it is a full-share row or carries direct mass, so that remap-only
        penalty and incidental as-written spellings keep the remapped key their
        downstream ``_surname_spelling_share_logp`` discount and comparative
        order scoring depend on. See that method for why the two sides cannot
        share one literal key.
        """
        if len(surname_tokens) == 1:
            token = surname_tokens[0]

            # If token contains Chinese characters, try Chinese character lookup first
            if self._config.cjk_pattern.search(token):
                if token in self._data.surname_frequencies:
                    return token

            # Prefer the as-written romanization for curated spellings
            # (surname_romanizations.csv) that carry their own frequency
            # entry: initialization guarantees full-share rows the mandarin
            # target's full mass under this key (e.g. chien -> qian mass, not
            # the remapped jian mass) and Korean-dominant trimmed penalty rows
            # their discounted as-written mass (e.g. jung, not zheng's mass).
            light_key = self._normalizer.norm_light(token)
            resolution = self._data.resolve_surname_spelling(light_key)
            if resolution is not None and (resolution.target_share == 1.0 or light_key in self._data.surname_frequencies):
                return light_key
            if self._is_wade_giles_initial_remapped_surname_token(token):
                return light_key

            # Try normalized form next.
            original_key = self._normalizer.norm(token)
            if original_key in self._data.surname_frequencies:
                return original_key
            # Fall back to normalized form
            return StringManipulationUtils.remove_spaces(
                self._normalizer.get_normalized(token, normalized_cache),
            )
        # Compound surname - join with space
        normalized_tokens = [normalized_cache.get(t, self._normalizer.norm(t)) for t in surname_tokens]
        return StringManipulationUtils.join_with_spaces(normalized_tokens)

    def _given_name_key(self, given_token: str, normalized_cache: dict[str, str]) -> str:
        """Convert given name token to lookup key, preferring Chinese characters when available."""
        # If token contains Chinese characters, try Chinese character lookup first
        if self._config.cjk_pattern.search(given_token):
            if given_token in self._data.given_log_probabilities:
                return given_token

        light_key = self._normalizer.norm_light(given_token)
        remapped_key = self._normalizer.norm(given_token)
        light_logp = self._data.get_given_logp(light_key, float("-inf"))
        remapped_logp = self._data.get_given_logp(remapped_key, float("-inf"))
        if light_logp > float("-inf") or remapped_logp > float("-inf"):
            return light_key if light_logp >= remapped_logp else remapped_key
        # Fall back to normalized form
        return self._normalizer.norm(self._normalizer.get_normalized(given_token, normalized_cache))

    def _calculate_format_alignment_bonus(
        self,
        surname_tokens: list[str],
        given_tokens: list[str],
        original_tokens: list[str],
    ) -> float:
        """
        Calculate a small bonus for parses that align with the original input format.
        This is used purely for deterministic tie-breaking when scores are identical.
        """
        if len(original_tokens) != 2 or len(surname_tokens) != 1 or len(given_tokens) != 1:
            return 0.0  # Only apply to simple 2-token cases

        surname_token = surname_tokens[0]
        given_token = given_tokens[0]

        # Check which parse preserves the original token order
        if given_token == original_tokens[0] and surname_token == original_tokens[1]:
            # This parse maintains given-surname order (Western style)
            return 0.001  # Small bonus for format alignment
        if surname_token == original_tokens[0] and given_token == original_tokens[1]:
            # Keep a neutral score for the opposite parse so ties are actually discriminated.
            return 0.0
        # This parse changes the token order
        return 0.0
