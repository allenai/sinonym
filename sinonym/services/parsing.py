"""
Name parsing service for Chinese name processing.

This module provides sophisticated name parsing algorithms to identify
surname/given name boundaries using probabilistic scoring and cultural
validation patterns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sinonym.chinese_names_data import COMPOUND_VARIANTS
from sinonym.coretypes import ParseResult
from sinonym.utils.cjk_cache import CJKCharacterCache
from sinonym.utils.string_manipulation import StringManipulationUtils

if TYPE_CHECKING:
    from sinonym.services.normalization import CompoundMetadata


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

        # Weight parameters - can be overridden
        if weights and len(weights) == 8:
            self._weights = weights
        else:
            # Default weights (optimized 8-feature system - 118 failures)
            # Removed given name comparative features as they don't improve performance
            self._weights = [
                0.465,  # surname_logp
                0.395,  # given_logp_sum
                -0.888,  # surname_rank_bonus
                1.348,  # compound_given_bonus
                1.102,  # order_preservation_bonus
                0.425,  # surname_first_bonus
                -0.042,  # surname_freq_log_ratio (log-based comparative feature)
                -0.873,  # surname_rank_difference (comparative feature)
            ]

        # Initialize centralized CJK character cache
        self._cjk_cache = CJKCharacterCache(self._config.cjk_pattern)

    def parse_name_order(
        self,
        order: list[str],
        normalized_cache: dict[str, str],
        compound_metadata: dict[str, CompoundMetadata],
    ) -> ParseResult:
        """Parse using rule-based system with fallback - pattern matching style."""
        # Try probabilistic parsing first
        parse_result = self._best_parse(order, normalized_cache, compound_metadata)

        # Pattern match on result type (Scala-like)
        if parse_result.success and isinstance(parse_result.result, tuple):
            surname_tokens, given_tokens = parse_result.result
            return ParseResult.success_with_parse(surname_tokens, given_tokens, parse_result.original_compound_surname)

        # Fallback parsing - try different surname positions
        fallback_attempts = [
            (-1, slice(None, -1)),  # surname-last pattern
            (0, slice(1, None)),  # surname-first pattern
        ]

        for surname_pos, given_slice in fallback_attempts:
            result = self._try_fallback_parse(order, surname_pos, given_slice, normalized_cache)
            if result.success:
                return result

        return ParseResult.failure("surname not recognised")

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
                score = self.calculate_parse_score(surname_tokens, given_tokens, order, normalized_cache, False)

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
                False,
                original_compound_format,
            )

            if score > float("-inf"):
                scored_parses.append((surname_tokens, given_tokens, score, original_compound_format))

        if not scored_parses:
            return ParseResult.failure("no valid parse found")

        # Find best scoring parse with deterministic tie-breaking
        # Sort by: (1) score descending, (2) format alignment descending, (3) deterministic secondary key
        def parse_sort_key(parse_data):
            surname_tokens, given_tokens, score, original_compound_format = parse_data

            # Calculate format alignment bonus for tie-breaking
            format_alignment = self._calculate_format_alignment_bonus(surname_tokens, given_tokens, tokens)

            # Deterministic secondary key (string representation for consistent ordering)
            secondary_key = f"{surname_tokens}|{given_tokens}"

            return (score, format_alignment, secondary_key)

        best_parse_result = max(scored_parses, key=parse_sort_key)

        return ParseResult.success_with_parse(best_parse_result[0], best_parse_result[1], best_parse_result[3])

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
        elif len(first_token) > 1:
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
            elif len(last_token) > 1:
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
        if first_meta and first_meta.is_compound and first_meta.format_type == "hyphenated":
            target_compound = first_meta.compound_target
            compound_parts = [StringManipulationUtils.capitalize_name_part(part) for part in target_compound.split()]
            if len(compound_parts) == 2 and len(tokens) > 1:
                original_format = tokens[0]  # Keep the hyphenated format
                parses.append((compound_parts, tokens[1:], original_format))

        # End position
        if len(tokens) >= 2:
            last_meta = compound_metadata.get(tokens[-1])
            if last_meta and last_meta.is_compound and last_meta.format_type == "hyphenated":
                target_compound = last_meta.compound_target
                compound_parts = [
                    StringManipulationUtils.capitalize_name_part(part) for part in target_compound.split()
                ]
                if len(compound_parts) == 2:
                    original_format = tokens[-1]  # Keep the hyphenated format
                    parses.append((compound_parts, tokens[:-1], original_format))

        return parses

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
        is_all_chinese: bool = False,
        original_compound_format: str | None = None,
    ) -> float:
        """Calculate unified score for a parse candidate."""
        if not given_tokens:
            return float("-inf")

        surname_key = self._surname_key(surname_tokens, normalized_cache)
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

        given_logp_sum = sum(
            self._data.get_given_logp(
                self._given_name_key(g_token, normalized_cache),
                self._config.default_given_logp,
            )
            for g_token in given_tokens
        )

        compound_given_bonus = 0.0
        if len(given_tokens) == 2 and all(
            self._data.is_given_name(self._normalizer.get_normalized(t, normalized_cache)) for t in given_tokens
        ):
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

        # Order preservation bonus for ambiguous romanized cases
        order_preservation_bonus = 0.0
        if not is_all_chinese and len(tokens) == 2 and len(surname_tokens) == 1 and len(given_tokens) == 1:
            # Check if this parse maintains the original given-surname order (given first, surname last)
            if given_tokens[0] == tokens[0] and surname_tokens[0] == tokens[1]:
                # This maintains the original order - check if case is ambiguous
                if self._is_ambiguous_case(surname_tokens[0], given_tokens[0], normalized_cache):
                    order_preservation_bonus = 1.0  # Strong bonus for preserving original order in ambiguous cases

        # Percentile rank-based scoring for surnames only
        surname_rank_bonus = 0.0
        if surname_tokens:
            surname_rank_bonus = self._data.get_surname_rank(surname_key)

        # NEW: Comparative features - scaled and balanced approach
        surname_freq_log_ratio = 0.0
        surname_rank_difference = 0.0

        if len(surname_tokens) == 1 and len(given_tokens) == 1:
            # Get the alternative key
            given_as_surname_key = self._surname_key(given_tokens, normalized_cache)

            # SURNAME comparative features
            # Calculate log frequency ratio to keep values in reasonable range
            import math

            my_surname_freq = self._data.get_surname_freq(surname_key, 0.001)  # Small default
            alt_surname_freq = self._data.get_surname_freq(given_as_surname_key, 0.001)
            surname_freq_log_ratio = math.log(my_surname_freq / alt_surname_freq) if alt_surname_freq > 0 else 0.0

            # Calculate rank difference: my_surname_rank - alternative_surname_rank
            my_surname_rank = self._data.get_surname_rank(surname_key)
            alt_surname_rank = self._data.get_surname_rank(given_as_surname_key)
            surname_rank_difference = my_surname_rank - alt_surname_rank

        total_score = (
            surname_logp * self._weights[0]  # Surname frequency weight
            + given_logp_sum * self._weights[1]  # Given name frequency weight
            + surname_rank_bonus * self._weights[2]  # Surname rank bonus weight
            + compound_given_bonus * self._weights[3]  # Compound given bonus weight
            + order_preservation_bonus * self._weights[4]  # Order preservation weight
            + surname_first_bonus * self._weights[5]  # Surname first bonus weight
            + surname_freq_log_ratio * self._weights[6]  # Surname frequency log ratio weight
            + surname_rank_difference * self._weights[7]  # Surname rank difference weight
        )

        return total_score

    def _is_ambiguous_case(self, surname_token: str, given_token: str, normalized_cache: dict[str, str]) -> bool:
        """
        Determine if a case is ambiguous enough to warrant order preservation.

        A case is considered ambiguous when:
        1. Both tokens are valid as surnames AND given names
        2. Surname frequencies are similar (ratio < 3x)
        """
        # Get normalized forms
        surname_norm = self._normalizer.get_normalized(surname_token, normalized_cache)
        given_norm = self._normalizer.get_normalized(given_token, normalized_cache)

        # Check if both can be surnames
        surname_is_surname = self._data.is_surname(surname_token, surname_norm)
        given_is_surname = self._data.is_surname(given_token, given_norm)

        if not (surname_is_surname and given_is_surname):
            return False  # Not ambiguous if one clearly can't be a surname

        # Check if both can be given names
        surname_is_given = self._data.is_given_name(surname_norm)
        given_is_given = self._data.is_given_name(given_norm)

        if not (surname_is_given and given_is_given):
            return False  # Not ambiguous if one clearly can't be a given name

        # Check frequency similarity - ambiguous if frequencies are similar
        surname_freq = self._data.get_surname_freq(surname_norm)
        given_freq = self._data.get_surname_freq(given_norm)

        if surname_freq == 0 or given_freq == 0:
            return True  # Ambiguous if we lack frequency data for either

        freq_ratio = max(surname_freq, given_freq) / min(surname_freq, given_freq)
        return freq_ratio < 5.0  # Ambiguous if frequencies are within 5x of each other

    def _surname_key(self, surname_tokens: list[str], normalized_cache: dict[str, str]) -> str:
        """Convert surname tokens to lookup key, preferring Chinese characters when available."""
        if len(surname_tokens) == 1:
            token = surname_tokens[0]

            # If token contains Chinese characters, try Chinese character lookup first
            if self._cjk_cache.has_cjk_characters(token):
                if token in self._data.surname_frequencies:
                    return token

            # Try original form first (more likely to preserve correct romanization)
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
        if self._cjk_cache.has_cjk_characters(given_token):
            if given_token in self._data.given_log_probabilities:
                return given_token

        # Try original form first (more likely to preserve correct romanization)
        original_key = self._normalizer.norm(given_token)
        if original_key in self._data.given_log_probabilities:
            return original_key
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
            # This parse maintains surname-given order (Chinese style)
            return 0.001  # Small bonus for format alignment
        # This parse changes the token order
        return 0.0
