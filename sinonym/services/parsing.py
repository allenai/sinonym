"""
Name parsing service for Chinese name processing.

This module provides sophisticated name parsing algorithms to identify
surname/given name boundaries using probabilistic scoring and cultural
validation patterns.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from sinonym.chinese_names_data import COMPOUND_VARIANTS
from sinonym.types import ParseResult
from sinonym.utils.string_manipulation import StringManipulationUtils

if TYPE_CHECKING:
    from sinonym.services.normalization import CompoundMetadata


class NameParsingService:
    """Service for parsing Chinese names into surname and given name components."""

    def __init__(self, config, normalizer, data):
        self._config = config
        self._normalizer = normalizer
        self._data = data

    def parse_name_order(
        self,
        order: list[str],
        normalized_cache: dict[str, str],
        compound_metadata: dict[str, CompoundMetadata],
    ) -> ParseResult:
        """Parse using probabilistic system with fallback - pattern matching style."""
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
        normalized_surname = normalized_cache.get(surname_token, self._normalizer.norm(surname_token))

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
                    and (
                        self._normalizer.norm(token) in self._data.surnames
                        or StringManipulationUtils.remove_spaces(normalized_cache.get(token, self._normalizer.norm(token)))
                        in self._data.surnames_normalized
                    )
                    for token in order
                )

                if (
                    has_single_letter_given
                    and has_multi_syllable_tokens
                    and score < -25.0
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
        if len(tokens) < 2:
            return ParseResult.failure("needs at least 2 tokens")

        parses_with_format = self._generate_all_parses_with_format(
            tokens, normalized_cache, compound_metadata,
        )
        parses = [(surname, given) for surname, given, _ in parses_with_format]
        if not parses:
            return ParseResult.failure("surname not recognised")

        # Score all parses
        scored_parses = []
        for surname_tokens, given_tokens, original_compound_format in parses_with_format:
            score = self.calculate_parse_score(surname_tokens, given_tokens, tokens, normalized_cache, False, original_compound_format)

            # Additional validation: reject parses where single letters are used as given names
            # when there are multi-syllable alternatives available (likely Western names)
            has_single_letter_given = any(len(token) == 1 for token in given_tokens)
            has_multi_syllable_tokens = any(len(token) > 3 for token in tokens)

            # Check if any multi-syllable token is a known Chinese surname
            has_chinese_surname_in_tokens = any(
                len(token) > 3
                and (
                    self._normalizer.norm(token) in self._data.surnames
                    or StringManipulationUtils.remove_spaces(normalized_cache.get(token, self._normalizer.norm(token)))
                    in self._data.surnames_normalized
                )
                for token in tokens
            )

            if (
                has_single_letter_given
                and has_multi_syllable_tokens
                and score < -25.0
                and not has_chinese_surname_in_tokens
            ):
                # This looks like a Western name where single letters are initials
                continue

            if score > float("-inf"):
                scored_parses.append((surname_tokens, given_tokens, score, original_compound_format))

        if not scored_parses:
            return ParseResult.failure("no valid parse found")

        # Find best score and handle ties
        best_score = max(scored_parses, key=lambda x: x[2])[2]
        best_parses = [p for p in scored_parses if abs(p[2] - best_score) <= 0.1]

        if len(best_parses) > 1:
            # Tie-break with surname frequency
            best_parse_result = max(
                best_parses,
                key=lambda x: self._data.surname_frequencies.get(
                    StringManipulationUtils.remove_spaces(self._surname_key(x[0], normalized_cache)),
                    0,
                ),
            )
        else:
            best_parse_result = best_parses[0]

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
        first_normalized = normalized_cache.get(first_token, self._normalizer.norm(first_token))

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
            is_regular_surname = (
                self._normalizer.norm(first_token) in self._data.surnames
                or StringManipulationUtils.remove_spaces(first_normalized) in self._data.surnames_normalized
            )
            if is_regular_surname:
                parses.append(([first_token], tokens[1:], None))

        # Surname-last pattern: given_names + surname
        if len(tokens) >= 2:
            last_token = tokens[-1]
            last_normalized = normalized_cache.get(last_token, self._normalizer.norm(last_token))

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
                is_regular_surname = (
                    self._normalizer.norm(last_token) in self._data.surnames
                    or StringManipulationUtils.remove_spaces(last_normalized) in self._data.surnames_normalized
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
                compound_parts = [StringManipulationUtils.capitalize_name_part(part) for part in target_compound.split()]
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
        surname_logp = self._data.surname_log_probabilities.get(surname_key, self._config.default_surname_logp)

        # Handle compound surname mapping mismatches
        if surname_logp == self._config.default_surname_logp and len(surname_tokens) > 1:
            # First try using original compound format if available (e.g., "szeto" -> "si tu")
            if original_compound_format and original_compound_format.lower() in COMPOUND_VARIANTS:
                compound_target = COMPOUND_VARIANTS[original_compound_format.lower()]
                surname_logp = self._data.surname_log_probabilities.get(
                    compound_target,
                    self._config.default_surname_logp,
                )
            else:
                # Fallback to old behavior
                original_compound = StringManipulationUtils.lowercase_join_with_spaces(surname_tokens)
                surname_logp = self._data.surname_log_probabilities.get(
                    original_compound,
                    self._config.default_surname_logp,
                )

        given_logp_sum = sum(
            self._data.given_log_probabilities.get(
                self._given_name_key(g_token, normalized_cache),
                self._config.default_given_logp,
            )
            for g_token in given_tokens
        )

        validation_penalty = 0.0 if self._normalizer.validate_given_tokens(given_tokens, normalized_cache) else -3.0

        compound_given_bonus = 0.0
        if len(given_tokens) == 2 and all(
            normalized_cache.get(t, self._normalizer.norm(t)) in self._data.given_names_normalized for t in given_tokens
        ):
            compound_given_bonus = 0.8

        cultural_score = self._cultural_plausibility_score(surname_tokens, given_tokens, normalized_cache)

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
                surname_first_bonus = 2.0  # Strong bonus for correct Chinese order

        return surname_logp + given_logp_sum + validation_penalty + compound_given_bonus + cultural_score + surname_first_bonus

    def _surname_key(self, surname_tokens: list[str], normalized_cache: dict[str, str]) -> str:
        """Convert surname tokens to lookup key, preferring original form when available."""
        if len(surname_tokens) == 1:
            # Try original form first (more likely to preserve correct romanization)
            original_key = self._normalizer.norm(surname_tokens[0])
            if original_key in self._data.surname_frequencies:
                return original_key
            # Fall back to normalized form
            return StringManipulationUtils.remove_spaces(
                normalized_cache.get(surname_tokens[0], self._normalizer.norm(surname_tokens[0])),
            )
        # Compound surname - join with space
        normalized_tokens = [normalized_cache.get(t, self._normalizer.norm(t)) for t in surname_tokens]
        return StringManipulationUtils.join_with_spaces(normalized_tokens)

    def _given_name_key(self, given_token: str, normalized_cache: dict[str, str]) -> str:
        """Convert given name token to lookup key, preferring original form when available."""
        # Try original form first (more likely to preserve correct romanization)
        original_key = self._normalizer.norm(given_token)
        if original_key in self._data.given_log_probabilities:
            return original_key
        # Fall back to normalized form
        return self._normalizer.norm(normalized_cache.get(given_token, self._normalizer.norm(given_token)))

    def _cultural_plausibility_score(
        self,
        surname_tokens: list[str],
        given_tokens: list[str],
        normalized_cache: dict[str, str],
    ) -> float:
        """Calculate cultural plausibility score for a Chinese name parse."""
        if not surname_tokens or not given_tokens:
            return -10.0

        score = 0.0
        surname_key = self._surname_key(surname_tokens, normalized_cache)

        # Surname frequency bonus
        surname_freq = self._data.surname_frequencies.get(StringManipulationUtils.remove_spaces(surname_key), 0)
        if surname_freq == 0 and " " in surname_key:
            surname_freq = self._data.surname_frequencies.get(surname_key, 0)

        # Cross-semantic frequency capping to prevent given names from inheriting
        # high surname frequencies through normalization (e.g., 'fai' → 'hui')
        if len(surname_tokens) == 1 and surname_freq > 10000:
            original_token = surname_tokens[0].lower()
            normalized_token = StringManipulationUtils.remove_spaces(surname_key).lower()

            # Check if this is cross-semantic inheritance: original not a surname, normalized is
            original_is_surname = (
                original_token in self._data.surnames or
                original_token in self._data.surname_frequencies
            )
            if not original_is_surname and original_token != normalized_token:
                # Cap high-frequency cross-semantic inheritance to level playing field
                surname_freq = min(surname_freq, 1000)

        if surname_freq > 0:
            score += min(5.0, math.log10(surname_freq + 1) * 1.2)
        else:
            score -= 3.0

        # Compound surname validation
        if len(surname_tokens) == 2:
            compound_original = StringManipulationUtils.lowercase_join_with_spaces(surname_tokens)
            is_valid_compound = (
                surname_key in self._data.compound_surnames_normalized
                or StringManipulationUtils.remove_spaces(surname_key) in self._data.compound_surnames_normalized
                or (
                    compound_original in COMPOUND_VARIANTS
                    and COMPOUND_VARIANTS[compound_original] in self._data.compound_surnames_normalized
                )
            )
            score += 5.0 if is_valid_compound else -2.0

        # Given name structure scoring
        if len(given_tokens) == 1:
            token = given_tokens[0]
            if len(token) > 6:
                score -= 1.0
            elif StringManipulationUtils.split_concatenated_name(
                token, normalized_cache, self._data, self._normalizer, self._config,
            ):
                score += 0.5
        elif len(given_tokens) == 2:
            score += 1.0
            if all(
                normalized_cache.get(t, self._normalizer.norm(t)) in self._data.given_names_normalized
                for t in given_tokens
            ):
                score += 1.5
        elif len(given_tokens) > 2:
            score -= 1.5

        # Avoid role confusion
        for token in surname_tokens:
            key = normalized_cache.get(token, self._normalizer.norm(token))
            if (
                key in self._data.given_names_normalized
                and StringManipulationUtils.remove_spaces(normalized_cache.get(token, self._normalizer.norm(token)))
                not in self._data.surnames_normalized
            ):
                score -= 2.0

        for token in given_tokens:
            key = StringManipulationUtils.remove_spaces(normalized_cache.get(token, self._normalizer.norm(token)))
            if key in self._data.surnames and self._data.surname_frequencies.get(key, 0) > 1000:
                score -= 1.5

        return score
