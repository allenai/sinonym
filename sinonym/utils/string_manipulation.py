"""
Centralized string manipulation utilities for Chinese name processing.

RESPONSIBILITIES (After Service Responsibility Clarification):
- Pure string operations: joining, splitting, case conversion, capitalization
- Context-aware Chinese name splitting with database validation
- Format conversion utilities for compound surnames
- NO format detection for primary business logic (CompoundDetector is authority)
- Utility methods only - called by services, doesn't make business decisions

This module consolidates all string manipulation functions that were previously
scattered across normalization, parsing, and formatting services.
"""

from __future__ import annotations

import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sinonym.services.normalization import CompoundMetadata


class StringManipulationUtils:
    """Centralized utilities for string manipulation in Chinese name processing."""

    # ====================================================================
    # SPLITTING FUNCTIONS
    # ====================================================================

    @staticmethod
    def smart_split_concatenated(token: str) -> str:
        """
        Split concatenated names with various capitalization patterns.

        Handles:
        - Simple CamelCase: LinShu → Lin Shu
        - Mixed caps: XIAOChen → XIAO Chen, LuWANG → Lu WANG
        - Multiple transitions: FurukawaKoichi → Furukawa Koichi
        - Preserve existing hyphens: LU-Wang → LU Wang
        """
        # If already has spaces or hyphens, don't split further
        if " " in token or "-" in token:
            return token

        # Use a different approach: insert spaces at transition points
        result = []
        i = 0

        while i < len(token):
            char = token[i]
            result.append(char)

            # Look ahead to see if we need to insert a space
            if i < len(token) - 1:
                next_char = token[i + 1]

                # Case 1: lowercase followed by uppercase (camelCase)
                if char.islower() and next_char.isupper():
                    result.append(" ")

                # Case 2: multiple uppercase followed by uppercase+lowercase (XMLParser → XML Parser)
                elif char.isupper() and next_char.isupper() and i < len(token) - 2:
                    next_next_char = token[i + 2]
                    if next_next_char.islower():
                        result.append(" ")

            i += 1

        return "".join(result)

    @staticmethod
    def _get_normalized_parts(a: str, b: str, normalized_cache: dict[str, str] | None, normalizer) -> tuple[str, str]:
        """Get normalized versions of two string parts."""
        if normalized_cache:
            norm_a = normalizer.get_normalized(a, normalized_cache)
            norm_b = normalizer.get_normalized(b, normalized_cache)
        else:
            norm_a = normalizer.norm(a)
            norm_b = normalizer.norm(b)
        return norm_a, norm_b

    @staticmethod
    def _is_valid_component_pair(norm_a: str, norm_b: str, data_context, orig_a: str = None, orig_b: str = None) -> bool:
        """Check if both parts are valid plausible components.
        
        Tries both original and normalized forms to handle cases where
        normalization changes valid components (e.g., kun -> gun).
        """
        # First try normalized forms
        if norm_a in data_context.plausible_components and norm_b in data_context.plausible_components:
            return True

        # If we have original forms, try those too
        if orig_a and orig_b:
            orig_a_lower = orig_a.lower()
            orig_b_lower = orig_b.lower()
            if orig_a_lower in data_context.plausible_components and orig_b_lower in data_context.plausible_components:
                return True

        return False

    @staticmethod
    def _should_skip_splitting(token: str, normalized_cache: dict[str, str] | None, normalizer, data_context) -> bool:
        """Check early exit conditions that prevent splitting - optimized validation chain."""
        # Get normalized form once
        if normalized_cache and token in normalized_cache:
            tok_normalized = StringManipulationUtils.remove_spaces(normalized_cache[token])
        else:
            tok_normalized = StringManipulationUtils.remove_spaces(normalizer.norm(token))

        # Optimized validation chain: combine multiple checks with OR short-circuit
        from sinonym.chinese_names_data import HIGH_CONFIDENCE_ANCHORS
        original_lower = token.lower()

        return (
            # Don't split known surnames
            tok_normalized in data_context.surnames_normalized or
            # Don't split HIGH_CONFIDENCE_ANCHORS - they should remain intact
            tok_normalized in HIGH_CONFIDENCE_ANCHORS or
            # Don't split existing valid Chinese given name components
            data_context.is_given_name(tok_normalized) or
            data_context.is_given_name(original_lower)
        )

    @staticmethod
    def _check_cultural_plausibility_if_needed(norm_a: str, norm_b: str, data_context, config, raw_length: int) -> bool:
        """Check cultural plausibility only if the raw token is long enough."""
        if raw_length >= 3:
            return StringManipulationUtils.is_plausible_chinese_split(norm_a, norm_b, data_context, config)
        return True  # Skip cultural check for short tokens

    @staticmethod
    def split_concatenated_name(
        token: str,
        normalized_cache: dict[str, str] | None,
        data_context,
        normalizer,
        config,
    ) -> list[str] | None:
        """
        Split concatenated Chinese given names using a sophisticated tiered confidence system.

        This function uses multiple validation layers to distinguish authentic Chinese name
        combinations from Western names that shouldn't be split.

        **Tiered Confidence System:**
        - **Gold Standard**: Both parts are HIGH_CONFIDENCE_ANCHORS (highest confidence)
        - **Silver Standard**: One part is anchor + cultural plausibility check
        - **Bronze Standard**: Both parts plausible + cultural check + length ≥4

        **Processing Pipeline:**
        1. Early exit conditions (surnames, anchors, valid given names)
        2. Pattern-based splitting (repeated syllables, hyphens, camelCase)
        3. Brute-force splitting with tiered validation

        Args:
            token: The concatenated token to potentially split (e.g., "Minghua")
            normalized_cache: Optional cache of token→normalized mappings for performance
            data_context: Database context containing surnames, given names, etc.
            normalizer: Text normalizer for converting tokens to normalized form
            config: Configuration containing patterns and validation rules

        Returns:
            List of split parts if valid split found (e.g., ["Ming", "Hua"]),
            None if token should remain unsplit

        Examples:
            - "Minghua" → ["Ming", "Hua"] (valid Chinese compound)
            - "Alan" → None (Western name, should not split)
            - "huihui" → ["hui", "hui"] (repeated syllable pattern)
            - "Ka-Fai" → ["Ka", "Fai"] (hyphenated with valid parts)
        """
        if not data_context:
            return None

        # ================================================================
        # EARLY EXIT CONDITIONS: Skip splitting for certain token types
        # ================================================================
        if StringManipulationUtils._should_skip_splitting(token, normalized_cache, normalizer, data_context):
            return None

        # ================================================================
        # PATTERN-BASED SPLITTING: Handle explicit structural patterns
        # ================================================================

        # Pattern 1: Repeated syllable patterns (e.g., "huihui" → ["hui", "hui"])
        raw = token.translate(config.hyphens_apostrophes_tr)
        if len(raw) >= 4 and len(raw) % 2 == 0:
            mid = len(raw) // 2
            first_half = raw[:mid]
            second_half = raw[mid:]

            if first_half.lower() == second_half.lower():
                # Check if the repeated syllable is valid - inline for performance
                if normalized_cache and first_half in normalized_cache:
                    norm_syllable = normalized_cache[first_half]
                else:
                    norm_syllable = normalizer.norm(first_half)
                # For repeated syllables, we only need to check if one is valid (they're the same)
                if norm_syllable in data_context.plausible_components or first_half.lower() in data_context.plausible_components:
                    return [first_half, second_half]

        # Check for forbidden phonetic patterns
        has_forbidden_patterns = bool(config.forbidden_patterns_regex.search(token.lower()))

        # Pattern 2: Explicit hyphen splitting (e.g., "Ka-Fai" → ["Ka", "Fai"])
        if "-" in token and token.count("-") == 1:
            a, b = token.split("-")
            # Inline normalization for performance
            if normalized_cache:
                norm_a = normalizer.get_normalized(a, normalized_cache)
                norm_b = normalizer.get_normalized(b, normalized_cache)
            else:
                norm_a = normalizer.norm(a)
                norm_b = normalizer.norm(b)
            # Component validation with fallback to original forms
            if StringManipulationUtils._is_valid_component_pair(norm_a, norm_b, data_context, a, b):
                return [a, b]

        # Pattern 3: CamelCase detection (e.g., "MingHua" → ["Ming", "Hua"])
        camel = config.camel_case_pattern.findall(raw)
        if len(camel) == 2:
            # Inline normalization for performance
            if normalized_cache:
                norm_a = normalizer.get_normalized(camel[0], normalized_cache)
                norm_b = normalizer.get_normalized(camel[1], normalized_cache)
            else:
                norm_a = normalizer.norm(camel[0])
                norm_b = normalizer.norm(camel[1])
            # Inline component validation for performance
            if StringManipulationUtils._is_valid_component_pair(norm_a, norm_b, data_context, camel[0], camel[1]):
                return camel

        # ================================================================
        # TIERED CONFIDENCE VALIDATION: Brute-force split with quality ranking
        # ================================================================

        # Cache frequently accessed values for performance
        from sinonym.chinese_names_data import HIGH_CONFIDENCE_ANCHORS

        plausible_components = data_context.plausible_components
        raw_len = len(raw)

        for i in range(1, raw_len):
            a, b = raw[:i], raw[i:]

            # Short-circuit: skip very unbalanced splits (optimize for common balanced cases)
            if len(a) == 1 and len(b) > 4:  # Very unbalanced split (e.g., "W" + "eiming")
                continue
            if len(b) == 1 and len(a) > 4:  # Very unbalanced split (e.g., "Weimi" + "g")
                continue

            # Inline normalization for hot path performance
            if normalized_cache:
                norm_a = normalizer.get_normalized(a, normalized_cache)
                norm_b = normalizer.get_normalized(b, normalized_cache)
            else:
                norm_a = normalizer.norm(a)
                norm_b = normalizer.norm(b)

            # Both halves must be known plausible syllables (check both original and normalized forms)
            if not StringManipulationUtils._is_valid_component_pair(norm_a, norm_b, data_context, a, b):
                continue

            # Cultural plausibility check (initial screening) - inline for performance
            if raw_len >= 3:
                if not StringManipulationUtils.is_plausible_chinese_split(norm_a, norm_b, data_context, config):
                    continue

            # Cache anchor checks
            is_a_anchor = norm_a in HIGH_CONFIDENCE_ANCHORS
            is_b_anchor = norm_b in HIGH_CONFIDENCE_ANCHORS

            # Gold Standard (Anchor + Anchor)
            if is_a_anchor and is_b_anchor:
                return [a, b]

            # Silver Standard (Anchor + Plausible)
            if is_a_anchor or is_b_anchor:
                # For Silver Standard, always check cultural plausibility regardless of length
                if StringManipulationUtils.is_plausible_chinese_split(norm_a, norm_b, data_context, config):
                    return [a, b]

            # Bronze Standard (Plausible + Plausible) - requires length >= 4
            elif raw_len >= 4:
                if StringManipulationUtils.is_plausible_chinese_split(norm_a, norm_b, data_context, config):
                    return [a, b]

        # No valid split found
        if has_forbidden_patterns:
            return None

        return None

    @staticmethod
    def is_plausible_chinese_split(norm_a: str, norm_b: str, data_context, config) -> bool:
        """
        Check if a split represents an authentic Chinese name combination vs Western name decomposition.
        """
        if not data_context:
            return False

        # At least one component should be in the actual given names database
        is_a_in_db = norm_a in data_context.given_names_normalized
        is_b_in_db = norm_b in data_context.given_names_normalized

        if not (is_a_in_db or is_b_in_db):
            return False

        # Frequency-based validation: reject if both parts are very uncommon
        freq_a = data_context.get_given_logp(norm_a, config.default_given_logp)
        freq_b = data_context.get_given_logp(norm_b, config.default_given_logp)

        # If both parts are very rare (below -12), it's suspicious
        return not (freq_a < -12.0 and freq_b < -12.0)

    @staticmethod
    def split_compound_token(token: str, compound_meta: CompoundMetadata) -> list[str]:
        """Split a compound token while preserving its original structure.

        Args:
            token: The original compound token (e.g., "AuYeung", "auyeung", "aUyEUNG")
            compound_meta: Metadata about the compound

        Returns:
            List of compound parts preserving original structure when possible
        """
        if compound_meta.format_type == "camelCase":
            # For camelCase, first try splitting on uppercase boundaries
            parts = StringManipulationUtils.split_camel_case(token)
            if len(parts) == 2 and StringManipulationUtils.is_proper_camel_case(token, parts):
                # Proper camelCase - preserve exact structure
                return parts
            # Malformed camelCase like "aUyEUNG" - normalize first, then split
            normalized_token = token.lower()  # "auyeung"
            target_parts = compound_meta.compound_target.split()
            if len(target_parts) == 2:
                return StringManipulationUtils.split_compact_format(normalized_token, target_parts)

        # For compact cases, split based on compound_target length
        target_parts = compound_meta.compound_target.split()
        if len(target_parts) == 2:
            return StringManipulationUtils.split_compact_format(token, target_parts)

        # Fallback: use compound_target (old behavior)
        return [part.title() for part in target_parts]

    @staticmethod
    def is_proper_camel_case(token: str, parts: list[str]) -> bool:
        """Check if a camelCase token has proper boundaries.

        Proper camelCase: "AuYeung" -> ["Au", "Yeung"]
        Malformed camelCase: "aUyEUNG" -> ["aUy", "EUNG"]
        """
        if len(parts) != 2:
            return False

        # Reconstruct the token from parts
        reconstructed = "".join(parts)
        if reconstructed != token:
            return False

        # Check if the split makes sense (no weird case patterns within parts)
        part1, part2 = parts

        # Part 1 should start with lowercase or uppercase, but not have weird mixed case
        # Part 2 should start with uppercase
        if not part2[0].isupper():
            return False

        # Check for reasonable case patterns (not too many case changes within parts)
        def has_reasonable_case_pattern(part):
            if len(part) <= 2:
                return True
            case_changes = sum(1 for i in range(1, len(part)) if part[i].isupper() != part[i - 1].isupper())
            return case_changes <= 1

        return has_reasonable_case_pattern(part1) and has_reasonable_case_pattern(part2)

    @staticmethod
    def split_camel_case(token: str) -> list[str]:
        """Split a camelCase token on uppercase boundaries.

        Examples:
            "AuYeung" -> ["Au", "Yeung"]
            "DuanMu" -> ["Duan", "Mu"]
        """
        if len(token) < 2:
            return [token]

        parts = []
        current = token[0]

        for i in range(1, len(token)):
            if token[i].isupper() and not token[i - 1].isupper():
                # Found camelCase boundary
                parts.append(current)
                current = token[i]
            else:
                current += token[i]

        if current:
            parts.append(current)

        # If we got exactly 2 parts, return them
        if len(parts) == 2:
            return parts

        # Fallback: split in middle
        mid = len(token) // 2
        return [token[:mid], token[mid:]]

    @staticmethod
    def split_compact_format(token: str, target_parts: list[str]) -> list[str]:
        """Split a compact token based on target part lengths.

        Examples:
            "auyeung" with target ["ou", "yang"] -> try to split as ["au", "yeung"]
        """
        if len(target_parts) != 2:
            return [token]  # Can't split

        # Use target lengths as a guide, but preserve original structure
        target_len1, _target_len2 = len(target_parts[0]), len(target_parts[1])

        # Try splitting at the boundary suggested by target lengths
        split_point = target_len1
        if split_point < len(token):
            part1 = token[:split_point]
            part2 = token[split_point:]
            return [part1, part2]

        # Fallback: split in middle
        mid = len(token) // 2
        return [token[:mid], token[mid:]]

    # ====================================================================
    # COMMON STRING OPERATIONS
    # ====================================================================

    @staticmethod
    def join_with_hyphens(parts: list[str]) -> str:
        """Join string parts with hyphens."""
        return "-".join(parts)

    @staticmethod
    def join_with_spaces(parts: list[str]) -> str:
        """Join string parts with spaces."""
        return " ".join(parts)

    @staticmethod
    def join_compact(parts: list[str]) -> str:
        """Join string parts with no separator (compact)."""
        return "".join(parts)

    @staticmethod
    def lowercase_join_with_spaces(tokens: list[str]) -> str:
        """Convert tokens to lowercase and join with spaces - common pattern."""
        return " ".join(t.lower() for t in tokens)

    @staticmethod
    def clean_hyphen_boundaries(text: str) -> str:
        """Remove leading/trailing hyphens from text."""
        return text.strip("-")

    @staticmethod
    def split_and_clean_hyphens(text: str) -> list[str]:
        """Split on hyphens and clean empty parts - common pattern."""
        return [part.strip() for part in text.split("-") if part.strip()]

    @staticmethod
    def remove_spaces(text: str) -> str:
        """Remove spaces from text - centralized space removal."""
        return text.replace(" ", "")

    # ====================================================================
    # CAPITALIZATION FUNCTIONS
    # ====================================================================

    @staticmethod
    def capitalize_name_part(part: str) -> str:
        """Properly capitalize a name part, handling apostrophes and diacritics correctly.

        Standard .title() incorrectly capitalizes after apostrophes (ts'ai -> Ts'Ai).
        This function:
        1. Removes diacritical marks for consistent output
        2. Only capitalizes the first letter: ts'ai -> Ts'ai

        Args:
            part: The name part to capitalize (may contain diacritics)

        Returns:
            Capitalized name part without diacritics
        """
        if not part:
            return part

        # Normalize Unicode and remove diacritical marks
        normalized = unicodedata.normalize("NFD", part)
        without_diacritics = "".join(c for c in normalized if unicodedata.category(c) != "Mn")

        return without_diacritics[0].upper() + without_diacritics[1:].lower()

    @staticmethod
    def is_camel_case(token: str) -> bool:
        """Check if a token follows camelCase pattern (e.g., AuYeung)."""
        if len(token) <= 1:
            return False

        # Standard camelCase: starts with uppercase, has at least one more uppercase
        return token[0].isupper() and any(c.isupper() for c in token[1:]) and not token.isupper()  # Not all uppercase

    @staticmethod
    def is_malformed_camel_case(token: str) -> bool:
        """Check if token might be malformed camelCase (e.g., aUyEUNG).

        This detects patterns that suggest camelCase intent but don't follow
        standard capitalization rules.
        """
        if len(token) <= 2:
            return False

        # Exclude all-caps words - they should be compact, not camelCase
        if token.isupper():
            return False

        # Look for multiple uppercase letters that suggest word boundaries
        uppercase_count = sum(1 for c in token if c.isupper())
        return uppercase_count >= 2

    # ====================================================================
    # FORMAT CONVERSION FUNCTIONS
    # ====================================================================

    @staticmethod
    def format_camel_case_compound(surname_tokens: list[str], original: str) -> str:
        """Format compound surname in camelCase preserving original casing pattern."""
        # For camelCase, we want to preserve the original casing structure
        # Use the original token to preserve the exact capitalization
        # If we're here, it means the compound metadata detected this as camelCase
        # (including malformed camelCase), so preserve the original exactly
        if StringManipulationUtils.is_camel_case(original):
            # For proper camelCase, preserve the exact original capitalization
            return original
        if StringManipulationUtils.is_malformed_camel_case(original):
            # For malformed camelCase, normalize to proper camelCase format
            # Create proper camelCase: first part capitalized, second part capitalized, no separator
            if len(surname_tokens) == 2:
                return StringManipulationUtils.capitalize_name_part(
                    surname_tokens[0],
                ) + StringManipulationUtils.capitalize_name_part(surname_tokens[1])
            # Fallback for unexpected cases
            return "".join(StringManipulationUtils.capitalize_name_part(token) for token in surname_tokens)

        # Fallback for non-camelCase originals
        if len(surname_tokens) != 2:
            return "".join(StringManipulationUtils.capitalize_name_part(token) for token in surname_tokens)

        # Create camelCase from parts
        return "".join(StringManipulationUtils.capitalize_name_part(token) for token in surname_tokens)
