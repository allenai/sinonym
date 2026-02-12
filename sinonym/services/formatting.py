"""
Name formatting service for Chinese name processing.

RESPONSIBILITIES (After Service Responsibility Clarification):
- Context-aware given name splitting using full database (AFTER parsing)
- Proper capitalization and formatting of all name parts
- Compound surname formatting using metadata from CompoundDetector
- Trust CompoundDetector metadata as single authority
- NO structural/pattern-based splitting (that belongs in TextPreprocessor)
- Operates AFTER parsing when surname/given context is known

This module provides sophisticated name formatting including proper capitalization,
compound name splitting, and output standardization to "Given-Name Surname" format.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sinonym.utils.string_manipulation import StringManipulationUtils

if TYPE_CHECKING:
    from sinonym.services.normalization import CompoundMetadata


class NameFormattingService:
    """Service for formatting Chinese names into standardized output."""

    def __init__(self, context_or_config, normalizer=None, data=None):
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

    def format_name_output(
        self,
        surname_tokens: list[str],
        given_tokens: list[str],
        normalized_cache: dict[str, str] | None = None,
        compound_metadata: dict[str, CompoundMetadata] | None = None,
    ) -> str:
        """
        Format parsed name components into final output string.

        Responsibilities:
        - Context-aware given name splitting using full database (after parsing)
        - Proper capitalization and formatting of all name parts
        - Compound surname formatting using metadata from CompoundDetector
        - NO structural/pattern-based splitting (that belongs in TextPreprocessor)
        """
        # First validate that given tokens could plausibly be Chinese
        if not self._normalizer.validate_given_tokens(given_tokens, normalized_cache):
            msg = "given name tokens are not plausibly Chinese"
            raise ValueError(msg)

        # Process given name tokens: context-aware splitting using full database
        parts = []
        for token in given_tokens:
            # Check if token is already a valid given name (no splitting needed)
            if normalized_cache and token in normalized_cache:
                normalized_token = normalized_cache[token]
            else:
                normalized_token = self._normalizer.norm(token)

            if self._data.is_given_name(normalized_token):
                parts.append(token)
                continue

            # Check if token is already a valid Chinese syllable (no splitting needed)
            if self._normalizer.is_valid_chinese_phonetics(token):
                parts.append(token)
                continue

            # Context-aware splitting: use full database for intelligent given name splitting
            split = StringManipulationUtils.split_concatenated_name(
                token,
                normalized_cache,
                self._data,
                self._normalizer,
                self._config,
            )
            if split:
                parts.extend(split)
            # Final validation: accept if it's a valid Chinese token
            elif self._normalizer.is_valid_given_name_token(token, normalized_cache):
                parts.append(token)
            else:
                msg = f"given name token '{token}' is not valid Chinese"
                raise ValueError(msg)

        if not parts:
            msg = "given name invalid"
            raise ValueError(msg)

        # Capitalize each part properly, handling hyphens within parts
        formatted_parts = []
        for part in parts:
            # Clean up any leading/trailing hyphens that may have come from tokenization
            clean_part = StringManipulationUtils.clean_hyphen_boundaries(part)
            if not clean_part:  # Skip empty parts after stripping hyphens
                continue

            if "-" in clean_part:
                sub_parts = StringManipulationUtils.split_and_clean_hyphens(clean_part)
                capitalized_parts = [StringManipulationUtils.capitalize_name_part(sub) for sub in sub_parts]
                formatted_part = StringManipulationUtils.join_with_hyphens(capitalized_parts)
                formatted_parts.append(formatted_part)
            else:
                formatted_parts.append(StringManipulationUtils.capitalize_name_part(clean_part))

        # Determine separator based on part lengths
        # Use spaces when we have mixed-length parts (some single chars, some multi-char)
        if len(formatted_parts) > 1:
            part_lengths = [
                len(part.replace("-", "")) for part in formatted_parts
            ]  # Count chars, ignoring internal hyphens
            has_single_char = any(length == 1 for length in part_lengths)
            has_multi_char = any(length > 1 for length in part_lengths)

            # Special-case: one or more trailing single-letter initials
            trailing_count = 0
            for length in reversed(part_lengths):
                if length == 1:
                    trailing_count += 1
                else:
                    break

            if trailing_count > 0 and any(l > 1 for l in part_lengths[:-trailing_count]):
                primary_parts = formatted_parts[:-trailing_count]
                middle_parts = formatted_parts[-trailing_count:]

                # Join primary parts using mixed-length rule
                if len(primary_parts) > 1:
                    p_lengths = [len(p.replace("-", "")) for p in primary_parts]
                    p_has_single = any(l == 1 for l in p_lengths)
                    p_has_multi = any(l > 1 for l in p_lengths)
                    if p_has_single and p_has_multi:
                        primary_given_str = StringManipulationUtils.join_with_spaces(primary_parts)
                    else:
                        primary_given_str = StringManipulationUtils.join_with_hyphens(primary_parts)
                else:
                    primary_given_str = primary_parts[0]

                middle_str = StringManipulationUtils.join_with_spaces(middle_parts)
                given_str = f"{primary_given_str} {middle_str}".strip()
            elif has_single_char and has_multi_char:
                # Mixed lengths: use spaces (e.g., "Bin B" not "Bin-B")
                given_str = StringManipulationUtils.join_with_spaces(formatted_parts)
            else:
                # All same length category: use hyphens (e.g., "Yu-Ming" or "A-B")
                given_str = StringManipulationUtils.join_with_hyphens(formatted_parts)
        else:
            given_str = formatted_parts[0] if formatted_parts else ""

        # Handle compound surnames using centralized metadata
        if len(surname_tokens) > 1:
            # CompoundDetector should always provide metadata - trust it as the single authority
            if compound_metadata:
                surname_str = self._format_compound_with_metadata(surname_tokens, compound_metadata)
            else:
                # This should rarely happen if CompoundDetector is working correctly
                # Log a warning and use default behavior
                # TODO: Add logging: "Missing compound metadata for multi-token surname"
                capitalized_tokens = [StringManipulationUtils.capitalize_name_part(t) for t in surname_tokens]
                surname_str = StringManipulationUtils.join_with_hyphens(capitalized_tokens)
        # Single token surname - check if it's a compact compound
        elif compound_metadata:
            surname_str = self._format_single_token_with_metadata(surname_tokens[0], compound_metadata)
        else:
            surname_str = StringManipulationUtils.capitalize_name_part(surname_tokens[0])

        return f"{given_str} {surname_str}"

    def format_name_output_with_tokens(
        self,
        surname_tokens: list[str],
        given_tokens: list[str],
        normalized_cache: dict[str, str] | None = None,
        compound_metadata: dict[str, CompoundMetadata] | None = None,
    ) -> tuple[str, list[str], list[str], str, str, list[str]]:
        """
        Format parsed name components and also return the individual tokens.

        Returns:
            (full_formatted_name, given_tokens_final, surname_tokens_final, surname_str, given_str)

        - given_tokens_final: individual given name tokens after splitting and capitalization
        - surname_tokens_final: individual surname tokens (capitalized)
        - surname_str / given_str: component strings as used in full_formatted_name
        """
        # Validate given name tokens first
        if not self._normalizer.validate_given_tokens(given_tokens, normalized_cache):
            msg = "given name tokens are not plausibly Chinese"
            raise ValueError(msg)

        # Process given tokens with splitting
        parts: list[str] = []
        for token in given_tokens:
            if normalized_cache and token in normalized_cache:
                normalized_token = normalized_cache[token]
            else:
                normalized_token = self._normalizer.norm(token)

            if self._data.is_given_name(normalized_token):
                parts.append(token)
                continue

            if self._normalizer.is_valid_chinese_phonetics(token):
                parts.append(token)
                continue

            split = StringManipulationUtils.split_concatenated_name(
                token, normalized_cache, self._data, self._normalizer, self._config,
            )
            if split:
                parts.extend(split)
            elif self._normalizer.is_valid_given_name_token(token, normalized_cache):
                parts.append(token)
            else:
                msg = f"given name token '{token}' is not valid Chinese"
                raise ValueError(msg)

        if not parts:
            raise ValueError("given name invalid")

        # Build tokens and formatted parts
        formatted_parts: list[str] = []
        given_tokens_final: list[str] = []
        for part in parts:
            clean_part = StringManipulationUtils.clean_hyphen_boundaries(part)
            if not clean_part:
                continue
            if "-" in clean_part:
                sub_parts = StringManipulationUtils.split_and_clean_hyphens(clean_part)
                capitalized_parts = [StringManipulationUtils.capitalize_name_part(sub) for sub in sub_parts]
                given_tokens_final.extend(capitalized_parts)
                formatted_parts.append(StringManipulationUtils.join_with_hyphens(capitalized_parts))
            else:
                cap = StringManipulationUtils.capitalize_name_part(clean_part)
                formatted_parts.append(cap)
                given_tokens_final.append(cap)

        # Determine given separator (mirror format_name_output)
        # Also peel leading/trailing single-letter initials into middle tokens when appropriate
        middle_tokens_final: list[str] = []
        if len(formatted_parts) > 1:
            part_lengths = [len(part.replace("-", "")) for part in formatted_parts]
            has_single_char = any(length == 1 for length in part_lengths)
            has_multi_char = any(length > 1 for length in part_lengths)

            # leading run of single-letter initials → middle tokens (only if followed by any multi-char part)
            leading_count = 0
            for length in part_lengths:
                if length == 1:
                    leading_count += 1
                else:
                    break

            if leading_count > 0 and any(l > 1 for l in part_lengths[leading_count:]):
                middle_tokens_final.extend(formatted_parts[:leading_count])
                formatted_parts = formatted_parts[leading_count:]
                part_lengths = part_lengths[leading_count:]
                if len(given_tokens_final) >= leading_count:
                    given_tokens_final = given_tokens_final[leading_count:]

            # trailing run of single-letter initials → middle tokens (only if preceded by any multi-char part)
            trailing_count = 0
            for length in reversed(part_lengths):
                if length == 1:
                    trailing_count += 1
                else:
                    break

            if trailing_count > 0 and any(l > 1 for l in part_lengths[:-trailing_count]):
                middle_tokens_final.extend(formatted_parts[-trailing_count:])
                primary_parts = formatted_parts[:-trailing_count]

                if len(primary_parts) > 1:
                    p_lengths = [len(p.replace("-", "")) for p in primary_parts]
                    p_has_single = any(l == 1 for l in p_lengths)
                    p_has_multi = any(l > 1 for l in p_lengths)
                    if p_has_single and p_has_multi:
                        given_str = StringManipulationUtils.join_with_spaces(primary_parts)
                    else:
                        given_str = StringManipulationUtils.join_with_hyphens(primary_parts)
                else:
                    given_str = primary_parts[0] if primary_parts else ""

                # Remove the trailing middle initials from the given token list
                if trailing_count > 0 and len(given_tokens_final) >= trailing_count:
                    given_tokens_final = given_tokens_final[:-trailing_count]
            elif len(formatted_parts) > 1:
                # Mixed lengths remain in given – choose separator
                part_lengths = [len(part.replace("-", "")) for part in formatted_parts]
                has_single_char = any(length == 1 for length in part_lengths)
                has_multi_char = any(length > 1 for length in part_lengths)
                if has_single_char and has_multi_char:
                    given_str = StringManipulationUtils.join_with_spaces(formatted_parts)
                else:
                    given_str = StringManipulationUtils.join_with_hyphens(formatted_parts)
            else:
                given_str = formatted_parts[0] if formatted_parts else ""
        else:
            given_str = formatted_parts[0] if formatted_parts else ""

        # Surname formatting
        if len(surname_tokens) > 1:
            if compound_metadata:
                surname_str = self._format_compound_with_metadata(surname_tokens, compound_metadata)
            else:
                capitalized_tokens = [StringManipulationUtils.capitalize_name_part(t) for t in surname_tokens]
                surname_str = StringManipulationUtils.join_with_hyphens(capitalized_tokens)
        elif compound_metadata:
            surname_str = self._format_single_token_with_metadata(surname_tokens[0], compound_metadata)
        else:
            surname_str = StringManipulationUtils.capitalize_name_part(surname_tokens[0])

        # If we have middle tokens, include them between given and surname
        if middle_tokens_final:
            middle_str = StringManipulationUtils.join_with_spaces(middle_tokens_final)
            full_formatted = f"{given_str} {middle_str} {surname_str}".strip()
        else:
            full_formatted = f"{given_str} {surname_str}".strip()
        surname_tokens_final = [StringManipulationUtils.capitalize_name_part(t) for t in surname_tokens]

        return full_formatted, given_tokens_final, surname_tokens_final, surname_str, given_str, middle_tokens_final

    def capitalize_name_part(self, part: str) -> str:
        """Properly capitalize a name part - delegated to centralized utility."""
        return StringManipulationUtils.capitalize_name_part(part)

    def _format_compound_with_metadata(
        self,
        surname_tokens: list[str],
        compound_metadata: dict[str, CompoundMetadata],
    ) -> str:
        """Format compound surname using centralized metadata.

        Args:
            surname_tokens: List of surname token parts (e.g., ["Au", "Yeung"])
            compound_metadata: Centralized compound metadata

        Returns:
            Formatted surname string using metadata-driven formatting
        """
        def _get_meta_for_token(token: str) -> CompoundMetadata | None:
            meta = compound_metadata.get(token)
            if meta is not None:
                return meta

            token_lower = token.lower()
            for original_token, candidate in compound_metadata.items():
                if original_token.lower() == token_lower:
                    return candidate
            return None

        first_token_meta = _get_meta_for_token(surname_tokens[0])
        second_token_meta = _get_meta_for_token(surname_tokens[1]) if len(surname_tokens) > 1 else None

        # Prefer direct token-linked metadata when available.
        if not first_token_meta or not first_token_meta.is_compound:
            first_token_meta = None
        elif len(surname_tokens) > 1:
            if not second_token_meta or not second_token_meta.is_compound:
                first_token_meta = None
            elif first_token_meta.compound_target != second_token_meta.compound_target:
                first_token_meta = None

        # Compact/camelCase compounds are represented by a single original token in metadata.
        # Match by split result so we don't accidentally pick unrelated compound metadata.
        if not first_token_meta:
            surname_tokens_lower = [token.lower() for token in surname_tokens]
            for original_token, meta in compound_metadata.items():
                if not meta.is_compound:
                    continue

                split_parts = StringManipulationUtils.split_compound_token(original_token, meta)
                if len(split_parts) != len(surname_tokens):
                    continue
                if [part.lower() for part in split_parts] == surname_tokens_lower:
                    first_token_meta = meta
                    break

        if not first_token_meta:
            # Fallback: join with hyphens
            capitalized_tokens = [StringManipulationUtils.capitalize_name_part(t) for t in surname_tokens]
            return StringManipulationUtils.join_with_hyphens(capitalized_tokens)

        # Format based on detected type
        if first_token_meta.format_type == "hyphenated":
            capitalized_tokens = [StringManipulationUtils.capitalize_name_part(token) for token in surname_tokens]
            return StringManipulationUtils.join_with_hyphens(capitalized_tokens)
        if first_token_meta.format_type == "spaced":
            capitalized_tokens = [StringManipulationUtils.capitalize_name_part(token) for token in surname_tokens]
            return StringManipulationUtils.join_with_spaces(capitalized_tokens)
        if first_token_meta.format_type == "camelCase":
            # Need to find the original token to preserve camelCase
            original_token = self._find_original_compound_token(compound_metadata, first_token_meta.compound_target)
            if original_token:
                return StringManipulationUtils.format_camel_case_compound(surname_tokens, original_token)
            capitalized_tokens = [StringManipulationUtils.capitalize_name_part(token) for token in surname_tokens]
            return StringManipulationUtils.join_compact(capitalized_tokens)
        if first_token_meta.format_type == "compact":
            # Need to find the original token to preserve compact format
            original_token = self._find_original_compound_token(compound_metadata, first_token_meta.compound_target)
            if original_token:
                return StringManipulationUtils.capitalize_name_part(original_token)
            return StringManipulationUtils.join_compact(surname_tokens).capitalize()
        # Unknown format, use default
        capitalized_tokens = [StringManipulationUtils.capitalize_name_part(t) for t in surname_tokens]
        return StringManipulationUtils.join_with_hyphens(capitalized_tokens)

    def _format_single_token_with_metadata(
        self,
        surname_token: str,
        compound_metadata: dict[str, CompoundMetadata],
    ) -> str:
        """Format single token surname using centralized metadata.

        This handles cases where parsing converts a compact compound like "Sima"
        into separate tokens ["Si", "Ma"] but we need to format it as "Sima".

        Args:
            surname_token: The surname token
            compound_metadata: Centralized compound metadata

        Returns:
            Formatted surname string
        """
        # Check if this token appears in compound metadata
        meta = compound_metadata.get(surname_token)
        if meta and meta.is_compound and meta.format_type == "compact":
            # This was originally a compact compound, find the original token
            original_token = self._find_original_compound_token(compound_metadata, meta.compound_target)
            if original_token:
                return StringManipulationUtils.capitalize_name_part(original_token)

        # Default single token formatting
        return StringManipulationUtils.capitalize_name_part(surname_token)

    def _find_original_compound_token(
        self,
        compound_metadata: dict[str, CompoundMetadata],
        target_compound: str,
    ) -> str | None:
        """Find the original token that corresponds to a compound target.

        Args:
            compound_metadata: Centralized compound metadata
            target_compound: The target compound (e.g., "ou yang")

        Returns:
            Original token (e.g., "AuYeung") or None if not found
        """
        for token, meta in compound_metadata.items():
            if meta.is_compound and meta.compound_target == target_compound:
                return token
        return None
