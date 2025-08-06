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
        original_compound_surname: str | None = None,
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
                token, normalized_cache, self._data, self._normalizer, self._config,
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

            if has_single_char and has_multi_char:
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

    def capitalize_name_part(self, part: str) -> str:
        """Properly capitalize a name part - delegated to centralized utility."""
        return StringManipulationUtils.capitalize_name_part(part)

    def _format_compound_with_metadata(
        self, surname_tokens: list[str], compound_metadata: dict[str, CompoundMetadata],
    ) -> str:
        """Format compound surname using centralized metadata.

        Args:
            surname_tokens: List of surname token parts (e.g., ["Au", "Yeung"])
            compound_metadata: Centralized compound metadata

        Returns:
            Formatted surname string using metadata-driven formatting
        """
        # Find the first token's metadata to determine format type
        first_token_meta = None
        for meta in compound_metadata.values():
            if meta.is_compound:
                # Check if this metadata matches our surname tokens
                target_parts = meta.compound_target.split()
                if len(target_parts) == len(surname_tokens):
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
        self, surname_token: str, compound_metadata: dict[str, CompoundMetadata],
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
        self, compound_metadata: dict[str, CompoundMetadata], target_compound: str,
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
