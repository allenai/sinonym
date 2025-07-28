"""
Name formatting service for Chinese name processing.

This module provides sophisticated name formatting including proper capitalization,
compound name splitting, and output standardization to "Given-Name Surname" format.
"""
from __future__ import annotations


class NameFormattingService:
    """Service for formatting Chinese names into standardized output."""

    def __init__(self, config, normalizer, data):
        self._config = config
        self._normalizer = normalizer
        self._data = data

    def format_name_output(
        self,
        surname_tokens: list[str],
        given_tokens: list[str],
        normalized_cache: dict[str, str] | None = None,
    ) -> str:
        """Format parsed name components into final output string."""
        # First validate that given tokens could plausibly be Chinese
        if not self._normalizer.validate_given_tokens(given_tokens, normalized_cache):
            raise ValueError("given name tokens are not plausibly Chinese")

        parts = []
        for token in given_tokens:
            # If the token itself is a valid given name, don't try to split it.
            if normalized_cache and token in normalized_cache:
                normalized_token = normalized_cache[token]
            else:
                normalized_token = self._normalizer.norm(token)

            if normalized_token in self._data.given_names_normalized:
                parts.append(token)
                continue

            # NEW: Before trying to split, check if token is already a valid Chinese syllable
            if self._normalizer.is_valid_chinese_phonetics(token):
                # It's a valid syllable, don't split it
                parts.append(token)
                continue

            # Only try splitting if it's not already a valid syllable
            split = self._normalizer.split_concat(token, normalized_cache)
            if split:
                parts.extend(split)
            # Strict validation: only accept if it's a valid Chinese token
            elif self._normalizer.is_valid_given_name_token(token, normalized_cache):
                parts.append(token)
            else:
                raise ValueError(f"given name token '{token}' is not valid Chinese")

        if not parts:
            raise ValueError("given name invalid")

        # Capitalize each part properly, handling hyphens within parts
        formatted_parts = []
        for part in parts:
            # Clean up any leading/trailing hyphens that may have come from tokenization
            clean_part = part.strip("-")
            if not clean_part:  # Skip empty parts after stripping hyphens
                continue

            if "-" in clean_part:
                sub_parts = clean_part.split("-")
                formatted_part = "-".join(self.capitalize_name_part(sub) for sub in sub_parts)
                formatted_parts.append(formatted_part)
            else:
                formatted_parts.append(self.capitalize_name_part(clean_part))

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
                given_str = " ".join(formatted_parts)
            else:
                # All same length category: use hyphens (e.g., "Yu-Ming" or "A-B")
                given_str = "-".join(formatted_parts)
        else:
            given_str = formatted_parts[0] if formatted_parts else ""

        # Handle compound surnames properly
        if len(surname_tokens) > 1:
            surname_str = "-".join(self.capitalize_name_part(t) for t in surname_tokens)
        else:
            surname_str = self.capitalize_name_part(surname_tokens[0])

        return f"{given_str} {surname_str}"

    def capitalize_name_part(self, part: str) -> str:
        """Properly capitalize a name part, handling apostrophes correctly.

        Standard .title() incorrectly capitalizes after apostrophes (ts'ai -> Ts'Ai).
        This function only capitalizes the first letter: ts'ai -> Ts'ai.
        """
        if not part:
            return part
        return part[0].upper() + part[1:].lower()

