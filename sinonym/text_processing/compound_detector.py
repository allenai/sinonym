"""
Compound surname detection utilities for Chinese name processing.

RESPONSIBILITIES (After Service Responsibility Clarification):
- SINGLE AUTHORITY for compound surname detection and format classification
- Generate compound metadata for ALL tokens during normalization
- Provide unidirectional metadata flow to downstream services
- NO other service should re-detect compound formats
- Self-contained format detection (no dependencies on StringManipulationUtils)

This module handles the detection and classification of compound surnames
in various formats (compact, spaced, hyphenated, camelCase).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sinonym.chinese_names_data import COMPOUND_VARIANTS
from sinonym.utils.string_manipulation import StringManipulationUtils

if TYPE_CHECKING:
    from sinonym.services.normalization import CompoundMetadata, SpacedCompoundSpan


class CompoundDetector:
    """
    Single authority for detecting and classifying compound surnames.

    Responsibilities:
    - Generate compound metadata with format detection for all tokens
    - Provide unidirectional metadata flow to downstream services
    - NO other service should re-detect compound formats
    """

    def __init__(self, config):
        self._config = config

    def generate_compound_metadata(self, roman_tokens: tuple[str, ...], data_context=None) -> dict[str, CompoundMetadata]:
        """Generate token-keyed compound metadata for compatibility callers."""
        compound_metadata, _spaced_spans = self.generate_compound_analysis(roman_tokens, data_context)
        return compound_metadata

    def generate_compound_analysis(
        self,
        roman_tokens: tuple[str, ...],
        data_context=None,
    ) -> tuple[dict[str, CompoundMetadata], tuple[SpacedCompoundSpan, ...]]:
        """Generate token metadata and occurrence-level spaced compound spans.

        Args:
            roman_tokens: The roman tokens to analyze
            data_context: Optional data context for validation

        Returns:
            Token-keyed display metadata and exact spaced-compound occurrences.
        """
        # Import here to avoid circular imports

        compound_metadata = {}

        # First, detect single-token compounds
        for token in roman_tokens:
            metadata = self._detect_compound_for_token(token, data_context)
            compound_metadata[token] = metadata

        # Second, detect multi-token compounds.  Detection finishes before the
        # token-keyed display summary is updated so one occurrence cannot hide
        # a later repeated occurrence with the same token text.
        spaced_spans = self._detect_multi_token_compounds(roman_tokens, compound_metadata, data_context)
        self._attach_spaced_compound_summaries(roman_tokens, compound_metadata, spaced_spans)

        return compound_metadata, spaced_spans

    def _detect_compound_for_token(self, token: str, data_context=None) -> CompoundMetadata:
        """
        Detect compound surname information for a single token.

        Args:
            token: The token to analyze
            data_context: Optional data context for validation

        Returns:
            CompoundMetadata with detection results
        """
        # Import here to avoid circular imports
        from sinonym.services.normalization import CompoundMetadata

        token_lower = token.strip().lower()

        # 1. Check COMPOUND_VARIANTS
        if token_lower in COMPOUND_VARIANTS:
            target_compound = COMPOUND_VARIANTS[token_lower]
            format_type = self._detect_format_type(token)
            return CompoundMetadata(
                is_compound=True,
                format_type=format_type,
                compound_target=target_compound,
            )

        # 2. Check hyphenated compounds
        if "-" in token and data_context:
            if hasattr(data_context, "compound_hyphen_map") and token_lower in data_context.compound_hyphen_map:
                target_compound = data_context.compound_hyphen_map[token_lower]
                return CompoundMetadata(
                    is_compound=True,
                    format_type="hyphenated",
                    compound_target=target_compound,
                )

        # 3. Check if token is already a compound in standard form
        if (
            data_context
            and hasattr(data_context, "compound_surnames_normalized")
            and token_lower in data_context.compound_surnames_normalized
        ):
            return CompoundMetadata(
                is_compound=True,
                format_type="spaced",
                compound_target=token_lower,
            )

        # 4. Not a compound surname
        return CompoundMetadata(is_compound=False)

    def _detect_format_type(self, token: str) -> str:
        """
        Detect the format type of a compound surname token.

        CompoundDetector is the single authority for format detection.
        NO dependencies on StringManipulationUtils for detection logic.

        Args:
            token: The original token (with original capitalization)

        Returns:
            Format type: "compact", "spaced", "hyphenated", "camelCase"
        """
        if " " in token:
            return "spaced"
        if "-" in token:
            return "hyphenated"
        if StringManipulationUtils.is_camel_case(token):
            return "camelCase"
        if StringManipulationUtils.is_malformed_camel_case(token):
            return "camelCase"
        return "compact"

    def _detect_multi_token_compounds(
        self,
        roman_tokens: tuple[str, ...],
        compound_metadata: dict[str, CompoundMetadata],
        data_context=None,
    ) -> tuple[SpacedCompoundSpan, ...]:
        """
        Detect multi-token compound surnames (like "au yeung").

        Args:
            roman_tokens: The roman tokens to analyze
            compound_metadata: Single-token metadata used to avoid overlapping
                compact or hyphenated compounds.
            data_context: Optional data context for validation
        """
        # Import here to avoid circular imports
        from sinonym.services.normalization import SpacedCompoundSpan

        spans: list[SpacedCompoundSpan] = []
        i = 0
        while i < len(roman_tokens) - 1:
            token1 = roman_tokens[i]
            token2 = roman_tokens[i + 1]

            # Skip if either token is already part of a compound
            if compound_metadata[token1].is_compound or compound_metadata[token2].is_compound:
                i += 1
                continue

            pair_lower = f"{token1.lower()} {token2.lower()}"
            is_compound_pair = False
            target_compound = ""

            # Check in COMPOUND_VARIANTS
            if pair_lower in COMPOUND_VARIANTS:
                is_compound_pair = True
                target_compound = COMPOUND_VARIANTS[pair_lower]
            # Check in compound_surnames_normalized
            elif (
                data_context
                and hasattr(data_context, "compound_surnames_normalized")
                and pair_lower in data_context.compound_surnames_normalized
            ) or pair_lower in COMPOUND_VARIANTS.values():
                is_compound_pair = True
                target_compound = pair_lower

            if is_compound_pair:
                spans.append(
                    SpacedCompoundSpan(
                        start=i,
                        end=i + 2,
                        compound_target=target_compound,
                    ),
                )
                i += 2  # Skip the next token as it's part of this compound
            else:
                i += 1

        return tuple(spans)

    @staticmethod
    def _attach_spaced_compound_summaries(
        roman_tokens: tuple[str, ...],
        compound_metadata: dict[str, CompoundMetadata],
        spans: tuple[SpacedCompoundSpan, ...],
    ) -> None:
        """Attach formatting summaries while retaining exact occurrence spans."""
        if not spans:
            return

        # Import here to avoid the normalization/compound-detector import cycle.
        from sinonym.services.normalization import CompoundMetadata

        target_by_token: dict[str, str] = {}
        for span in spans:
            for token in set(roman_tokens[span.start : span.end]):
                target_by_token.setdefault(token, span.compound_target)

        for token, target in target_by_token.items():
            # The token-keyed value is only a display summary. Positional
            # consumers use the separately returned occurrence tuple.
            compound_metadata[token] = CompoundMetadata(
                is_compound=True,
                format_type="spaced",
                compound_target=target,
            )
