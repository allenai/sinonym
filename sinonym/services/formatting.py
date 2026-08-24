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

from dataclasses import replace
from typing import TYPE_CHECKING

from sinonym.chinese_names_data import ETHNICITY_CHINESE_SURNAME_ROMANIZATION_ALIASES, REVIEWED_ATOMIC_GIVEN_FORMS
from sinonym.coretypes import NameFormat, ParsedName, ParseResult
from sinonym.services.name_lookup import DOMINANT_CHINESE_SURNAME_FREQ_MIN, SurnameResolver
from sinonym.services.order_metadata import original_component_order
from sinonym.services.person_name_normalization import is_reviewed_compact_initial_boundary
from sinonym.utils.string_manipulation import StringManipulationUtils

if TYPE_CHECKING:
    from sinonym.services.normalization import CompoundMetadata

REVIEWED_UNBOUNDED_PREFIX_GIVEN_FORMS = frozenset({"alei"})

# Authored compound surfaces with a reviewed regional syllable that is not in
# the Mandarin lexicon. Preserve their explicit boundary without requiring an
# invented split through the standalone ``Ng`` surname alias.
REVIEWED_EXPLICIT_BOUND_GIVEN_FORMS = frozenset({"chung-chieng"})

# The only Mandarin syllables spelled with a single Roman letter. A lone "a"/"e"
# may therefore be a real syllable rather than an initial.
SINGLE_LETTER_PINYIN_SYLLABLES = frozenset({"a", "e"})
EXPLICIT_HYPHEN_SINGLE_LETTER_SYLLABLES = SINGLE_LETTER_PINYIN_SYLLABLES | {"i"}


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
        self._surname_resolver = SurnameResolver(self._data, self._normalizer)

    def _is_reviewed_atomic_given_form(self, token: str) -> bool:
        """Match reviewed atomics case-insensitively and with tone marks folded."""
        return token.casefold() in REVIEWED_ATOMIC_GIVEN_FORMS or (
            token.isalpha() and self._normalizer.norm_light(token) in REVIEWED_ATOMIC_GIVEN_FORMS
        )

    def materialize_parse_result(  # noqa: PLR0913 - formatter inputs are independent policy evidence
        self,
        surname_tokens: list[str],
        given_tokens: list[str],
        selected_format: NameFormat,
        normalized_cache: dict[str, str] | None = None,
        compound_metadata: dict[str, CompoundMetadata] | None = None,
        *,
        original_compound_format: str | None = None,
        allow_surname_like_given_split: bool = True,
        syllabic_single_letter_tokens: frozenset[str] | None = None,
    ) -> ParseResult:
        """Format prepared components and build their public parse result."""
        try:
            formatted_name, given_final, surname_final, surname_str, given_str, middle_tokens = (
                self.format_name_output_with_tokens(
                    surname_tokens,
                    given_tokens,
                    normalized_cache,
                    compound_metadata,
                    original_compound_format=original_compound_format,
                    allow_surname_like_given_split=allow_surname_like_given_split,
                    syllabic_single_letter_tokens=syllabic_single_letter_tokens,
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
            parsed_original_order = replace(
                parsed,
                order=original_component_order(selected_format, given_tokens, middle_tokens),
            )
            return ParseResult.success_with_name(
                formatted_name,
                original_compound_surname=original_compound_format,
                parsed=parsed,
                parsed_original_order=parsed_original_order,
            )
        except ValueError as error:
            return ParseResult.failure(str(error))

    def format_name_output_with_tokens(  # noqa: PLR0913 - formatter inputs are independent policy evidence
        self,
        surname_tokens: list[str],
        given_tokens: list[str],
        normalized_cache: dict[str, str] | None = None,
        compound_metadata: dict[str, CompoundMetadata] | None = None,
        *,
        original_compound_format: str | None = None,
        allow_surname_like_given_split: bool = True,
        syllabic_single_letter_tokens: frozenset[str] | None = None,
    ) -> tuple[str, list[str], list[str], str, str, list[str]]:
        """
        Format parsed name components and also return the individual tokens.

        Returns:
            A tuple containing the formatted name, final given tokens, final
            surname tokens, surname string, given string, and middle tokens.

        - given_tokens_final: individual given name tokens after splitting and capitalization
        - surname_tokens_final: individual surname tokens (capitalized)
        - surname_str / given_str: component strings as used in full_formatted_name
        - middle_tokens_final: canonical standalone initials outside the Chinese given span
        """
        surname_tokens = [token.strip("'") for token in surname_tokens]
        given_tokens = [token.strip("'") for token in given_tokens]
        if not all(surname_tokens) or not all(given_tokens):
            raise ValueError("name token invalid")

        apostrophe_lineage = {
            token: lineage
            for token in given_tokens
            if (lineage := self._direct_apostrophe_lineage(token, normalized_cache)) is not None
        }
        # Validate given name tokens first
        compact_initial = self._accepts_compact_initial(surname_tokens, given_tokens)
        expand_compact_initial = compact_initial and not is_reviewed_compact_initial_boundary(given_tokens[0])
        alias_given_parts = self._reviewed_alias_compact_given_parts(surname_tokens, given_tokens)
        wade_giles_single_given = self._accepts_wade_giles_single_given(surname_tokens, given_tokens)
        unbounded_syllabic_prefix = bool(
            len(given_tokens) == 1 and self._accepts_unbounded_syllabic_prefix(surname_tokens, given_tokens, given_tokens[0]),
        )
        if (
            not all(
                token in apostrophe_lineage
                or self._is_reviewed_atomic_given_form(token)
                or token.casefold() in REVIEWED_EXPLICIT_BOUND_GIVEN_FORMS
                or self._normalizer.is_valid_given_name_token(token, normalized_cache)
                for token in given_tokens
            )
            and not compact_initial
            and not alias_given_parts
            and not wade_giles_single_given
            and not unbounded_syllabic_prefix
        ):
            msg = "given name tokens are not plausibly Chinese"
            raise ValueError(msg)

        # Process given tokens with splitting
        parts: list[str] = []
        syllabic_keys = {token.casefold().rstrip(".") for token in syllabic_single_letter_tokens or ()}
        for token in given_tokens:
            if token in apostrophe_lineage:
                parts.append(token)
                continue
            if normalized_cache and token in normalized_cache:
                normalized_token = normalized_cache[token]
            else:
                normalized_token = self._normalizer.norm(token)

            if compact_initial and len(given_tokens) == 1 and token == given_tokens[0]:
                if expand_compact_initial:
                    parts.extend(token)
                else:
                    parts.append(token)
                continue

            if self._is_reviewed_atomic_given_form(token):
                parts.append(token)
                continue

            if token.casefold() in REVIEWED_EXPLICIT_BOUND_GIVEN_FORMS:
                parts.append(token)
                continue

            if self._data.is_given_name(normalized_token):
                parts.append(token)
                continue

            if alias_given_parts and len(given_tokens) == 1 and token == given_tokens[0]:
                parts.extend(alias_given_parts)
                continue

            if wade_giles_single_given and len(given_tokens) == 1 and token == given_tokens[0]:
                parts.append(token)
                continue

            if unbounded_syllabic_prefix:
                parts.append(token)
                continue

            if self._normalizer.is_valid_chinese_phonetics(token):
                parts.append(token)
                continue

            split = StringManipulationUtils.split_concatenated_name(
                token,
                normalized_cache,
                self._data,
                self._normalizer,
                self._config,
            )
            if not split and allow_surname_like_given_split:
                split = StringManipulationUtils.split_surname_like_given_name(
                    token,
                    normalized_cache,
                    self._data,
                    self._normalizer,
                    self._config,
                )
            if split and (explicit_lineage := self._explicit_apostrophe_lineage(token, split)):
                parts.append(token)
                apostrophe_lineage[token] = explicit_lineage
            elif split:
                trailing = split[-1].casefold().rstrip(".")
                if len(split) > 1 and len(split[-1]) == 1 and trailing not in syllabic_keys:
                    parts.append(token)
                else:
                    parts.extend(split)
            elif self._normalizer.is_valid_given_name_token(token, normalized_cache):
                parts.append(token)
            else:
                msg = f"given name token '{token}' is not valid Chinese"
                raise ValueError(msg)

        if not parts:
            raise ValueError("given name invalid")

        # Build formatter parts while preserving whether each source component
        # is a standalone initial. Native-script alignment can explicitly mark
        # a one-letter Roman token as a complete syllable instead.
        formatted_parts: list[str] = []
        formatted_part_tokens: list[list[str]] = []
        initial_parts: list[bool] = []
        for part in parts:
            clean_part = StringManipulationUtils.clean_hyphen_boundaries(part)
            if not clean_part:
                continue
            if explicit_lineage := apostrophe_lineage.get(part):
                capitalized_parts, subpart_initials = self._format_bound_given_parts(
                    explicit_lineage,
                    syllabic_keys,
                )
                display_parts = StringManipulationUtils.capitalize_name_part(clean_part).split("'")
                formatted_parts.append(
                    "'".join(
                        formatted if is_initial else display
                        for formatted, display, is_initial in zip(
                            capitalized_parts,
                            display_parts,
                            subpart_initials,
                            strict=True,
                        )
                    ),
                )
                formatted_part_tokens.append(capitalized_parts)
                initial_parts.append(False)
            elif self._is_reviewed_atomic_given_form(clean_part):
                capitalized = StringManipulationUtils.capitalize_name_part(clean_part)
                formatted_part_tokens.append([capitalized])
                formatted_parts.append(capitalized)
                initial_parts.append(False)
            elif "-" in clean_part:
                sub_parts = StringManipulationUtils.split_and_clean_hyphens(clean_part)
                capitalized_parts, _subpart_initials = self._format_bound_given_parts(
                    sub_parts,
                    syllabic_keys,
                    single_letter_syllables=EXPLICIT_HYPHEN_SINGLE_LETTER_SYLLABLES,
                )
                formatted_part_tokens.append(capitalized_parts)
                formatted_parts.append(StringManipulationUtils.join_with_hyphens(capitalized_parts))
                # An explicit hyphen binds every subpart into the first name.
                initial_parts.append(False)
            else:
                letter = self._initial_letter(clean_part)
                is_syllable = clean_part.casefold().rstrip(".") in syllabic_keys
                cap = (
                    f"{letter}."
                    if letter is not None and not is_syllable
                    else StringManipulationUtils.capitalize_name_part(clean_part)
                )
                formatted_parts.append(cap)
                formatted_part_tokens.append([cap])
                initial_parts.append(letter is not None and not is_syllable)

        if not formatted_parts:
            raise ValueError("given name invalid")

        # Chinese all-initial spans are one compound first name. In mixed
        # spans, spelled/source-bound components form the hyphenated first name
        # and all standalone initials occupy the operational middle field.
        all_initials = all(initial_parts)
        primary_indices = [index for index, is_initial in enumerate(initial_parts) if all_initials or not is_initial]
        middle_indices = [] if all_initials else [index for index, is_initial in enumerate(initial_parts) if is_initial]
        given_str = StringManipulationUtils.join_with_hyphens([formatted_parts[index] for index in primary_indices])
        given_tokens_final = [token for index in primary_indices for token in formatted_part_tokens[index]]
        middle_tokens_final = [formatted_part_tokens[index][0] for index in middle_indices]

        # Surname formatting
        if len(surname_tokens) > 1:
            if original_compound_format:
                surname_str = self._format_compound_from_source(surname_tokens, original_compound_format)
            elif compound_metadata:
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

    def _format_bound_given_parts(
        self,
        parts: list[str],
        syllabic_keys: set[str],
        *,
        single_letter_syllables: frozenset[str] = SINGLE_LETTER_PINYIN_SYLLABLES,
    ) -> tuple[list[str], list[bool]]:
        """Format explicitly bound given-name parts without manufacturing initials."""
        all_initial_parts = bool(parts) and all(self._initial_letter(part) is not None for part in parts)
        formatted: list[str] = []
        initial_flags: list[bool] = []
        for part in parts:
            letter = self._initial_letter(part)
            key = part.casefold().rstrip(".")
            is_syllable = key in syllabic_keys or (
                not all_initial_parts and key in single_letter_syllables and "." not in part
            )
            is_initial = letter is not None and not is_syllable
            formatted.append(f"{letter}." if is_initial else StringManipulationUtils.capitalize_name_part(part))
            initial_flags.append(is_initial)
        return formatted, initial_flags

    def _direct_apostrophe_lineage(
        self,
        token: str,
        normalized_cache: dict[str, str] | None,
    ) -> list[str] | None:
        """Return an explicit boundary without misreading Wade-Giles aspiration."""
        if token.count("'") != 1 or token.startswith("'") or token.endswith("'"):
            return None
        pieces = token.split("'")
        if all(
            self._initial_letter(piece) is not None and (piece.isupper() or "." in piece) for piece in pieces
        ) and not self._normalizer.is_valid_given_name_token(token, normalized_cache):
            return pieces
        if any(len(piece) == 1 and piece.casefold() not in SINGLE_LETTER_PINYIN_SYLLABLES for piece in pieces):
            return None
        return pieces if all(self._normalizer.is_valid_given_name_token(piece, normalized_cache) for piece in pieces) else None

    def _explicit_apostrophe_lineage(self, token: str, split: list[str]) -> list[str] | None:
        """Keep an explicit apostrophe in display while retaining validated split tokens."""
        if token.count("'") != 1 or token.startswith("'") or token.endswith("'"):
            return None
        pieces = [piece for split_part in split for piece in StringManipulationUtils.split_and_clean_hyphens(split_part)]
        if len(pieces) != len(token.split("'")):
            return None
        source_key = self._normalizer.norm_light(token)
        split_key = "".join(self._normalizer.norm_light(piece) for piece in pieces)
        return pieces if source_key == split_key else None

    def _accepts_compact_initial(self, surname_tokens: list[str], given_tokens: list[str]) -> bool:
        """Return dominant surname plus a single vowelless 2-3 letter bundle."""
        if len(surname_tokens) != 1 or len(given_tokens) != 1:
            return False
        abbreviation = given_tokens[0]
        surname_key = self._normalizer.norm_light(surname_tokens[0])
        return bool(
            self._data.get_surname_freq_as_written(surname_key) >= DOMINANT_CHINESE_SURNAME_FREQ_MIN
            and self._normalizer.is_vowelless_compact_initial(abbreviation),
        )

    @staticmethod
    def _format_compound_from_source(surname_tokens: list[str], source_format: str) -> str:
        """Format a compound surname from the selected source occurrence."""
        capitalized = [StringManipulationUtils.capitalize_name_part(token) for token in surname_tokens]
        if " " in source_format:
            return StringManipulationUtils.join_with_spaces(capitalized)
        if "-" in source_format:
            return StringManipulationUtils.join_with_hyphens(capitalized)
        if any(character.isupper() for character in source_format[1:]) and not source_format.isupper():
            return StringManipulationUtils.format_camel_case_compound(surname_tokens, source_format)
        return StringManipulationUtils.capitalize_name_part(source_format)

    @staticmethod
    def _initial_letter(token: str) -> str | None:
        """Return the canonical letter for one optional-period initial."""
        stripped = token.strip()
        letters = [character for character in stripped if character.isalpha()]
        if len(letters) != 1 or any(not (character.isalpha() or character == ".") for character in stripped):
            return None
        return letters[0].upper()

    def _accepts_unbounded_syllabic_prefix(
        self,
        surname_tokens: list[str],
        given_tokens: list[str],
        token: str,
    ) -> bool:
        """Preserve an A/E-prefixed given behind an identified Chinese surname."""
        return bool(
            len(surname_tokens) == 1
            and len(given_tokens) == 1
            and token.isalpha()
            and len(token) > 2  # noqa: PLR2004
            and self._normalizer.norm_light(token) in REVIEWED_UNBOUNDED_PREFIX_GIVEN_FORMS
            and token[0].casefold() in SINGLE_LETTER_PINYIN_SYLLABLES
            and (
                self._data.is_given_name(self._normalizer.norm(token[1:]))
                or self._normalizer.is_valid_chinese_phonetics(token[1:])
            )
            and self._surname_resolver.evidence_is_surname(surname_tokens[0]),
        )

    def _accepts_wade_giles_single_given(
        self,
        surname_tokens: list[str],
        given_tokens: list[str],
    ) -> bool:
        """Allow one alphabetic given token behind exact Wade-Giles surname evidence."""
        return bool(
            len(surname_tokens) == 1
            and len(given_tokens) == 1
            and given_tokens[0].isalpha()
            and self._surname_resolver.evidence_is_wade_giles_apostrophe_surname(surname_tokens[0]),
        )

    def _reviewed_alias_compact_given_parts(
        self,
        surname_tokens: list[str],
        given_tokens: list[str],
    ) -> list[str]:
        """Split one compact given token only under reviewed surname-alias evidence."""
        if (
            len(surname_tokens) != 1
            or len(given_tokens) != 1
            or surname_tokens[0].lower() not in ETHNICITY_CHINESE_SURNAME_ROMANIZATION_ALIASES
        ):
            return []
        token = given_tokens[0]
        candidates = [
            [token[:index], token[index:]]
            for index in range(1, len(token))
            if self._normalizer.is_valid_chinese_phonetics(token[:index])
            and self._normalizer.is_valid_chinese_phonetics(token[index:])
        ]
        return candidates[0] if len(candidates) == 1 else []

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
        if meta and meta.is_compound:
            if meta.format_type == "compact":
                return StringManipulationUtils.capitalize_name_part(surname_token)
            if meta.format_type == "camelCase":
                surname_parts = StringManipulationUtils.split_compound_token(surname_token, meta)
                return StringManipulationUtils.format_camel_case_compound(surname_parts, surname_token)
            if meta.format_type == "hyphenated":
                return StringManipulationUtils.capitalize_name_part(surname_token)

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
