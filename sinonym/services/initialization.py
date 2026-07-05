"""
Data initialization service for Chinese name processing.

This module handles loading and preprocessing of Chinese name databases,
building frequency mappings, and creating immutable data structures.
"""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple

from sinonym.chinese_names_data import CANTONESE_SURNAMES, COMPOUND_VARIANTS, PYPINYIN_FREQUENCY_ALIASES
from sinonym.resources import open_csv_reader
from sinonym.utils.string_manipulation import StringManipulationUtils

if TYPE_CHECKING:
    from sinonym.coretypes import ChineseNameConfig


class SurnameRomanization(NamedTuple):
    """One curated row of surname_romanizations.csv, keyed by as-written spelling.

    ``target_share`` is the share of ``mandarin_target``'s surname mass the
    as-written spelling keeps when scored as a surname (1.0 for attested
    romanizations, ~e^-4 for remap-only spellings). ``as_written_ppm`` is the
    already-discounted mass: target mass times ``target_share``.
    """

    mandarin_target: str
    target_share: float
    as_written_ppm: float


@dataclass(frozen=True)
class NameDataStructures:
    """Immutable container for all name-related data structures."""

    # Core surname and given name sets
    surnames: frozenset[str]
    surnames_normalized: frozenset[str]
    compound_surnames: frozenset[str]
    compound_surnames_normalized: frozenset[str]
    given_names: frozenset[str]
    given_names_normalized: frozenset[str]

    # Dynamically generated plausible components from givenname.csv
    plausible_components: frozenset[str]

    # Frequency and probability mappings
    surname_frequencies: Mapping[str, float]
    surname_log_probabilities: Mapping[str, float]
    given_log_probabilities: Mapping[str, float]

    # Position-conditional given-syllable log probabilities (as-written romanization)
    given_initial_log_probabilities: Mapping[str, float]
    given_final_log_probabilities: Mapping[str, float]
    given_initial_default_logp: float
    given_final_default_logp: float

    # Pre-computed percentile ranks for ML features (0-1 scale)
    surname_percentile_ranks: Mapping[str, float]

    # Compound surname mappings
    compound_hyphen_map: Mapping[str, str]
    # Maps normalized compound surnames back to their original input format
    compound_original_format_map: Mapping[str, str]

    # Canonical per-spelling romanization resolution (surname_romanizations.csv),
    # keyed by as-written (norm_light) spelling. Shared by parser scoring and
    # batch/routing evidence via ``resolve_surname_spelling``. The as-written-vs-
    # remapped key decision differs by consumer: batch/routing evidence uses the
    # shared ``surname_lookup_key`` resolver, while parser scoring applies a
    # stricter romanization-conditional key so its ``_surname_spelling_share_logp``
    # discount can operate on remapped penalty-spelling mass. The two converge on
    # the same effective surname mass through different keys (see
    # ``surname_lookup_key``), not the same literal key.
    surname_romanizations: Mapping[str, SurnameRomanization]
    surname_as_written_aliases: frozenset[str]

    def get_surname_logp(self, surname_key: str, default: float) -> float:
        """Get surname log probability with default fallback."""
        return self.surname_log_probabilities.get(surname_key, default)

    def get_given_logp(self, given_key: str, default: float) -> float:
        """Get given name log probability with default fallback."""
        return self.given_log_probabilities.get(given_key, default)

    def get_given_initial_logp(self, given_key: str) -> float:
        """Get log probability of a syllable in the first given-name position."""
        return self.given_initial_log_probabilities.get(given_key, self.given_initial_default_logp)

    def get_given_final_logp(self, given_key: str) -> float:
        """Get log probability of a syllable in the second given-name position."""
        return self.given_final_log_probabilities.get(given_key, self.given_final_default_logp)

    def get_surname_freq(self, surname_key: str, default: float = 0.0) -> float:
        """Get surname frequency with default fallback."""
        return self.surname_frequencies.get(surname_key, default)

    def resolve_surname_spelling(self, spelling: str) -> SurnameRomanization | None:
        """Resolve an as-written (norm_light) spelling against the curated romanization table."""
        return self.surname_romanizations.get(spelling)

    def surname_lookup_key(self, light_key: str, remapped_key: str) -> str:
        """Resolve a surname spelling to a single frequency-table lookup key.

        The as-written (norm_light) ``light_key`` wins when the spelling has a
        curated romanization-table row or carries direct surname frequency mass;
        otherwise the Wade-Giles/pinyin ``remapped_key`` applies. This is the
        resolver batch/routing evidence uses (paired with
        ``get_surname_freq_as_written``) so a spelling resolves to its attested
        as-written mass. Parser scoring deliberately does *not* call this: it
        keeps the remapped key for remap-only penalty spellings (e.g. fai, kung)
        and incidental as-written mass (e.g. cha) so its
        ``_surname_spelling_share_logp`` discount and comparative order scoring
        operate on the remapped target mass. Both sides reach the same effective
        surname mass, one via this as-written key, the other via the remapped key
        plus a discount; a single literal key cannot serve both.
        """
        if self.resolve_surname_spelling(light_key) is not None or self.get_surname_freq(light_key, 0.0) > 0:
            return light_key
        return remapped_key

    def get_surname_freq_as_written(self, spelling: str, default: float = 0.0) -> float:
        """Surname frequency for an as-written (norm_light) spelling.

        Curated table rows honor the mandarin target's mass (full-share rows) or
        the discounted as-written mass (penalty rows), even when the spelling's
        remapped form has incidental mass of its own. Spellings outside the
        table fall back to the runtime frequency table.
        """
        direct = self.surname_frequencies.get(spelling, 0.0)
        resolution = self.surname_romanizations.get(spelling)
        if resolution is not None:
            return max(direct, resolution.as_written_ppm)
        return direct if direct > 0 else default

    def get_surname_rank(self, surname_key: str, default: float = 0.0) -> float:
        """Get surname percentile rank with default fallback."""
        return self.surname_percentile_ranks.get(surname_key, default)

    def is_surname(self, token: str, normalized_token: str) -> bool:
        """Check if token is a surname using both original and normalized forms."""
        token_key = token.lower()
        return (
            token_key in self.surname_as_written_aliases
            or token in self.surnames
            or normalized_token in self.surnames
            or normalized_token in self.surnames_normalized
        )

    def is_given_name(self, normalized_token: str) -> bool:
        """Check if normalized token is a given name."""
        return normalized_token in self.given_names_normalized


class DataInitializationService:
    """Service to initialize all name data structures."""

    def __init__(self, config: ChineseNameConfig, cache_service, normalizer):
        self._config = config
        self._cache_service = cache_service
        self._normalizer = normalizer

        # Memoized Pinyin conversion for performance
        @cache
        def _pinyin_clean(han: str) -> str:
            """Cached Pinyin conversion with tone removal."""
            lst = cache_service.han_to_pinyin_fast(han)
            if not lst:
                return ""
            return "".join(c for c in lst[0].lower() if not c.isdigit())

        self._pinyin_clean = _pinyin_clean

    def initialize_data_structures(self) -> NameDataStructures:
        """Initialize all immutable data structures."""

        # Canonical romanization table: one load shared by scoring and evidence
        surname_romanizations = self._load_surname_romanizations()

        # Build core surname data
        surnames_raw, surname_frequencies, surname_as_written_aliases = self._build_surname_data(surname_romanizations)
        surnames = frozenset(StringManipulationUtils.remove_spaces(s.lower()) for s in surnames_raw)
        compound_surnames = frozenset(s.lower() for s in surnames_raw if " " in s)

        # Add all compound variants from COMPOUND_VARIANTS to ensure they're available
        compound_surnames_with_variants = set(compound_surnames)
        for standard_compound in COMPOUND_VARIANTS.values():
            compound_surnames_with_variants.add(standard_compound.lower())
        compound_surnames = frozenset(compound_surnames_with_variants)

        # Build normalized versions
        surnames_normalized = frozenset(StringManipulationUtils.remove_spaces(self._normalizer.norm(s)) for s in surnames_raw)
        compound_surnames_normalized = frozenset(self._normalizer.norm(s) for s in surnames_raw if " " in s)

        # Add normalized compound variants
        compound_surnames_normalized_with_variants = set(compound_surnames_normalized)
        for standard_compound in COMPOUND_VARIANTS.values():
            compound_surnames_normalized_with_variants.add(self._normalizer.norm(standard_compound))
        compound_surnames_normalized = frozenset(compound_surnames_normalized_with_variants)

        # Build given name data and plausible components
        given_names, given_log_probabilities, plausible_components = self._build_given_name_data()
        (
            given_initial_log_probabilities,
            given_final_log_probabilities,
            given_initial_default_logp,
            given_final_default_logp,
        ) = self._build_given_position_data()
        given_names_normalized = given_names  # Already normalized from pinyin data

        # Build compound surname mappings
        compound_hyphen_map = self._build_compound_hyphen_map(compound_surnames)
        compound_original_format_map = self._build_compound_original_format_map()

        # Build surname log probabilities
        surname_log_probabilities = self._build_surname_log_probabilities(
            surname_frequencies,
            compound_surnames,
            surname_romanizations,
        )

        # Build pre-computed percentile ranks for ML features
        surname_percentile_ranks = self._build_percentile_ranks(surname_frequencies)

        return NameDataStructures(
            surnames=surnames,
            surnames_normalized=surnames_normalized,
            compound_surnames=compound_surnames,
            compound_surnames_normalized=compound_surnames_normalized,
            given_names=given_names,
            given_names_normalized=given_names_normalized,
            plausible_components=plausible_components,
            surname_frequencies=MappingProxyType(dict(surname_frequencies)),
            surname_log_probabilities=MappingProxyType(dict(surname_log_probabilities)),
            given_log_probabilities=MappingProxyType(dict(given_log_probabilities)),
            given_initial_log_probabilities=MappingProxyType(dict(given_initial_log_probabilities)),
            given_final_log_probabilities=MappingProxyType(dict(given_final_log_probabilities)),
            given_initial_default_logp=given_initial_default_logp,
            given_final_default_logp=given_final_default_logp,
            surname_percentile_ranks=MappingProxyType(dict(surname_percentile_ranks)),
            compound_hyphen_map=MappingProxyType(dict(compound_hyphen_map)),
            compound_original_format_map=MappingProxyType(dict(compound_original_format_map)),
            surname_romanizations=MappingProxyType(surname_romanizations),
            surname_as_written_aliases=frozenset(surname_as_written_aliases),
        )

    def _is_plausible_chinese_syllable(self, component: str) -> bool:
        """
        Check if a component is a plausible Chinese syllable suitable for compound splitting.
        """
        if not component or len(component) > 7:
            return False

        component_lower = component.lower()
        if self._config.forbidden_patterns_regex.search(component_lower):
            return False

        # Prefer strict phonetic validation when available.
        if self._normalizer.is_valid_chinese_phonetics(component_lower):
            return True

        # Fallback: retain permissive coverage for known-valid entries while
        # still filtering clearly implausible components.
        if not component_lower.isalpha():
            return False

        vowels = set("aeiouü")
        if not any(ch in vowels for ch in component_lower):
            return False

        consonant_run = 0
        for ch in component_lower:
            if ch in vowels:
                consonant_run = 0
            else:
                consonant_run += 1
                if consonant_run >= 4:
                    return False

        return True

    def _load_surname_romanizations(self) -> dict[str, SurnameRomanization]:
        """Load surname_romanizations.csv into the canonical per-spelling map."""
        return {
            row["spelling"]: SurnameRomanization(
                mandarin_target=row["mandarin_target"],
                target_share=float(row["target_share"]),
                as_written_ppm=float(row["surname_ppm_as_written"]),
            )
            for row in open_csv_reader("surname_romanizations.csv")
        }

    def _build_surname_data(
        self,
        surname_romanizations: dict[str, SurnameRomanization],
    ) -> tuple[set[str], dict[str, float], set[str]]:
        """Build surname sets and frequency data."""
        surnames_raw = set()
        surname_frequencies = {}
        surname_as_written_aliases = set()

        for row in open_csv_reader("familyname_orcid.csv"):
            han = row["surname"]
            pinyin_list = self._cache_service.han_to_pinyin_fast(han)
            if pinyin_list:
                romanized = " ".join(pinyin_list).title()
                surnames_raw.update({romanized, StringManipulationUtils.remove_spaces(romanized)})
            else:
                continue

            # Store frequency data for both Chinese characters and romanized forms
            ppm = float(row.get("ppm", 0))

            # Store frequency for Chinese characters (original)
            surname_frequencies[han] = max(surname_frequencies.get(han, 0), ppm)

            # Store frequency for romanized form (existing behavior)
            freq_key = StringManipulationUtils.remove_spaces(romanized.lower())
            surname_frequencies[freq_key] = max(surname_frequencies.get(freq_key, 0), ppm)

        # Add frequency aliases where pypinyin output differs from expected romanization
        # These handle cases where Han characters produce different pinyin than the romanization system expects

        for pypinyin_key, alias_key in PYPINYIN_FREQUENCY_ALIASES:
            if pypinyin_key in surname_frequencies:
                surname_frequencies[alias_key] = max(
                    surname_frequencies.get(alias_key, 0),
                    surname_frequencies[pypinyin_key],
                )
                surnames_raw.add(alias_key.title())

        # Add Cantonese surnames (frequency aliasing driven by the romanization table).
        self._add_cantonese_surname_aliases(
            surname_romanizations,
            surnames_raw,
            surname_frequencies,
            surname_as_written_aliases,
        )

        # Add curated full-share romanization spellings as as-written surname aliases.
        # Remap-only penalty spellings (e.g. fai) stay out of the surname tables so
        # they still inherit discounted target mass through the parser's scoring hook;
        # Korean-dominant penalty Cantonese keys were already added above as attested
        # as-written spellings carrying their discounted frequency directly.
        for spelling, resolution in surname_romanizations.items():
            if resolution.target_share != 1.0:
                continue
            surnames_raw.add(spelling.title())
            surname_as_written_aliases.add(spelling)

        return surnames_raw, surname_frequencies, surname_as_written_aliases

    @staticmethod
    def _add_cantonese_surname_aliases(
        surname_romanizations: dict[str, SurnameRomanization],
        surnames_raw: set[str],
        surname_frequencies: dict[str, float],
        surname_as_written_aliases: set[str],
    ) -> None:
        """Alias each CANTONESE_SURNAMES key to a surname frequency, per the table.

        ``surname_romanizations.csv`` is the source of truth for how much of the
        Mandarin target's mass each single-token spelling keeps: full-share rows
        alias the target's full mass (pre-existing behavior), while penalty rows
        (Korean-dominant spellings trimmed by
        ``scripts/generate_surname_romanizations.py``, e.g. jung/moon/im) keep only
        the discounted as-written mass, scoring like genuinely rare Chinese surnames
        instead of their Mandarin targets. Every key is still an attested as-written
        surname; only its frequency differs.
        """
        for cant_surname, (mand_surname, _han_char) in CANTONESE_SURNAMES.items():
            surnames_raw.add(cant_surname.title())
            resolution = surname_romanizations.get(cant_surname)
            penalty = resolution is not None and resolution.target_share != 1.0
            source_freq = resolution.as_written_ppm if penalty else surname_frequencies.get(mand_surname.lower(), 0.0)
            if source_freq <= 0:
                continue
            surname_frequencies[cant_surname] = max(surname_frequencies.get(cant_surname, 0), source_freq)
            surname_as_written_aliases.add(cant_surname)

    def _build_given_name_data(self) -> tuple[frozenset[str], dict[str, float], frozenset[str]]:
        """Build given name data, log probabilities, and dynamically generate plausible components."""
        given_names = set()
        given_frequencies = {}

        for row in open_csv_reader("givenname_orcid.csv"):
            han_char = row["character"]
            # Strip tone markers from pinyin string
            normalized = unicodedata.normalize("NFKD", row["pinyin"])
            pinyin = self._config.digits_pattern.sub(
                "",
                "".join(c for c in normalized if not unicodedata.combining(c)),
            ).lower()
            given_names.add(pinyin)

            ppm = float(row.get("ppm", 0))
            if ppm > 0:
                # Store frequency for both Chinese character and pinyin
                given_frequencies[han_char] = max(given_frequencies.get(han_char, 0), ppm)
                given_frequencies[pinyin] = max(given_frequencies.get(pinyin, 0), ppm)

        # Keep denominator on the same aggregation basis as given_frequencies (max per key).
        total_given_freq = sum(given_frequencies.values())

        # Convert to log probabilities for both Chinese characters and pinyin
        given_log_probabilities = {}
        for given_name, freq in given_frequencies.items():
            prob = freq / total_given_freq if total_given_freq > 0 else 1e-15
            given_log_probabilities[given_name] = math.log(prob)

        # Generate plausible components dynamically from givenname_orcid.csv data
        # This replaces the static PLAUSIBLE_COMPONENTS with real-world usage data

        # Filter multi-syllable entries out of plausible_components
        # They leak in via manual supplements; restrict to ≤7 letters & exactly one onset–rime split
        # to avoid false "split-happy" behaviour with names like Weibian
        filtered_components = set()

        for component in given_names:
            # Check length constraint
            if len(component) > 7:
                continue

            # Check if component is actually usable for splitting
            # Some entries from givenname.csv might not be suitable for compound splitting
            # Use a more lenient approach: include if it passes basic phonetic validation
            # rather than strict onset-rime decomposition

            # Basic phonetic validation - check if it could plausibly be Chinese
            if self._is_plausible_chinese_syllable(component):
                filtered_components.add(component)

        plausible_components = frozenset(filtered_components)

        return frozenset(given_names), given_log_probabilities, plausible_components

    def _build_given_position_data(self) -> tuple[dict[str, float], dict[str, float], float, float]:
        """Build add-k smoothed positional log probabilities from given_position.csv.

        Keys are as-written romanized syllables (no Mandarin remapping), so
        Cantonese/Wade-Giles forms keep their own positional statistics. The
        only consumer (surname_first_delta in NameParsingService) takes
        differences of these values, so no baseline centering is needed.
        """
        smoothing_k = 0.5
        initial_counts: dict[str, int] = {}
        final_counts: dict[str, int] = {}
        for row in open_csv_reader("given_position.csv"):
            syllable = row["syllable"]
            initial_counts[syllable] = int(row["initial_count"])
            final_counts[syllable] = int(row["final_count"])

        vocabulary_size = len(initial_counts) + 1  # reserve one slot for unseen syllables
        initial_total = sum(initial_counts.values()) + smoothing_k * vocabulary_size
        final_total = sum(final_counts.values()) + smoothing_k * vocabulary_size

        initial_logp = {s: math.log((c + smoothing_k) / initial_total) for s, c in initial_counts.items()}
        final_logp = {s: math.log((c + smoothing_k) / final_total) for s, c in final_counts.items()}
        initial_default = math.log(smoothing_k / initial_total)
        final_default = math.log(smoothing_k / final_total)

        return initial_logp, final_logp, initial_default, final_default

    def _build_compound_hyphen_map(self, compound_surnames: frozenset[str]) -> dict[str, str]:
        """Build mapping for hyphenated compound surnames (stores lowercase keys only)."""
        compound_hyphen_map = {}

        for compound in compound_surnames:
            if " " in compound:
                parts = compound.split()
                if len(parts) == 2:
                    # Store only lowercase hyphenated form
                    lowercase_parts = [part.lower() for part in parts]
                    hyphen_form = StringManipulationUtils.join_with_hyphens(lowercase_parts)
                    # Store lowercase space form (will be title-cased on demand)
                    space_form = StringManipulationUtils.join_with_spaces(lowercase_parts)
                    compound_hyphen_map[hyphen_form] = space_form

        return compound_hyphen_map

    def _build_compound_original_format_map(self) -> dict[str, str]:
        """Build mapping from normalized compound surnames to original format."""
        compound_original_format_map = {}

        # Create reverse mapping from COMPOUND_VARIANTS to preserve original format
        # This maps the normalized spaced version back to the original compact form
        for original_form, normalized_form in COMPOUND_VARIANTS.items():
            # Store mapping from normalized form to original form
            compound_original_format_map[normalized_form.lower()] = original_form.lower()

            # Also create mapping from each component back to original when appropriate
            # This handles cases where we parse "duan mu" but want to output "duanmu"
            if " " in normalized_form:
                parts = normalized_form.lower().split()
                if len(parts) == 2:
                    # Only add this mapping if the original form doesn't contain spaces
                    # This preserves compound surnames like "duanmu" vs spaced forms like "au yeung"
                    if " " not in original_form:
                        joined_form = "".join(parts)
                        compound_original_format_map[joined_form] = original_form.lower()

        return compound_original_format_map

    def _build_surname_log_probabilities(
        self,
        surname_frequencies: dict[str, float],
        compound_surnames: frozenset[str],
        surname_romanizations: dict[str, SurnameRomanization],
    ) -> dict[str, float]:
        """Build surname log probabilities including compound surnames."""
        surname_log_probabilities = {}
        total_surname_freq = sum(surname_frequencies.values())
        compound_freq_delta = 0.0

        # Base surname probabilities
        for surname, freq in surname_frequencies.items():
            if freq > 0:
                prob = freq / total_surname_freq
                surname_log_probabilities[surname] = math.log(prob)
            else:
                surname_log_probabilities[surname] = self._config.default_surname_logp

        # Add compound surname probabilities
        for compound_surname in compound_surnames:
            parts = compound_surname.split()
            if len(parts) == 2:
                existing_freq = surname_frequencies.get(compound_surname, 0.0)
                if existing_freq > 0:
                    continue

                # Use reasonable fallback frequency for missing parts (1.0 instead of 1e-6)
                freq1 = surname_frequencies.get(parts[0], 1.0)
                freq2 = surname_frequencies.get(parts[1], 1.0)
                compound_freq = math.sqrt(freq1 * freq2) * self._config.compound_penalty

                # Apply minimum frequency floor to avoid extremely low scores
                min_compound_freq = 0.1  # Reasonable floor for compound surnames
                compound_freq = max(compound_freq, min_compound_freq)

                surname_frequencies[compound_surname] = compound_freq
                compound_freq_delta += compound_freq

        total_with_compounds = total_surname_freq + compound_freq_delta
        if total_with_compounds <= 0:
            total_with_compounds = total_surname_freq

        # Add/refresh compound surname probabilities using the updated denominator.
        for compound_surname in compound_surnames:
            compound_freq = surname_frequencies.get(compound_surname)
            if compound_freq is None or compound_freq <= 0:
                continue
            prob = compound_freq / total_with_compounds
            surname_log_probabilities[compound_surname] = math.log(prob)

        # Add frequency mappings for compound variants
        for variant_compound, standard_compound in COMPOUND_VARIANTS.items():
            if standard_compound in surname_log_probabilities:
                surname_log_probabilities[variant_compound] = surname_log_probabilities[standard_compound]
            if standard_compound in surname_frequencies:
                surname_frequencies[variant_compound] = surname_frequencies[standard_compound]

        # Add full-share as-written romanization aliases after the base
        # distribution is built. Each full-share spelling ends up carrying at
        # least the mandarin target's mass under its as-written key (even when
        # the remapped spelling has incidental mass of its own, e.g. chien ->
        # qian, not jian) without inflating the global denominator. Spellings
        # already at or above the target mass keep their own entries.
        for spelling, resolution in surname_romanizations.items():
            if resolution.target_share != 1.0 or resolution.as_written_ppm <= 0:
                continue
            if surname_frequencies.get(spelling, 0.0) >= resolution.as_written_ppm:
                continue
            surname_frequencies[spelling] = resolution.as_written_ppm
            surname_log_probabilities[spelling] = surname_log_probabilities.get(
                resolution.mandarin_target,
                math.log(resolution.as_written_ppm / total_with_compounds),
            )

        return surname_log_probabilities

    def _percentiles(self, vals: dict[str, float]) -> dict[str, float]:
        """Compute percentile ranks in O(n log n) time."""
        # Sort once by frequency (ascending - low to high)
        items = sorted(vals.items(), key=lambda kv: kv[1])
        n = len(items)
        out = {}
        for rank, (k, _) in enumerate(items):  # rank 0 = rarest
            out[k] = rank / (n - 1) if n > 1 else 0.0  # 0-1 scale
        return out

    def _build_percentile_ranks(self, surname_frequencies: dict[str, float]) -> dict[str, float]:
        """Build pre-computed percentile ranks for ML features (0-1 scale)."""
        return self._percentiles(surname_frequencies)
