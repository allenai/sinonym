"""
Data initialization service for Chinese name processing.

This module handles loading and preprocessing of Chinese name databases,
building frequency mappings, and creating immutable data structures.
"""
from __future__ import annotations

import csv
import math
import unicodedata
from dataclasses import dataclass

from sinonym.chinese_names_data import CANTONESE_SURNAMES, COMPOUND_VARIANTS
from sinonym.paths import DATA_PATH
from sinonym.types import ChineseNameConfig


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
    surname_frequencies: dict[str, float]
    surname_log_probabilities: dict[str, float]
    given_log_probabilities: dict[str, float]

    # Pre-computed surname bonuses for cultural plausibility scoring
    surname_bonus_map: dict[str, float]

    # Compound surname mappings
    compound_hyphen_map: dict[str, str]


class DataInitializationService:
    """Service to initialize all name data structures."""

    def __init__(self, config: ChineseNameConfig, cache_service, normalizer):
        self._config = config
        self._cache_service = cache_service
        self._normalizer = normalizer

    def initialize_data_structures(self) -> NameDataStructures:
        """Initialize all immutable data structures."""

        # Build core surname data
        surnames_raw, surname_frequencies = self._build_surname_data()
        surnames = frozenset(self._normalizer.remove_spaces(s.lower()) for s in surnames_raw)
        compound_surnames = frozenset(s.lower() for s in surnames_raw if " " in s)

        # Build normalized versions
        surnames_normalized = frozenset(self._normalizer.remove_spaces(self._normalizer.norm(s)) for s in surnames_raw)
        compound_surnames_normalized = frozenset(self._normalizer.norm(s) for s in surnames_raw if " " in s)

        # Build given name data and plausible components
        given_names, given_log_probabilities, plausible_components = self._build_given_name_data()
        given_names_normalized = given_names  # Already normalized from pinyin data

        # Build compound surname mappings
        compound_hyphen_map = self._build_compound_hyphen_map(compound_surnames)

        # Build surname log probabilities
        surname_log_probabilities = self._build_surname_log_probabilities(
            surname_frequencies,
            compound_surnames,
            compound_hyphen_map,
        )

        # Pre-compute surname bonuses for cultural plausibility scoring (micro-optimization)
        surname_bonus_map = self._build_surname_bonus_map(surname_frequencies)

        return NameDataStructures(
            surnames=surnames,
            surnames_normalized=surnames_normalized,
            compound_surnames=compound_surnames,
            compound_surnames_normalized=compound_surnames_normalized,
            given_names=given_names,
            given_names_normalized=given_names_normalized,
            plausible_components=plausible_components,
            surname_frequencies=surname_frequencies,
            surname_log_probabilities=surname_log_probabilities,
            given_log_probabilities=given_log_probabilities,
            surname_bonus_map=surname_bonus_map,
            compound_hyphen_map=compound_hyphen_map,
        )

    def _is_plausible_chinese_syllable(self, component: str) -> bool:
        """
        Check if a component is a plausible Chinese syllable suitable for compound splitting.
        Uses a more lenient approach than strict onset-rime decomposition to handle
        romanization variations and valid Chinese syllables.
        """
        if not component or len(component) > 7:
            return False

        # Reject components with forbidden Western patterns
        component_lower = component.lower()
        if self._config.forbidden_patterns_regex.search(component_lower):
            return False

        # Accept if it's a known Chinese syllable (from the given names database)
        # This handles cases like 'xue', 'yue', 'jue' which are valid Chinese syllables
        # even if they don't decompose cleanly in the onset-rime system we're using
        return True  # Since we're already filtering from given_names, they should be valid

    def _build_surname_data(self) -> tuple[set[str], dict[str, float]]:
        """Build surname sets and frequency data."""
        surnames_raw = set()
        surname_frequencies = {}

        with (DATA_PATH / "familyname_orcid.csv").open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                han = row["surname"]
                romanized = " ".join(self._cache_service.han_to_pinyin_fast(han)).title()
                surnames_raw.update({romanized, self._normalizer.remove_spaces(romanized)})

                # Store frequency data
                ppm = float(row.get("ppm.1930_2008", 0))
                freq_key = self._normalizer.remove_spaces(romanized.lower())
                surname_frequencies[freq_key] = max(surname_frequencies.get(freq_key, 0), ppm)

        # Add frequency alias: zeng should inherit ceng's frequency from Han character processing
        if "ceng" in surname_frequencies:
            surname_frequencies["zeng"] = surname_frequencies["ceng"]

        # Add Cantonese surnames
        for cant_surname, (mand_surname, han_char) in CANTONESE_SURNAMES.items():
            surnames_raw.add(cant_surname.title())
            # Use lowercase key to match the frequency mapping format
            mand_key = mand_surname.lower()
            if mand_key in surname_frequencies:
                surname_frequencies[cant_surname] = max(
                    surname_frequencies.get(cant_surname, 0),
                    surname_frequencies[mand_key],
                )

        return surnames_raw, surname_frequencies

    def _build_given_name_data(self) -> tuple[frozenset[str], dict[str, float], frozenset[str]]:
        """Build given name data, log probabilities, and dynamically generate plausible components."""
        given_names = set()
        given_frequencies = {}
        total_given_freq = 0

        with (DATA_PATH / "givenname_orcid.csv").open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pinyin = self._strip_tone(row["pinyin"])
                given_names.add(pinyin)

                ppm = float(row.get("name.ppm", 0))
                if ppm > 0:
                    given_frequencies[pinyin] = given_frequencies.get(pinyin, 0) + ppm
                    total_given_freq += ppm

        # Convert to log probabilities
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

    def _build_compound_hyphen_map(self, compound_surnames: frozenset[str]) -> dict[str, str]:
        """Build mapping for hyphenated compound surnames (stores lowercase keys only)."""
        compound_hyphen_map = {}

        for compound in compound_surnames:
            if " " in compound:
                parts = compound.split()
                if len(parts) == 2:
                    # Store only lowercase hyphenated form
                    hyphen_form = f"{parts[0].lower()}-{parts[1].lower()}"
                    # Store lowercase space form (will be title-cased on demand)
                    space_form = f"{parts[0].lower()} {parts[1].lower()}"
                    compound_hyphen_map[hyphen_form] = space_form

        return compound_hyphen_map

    def _build_surname_log_probabilities(
        self,
        surname_frequencies: dict[str, float],
        compound_surnames: frozenset[str],
        compound_hyphen_map: dict[str, str],
    ) -> dict[str, float]:
        """Build surname log probabilities including compound surnames."""
        surname_log_probabilities = {}
        total_surname_freq = sum(surname_frequencies.values())

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
                # Use reasonable fallback frequency for missing parts (1.0 instead of 1e-6)
                freq1 = surname_frequencies.get(parts[0], 1.0)
                freq2 = surname_frequencies.get(parts[1], 1.0)
                compound_freq = math.sqrt(freq1 * freq2) * self._config.compound_penalty

                # Apply minimum frequency floor to avoid extremely low scores
                min_compound_freq = 0.1  # Reasonable floor for compound surnames
                compound_freq = max(compound_freq, min_compound_freq)

                surname_frequencies[compound_surname] = compound_freq
                prob = compound_freq / total_surname_freq
                surname_log_probabilities[compound_surname] = math.log(prob)

        # Add frequency mappings for compound variants
        for variant_compound, standard_compound in COMPOUND_VARIANTS.items():
            if standard_compound in surname_log_probabilities:
                surname_log_probabilities[variant_compound] = surname_log_probabilities[standard_compound]
            if standard_compound in surname_frequencies:
                surname_frequencies[variant_compound] = surname_frequencies[standard_compound]

        return surname_log_probabilities

    def _build_surname_bonus_map(self, surname_frequencies: dict[str, float]) -> dict[str, float]:
        """Pre-compute surname bonuses for cultural plausibility scoring - performance optimization."""
        surname_bonus_map = {}

        for surname, freq in surname_frequencies.items():
            # Pre-compute the log10(freq+1)*1.2 calculation for fast lookup
            surname_bonus_map[surname] = math.log10(freq + 1) * 1.2

        return surname_bonus_map

    def _strip_tone(self, pinyin_str: str) -> str:
        """Strip tone markers from pinyin string."""
        normalized = unicodedata.normalize("NFKD", pinyin_str)
        return self._config.digits_pattern.sub(
            "",
            "".join(c for c in normalized if not unicodedata.combining(c)),
        ).lower()
