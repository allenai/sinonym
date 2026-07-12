"""Surname lookup resolver with explicit parser and evidence policies."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from sinonym.utils.string_manipulation import StringManipulationUtils

DOMINANT_CHINESE_SURNAME_FREQ_MIN = 10_000.0
_WADE_GILES_APOSTROPHE_SURNAME_RE = re.compile(r"(?:ch|ts|tz|k|p|t)'[a-z]+", re.IGNORECASE)
_APOSTROPHE_TRANSLATION = str.maketrans(
    {
        "\u02b9": "'",
        "\u02bb": "'",
        "\u02bc": "'",
        "\u2018": "'",
        "\u2019": "'",
        "\u2032": "'",
        "\uff07": "'",
        "`": "'",
        "\u00b4": "'",
    },
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sinonym.services.initialization import NameDataStructures, SurnameRomanization
    from sinonym.services.normalization import NormalizationService


class SurnameResolver:
    """Answer surname lookup questions under named parser and evidence policies.

    Args:
        data: Initialized name lookup data structures.
        normalizer: Normalization service used to derive lookup forms.
    """

    def __init__(self, data: NameDataStructures, normalizer: NormalizationService) -> None:
        self._data = data
        self._normalizer = normalizer
        self._parser_key_cache: dict[tuple[str, ...], str] = {}
        self._evidence_key_cache: dict[str, str] = {}
        self._wade_initial_remap_cache: dict[str, bool] = {}

    def parser_frequency(self, surname_tokens: Sequence[str]) -> float:
        """Return surname frequency under the parser-strict lookup policy."""
        return self._data.get_surname_freq(self._parser_key(surname_tokens))

    def parser_logp(self, surname_tokens: Sequence[str], default: float) -> float:
        """Return surname log probability under the parser-strict lookup policy."""
        return self._data.get_surname_logp(self._parser_key(surname_tokens), default)

    def parser_rank(self, surname_tokens: Sequence[str], default: float = 0.0) -> float:
        """Return surname percentile rank under the parser-strict lookup policy."""
        return self._data.get_surname_rank(self._parser_key(surname_tokens), default)

    def parser_is_surname(self, surname_tokens: Sequence[str]) -> bool:
        """Return whether the first parser surname token is recognized as a surname."""
        self._require_tokens(surname_tokens)
        return self._data.is_surname(surname_tokens[0], self._parser_key(surname_tokens))

    def evidence_frequency(self, token: str) -> float:
        """Return surname frequency under the as-written evidence lookup policy."""
        return self._data.get_surname_freq_as_written(self._evidence_key(token))

    def evidence_frequency_for_key(self, key: str) -> float:
        """Return surname frequency for an already-derived evidence lookup key."""
        return self._data.get_surname_freq_as_written(key)

    def evidence_is_surname(self, token: str) -> bool:
        """Return whether ``token`` is recognized under the evidence lookup policy."""
        return self._data.is_surname(token, self._evidence_key(token))

    def evidence_is_dominant_surname(self, token: str) -> bool:
        """Return whether ``token`` carries dominant as-written Chinese surname evidence."""
        return self.evidence_frequency(token) >= DOMINANT_CHINESE_SURNAME_FREQ_MIN

    def evidence_is_wade_giles_apostrophe_surname(self, token: str) -> bool:
        """Return exact-token Wade-Giles spelling backed by existing surname data."""
        normalized_apostrophe = token.translate(_APOSTROPHE_TRANSLATION)
        return bool(
            _WADE_GILES_APOSTROPHE_SURNAME_RE.fullmatch(normalized_apostrophe) and self.evidence_is_surname(token),
        )

    def evidence_span_key(self, token: str) -> str:
        """Return the evidence key for batch span assembly only."""
        return self._evidence_key(token)

    def spelling_for_light_key(self, spelling: str) -> SurnameRomanization | None:
        """Return the curated surname romanization row for an already norm_light spelling."""
        return self._data.resolve_surname_spelling(spelling)

    def spelling(self, token: str) -> SurnameRomanization | None:
        """Return the curated as-written surname romanization row for ``token``."""
        return self.spelling_for_light_key(self._normalizer.norm_light(token))

    def parser_is_wade_giles_initial_remapped_surname(self, token: str) -> bool:
        """Return whether parser policy keeps a Wade-Giles initial-remapped surname as-written."""
        return self._is_wade_giles_initial_remapped_surname_token(token)

    @staticmethod
    def _require_tokens(surname_tokens: Sequence[str]) -> None:
        if surname_tokens:
            return
        message = "surname_tokens must contain at least one token"
        raise ValueError(message)

    def _parser_key(self, surname_tokens: Sequence[str]) -> str:
        """Derive the parser-strict surname key used by parse scoring."""
        self._require_tokens(surname_tokens)
        cache_key = tuple(surname_tokens)
        if cache_key in self._parser_key_cache:
            return self._parser_key_cache[cache_key]

        if len(cache_key) == 1:
            parser_key = self._single_parser_key(cache_key[0])
        else:
            normalized_tokens = [self._normalizer.norm(token) for token in cache_key]
            parser_key = StringManipulationUtils.join_with_spaces(normalized_tokens)

        self._parser_key_cache[cache_key] = parser_key
        return parser_key

    def _single_parser_key(self, token: str) -> str:
        if self._contains_cjk(token) and token in self._data.surname_frequencies:
            return token

        light_key = self._normalizer.norm_light(token)
        resolution = self._data.resolve_surname_spelling(light_key)
        if resolution is not None and (resolution.target_share == 1.0 or light_key in self._data.surname_frequencies):
            return light_key
        if self.parser_is_wade_giles_initial_remapped_surname(token):
            return light_key

        original_key = self._normalizer.norm(token)
        if original_key in self._data.surname_frequencies:
            return original_key
        return StringManipulationUtils.remove_spaces(self._normalizer.norm(token))

    def _contains_cjk(self, token: str) -> bool:
        """Return whether ``token`` matches the normalizer's configured CJK pattern."""
        return self._normalizer.contains_cjk(token)

    def _evidence_key(self, token: str) -> str:
        """Derive the as-written surname evidence key."""
        if token not in self._evidence_key_cache:
            self._evidence_key_cache[token] = self._data.surname_lookup_key(
                self._normalizer.norm_light(token),
                self._normalizer.norm(token),
            )
        return self._evidence_key_cache[token]

    def _is_wade_giles_initial_remapped_surname_token(self, token: str) -> bool:
        """Return whether a direct surname was remapped by a Wade-Giles initial rule."""
        if token in self._wade_initial_remap_cache:
            return self._wade_initial_remap_cache[token]

        light_key = self._normalizer.norm_light(token)
        remapped_key = self._normalizer.norm(token)
        result = (
            light_key != remapped_key
            and bool(light_key)
            and bool(remapped_key)
            and light_key[0] in {"k", "p", "t"}
            and remapped_key[0] in {"g", "b", "d"}
            and self._data.get_surname_freq(light_key) > 0
            and self._data.get_surname_freq(remapped_key) == 0
        )
        self._wade_initial_remap_cache[token] = result
        return result
