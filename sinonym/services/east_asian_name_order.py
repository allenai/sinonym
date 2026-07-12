"""Conservative family-first routing for Japanese, Korean, and Vietnamese names."""

from __future__ import annotations

import gzip
import json
import re
import unicodedata
from bisect import bisect_left
from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise
from typing import TYPE_CHECKING, Any

from sinonym.chinese_names_data import (
    KOREAN_AMBIGUOUS_PATTERNS,
    KOREAN_GIVEN_PATTERNS,
    KOREAN_ONLY_SURNAMES,
    KOREAN_SPECIFIC_PATTERNS,
    NAME_ORDER_ROUTING_KOREAN_GIVEN_SYLLABLES,
    NAME_ORDER_ROUTING_KOREAN_SURNAMES,
    OVERLAPPING_KOREAN_SURNAMES,
    VIETNAMESE_ONLY_SURNAMES,
)
from sinonym.coretypes import NameComponents
from sinonym.resources import read_bytes

if TYPE_CHECKING:
    from collections.abc import Callable

JAPANESE_ML_THRESHOLD = 0.8
KOREAN_NATIVE_TOKEN_LENGTH = 3
MAX_ROMANIZED_TOKENS = 5
MAX_KOREAN_ROMANIZED_TOKENS = 3
MIN_ROMANIZED_TOKENS = 2
ROMAN_ASSET = "east_asian_roman_lexicons.json.gz"
NATIVE_ASSET = "japanese_native_lexicons.json.gz"
WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class EastAsianNameOrderDecision:
    """One high-confidence semantic component assignment."""

    surface: str
    given_tokens: tuple[str, ...]
    middle_tokens: tuple[str, ...]
    surname_tokens: tuple[str, ...]
    source_order: tuple[str, ...]
    reason: str

    @property
    def first_name(self) -> str:
        """Return the semantic given component."""
        return " ".join(self.given_tokens)

    @property
    def middle_name(self) -> str:
        """Return the semantic middle component."""
        return " ".join(self.middle_tokens)

    @property
    def last_name(self) -> str:
        """Return the semantic surname component."""
        return " ".join(self.surname_tokens)

    def source_components(self) -> NameComponents:
        """Return source-role lineage in the original family-first order."""
        return NameComponents(
            given_name=self.first_name,
            middle_name=self.middle_name,
            surname=self.last_name,
            suffix="",
            given_tokens=self.given_tokens,
            middle_tokens=self.middle_tokens,
            surname_tokens=self.surname_tokens,
            suffix_tokens=(),
            order=self.source_order,
        )


@dataclass(frozen=True)
class _RomanLexicons:
    japanese_surnames: tuple[str, ...]
    japanese_given_names: tuple[str, ...]
    korean_surnames: tuple[str, ...]
    vietnamese_surnames: tuple[str, ...]


@dataclass(frozen=True)
class _NativeLexicons:
    japanese_surnames: tuple[str, ...]
    japanese_given_names: tuple[str, ...]


def _load_payload(name: str) -> dict[str, Any]:
    """Load one strict gzip JSON package resource."""
    payload = json.loads(gzip.decompress(read_bytes(name)).decode("utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        message = f"unsupported East Asian lexicon schema in {name}"
        raise ValueError(message)
    return payload


def _validated_values(payload: dict[str, Any], key: str, asset: str) -> tuple[str, ...]:
    """Validate one sorted, unique string list at the package boundary."""
    values = payload.get(key)
    if not isinstance(values, list) or not all(isinstance(value, str) and value for value in values):
        message = f"invalid {key} in {asset}"
        raise ValueError(message)
    if any(left >= right for left, right in pairwise(values)):
        message = f"{key} must be sorted and unique in {asset}"
        raise ValueError(message)
    return tuple(values)


@lru_cache(maxsize=1)
def _roman_lexicons() -> _RomanLexicons:
    payload = _load_payload(ROMAN_ASSET)
    external_korean = _validated_values(payload, "korean_surnames", ROMAN_ASSET)
    korean = sorted(
        set(external_korean) | NAME_ORDER_ROUTING_KOREAN_SURNAMES | KOREAN_ONLY_SURNAMES | OVERLAPPING_KOREAN_SURNAMES,
    )
    vietnamese = sorted(
        set(_validated_values(payload, "vietnamese_surnames", ROMAN_ASSET))
        | {_fold(value) for value in VIETNAMESE_ONLY_SURNAMES},
    )
    return _RomanLexicons(
        japanese_surnames=_validated_values(payload, "japanese_surnames", ROMAN_ASSET),
        japanese_given_names=_validated_values(payload, "japanese_given_names", ROMAN_ASSET),
        korean_surnames=tuple(korean),
        vietnamese_surnames=tuple(vietnamese),
    )


@lru_cache(maxsize=1)
def _native_lexicons() -> _NativeLexicons:
    payload = _load_payload(NATIVE_ASSET)
    return _NativeLexicons(
        japanese_surnames=_validated_values(payload, "japanese_surnames", NATIVE_ASSET),
        japanese_given_names=_validated_values(payload, "japanese_given_names", NATIVE_ASSET),
    )


def _contains(values: tuple[str, ...], key: str) -> bool:
    index = bisect_left(values, key)
    return index < len(values) and values[index] == key


def _fold(value: str) -> str:
    translated = value.translate(str.maketrans({"\u0110": "D", "\u0111": "d"}))
    return "".join(
        character for character in unicodedata.normalize("NFD", translated).casefold() if not unicodedata.combining(character)
    )


def _japanese_roman_keys(value: str) -> tuple[str, ...]:
    exact = _fold(value)
    collapsed = exact.replace("ou", "o").replace("oo", "o").replace("uu", "u")
    return (exact,) if exact == collapsed else (exact, collapsed)


def _contains_any(values: tuple[str, ...], keys: tuple[str, ...]) -> bool:
    return any(_contains(values, key) for key in keys)


def _is_han(character: str) -> bool:
    return "\u3400" <= character <= "\u4dbf" or "\u4e00" <= character <= "\u9fff" or "\uf900" <= character <= "\ufaff"


def _is_kana(character: str) -> bool:
    return "\u3040" <= character <= "\u30ff" or "\u31f0" <= character <= "\u31ff"


def _is_hangul(value: str) -> bool:
    return bool(value) and all("\uac00" <= character <= "\ud7a3" for character in value)


def _is_compact_japanese(value: str) -> bool:
    return bool(value) and " " not in value and all(_is_han(character) or _is_kana(character) for character in value)


def _han_to_kana_boundary(value: str) -> int | None:
    index = 0
    while index < len(value) and _is_han(value[index]):
        index += 1
    if index and index < len(value) and all(_is_kana(character) for character in value[index:]):
        return index
    return None


class EastAsianNameOrderService:
    """Infer only the family-first cases supported by conservative evidence."""

    def infer(
        self,
        raw_name: str,
        *,
        japanese_probability: Callable[[str], float],
    ) -> EastAsianNameOrderDecision | None:
        """Return semantic components or abstain without changing visible order."""
        surface = WHITESPACE_RE.sub(" ", unicodedata.normalize("NFKC", raw_name)).strip()
        if not surface or "," in surface:
            return None

        native = self._infer_native(surface, japanese_probability)
        if native is not None:
            return native
        return self._infer_romanized(surface)

    def _infer_native(
        self,
        surface: str,
        japanese_probability: Callable[[str], float],
    ) -> EastAsianNameOrderDecision | None:
        if _is_hangul(surface):
            if len(surface) != KOREAN_NATIVE_TOKEN_LENGTH:
                return None
            return EastAsianNameOrderDecision(
                surface=surface,
                given_tokens=(surface[1:],),
                middle_tokens=(),
                surname_tokens=(surface[:1],),
                source_order=("surname", "given"),
                reason="korean_native_three_syllable",
            )
        if not _is_compact_japanese(surface) or japanese_probability(surface) < JAPANESE_ML_THRESHOLD:
            return None
        boundary = self._japanese_native_boundary(surface)
        return EastAsianNameOrderDecision(
            surface=surface,
            given_tokens=(surface[boundary:],),
            middle_tokens=(),
            surname_tokens=(surface[:boundary],),
            source_order=("surname", "given"),
            reason="japanese_native_dictionary",
        )

    @staticmethod
    def _japanese_native_boundary(surface: str) -> int:
        transition = _han_to_kana_boundary(surface)
        if transition is not None:
            return transition
        lexicons = _native_lexicons()
        default_boundary = 1 if len(surface) == 2 else 2  # noqa: PLR2004
        candidates: list[tuple[int, int, int, int]] = []
        for boundary in range(1, len(surface)):
            score = (
                4 * _contains(lexicons.japanese_surnames, surface[:boundary])
                + 2 * _contains(lexicons.japanese_given_names, surface[boundary:])
                + (boundary == default_boundary)
            )
            candidates.append((score, -abs(boundary - default_boundary), -boundary, boundary))
        return max(candidates)[-1]

    def _infer_romanized(self, surface: str) -> EastAsianNameOrderDecision | None:
        tokens = surface.split()
        if not MIN_ROMANIZED_TOKENS <= len(tokens) <= MAX_ROMANIZED_TOKENS:
            return None
        if not all(all(character.isalpha() or character in "-'" for character in token) for token in tokens):
            return None
        lexicons = _roman_lexicons()

        vietnamese = self._infer_vietnamese(tokens, surface, lexicons)
        if vietnamese is not None:
            return vietnamese
        korean = self._infer_korean(tokens, lexicons)
        if korean is not None:
            return korean
        return self._infer_japanese_romanized(tokens, lexicons)

    @staticmethod
    def _infer_vietnamese(
        tokens: list[str],
        surface: str,
        lexicons: _RomanLexicons,
    ) -> EastAsianNameOrderDecision | None:
        if surface.isascii() or not _contains(lexicons.vietnamese_surnames, _fold(tokens[0])):
            return None
        middle_tokens = tuple(tokens[1:-1])
        return EastAsianNameOrderDecision(
            surface=surface,
            given_tokens=(tokens[-1],),
            middle_tokens=middle_tokens,
            surname_tokens=(tokens[0],),
            source_order=("surname", *("middle" for _ in middle_tokens), "given"),
            reason="vietnamese_unicode_surname_first",
        )

    @staticmethod
    def _infer_korean(
        tokens: list[str],
        lexicons: _RomanLexicons,
    ) -> EastAsianNameOrderDecision | None:
        if len(tokens) > MAX_KOREAN_ROMANIZED_TOKENS:
            return None
        if not _contains(lexicons.korean_surnames, _fold(tokens[0])):
            return None
        if _contains(lexicons.korean_surnames, _fold(tokens[-1])):
            return None
        known_given = {
            _fold(value)
            for value in (
                NAME_ORDER_ROUTING_KOREAN_GIVEN_SYLLABLES
                | KOREAN_GIVEN_PATTERNS
                | KOREAN_SPECIFIC_PATTERNS
                | KOREAN_AMBIGUOUS_PATTERNS
            )
        }
        given_parts = [_fold(part) for token in tokens[1:] for part in token.split("-") if part]
        has_hyphen = any("-" in token for token in tokens[1:])
        all_known = bool(given_parts) and all(part in known_given for part in given_parts)
        if not (has_hyphen or (len(tokens) == MAX_KOREAN_ROMANIZED_TOKENS and all_known)):
            return None
        given_tokens = tuple(tokens[1:])
        return EastAsianNameOrderDecision(
            surface=" ".join(tokens),
            given_tokens=given_tokens,
            middle_tokens=(),
            surname_tokens=(tokens[0],),
            source_order=("surname", *("given" for _ in given_tokens)),
            reason="korean_romanized_strict",
        )

    @staticmethod
    def _infer_japanese_romanized(
        tokens: list[str],
        lexicons: _RomanLexicons,
    ) -> EastAsianNameOrderDecision | None:
        if len(tokens) != 2:  # noqa: PLR2004
            return None
        first_keys = _japanese_roman_keys(tokens[0])
        last_keys = _japanese_roman_keys(tokens[1])
        surname_first = _contains_any(lexicons.japanese_surnames, first_keys) and _contains_any(
            lexicons.japanese_given_names,
            last_keys,
        )
        reverse_plausible = _contains_any(lexicons.japanese_given_names, first_keys) and _contains_any(
            lexicons.japanese_surnames,
            last_keys,
        )
        if not surname_first or reverse_plausible:
            return None
        return EastAsianNameOrderDecision(
            surface=" ".join(tokens),
            given_tokens=(tokens[1],),
            middle_tokens=(),
            surname_tokens=(tokens[0],),
            source_order=("surname", "given"),
            reason="japanese_romanized_directional_dictionary",
        )


__all__ = ["EastAsianNameOrderDecision", "EastAsianNameOrderService"]
