"""Conservative family-first routing for Japanese, Korean, and Vietnamese names."""

from __future__ import annotations

import gzip
import json
import math
import re
import unicodedata
from bisect import bisect_left
from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise
from typing import TYPE_CHECKING, Any

from sinonym.chinese_names_data import (
    COMPATIBILITY_IDEOGRAPH_FOLDS,
    KOREAN_AMBIGUOUS_PATTERNS,
    KOREAN_GIVEN_PATTERNS,
    KOREAN_ONLY_SURNAMES,
    KOREAN_SPECIFIC_PATTERNS,
    NAME_ORDER_ROUTING_COMMON_CHINESE_SURNAMES,
    NAME_ORDER_ROUTING_KOREAN_GIVEN_SYLLABLES,
    NAME_ORDER_ROUTING_KOREAN_SURNAMES,
    OVERLAPPING_KOREAN_SURNAMES,
    VIETNAMESE_ONLY_SURNAMES,
)
from sinonym.coretypes import NameComponents
from sinonym.coretypes.routing_resolution import EastAsianEvidenceReason, EvidenceFailure, ResolutionReason
from sinonym.resources import read_bytes
from sinonym.text_processing.text_normalizer import (
    exact_name_surface_key,
    is_name_variation_selector,
    strip_name_variation_selectors,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# Membership is per-token and earned by measurement, not by being on the Vietnamese surname list.
# The bar is that the token is near-exclusively a surname: where both tokens of a two-token name are
# surnames, the route has to pick the leading one, and these are wrong there only 1.8-6.8% of the
# time. `hoang` fails that bar at 66.7% ("Hoang Nguyen", "Hoang Pham" are given-name-first), and the
# short entries fail it outright.
ASCII_ROUTABLE_VIETNAMESE_SURNAMES = frozenset({"nguyen", "pham", "tran"})
# The pinned Vietnamese-surname/Japanese-given overlap is {do, mai, to} after
# excluding Korean surnames. ``do`` is also a Portuguese particle and an
# ambiguous credential, and its audited activations were malformed or
# non-Japanese. Freeze the positively adjudicated signal rather than allowing
# future lexicon additions to widen the rule silently.
CROSS_CULTURAL_CONFLICT_HEADS = frozenset({"mai", "to"})
VIETNAMESE_TOP_SURNAME_COUNT = 4
# These reviewed source-order exceptions are deliberately exact. Generalizing
# either shape would override correct family-first names in the same lexicons.
JAPANESE_GIVEN_FIRST_EXACT_SURFACES = frozenset({"智幸 小枝"})
VIETNAMESE_GIVEN_FIRST_EXACT_SURFACES = frozenset({"tuan le"})
# The sole corpus identity is independently verified and already receives this
# reason when supportive Japanese paper context is present.  Exact matching
# extends the same decision to singleton/PP-only use without changing that
# production result.
CROSS_CULTURAL_GIVEN_FIRST_EXACT_SURFACES = frozenset({"to keku"})
# These two released exact surfaces also suppress an early Japanese
# family-first inference. Newly censused surfaces are post-selection vetoes so
# already-correct context-supported parses keep their existing public metadata.
JAPANESE_PRESELECTION_GIVEN_FIRST_EXACT_SURFACES = frozenset({"kou hiroya", "takaya miwa"})
# Exact lexical false friends whose reviewed PP/VYS flips are family-first when
# another complete paper author supplies context. Token-level exclusions would
# suppress verified Japanese names such as Ma Kai. Yuan Tai is deliberately
# absent because its complete corpus census supports given-first order.
JAPANESE_FAMILY_FIRST_CONFLICT_SURFACES = frozenset({"gan kai", "shi kai", "yu mi"})
# The same bar, met only once the two-surname shape above is excluded rather than absorbed: these
# heads are absent from the Korean and Japanese lexicons, so admitting them preempts no later route,
# and outside that shape blind labelling puts the leading token in the surname. Corpus-wide they move
# 32,071 bare-ASCII names / 142,756 mentions that read "Van Minh" or "Huu Tai" as the surname today.
ASCII_ROUTABLE_VIETNAMESE_SURNAMES_WITHOUT_SURNAME_PARTNER = frozenset(
    {
        "bui",
        "dam",
        "dang",
        "dinh",
        "doan",
        "duong",
        "hoang",
        "huynh",
        "khuc",
        "luong",
        "luu",
        "ngo",
        "phan",
        "phi",
        "phung",
        "thach",
        "thuong",
        "trac",
        "trieu",
        "trinh",
        "truong",
        "quach",
        "tieu",
        "vo",
        "vu",
        "vuong",
    },
)
# Blind family-name judgments and identity evidence identified these exact
# surfaces as given-first. Keep the evidence surface-specific: tokens such as
# `Hy` are also valid given names in family-first names such as `Nguyen Hy`.
VIETNAMESE_GIVEN_FIRST_CONFLICT_SURFACES = frozenset(
    {
        "bùi hoàng thảo trần",
        "dam smith",
        "dam sunwoo",
        "doan nainggolan",
        "doan nugyen",
        "doan perdana",
        "hoá nguyễn",
        "hoang bao khanh chu",
        "huynh lien buia",
        "luu n'guyen",
        "nguyen van nhi tran",
        "phan datthuyawat",
        "phi goy",
        "thach tungnguyen",
        "tran duc le",
        "truong nghiem",
        "truong son hy",
        "vo ebuara",
        "vu sudakov",
    },
)
# Identity-backed exact assignments for corpus spellings whose correct roles
# cannot be inferred safely from token-level lexicons. Broad interior-surname
# and spaced-compound rules both fail on mostly-correct control populations.
IDENTITY_BACKED_EXACT_ROLES = {
    "huong yong ting": ("given", "given", "surname"),
    "ming hsien ou yang": ("given", "middle", "surname", "surname"),
    "shiu lun au yeung": ("given", "middle", "surname", "surname"),
    "thuong le thi": ("given", "surname", "middle"),
    "trinh nguyen duy": ("middle", "surname", "given"),
}
HOMOGRAPH_PRONE_SURNAME_LENGTH = 2
JAPANESE_ITERATION_MARK = "\u3005"
JAPANESE_ML_THRESHOLD = 0.8
JAPANESE_MARKED_SURNAME_LENGTH = 3
MIN_JAPANESE_ITERATION_COMPLEMENT_LENGTH = 1
MAX_JAPANESE_ITERATION_COMPLEMENT_LENGTH = 3
KOREAN_NATIVE_TOKEN_LENGTH = 3
# 남궁 / 황보 / 제갈 / 사공 / 선우 / 서문 / 독고: the surname occupies two of the three syllables, so the
# default 1+2 split lands inside it. Blind labelling of the class put the boundary after the second
# syllable on 19 of 25 items and called the rest genuinely ambiguous — never after the first.
KOREAN_NATIVE_COMPOUND_SURNAMES = frozenset({"남궁", "황보", "제갈", "사공", "선우", "서문", "독고"})
KOREAN_NATIVE_COMPOUND_SURNAME_LENGTH = 2
MAX_KOREAN_GIVEN_SYLLABLE_LENGTH = 6
MAX_ROMANIZED_TOKENS = 5
MAX_KOREAN_ROMANIZED_TOKENS = 3
MIN_ROMANIZED_TOKENS = 2
MIN_WESTERN_SUFFIX_TOKEN_LENGTH = 5
# Frozen before the fresh validation sample. These endings are positive
# Western-family evidence only when the proposed Korean given side has no
# known syllable evidence.
WESTERN_SURNAME_SUFFIXES = (
    "sen",
    "son",
    "sson",
    "ssen",
    "berg",
    "bergh",
    "lund",
    "gren",
    "strom",
    "quist",
    "qvist",
    "gaard",
    "gard",
    "holm",
    "dahl",
    "stad",
    "borg",
    "mann",
    "stein",
    "feld",
    "bauer",
    "meyer",
    "meier",
    "schmidt",
    "burg",
    "burger",
    "smith",
    "well",
    "ford",
    "wood",
    "field",
    "stone",
    "white",
    "worth",
    "ley",
    "ridge",
    "shaw",
    "cliffe",
    "combe",
)
KOREAN_ROMANIZATION_BREVE_LETTERS = frozenset({"Ŏ", "ŏ", "Ŭ", "ŭ"})
ROMAN_ASSET = "east_asian_roman_lexicons.json.gz"
NATIVE_ASSET = "japanese_native_lexicons.json.gz"
WHITESPACE_RE = re.compile(r"\s+")


def _normalized_surface(value: str) -> str:
    """Collapse whitespace and apply NFKC for exact-surface comparison."""
    return WHITESPACE_RE.sub(" ", unicodedata.normalize("NFKC", value)).strip()


@dataclass(frozen=True)
class EastAsianNameOrderDecision:
    """One high-confidence semantic component assignment."""

    surface: str
    given_tokens: tuple[str, ...]
    middle_tokens: tuple[str, ...]
    surname_tokens: tuple[str, ...]
    source_order: tuple[str, ...]
    reason: EastAsianEvidenceReason

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
class EastAsianNameOrderPreservation:
    """A proven veto that keeps the generic scalar baseline."""

    surface: str
    reason: EastAsianEvidenceReason


@dataclass(frozen=True)
class _RomanLexicons:
    japanese_surnames: tuple[str, ...]
    japanese_possible_surnames: tuple[str, ...]
    japanese_given_first_exact_surfaces: tuple[str, ...]
    japanese_given_names: tuple[str, ...]
    korean_surnames: tuple[str, ...]
    vietnamese_surnames: tuple[str, ...]
    vietnamese_top4_surnames: tuple[str, ...]


@dataclass(frozen=True)
class _NativeLexicons:
    japanese_surnames: tuple[str, ...]
    japanese_given_names: tuple[str, ...]


def _load_payload(name: str) -> dict[str, Any]:
    """Load one strict gzip JSON package resource."""
    payload = json.loads(gzip.decompress(read_bytes(name)).decode("utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 3:  # noqa: PLR2004 - asset contract
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
    japanese_surnames = _validated_values(payload, "japanese_surnames", ROMAN_ASSET)
    japanese_possible_surnames = _validated_values(payload, "japanese_possible_surnames", ROMAN_ASSET)
    if not set(japanese_surnames).issubset(japanese_possible_surnames):
        message = "japanese_surnames must be a subset of japanese_possible_surnames"
        raise ValueError(message)
    external_korean = _validated_values(payload, "korean_surnames", ROMAN_ASSET)
    korean = sorted(
        set(external_korean) | NAME_ORDER_ROUTING_KOREAN_SURNAMES | KOREAN_ONLY_SURNAMES | OVERLAPPING_KOREAN_SURNAMES,
    )
    external_vietnamese = _validated_values(payload, "vietnamese_surnames", ROMAN_ASSET)
    vietnamese_top4 = _validated_values(payload, "vietnamese_top4_surnames", ROMAN_ASSET)
    if len(vietnamese_top4) != VIETNAMESE_TOP_SURNAME_COUNT:
        message = (
            f"vietnamese_top4_surnames must contain exactly {VIETNAMESE_TOP_SURNAME_COUNT} entries; got {len(vietnamese_top4)}"
        )
        raise ValueError(message)
    if not set(vietnamese_top4).issubset(external_vietnamese):
        message = "vietnamese_top4_surnames must be a subset of the pinned Vietnamese surname asset"
        raise ValueError(message)
    vietnamese = sorted(set(external_vietnamese) | {_fold(value) for value in VIETNAMESE_ONLY_SURNAMES})
    return _RomanLexicons(
        japanese_surnames=japanese_surnames,
        japanese_possible_surnames=japanese_possible_surnames,
        japanese_given_first_exact_surfaces=_validated_values(
            payload,
            "japanese_given_first_exact_surfaces",
            ROMAN_ASSET,
        ),
        japanese_given_names=_validated_values(payload, "japanese_given_names", ROMAN_ASSET),
        korean_surnames=tuple(korean),
        vietnamese_surnames=tuple(vietnamese),
        vietnamese_top4_surnames=vietnamese_top4,
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


def _iteration_surname_only(
    value: str,
    lexicons: _NativeLexicons,
) -> bool:
    """Return whether a marked source span is exclusively a known surname."""
    key = _native_lookup_text(value)
    return (
        JAPANESE_ITERATION_MARK in value
        and _contains(lexicons.japanese_surnames, key)
        and not _contains(lexicons.japanese_given_names, key)
    )


def _one_sided_iteration_surname_evidence(
    marked: str,
    complement: str,
    lexicons: _NativeLexicons,
) -> bool:
    """Return whether one positive role plus two negative vetoes prove the marked surname."""
    marked_key = _native_lookup_text(marked)
    complement_key = _native_lookup_text(complement)
    if (
        len(marked_key) != JAPANESE_MARKED_SURNAME_LENGTH
        or not MIN_JAPANESE_ITERATION_COMPLEMENT_LENGTH <= len(complement_key) <= MAX_JAPANESE_ITERATION_COMPLEMENT_LENGTH
        or not all(any(_is_han(character) or _is_kana(character) for character in span) for span in (marked_key, complement_key))
    ):
        return False

    marked_is_surname = _contains(lexicons.japanese_surnames, marked_key)
    marked_is_given = _contains(lexicons.japanese_given_names, marked_key)
    complement_is_surname = _contains(lexicons.japanese_surnames, complement_key)
    complement_is_given = _contains(lexicons.japanese_given_names, complement_key)
    return not marked_is_given and not complement_is_surname and (marked_is_surname or complement_is_given)


def _one_sided_iteration_mark_decision(
    surface: str,
    tokens: list[str],
    lexicons: _NativeLexicons,
) -> EastAsianNameOrderDecision | None:
    """Build the spaced-only decision after the strict rule has abstained."""
    if len(tokens) != MIN_ROMANIZED_TOKENS:
        return None
    marked_indices = [index for index, token in enumerate(tokens) if JAPANESE_ITERATION_MARK in token]
    if len(marked_indices) != 1:
        return None
    marked_index = marked_indices[0]
    complement_index = 1 - marked_index
    marked = tokens[marked_index]
    complement = tokens[complement_index]
    if not _one_sided_iteration_surname_evidence(marked, complement, lexicons):
        return None
    return EastAsianNameOrderDecision(
        surface=surface,
        given_tokens=(complement,),
        middle_tokens=(),
        surname_tokens=(marked,),
        source_order=("surname", "given") if marked_index == 0 else ("given", "surname"),
        reason=EastAsianEvidenceReason.JAPANESE_ITERATION_MARK_ONE_SIDED_EXCLUSIVE,
    )


def _select_iteration_mark_decision(
    surface: str,
    tokens: list[str],
    lexicons: _NativeLexicons,
    strict_candidates: list[EastAsianNameOrderDecision],
) -> EastAsianNameOrderDecision | None:
    """Preserve one strict result and extend only after a clean strict abstention."""
    if len(strict_candidates) == 1:
        return strict_candidates[0]
    if strict_candidates:
        return None
    return _one_sided_iteration_mark_decision(surface, tokens, lexicons)


def _japanese_given_only(
    value: str,
    lexicons: _NativeLexicons,
) -> bool:
    """Return whether a source span is exclusively a known given name."""
    key = _native_lookup_text(value)
    return _contains(
        lexicons.japanese_given_names,
        key,
    ) and not _contains(lexicons.japanese_surnames, key)


_COMPATIBILITY_FOLD_TABLE = str.maketrans(COMPATIBILITY_IDEOGRAPH_FOLDS)


def _native_lookup_text(value: str) -> str:
    """Return selector-free, compatibility-folded native lookup text."""
    return strip_name_variation_selectors(value.translate(_COMPATIBILITY_FOLD_TABLE))


def _japanese_classifier_input(surface: str) -> str:
    """Return the deployed classifier's exact native-script input surface.

    Compatibility ideographs and variation selectors are lookup-only, while
    authored spacing is deliberately retained. Changing that representation
    requires a separately measured classifier migration rather than an
    incidental name-order rule change.
    """
    return _native_lookup_text(surface)


def _fold(value: str) -> str:
    if value.isascii():
        return value.casefold()
    translated = value.translate(str.maketrans({"\u0110": "D", "\u0111": "d"}))
    return "".join(
        character for character in unicodedata.normalize("NFD", translated).casefold() if not unicodedata.combining(character)
    )


def _endpoint_key(value: str) -> str:
    """Fold case, accents, and punctuation for exact endpoint comparison."""
    return "".join(character for character in _fold(unicodedata.normalize("NFKC", value)) if character.isalnum())


KOREAN_ROUTING_GIVEN_PARTS = frozenset(
    _fold(value)
    for value in (
        NAME_ORDER_ROUTING_KOREAN_GIVEN_SYLLABLES | KOREAN_GIVEN_PATTERNS | KOREAN_SPECIFIC_PATTERNS | KOREAN_AMBIGUOUS_PATTERNS
    )
)


def _japanese_roman_keys(value: str) -> tuple[str, ...]:
    exact = _fold(value)
    collapsed = exact.replace("ou", "o").replace("oo", "o").replace("uu", "u")
    return (exact,) if exact == collapsed else (exact, collapsed)


def _contains_any(values: tuple[str, ...], keys: tuple[str, ...]) -> bool:
    return any(_contains(values, key) for key in keys)


# Precomposed European letters with no base+combining-mark decomposition.
_EUROPEAN_EXCLUSIVE_LETTERS = frozenset("øØæÆœŒßþÞðÐłŁ")
# Combining marks that Vietnamese/East-Asian romanization never uses, so they are
# unambiguous European evidence regardless of base: diaeresis (ä/ö/ü/ë/ï), ring (å),
# cedilla (ç), ogonek (ą/ę), double acute (ő/ű).
_EUROPEAN_EXCLUSIVE_MARKS = frozenset({"̈", "̊", "̧", "̨", "̋"})
# Marks Vietnamese/McCune only ever place on a vowel; on a consonant they are European —
# acute → ć/ń/ś/ź, tilde → ñ, caron → č/ž/š, breve → Turkish ğ. (On a vowel these are
# routable: acute/tilde are Vietnamese tones, breve is McCune ŏ/ŭ, caron is only ever
# mojibake Vietnamese.)
_CONSONANT_EUROPEAN_MARKS = frozenset({"́", "̃", "̌", "̆"})
_VIETNAMESE_VOWELS = frozenset("aeiouy")


def _is_european_exclusive_diacritic(character: str) -> bool:
    """True if a character is a European-exclusive accented Latin letter.

    Only marks that cannot appear in Vietnamese/Korean/Japanese romanization count as
    European evidence (Nordic ø/å, German ä/ö/ü, Spanish ñ, Polish ć/ł, Czech č/ž). The
    Vietnamese repertoire (tone/quality marks on vowels, horn, đ), Hepburn/McCune macrons
    (ā/ō/ū/ŏ/ŭ), and non-European noise (Turkish İ, Cyrillic homoglyphs, mojibake like ƣ)
    are all left routable, so a genuine East-Asian name is never blocked by a stray glyph.
    """
    if character in _EUROPEAN_EXCLUSIVE_LETTERS:
        return True
    decomposed = unicodedata.normalize("NFD", character)
    marks = decomposed[1:]
    if not marks:
        return False
    if any(mark in _EUROPEAN_EXCLUSIVE_MARKS for mark in marks):
        return True
    base = decomposed[0].casefold()
    return base not in _VIETNAMESE_VOWELS and any(mark in _CONSONANT_EUROPEAN_MARKS for mark in marks)


def _has_east_asian_diacritic_evidence(surface: str) -> bool:
    """True when the surface has non-ASCII letters and none is European-exclusive.

    A single European-exclusive letter (ø/å/ö/ü/ñ/…) disqualifies the surface, so
    "Kim Brøsen" is not mistaken for Vietnamese; mojibake/Turkish/Cyrillic glyphs on an
    otherwise Vietnamese name (e.g. "Nguyễn Lan Hƣơng") stay routable.
    """
    return (not surface.isascii()) and not any(_is_european_exclusive_diacritic(character) for character in surface)


def _is_han(character: str) -> bool:
    return "\u3400" <= character <= "\u4dbf" or "\u4e00" <= character <= "\u9fff" or "\uf900" <= character <= "\ufaff"


def _is_kana(character: str) -> bool:
    return "\u3040" <= character <= "\u30ff" or "\u31f0" <= character <= "\u31ff"


def _is_hangul(value: str) -> bool:
    return bool(value) and all("\uac00" <= character <= "\ud7a3" for character in value)


def _is_compact_japanese(value: str) -> bool:
    lookup = _native_lookup_text(value)
    return bool(lookup) and " " not in lookup and all(_is_han(character) or _is_kana(character) for character in lookup)


def _source_index_after_lookup_boundary(surface: str, lookup_boundary: int) -> int:
    """Map a selector-free character boundary back onto the authored surface."""
    lookup_characters = 0
    for index, character in enumerate(surface):
        if is_name_variation_selector(character):
            continue
        lookup_characters += 1
        if lookup_characters == lookup_boundary:
            source_index = index + 1
            while source_index < len(surface) and is_name_variation_selector(surface[source_index]):
                source_index += 1
            return source_index
    message = f"lookup boundary {lookup_boundary} exceeds source surface {surface!r}"
    raise ValueError(message)


def _clears_japanese_classifier(
    surface: str,
    japanese_probability: Callable[[str], float],
) -> bool:
    """Validate one classifier response and apply the frozen Japanese gate."""
    probability = japanese_probability(
        _japanese_classifier_input(surface),
    )
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        message = f"Japanese classifier returned invalid probability {probability!r}"
        raise EvidenceFailure(message)
    return probability >= JAPANESE_ML_THRESHOLD


def _han_to_kana_boundary(value: str) -> int | None:
    index = 0
    while index < len(value) and _is_han(value[index]):
        index += 1
    if index and index < len(value) and all(_is_kana(character) for character in value[index:]):
        return index
    return None


class EastAsianNameOrderService:
    """Infer only the family-first cases supported by conservative evidence."""

    @staticmethod
    def _is_reviewed_japanese_given_first_exact_surface(raw_name: str) -> bool:
        """Return whether reviewed identity evidence fixes this exact surface as given-first."""
        surface_key = exact_name_surface_key(raw_name)
        return bool(
            surface_key
            and _contains(
                _roman_lexicons().japanese_given_first_exact_surfaces,
                surface_key,
            ),
        )

    def reorder_conflict_reason(
        self,
        raw_name: str,
        selected: NameComponents,
        *,
        paper_names: list[str],
        focal_index: int,
    ) -> ResolutionReason | None:
        """Veto a demonstrated endpoint reversal with strict role evidence.

        The source component labels are deliberately absent from this API. A
        conflict can veto only a candidate that visibly exchanges the first
        and final tokens of the flattened input; ordinary cleanup, dropped
        titles, and same-order candidates are outside its scope. Japanese
        given-first evidence can come from the focal spelling itself. Exact
        reviewed surfaces need no paper context. The remaining Vietnamese and
        cross-cultural rules require positional paper context: only the focal
        index is excluded, so duplicate name text cannot collapse or exclude
        another author.
        """
        if not 0 <= focal_index < len(paper_names):
            message = f"focal index {focal_index} is outside {len(paper_names)} paper names"
            raise IndexError(message)
        if paper_names[focal_index] != raw_name:
            message = "focal raw name does not match its positional paper-name slot"
            raise ValueError(message)

        surface = _normalized_surface(raw_name)
        if not surface or "," in surface:
            return None
        tokens = surface.split(" ")
        if len(tokens) < MIN_ROMANIZED_TOKENS or not self._reverses_endpoints(tokens, selected):
            return None

        first = _fold(tokens[0])
        last = _fold(tokens[-1])
        lexicons = _roman_lexicons()
        surface_key = exact_name_surface_key(surface)
        reviewed_surface = surface_key in JAPANESE_FAMILY_FIRST_CONFLICT_SURFACES
        reviewed_family_first = reviewed_surface and self._has_surname_bearing_peer(paper_names, focal_index)
        if not reviewed_family_first and self._japanese_given_first_veto_supported(surface, paper_names, focal_index):
            return ResolutionReason.JAPANESE_GIVEN_FIRST_REORDER_VETO_PRESERVE_INPUT
        exact_vietnamese = surface_key in VIETNAMESE_GIVEN_FIRST_EXACT_SURFACES
        exact_cross_cultural = surface_key in CROSS_CULTURAL_GIVEN_FIRST_EXACT_SURFACES
        if exact_vietnamese or exact_cross_cultural:
            return (
                ResolutionReason.VIETNAMESE_GIVEN_FIRST_REORDER_VETO_PRESERVE_INPUT
                if exact_vietnamese
                else ResolutionReason.CONTEXT_SUPPORTED_REORDER_VETO_PRESERVE_INPUT
            )
        if (
            len(tokens) == MIN_ROMANIZED_TOKENS
            and first in CROSS_CULTURAL_CONFLICT_HEADS
            and _contains(lexicons.vietnamese_surnames, first)
            and _contains(lexicons.japanese_given_names, first)
            and not _contains(lexicons.korean_surnames, first)
            and self._has_given_first_context(
                paper_names,
                focal_index,
                self._japanese_order_vote,
            )
        ):
            return ResolutionReason.CONTEXT_SUPPORTED_REORDER_VETO_PRESERVE_INPUT
        if (
            _contains(lexicons.vietnamese_top4_surnames, first)
            and _contains(lexicons.vietnamese_top4_surnames, last)
            and self._has_given_first_context(
                paper_names,
                focal_index,
                self._vietnamese_order_vote,
            )
        ):
            return ResolutionReason.CONTEXT_SUPPORTED_REORDER_VETO_PRESERVE_INPUT
        if (
            surface.isascii()
            and _contains(lexicons.vietnamese_surnames, first)
            and not _contains(lexicons.vietnamese_top4_surnames, first)
            and _contains(lexicons.vietnamese_top4_surnames, last)
            and self._has_given_first_context(
                paper_names,
                focal_index,
                self._vietnamese_order_vote,
            )
        ):
            return ResolutionReason.CONTEXT_SUPPORTED_REORDER_VETO_PRESERVE_INPUT
        return None

    @staticmethod
    def _has_given_first_context(
        paper_names: list[str],
        focal_index: int,
        vote: Callable[[str], str | None],
    ) -> bool:
        """Return whether other positional authors strictly favor given-first."""
        given_first = 0
        surname_first = 0
        for index, name in enumerate(paper_names):
            if index == focal_index:
                continue
            direction = vote(name)
            given_first += direction == "given_first"
            surname_first += direction == "surname_first"
        return given_first >= 1 and given_first > surname_first

    @staticmethod
    def _has_surname_bearing_peer(paper_names: list[str], focal_index: int) -> bool:
        """Return whether another paper author has a common Chinese surname endpoint."""
        for index, name in enumerate(paper_names):
            if index == focal_index:
                continue
            tokens = _normalized_surface(name).split(" ")
            if len(tokens) >= MIN_ROMANIZED_TOKENS and (
                _endpoint_key(tokens[0]) in NAME_ORDER_ROUTING_COMMON_CHINESE_SURNAMES
                or _endpoint_key(tokens[-1]) in NAME_ORDER_ROUTING_COMMON_CHINESE_SURNAMES
            ):
                return True
        return False

    @staticmethod
    def _vietnamese_order_vote(name: str) -> str | None:
        """Return one strict endpoint-only Vietnamese order vote."""
        tokens = _normalized_surface(name).split(" ")
        if len(tokens) < MIN_ROMANIZED_TOKENS:
            return None
        lexicons = _roman_lexicons()
        first_is_surname = _contains(lexicons.vietnamese_surnames, _fold(tokens[0]))
        last_is_surname = _contains(lexicons.vietnamese_surnames, _fold(tokens[-1]))
        if first_is_surname == last_is_surname:
            return None
        return "surname_first" if first_is_surname else "given_first"

    @staticmethod
    def _japanese_order_vote(name: str) -> str | None:
        """Return one unambiguous two-token Japanese role-pair vote."""
        tokens = _normalized_surface(name).split(" ")
        if len(tokens) != MIN_ROMANIZED_TOKENS:
            return None
        lexicons = _roman_lexicons()
        first_keys = _japanese_roman_keys(tokens[0])
        last_keys = _japanese_roman_keys(tokens[-1])
        given_first = _contains_any(lexicons.japanese_given_names, first_keys) and _contains_any(
            lexicons.japanese_surnames,
            last_keys,
        )
        surname_first = _contains_any(lexicons.japanese_surnames, first_keys) and _contains_any(
            lexicons.japanese_given_names,
            last_keys,
        )
        if given_first == surname_first:
            return None
        return "given_first" if given_first else "surname_first"

    @staticmethod
    def _japanese_given_first_plausible(name: str) -> bool:
        """Return whether broad surname evidence supports preserving input order."""
        tokens = _normalized_surface(name).split(" ")
        if len(tokens) != MIN_ROMANIZED_TOKENS:
            return False
        lexicons = _roman_lexicons()
        return _contains_any(lexicons.japanese_given_names, _japanese_roman_keys(tokens[0])) and _contains_any(
            lexicons.japanese_possible_surnames,
            _japanese_roman_keys(tokens[-1]),
        )

    def _japanese_given_first_veto_supported(
        self,
        surface: str,
        paper_names: list[str],
        focal_index: int,
    ) -> bool:
        """Combine exact evidence with context-gated possible-surname evidence."""
        lexicons = _roman_lexicons()
        surface_key = exact_name_surface_key(surface)
        if surface_key in JAPANESE_GIVEN_FIRST_EXACT_SURFACES or _contains(
            lexicons.japanese_given_first_exact_surfaces,
            surface_key,
        ):
            return True
        if not self._japanese_given_first_plausible(surface):
            return False
        if self._japanese_order_vote(surface) != "surname_first":
            return not self._has_strong_chinese_surname_first_context(surface, paper_names, focal_index)
        return self._has_given_first_context(
            paper_names,
            focal_index,
            self._japanese_order_vote,
        )

    @staticmethod
    def _has_strong_chinese_surname_first_context(
        surface: str,
        paper_names: list[str],
        focal_index: int,
    ) -> bool:
        """Return whether two peer endpoints override the broad Japanese veto."""
        focal_tokens = surface.split()
        if (
            len(focal_tokens) != MIN_ROMANIZED_TOKENS
            or not all(token.isascii() and token.isalpha() for token in focal_tokens)
            or _endpoint_key(focal_tokens[0]) not in NAME_ORDER_ROUTING_COMMON_CHINESE_SURNAMES
        ):
            return False

        surname_first = 0
        given_first = 0
        for index, name in enumerate(paper_names):
            if index == focal_index:
                continue
            tokens = _normalized_surface(name).split()
            if len(tokens) < MIN_ROMANIZED_TOKENS:
                continue
            first_is_surname = _endpoint_key(tokens[0]) in NAME_ORDER_ROUTING_COMMON_CHINESE_SURNAMES
            last_is_surname = _endpoint_key(tokens[-1]) in NAME_ORDER_ROUTING_COMMON_CHINESE_SURNAMES
            if first_is_surname == last_is_surname:
                continue
            surname_first += first_is_surname
            given_first += last_is_surname
        return surname_first >= 2 and surname_first > given_first  # noqa: PLR2004

    @staticmethod
    def _reverses_endpoints(tokens: list[str], selected: NameComponents) -> bool:
        """Return whether ``selected`` exactly exchanges the input endpoints."""
        first = _endpoint_key(tokens[0])
        last = _endpoint_key(tokens[-1])
        return bool(
            first
            and last
            and first != last
            and first == _endpoint_key(selected.surname)
            and last == _endpoint_key(selected.given_name),
        )

    def infer_iteration_mark(
        self,
        raw_name: str,
        *,
        japanese_probability: Callable[[str], float],
    ) -> EastAsianNameOrderDecision | None:
        """Resolve one iteration-mark surname from the frozen role-evidence rules."""
        surface = _normalized_surface(raw_name)
        if not surface or "," in surface:
            return None
        return self._infer_japanese_iteration_mark(
            surface,
            japanese_probability,
        )

    def infer_resolution(
        self,
        raw_name: str,
        *,
        japanese_probability: Callable[[str], float],
    ) -> EastAsianNameOrderDecision | EastAsianNameOrderPreservation | None:
        """Return one typed assignment, preservation veto, or non-applicability."""
        surface = _normalized_surface(raw_name)
        if not surface or "," in surface:
            return None

        iteration_mark = self._infer_japanese_iteration_mark(
            surface,
            japanese_probability,
        )
        if iteration_mark is not None:
            return iteration_mark
        native = self._infer_native(surface, japanese_probability)
        if native is not None:
            return native
        return self._infer_romanized_resolution(surface)

    def _infer_romanized_resolution(
        self,
        surface: str,
    ) -> EastAsianNameOrderDecision | EastAsianNameOrderPreservation | None:
        """Resolve Roman evidence without erasing a proven preservation veto."""
        romanized = self._infer_romanized(surface)
        if (
            romanized is not None
            and romanized.reason is EastAsianEvidenceReason.KOREAN_ROMANIZED_STRICT
            and self._family_first_western_suffix_conflict(surface)
        ):
            return EastAsianNameOrderPreservation(
                surface=surface,
                reason=EastAsianEvidenceReason.KOREAN_WESTERN_SUFFIX_CONFLICT,
            )
        return romanized

    def family_first_conflict_reason(self, raw_name: str) -> EastAsianEvidenceReason | None:
        """Return a frozen reason when the legacy Korean flip is unsafe."""
        surface = _normalized_surface(raw_name)
        if not surface or "," in surface:
            return None
        resolution = self._infer_romanized_resolution(surface)
        return resolution.reason if isinstance(resolution, EastAsianNameOrderPreservation) else None

    @staticmethod
    def _family_first_western_suffix_conflict(surface: str) -> bool:
        """Veto a family-first route with unsupported Korean and Western tail evidence."""
        tokens = surface.split()
        if len(tokens) < MIN_ROMANIZED_TOKENS:
            return False
        lexicons = _roman_lexicons()
        if not _contains(lexicons.korean_surnames, _fold(tokens[0])):
            return False
        if any(character in KOREAN_ROMANIZATION_BREVE_LETTERS for character in surface):
            return False
        tail_parts = tuple(_fold(part) for token in tokens[1:] for part in token.split("-") if part and part.isalpha())
        if not tail_parts or any(part in KOREAN_ROUTING_GIVEN_PARTS for part in tail_parts):
            return False
        return any(
            len(part) >= MIN_WESTERN_SUFFIX_TOKEN_LENGTH and part.endswith(WESTERN_SURNAME_SUFFIXES) for part in tail_parts
        )

    @staticmethod
    def _infer_japanese_iteration_mark(
        surface: str,
        japanese_probability: Callable[[str], float],
    ) -> EastAsianNameOrderDecision | None:
        """Resolve a marked surname from one unique complete source partition."""
        if JAPANESE_ITERATION_MARK not in surface:
            return None
        lookup_surface = _native_lookup_text(surface)
        if not all(
            character in {" ", JAPANESE_ITERATION_MARK} or _is_han(character) or _is_kana(character)
            for character in lookup_surface
        ):
            return None

        if not _clears_japanese_classifier(surface, japanese_probability):
            return None

        lexicons = _native_lexicons()
        tokens = surface.split(" ")
        partitions: list[tuple[str, str]] = []
        if len(tokens) == MIN_ROMANIZED_TOKENS:
            partitions.append((tokens[0], tokens[1]))
        elif len(tokens) == 1:
            for lookup_boundary in range(1, len(lookup_surface)):
                source_boundary = _source_index_after_lookup_boundary(surface, lookup_boundary)
                partitions.append((surface[:source_boundary], surface[source_boundary:]))
        else:
            return None

        candidates: list[EastAsianNameOrderDecision] = []
        for first, second in partitions:
            if _iteration_surname_only(
                first,
                lexicons,
            ) and _japanese_given_only(second, lexicons):
                candidates.append(
                    EastAsianNameOrderDecision(
                        surface=surface,
                        given_tokens=(second,),
                        middle_tokens=(),
                        surname_tokens=(first,),
                        source_order=("surname", "given"),
                        reason=EastAsianEvidenceReason.JAPANESE_ITERATION_MARK_DUAL_EXCLUSIVE,
                    ),
                )
            if _japanese_given_only(
                first,
                lexicons,
            ) and _iteration_surname_only(second, lexicons):
                candidates.append(
                    EastAsianNameOrderDecision(
                        surface=surface,
                        given_tokens=(first,),
                        middle_tokens=(),
                        surname_tokens=(second,),
                        source_order=("given", "surname"),
                        reason=EastAsianEvidenceReason.JAPANESE_ITERATION_MARK_DUAL_EXCLUSIVE,
                    ),
                )
        return _select_iteration_mark_decision(surface, tokens, lexicons, candidates)

    def _infer_native(
        self,
        surface: str,
        japanese_probability: Callable[[str], float],
    ) -> EastAsianNameOrderDecision | EastAsianNameOrderPreservation | None:
        if _is_hangul(surface):
            if len(surface) != KOREAN_NATIVE_TOKEN_LENGTH:
                return None
            boundary = (
                KOREAN_NATIVE_COMPOUND_SURNAME_LENGTH
                if surface[:KOREAN_NATIVE_COMPOUND_SURNAME_LENGTH] in KOREAN_NATIVE_COMPOUND_SURNAMES
                else 1
            )
            return EastAsianNameOrderDecision(
                surface=surface,
                given_tokens=(surface[boundary:],),
                middle_tokens=(),
                surname_tokens=(surface[:boundary],),
                source_order=("surname", "given"),
                reason=EastAsianEvidenceReason.KOREAN_NATIVE_THREE_SYLLABLE,
            )
        if not _is_compact_japanese(surface):
            return self._infer_spaced_japanese_native(surface, japanese_probability)
        lookup_surface = _native_lookup_text(surface)
        if len(lookup_surface) < MIN_ROMANIZED_TOKENS:
            return None
        if not _clears_japanese_classifier(surface, japanese_probability):
            return None
        lookup_boundary = self._japanese_native_boundary(lookup_surface)
        boundary = _source_index_after_lookup_boundary(surface, lookup_boundary)
        return EastAsianNameOrderDecision(
            surface=surface,
            given_tokens=(surface[boundary:],),
            middle_tokens=(),
            surname_tokens=(surface[:boundary],),
            source_order=("surname", "given"),
            reason=EastAsianEvidenceReason.JAPANESE_NATIVE_DICTIONARY,
        )

    @staticmethod
    def _infer_spaced_japanese_native(
        surface: str,
        japanese_probability: Callable[[str], float],
    ) -> EastAsianNameOrderDecision | EastAsianNameOrderPreservation | None:
        """Route a SPACED two-token kanji/kana name family-first when the native
        dictionary supports it ("佐藤 優" -> surname 佐藤, given 優).

        Only the compact form was handled before, so spaced kanji fell through to the
        generic given-first assumption and swapped the roles. Evidence is one-sided far
        more often than not, because the surname asset holds 2,000 entries against 69,002
        given names: requiring BOTH sides left 615,837 names / 3.74M occ reordered wrongly
        ("松中 成浩", "三浦 耕吉郎"), judged family-first 187/187 blind. So a single
        unopposed side routes too. ML-Japanese gated, so spaced Chinese is untouched.

        Only ONE two-sided shape still abstains: reverse-plausible, where the leading token
        is a known given name AND the trailing token a known surname ("剛 長谷川"). That is
        positive evidence of an inverted byline, and blind labelling agrees on 150 of 150.

        Both-surname and both-given pairs used to abstain as well, on the assumption that two
        signals cancel. They do not — a shared token is normally the given name, because the
        given asset is 35x the surname asset, so "both are surnames" usually means the
        trailing one is also a given name that the 2,000-entry surname list happens to list,
        and "both are given names" usually means the leading one is a surname the list
        happens to list. Blind labelling puts the family name FIRST on 91.6% of
        both-given-plausible occ (297 PPS-sampled names) and 89.3% of both-surname occ (159),
        so abstaining was wrong far more often than right: it mis-ordered 174,201 of 190,342
        occ across the two classes. Sampled at name level the shares are 84.0% (486 names)
        and 93.9% (428), across two independent rounds whose overlap agreed 60/60.
        """
        tokens = surface.split(" ")
        if len(tokens) != 2 or not all(_is_compact_japanese(token) for token in tokens):  # noqa: PLR2004
            return None
        if not _clears_japanese_classifier(surface, japanese_probability):
            return None
        lexicons = _native_lexicons()
        first, last = tokens
        first_key, last_key = (_native_lookup_text(token) for token in tokens)
        first_surname = _contains(lexicons.japanese_surnames, first_key)
        first_given = _contains(lexicons.japanese_given_names, first_key)
        last_surname = _contains(lexicons.japanese_surnames, last_key)
        last_given = _contains(lexicons.japanese_given_names, last_key)
        surname_first = first_surname and last_given
        reverse_plausible = first_given and last_surname
        strict_surname_first = surname_first and not first_given and not last_surname
        strict_given_first = reverse_plausible and not first_surname and not last_given
        if strict_given_first:
            return EastAsianNameOrderPreservation(
                surface=surface,
                reason=EastAsianEvidenceReason.JAPANESE_NATIVE_SPACED_STRICT_GIVEN_FIRST,
            )
        if not surname_first:
            if reverse_plausible:
                return None
            if not first_surname and not last_given:
                return None
        elif reverse_plausible:
            return None
        return EastAsianNameOrderDecision(
            surface=surface,
            given_tokens=(last,),
            middle_tokens=(),
            surname_tokens=(first,),
            source_order=("surname", "given"),
            reason=(
                EastAsianEvidenceReason.JAPANESE_NATIVE_SPACED_STRICT_FAMILY_FIRST
                if strict_surname_first
                else EastAsianEvidenceReason.JAPANESE_NATIVE_SPACED_DICTIONARY
            ),
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
        surface_key = exact_name_surface_key(" ".join(tokens))
        exact_roles = IDENTITY_BACKED_EXACT_ROLES.get(surface_key)
        if exact_roles is not None:
            return EastAsianNameOrderDecision(
                surface=surface,
                given_tokens=tuple(token for token, role in zip(tokens, exact_roles, strict=True) if role == "given"),
                middle_tokens=tuple(token for token, role in zip(tokens, exact_roles, strict=True) if role == "middle"),
                surname_tokens=tuple(token for token, role in zip(tokens, exact_roles, strict=True) if role == "surname"),
                source_order=exact_roles,
                reason=EastAsianEvidenceReason.IDENTITY_BACKED_EXACT_FULL_SURFACE,
            )
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
        head = _fold(tokens[0])
        if not _contains(lexicons.vietnamese_surnames, head):
            return None
        trailing = _fold(tokens[-1])
        surface_key = exact_name_surface_key(" ".join(tokens))
        if surface_key in VIETNAMESE_GIVEN_FIRST_CONFLICT_SURFACES:
            middle_tokens = tuple(tokens[1:-1])
            return EastAsianNameOrderDecision(
                surface=surface,
                given_tokens=(tokens[0],),
                middle_tokens=middle_tokens,
                surname_tokens=(tokens[-1],),
                source_order=("given", *("middle" for _ in middle_tokens), "surname"),
                reason=EastAsianEvidenceReason.VIETNAMESE_GIVEN_FIRST_EXACT_SURFACE,
            )
        # Bare-ASCII Vietnamese is otherwise left alone, because most of the surname list is short
        # and doubles as Korean, Chinese or Western given syllables ("Mai", "Le", "Do", "Kim"), so a
        # diacritic is what identifies the name as Vietnamese at all. The listed exceptions appear
        # in no other lexicon, so admitting them cannot preempt the Korean or Japanese routes that
        # run after this one, and each is near-exclusively a surname rather than a given name.
        # "Nguyen Van Hieu" and "Tran Quoc Khanh" parsed given-first before this, yielding surnames
        # "Van Hieu" and "Quoc Khanh"; blind labelling put the leading token as the surname in
        # 99.3-99.8% of sampled rows across the three.
        if head not in ASCII_ROUTABLE_VIETNAMESE_SURNAMES and not _has_east_asian_diacritic_evidence(surface):
            if head not in ASCII_ROUTABLE_VIETNAMESE_SURNAMES_WITHOUT_SURNAME_PARTNER:
                return None
            # A trailing surname means the byline was inverted for an English-language journal, at any
            # length: "Vu Nguyen", "Hoang Xuan Tran", "Truong Khang Nguyen" all carry the family name
            # last. Blind labelling of this shape put the family name in the trailing token on 74% of
            # mentions, and declining lifts the whole relaxation from 71.6% to 82.9% mention-weighted
            # accuracy — the leading token is the family name on 188 of the 197 rows that survive.
            # The surname may be the first half of a hyphenated compound rather than the whole token
            # ("Thuong Le-Tien", "Vu Thuy Khanh Le-Trilling", "Truong Nguyen-Ba" — judged 8/8 family
            # last), which is another 235 names / 984 mentions and takes a mention-weighted sample
            # from 96.4% to 98.5%. A hyphen whose first half is NOT a surname is a given name
            # ("Ngo Si-Huy"), so only the lexicon hit declines.
            if _contains(lexicons.vietnamese_surnames, trailing) or _contains(
                lexicons.vietnamese_surnames,
                trailing.split("-")[0],
            ):
                return None
        middle_tokens = tuple(tokens[1:-1])
        return EastAsianNameOrderDecision(
            surface=surface,
            given_tokens=(tokens[-1],),
            middle_tokens=middle_tokens,
            surname_tokens=(tokens[0],),
            source_order=("surname", *("middle" for _ in middle_tokens), "given"),
            reason=EastAsianEvidenceReason.VIETNAMESE_UNICODE_SURNAME_FIRST,
        )

    @staticmethod
    def _infer_korean(
        tokens: list[str],
        lexicons: _RomanLexicons,
    ) -> EastAsianNameOrderDecision | None:
        if len(tokens) > MAX_KOREAN_ROMANIZED_TOKENS:
            return None
        # A lone leading letter is an initial, not a surname ("O Braun-Falco" is Otto
        # Braun-Falco). `o` is the only single-letter entry in the Korean surname lexicon,
        # so without this the whole class routes as the Korean surname 오.
        if len(tokens[0]) == 1:
            return None
        # A European-exclusive diacritic (Nordic "Kim Hørslev-Petersen") is not romanized
        # Korean. McCune-Reischauer breve vowels (ŏ/ŭ) stay allowed — they are inside the
        # East-Asian-plausible repertoire, so only ø/å/ö/ü/ñ/… disqualify.
        if any(_is_european_exclusive_diacritic(character) for token in tokens for character in token):
            return None
        if not _contains(lexicons.korean_surnames, _fold(tokens[0])):
            return None
        if _contains(lexicons.korean_surnames, _fold(tokens[-1])):
            return None
        compact_given = EastAsianNameOrderService._unique_compact_korean_given(
            tokens,
        )
        if compact_given is not None:
            return EastAsianNameOrderDecision(
                surface=" ".join(tokens),
                given_tokens=(compact_given,),
                middle_tokens=(),
                surname_tokens=(tokens[0],),
                source_order=("surname", "given"),
                reason=EastAsianEvidenceReason.KOREAN_COMPACT_GIVEN_UNIQUE_SPLIT,
            )
        given_parts = [_fold(part) for token in tokens[1:] for part in token.split("-") if part]
        has_hyphen = any("-" in token for token in tokens[1:])
        all_known = bool(given_parts) and all(part in KOREAN_ROUTING_GIVEN_PARTS for part in given_parts)
        if not (has_hyphen or (len(tokens) == MAX_KOREAN_ROMANIZED_TOKENS and all_known)):
            return None
        # A surname-lexicon hit on token 0 is weak evidence by itself, because the short entries
        # double as ordinary Western given names ("Yu Alonso", "Ra Sanchez", "Kim Rudolph-Lund").
        # Demand corroboration from the given side in the two shapes blind labelling found
        # unreliable without it: a two-letter surname (34.3% wrong), and a given part longer than
        # a romanized Korean syllable can be — the syllabary tops out near six characters, so a
        # longer part is a Western surname element ("Gillespie-White", "Kramer-Johansen", 92% wrong).
        if not any(part in KOREAN_ROUTING_GIVEN_PARTS for part in given_parts) and (
            len(tokens[0]) == HOMOGRAPH_PRONE_SURNAME_LENGTH
            or any(len(part) > MAX_KOREAN_GIVEN_SYLLABLE_LENGTH for part in given_parts)
        ):
            return None
        given_tokens = tuple(tokens[1:])
        return EastAsianNameOrderDecision(
            surface=" ".join(tokens),
            given_tokens=given_tokens,
            middle_tokens=(),
            surname_tokens=(tokens[0],),
            source_order=("surname", *("given" for _ in given_tokens)),
            reason=EastAsianEvidenceReason.KOREAN_ROMANIZED_STRICT,
        )

    @staticmethod
    def _unique_compact_korean_given(tokens: list[str]) -> str | None:
        """Return an unsplit compact given token with one known two-part reading."""
        if len(tokens) != MIN_ROMANIZED_TOKENS:
            return None
        compact_given = tokens[1]
        if not compact_given.isalpha():
            return None
        folded = _fold(compact_given)
        splits = [
            boundary
            for boundary in range(1, len(folded))
            if folded[:boundary] in KOREAN_ROUTING_GIVEN_PARTS and folded[boundary:] in KOREAN_ROUTING_GIVEN_PARTS
        ]
        return compact_given if len(splits) == 1 else None

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
        exact_given_first = exact_name_surface_key(" ".join(tokens)) in JAPANESE_PRESELECTION_GIVEN_FIRST_EXACT_SURFACES
        reverse_plausible = _contains_any(lexicons.japanese_given_names, first_keys) and _contains_any(
            lexicons.japanese_surnames,
            last_keys,
        )
        if not surname_first or reverse_plausible or exact_given_first:
            return None
        return EastAsianNameOrderDecision(
            surface=" ".join(tokens),
            given_tokens=(tokens[1],),
            middle_tokens=(),
            surname_tokens=(tokens[0],),
            source_order=("surname", "given"),
            reason=EastAsianEvidenceReason.JAPANESE_ROMANIZED_DIRECTIONAL_DICTIONARY,
        )


__all__ = ["EastAsianNameOrderDecision", "EastAsianNameOrderService"]
