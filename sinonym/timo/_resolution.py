"""Structured TIMO contracts and terminal author resolution.

The resolver receives structured source fields so source fallback is lossless. Scalar
and batch inference run on ``SourceAuthorFields.full_name()``, matching the
existing ``fullNameOf`` sequence and deliberately excluding suffix. Source
labels ordinarily preserve lineage rather than establish semantic roles, but
a closed set of reviewed structured shapes may use their placement as direct
evidence.
"""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from dataclasses import replace
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, StrictStr, root_validator

from sinonym.chinese_names_data import HAN_SURNAME_POSITION_READINGS, REVIEWED_ATOMIC_KOREAN_GIVEN_FORMS
from sinonym.coretypes import NameComponents, ParseResult
from sinonym.coretypes.routing_resolution import (
    ApplyAssignment,
    EvidenceFailure,
    HardScalarConstraint,
    HardScalarMaterializationFailure,
    PreserveBaseline,
    ResolutionAction,
    ResolutionProvenance,
    ResolutionReason,
    resolution_decision_spec,
)
from sinonym.name_punctuation import ROMAN_HYPHEN_LIKE
from sinonym.pipeline.name_order_routing import pp_abstain_parsed
from sinonym.services.east_asian_name_order import JAPANESE_ML_THRESHOLD
from sinonym.services.non_person import (
    CJK_NON_PERSON_SUFFIX_MARKERS,
    REVIEWED_HANGUL_ORGANIZATION_MARKERS,
    reviewed_non_person_source_pattern,
)
from sinonym.services.person_name_normalization import (
    PersonNameNormalizationService,
    PersonNameOutcome,
    reviewed_closed_comma_credential_tail_head,
    reviewed_source_cleanup_pattern,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from sinonym.detector import ChineseNameDetector


LOGGER = logging.getLogger(__name__)

# Routed batch components must be fully Romanized. Scalar canonicals may retain
# CJK, but a CJK surname beside Latin components is an unsafe mixed-script split.
_CJK_LETTER_NAME_MARKERS = (
    "CJK UNIFIED IDEOGRAPH",
    "CJK COMPATIBILITY IDEOGRAPH",
    "HIRAGANA",
    "KATAKANA",
    "HANGUL",
    "BOPOMOFO LETTER",
    "IDEOGRAPHIC ITERATION MARK",
)
_KATAKANA_GENERATION_SUFFIXES = frozenset(
    {
        "ジュニア",
        "シニア",
        "ザサード",
        "ザセカンド",
        "ザフォース",
        "サード",
        "セカンド",
        "フォース",
    },
)
_CATALOG_SEPARATORS = frozenset({",", "\u3001", "\uff0c"})
_REVIEWED_JR_FORMS = frozenset({"Jr", "Jr.", "jr", "jr."})
_STRUCTURAL_ROMAN_HYPHEN_TRANSLATION = str.maketrans(dict.fromkeys(ROMAN_HYPHEN_LIKE, "-"))
_REVIEWED_JR_ORGANIZATION_WORDS = frozenset(
    {
        "academy",
        "board",
        "college",
        "department",
        "engineering",
        "institute",
        "laboratory",
        "school",
        "university",
    },
)
_MIDDLE_DOT = "\u00b7"
_MIDDLE_DOT_PERSON_PUNCTUATION = frozenset({"'", "\u2019", "-", "\u2010", "\u2011", "."})
_REVIEWED_JR_FAMILY_PARTICLES = frozenset(
    {
        "al",
        "ap",
        "ben",
        "bin",
        "da",
        "dal",
        "das",
        "de",
        "del",
        "della",
        "den",
        "der",
        "di",
        "dos",
        "du",
        "el",
        "ibn",
        "la",
        "las",
        "le",
        "los",
        "st",
        "ten",
        "ter",
        "ud",
        "ur",
        "van",
        "von",
        "zu",
        "zum",
        "zur",
    },
)
_UNICODE_LETTER_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_MULTI_INITIAL_TOKEN_RE = re.compile(r"(?:[^\W\d_]\.){2,}[^\W\d_]?\.?$", re.UNICODE)
_ASCII_WORD_RE = re.compile(r"[A-Za-z]+")
_ASCII_HYPHENATED_WORD_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")
_ASCII_INITIAL_RE = re.compile(r"[A-Za-z]\.?")
_DOTTED_INITIAL_RE = re.compile(r"[^\W\d_]\.", re.UNICODE)
_JOINED_UPPERCASE_SURNAME_RE = re.compile(r"([A-Z]{2,})([A-Z][a-z]+(?:-[a-z]+)*)")
_TITLED_GIVEN_RE = re.compile(r"(?:Dr\.|Prof\.) ([^\s]+)")
_TRAILING_AFFILIATION_SURNAME_RE = re.compile(r"[A-Z][a-z]+a")
_MIN_DOTTED_SOURCE_INITIALS = 2
_MAX_DOTTED_SOURCE_INITIALS = 3
_DETACHED_ACUTE = "\u00b4"
_DETACHED_TILDE = "\u02dc"
_DETACHED_DIACRITIC_TOKEN_RE = re.compile(rf"(?<!\S)[{_DETACHED_ACUTE}{_DETACHED_TILDE}](?!\S)")
_DETACHED_DIACRITIC_SPAN_RE = re.compile(
    rf"(?P<head>[^\W\d_]+)\s+(?P<mark>[{_DETACHED_ACUTE}{_DETACHED_TILDE}])\s+(?P<tail>[^\W\d_]+)",
    re.UNICODE,
)
_DETACHED_DIACRITIC_SPECS = {
    _DETACHED_ACUTE: ("\u0301", frozenset("aeiouy")),
    _DETACHED_TILDE: ("\u0303", frozenset("ano")),
}
_DETACHED_DIACRITIC_TRANSLITERATION_CONTROLS = frozenset(
    {
        ("cui", "e"),
        ("ma", "ayan"),
        ("o", "connor"),
        ("p", "eng"),
        ("sa", "di"),
        ("ts", "ai"),
    },
)
_REVIEWED_TRAILING_AFFILIATION_ROSTERS = frozenset(
    {
        (
            ("su-sui", "", "lina"),
            ("wei-shen", "", "taia"),
            ("kwo-ting", "", "fanga"),
        ),
        (
            ("ing", "", "zhanga"),
            ("yuan", "", "hua"),
            ("lei", "", "songa"),
            ("hongdian", "", "lua"),
            ("jian", "", "wanga"),
            ("qingqing", "", "liua"),
        ),
    },
)
_REVIEWED_CLEANUP_PREFIX_CREDENTIAL_KEYS = frozenset({"dnb"})
_CYRILLIC_NAME_TOKEN_RE = re.compile(r"[\u0410-\u042f\u0401][\u0410-\u044f\u0401\u0451'\u2019-]+")
_FEMININE_CYRILLIC_SURNAME_SUFFIXES = (
    "\u043e\u0432\u0430",
    "\u0435\u0432\u0430",
    "\u0441\u043a\u0430\u044f",
    "\u0446\u043a\u0430\u044f",
)
_MASCULINE_CYRILLIC_SURNAME_SUFFIXES = ("\u0441\u043a\u0438\u0439", "\u0446\u043a\u0438\u0439")
_FEMININE_CYRILLIC_PATRONYMIC_SUFFIXES = ("\u043e\u0432\u043d\u0430", "\u0435\u0432\u043d\u0430")
_MASCULINE_CYRILLIC_PATRONYMIC_SUFFIXES = ("\u043e\u0432\u0438\u0447", "\u0435\u0432\u0438\u0447")
_CYRILLIC_PATRONYMIC_SUFFIXES = (
    *_FEMININE_CYRILLIC_PATRONYMIC_SUFFIXES,
    *_MASCULINE_CYRILLIC_PATRONYMIC_SUFFIXES,
)
# Exact source tuples whose scalar fallback was manually verified to split a
# compound family name. The broader initials + multi-token-last shape includes
# thousands of legitimate bundled given/surname fields and is not safe.
REVIEWED_SCALAR_COMPOUND_SURNAMES = frozenset(
    {
        ("andrigo", "", "barboza de nardi"),
        ("angélica", "", "rico alonso"),
        ("r", "f", "lai a fat"),
        ("bahareh", "", "rezaei mirghaed"),
        ("camilo", "josé", "tamayo borray"),
        ("celso", "", "cancela outeda"),
        ("cristina", "", "bastidas redin"),
        ("dayana", "", "luna reyes"),
        ("elías", "alberto", "bedoya marrugo"),
        ("fahimeh", "", "asadi amoli"),
        ("federica", "", "li pomi"),
        ("gabith", "", "quispe fernández"),
        ("j", "", "aibar manero"),
        ("karolina", "", "hoppe gromadzka"),
        ("l", "", "carreras matas"),
        ("m.", "", "olmedo negrete"),
        ("magdalena", "sofía", "paláu cardona"),
        ("marina", "", "aguilar rubio"),
        ("mónica", "", "zamora zapata"),
        ("n.m.a.", "", "nik long"),
        ("natalia", "", "agudelo sep\u00falveda"),
        ("paulo josé", "", "mata pereira"),
        ("suzan", "", "gonçalves rosa"),
        ("a", "y f", "li yim"),
        ("valerio", "antonio", "pamplona salomon"),
        ("walter", "", "cardona maya"),
    },
)
REVIEWED_TRAILING_HYPHENATED_COMPOUND_SURNAMES = frozenset({"au-yeung", "ou-yang"})
# Every occurrence of these exact normalized source tuples was reviewed in the
# full corpus. The first set only exchanges the supplied endpoint components;
# the maps below record field-sourced and literal output roles.
REVIEWED_EXACT_SOURCE_ENDPOINT_REORDERS = frozenset(
    {
        ("abramova", "", "na"),
        ("arbabi", "", "masoud"),
        ("bakulina", "", "li"),
        ("barani", "", "hossein"),
        ("batista", "", "juanize matias da silva"),
        ("boriskova", "", "pi"),
        ("cho", "", "yk"),
        ("clyman", "", "mj"),
        ("dahl", "", "mm"),
        ("firmbach", "", "f-p."),
        ("gol'dman", "", "an"),
        ("grube", "", "mr"),
        ("ha", "", "sh"),
        ("hashimoto", "", "keiichi"),
        ("ho", "", "jm"),
        ("hori", "", "maiya"),
        ("im", "", "jj"),
        ("iwasaki", "", "tohru"),
        ("khurs", "", "en"),
        ("kim", "", "jina"),
        ("kim", "", "jy"),
        ("kim", "", "namseok"),
        ("kim", "", "woansub"),
        ("kim", "", "ys"),
        ("korolev", "", "vv"),
        ("kwon", "", "yong"),
        ("lemomu", "", "km"),
        ("nishanov", "", "d.a"),
        ("pappanikou", "", "aj"),
        ("park", "", "sh"),
        ("podol'nikova", "", "np"),
        ("rorem", "", "da"),
        ("seo", "", "myeongwhoon"),
        ("sodimu", "", "isiaka"),
        ("sugino", "", "eiichi"),
        ("veselov", "", "vf"),
        ("wada", "", "shin-ichi"),
        ("\u4e2d\u5c71", "", "\u8fc5"),
        ("\u6842", "", "\u7460\u4ee5"),
        ("برخورداری،", "", "وحید"),
        ("دعایی،", "", "فریما"),
        ("昌谷", "", "忠海"),
        ("赫勒", "", "m"),
    },
)
REVIEWED_EXACT_SOURCE_ROLE_ASSIGNMENTS = {
    ("augustin", "mary", "ann"): ("middle_names", "last_name", "first_name"),
    ("choi", "seung", "wook"): ("middle_names", "last_name", "first_name"),
    ("do", "thi kim", "lanh"): ("last_name", "middle_names", "first_name"),
    ("karkabounas", "spyridon", "ch."): ("middle_names", "last_name", "first_name"),
    ("kim", "sun", "hyoung"): ("middle_names", "last_name", "first_name"),
    ("lee", "joo", "youn"): ("middle_names", "last_name", "first_name"),
    ("mai", "dac", "bien"): ("last_name", "middle_names", "first_name"),
    ("kim", "tae", "in"): ("middle_names", "last_name", "first_name"),
    ("lee", "joo", "hee"): ("middle_names", "last_name", "first_name"),
    ("tormos", "josep", "maria"): ("middle_names", "last_name", "first_name"),
    ("jidong", "", "sung"): ("first_name", "middle_names", "last_name"),
    ("mung", "", "chiang"): ("first_name", "middle_names", "last_name"),
    ("onchee", "", "yu"): ("first_name", "middle_names", "last_name"),
    ("seah", "h.", "lim"): ("first_name", "middle_names", "last_name"),
    ("seah", "h", "lim"): ("first_name", "middle_names", "last_name"),
    ("nabiev", "valery", "sharifyanovich"): ("middle_names", "last_name", "first_name"),
    ("turaxodjayeva", "moxidil", "obidjonovna"): ("middle_names", "last_name", "first_name"),
    (
        "\u0430\u043b\u0435\u043a\u0441\u0435\u0435\u0432",
        "\u0433\u0435\u043d\u043d\u0430\u0434\u0438\u0439",
        "\u0432\u0430\u043b\u0435\u043d\u0442\u0438\u043d\u043e\u0432\u0438\u0447",
    ): ("middle_names", "last_name", "first_name"),
    (
        "\u0431\u0435\u043b\u043e\u0432",
        "\u0432\u043b\u0430\u0434\u0438\u043c\u0438\u0440",
        "\u043d\u0438\u043a\u043e\u043b\u0430\u0435\u0432\u0438\u0447",
    ): ("middle_names", "last_name", "first_name"),
    (
        "\u0432\u043e\u043b\u044b\u043d\u043e\u0432",
        "\u043c\u0438\u0445\u0430\u0438\u043b",
        "\u0430\u043d\u0430\u0442\u043e\u043b\u044c\u0435\u0432\u0438\u0447",
    ): ("middle_names", "last_name", "first_name"),
    (
        "\u043a\u0438\u0440\u0441\u0430\u043d\u043e\u0432",
        "\u0430\u043d\u0434\u0440\u0435\u0439",
        "\u0440\u043e\u043c\u0430\u043d\u043e\u0432\u0438\u0447",
    ): ("middle_names", "last_name", "first_name"),
    (
        "\u043a\u0438\u0441\u0442\u0435\u0440\u0441\u043a\u0438\u0439",
        "\u0430\u043b\u0435\u043a\u0441\u0430\u043d\u0434\u0440",
        "\u043f\u0435\u0442\u0440\u043e\u0432\u0438\u0447",
    ): ("middle_names", "last_name", "first_name"),
    (
        "\u043c\u0430\u043a\u0430\u0440\u043e\u0432\u0430",
        "\u0435\u043a\u0430\u0442\u0435\u0440\u0438\u043d\u0430",
        "\u0432\u043b\u0430\u0434\u0438\u043c\u0438\u0440\u043e\u0432\u043d\u0430",
    ): ("middle_names", "last_name", "first_name"),
    (
        "\u043d\u0435\u0432\u0435\u0440\u043e\u0432\u0430",
        "\u043e\u043b\u044c\u0433\u0430",
        "\u0430\u043b\u0435\u043a\u0441\u0430\u043d\u0434\u0440\u043e\u0432\u043d\u0430",
    ): ("middle_names", "last_name", "first_name"),
    (
        "\u0441\u0432\u0438\u0434\u0443\u043d\u043e\u0432\u0438\u0447",
        "\u043d\u0438\u043a\u043e\u043b\u0430\u0439",
        "\u0430\u043b\u0435\u043a\u0441\u0430\u043d\u0434\u0440\u043e\u0432\u0438\u0447",
    ): ("middle_names", "last_name", "first_name"),
    (
        "\u0442\u0440\u043e\u0444\u0438\u043c\u043e\u0432",
        "\u0430\u0440\u0442\u0435\u043c",
        "\u0430\u043b\u0435\u043a\u0441\u0430\u043d\u0434\u0440\u043e\u0432\u0438\u0447",
    ): ("middle_names", "last_name", "first_name"),
    (
        "\u0442\u0443\u0445\u0442\u0430\u043c\u0443\u0440\u043e\u0434",
        "\u0437\u0438\u0451\u0434\u0443\u043b\u043b\u0430",
        "\u0437\u0438\u043a\u0440\u0438\u043b\u043b\u0430",
    ): ("middle_names", "last_name", "first_name"),
    (
        "\u0448\u0435\u0432\u0447\u0435\u043d\u043a\u043e",
        "\u0435\u043b\u0435\u043d\u0430",
        "\u0432\u0438\u043a\u0442\u043e\u0440\u043e\u0432\u043d\u0430",
    ): ("middle_names", "last_name", "first_name"),
    (
        "\u044f\u043c\u0430\u043b\u0434\u0438\u043d\u043e\u0432",
        "\u0442\u0438\u043c\u0443\u0440",
        "\u0440\u0438\u0444\u0430\u0442\u043e\u0432\u0438\u0447",
    ): ("middle_names", "last_name", "first_name"),
}
# Reviewed tuples whose correct output needs literal text rather than source
# field selectors.
REVIEWED_EXACT_SOURCE_LITERAL_ASSIGNMENTS = {
    ("fernando", "del.", "pulgar"): NameComponents(
        given_name="Fernando",
        surname="del Pulgar",
    ),
    ("m.", "d", "services-reginald m. atwater"): NameComponents(
        given_name="Reginald",
        middle_name="M.",
        surname="Atwater",
    ),
    ("ibschons", "", "ioanna zimianiti mbbs"): NameComponents(
        given_name="Ioanna",
        surname="Zimianiti",
    ),
    ("johannes", "k steinweg", "mbbs"): NameComponents(
        given_name="Johannes",
        middle_name="K",
        surname="Steinweg",
    ),
    ("mao", "", "kai"): NameComponents(
        given_name="Kai",
        surname="Mao",
    ),
    ("min", "-fu", "tsan"): NameComponents(
        given_name="Min-Fu",
        surname="Tsan",
    ),
    ("ms", "frcs facs", "ronnie t. p. poon mbbs"): NameComponents(
        given_name="Ronnie",
        middle_name="T. P.",
        surname="Poon",
    ),
    ("phd", "msn rn aocnp", "carolyn s. phillips"): NameComponents(
        given_name="Carolyn",
        middle_name="S.",
        surname="Phillips",
    ),
    ("\u5289\u6bb7\u4f50", "", "i-ting wang"): NameComponents(
        given_name="Yin-Zuo",
        surname="Liu",
    ),
    ("モンゴメリー\uff0c", "エイチ\uff0e", "マンニング\uff0c"): NameComponents(
        given_name="エイチ.",
        middle_name="モンゴメリー",
        surname="マンニング",
    ),
}

# Exact empty-suffix corrections found by the PR29/PR30 manual-diff audit.
# These rows need semantic fields that cannot be expressed by the older
# endpoint-only tables. Keys use the same whitespace/case contract as
# ``_source_component_key``.
REVIEWED_EXACT_EMPTY_SUFFIX_ASSIGNMENTS = {
    ("35", "", "w.m.song"): NameComponents(
        given_name="W.",
        middle_name="M.",
        surname="Song",
    ),
    ("g.", "", "d andrea"): NameComponents(
        given_name="G.",
        surname="D'Andrea",
    ),
    ("graeme", "", "woodfield (chairman)"): NameComponents(
        given_name="Graeme",
        surname="Woodfield",
    ),
    ("albert", "guillén i", "fàbregas"): NameComponents(
        given_name="Albert",
        surname="Guillén i Fàbregas",
    ),
    ("ernest", "bladé i", "castellet"): NameComponents(
        given_name="Ernest",
        surname="Bladé i Castellet",
    ),
    ("jordi", "feu i", "gelis"): NameComponents(
        given_name="Jordi",
        surname="Feu i Gelis",
    ),
    ("jaime", "lluis y", "navas"): NameComponents(
        given_name="Jaime",
        surname="Lluis y Navas",
    ),
    ("alvaro", "d' ors y", "pérez-peix"): NameComponents(
        given_name="Alvaro",
        surname="d'Ors y Pérez-Peix",
    ),
    ("b.", "ó", "fearraigh"): NameComponents(
        given_name="B.",
        surname="Ó Fearraigh",
    ),
    ("olaru", "", "c. c."): NameComponents(
        given_name="C. C.",
        surname="Olaru",
    ),
    ("tran", "d", "huong"): NameComponents(
        given_name="Huong",
        middle_name="D",
        surname="Tran",
    ),
    ("l.", "s.", "lauria de cidre"): NameComponents(
        given_name="L.",
        middle_name="S.",
        surname="Lauria de Cidre",
    ),
    ("y.", "", "pépin dubois"): NameComponents(
        given_name="Y.",
        surname="Pépin Dubois",
    ),
    ("r", "", "uribe elías"): NameComponents(
        given_name="R.",
        surname="Uribe Elías",
    ),
    ("m.", "", "ravonel salzgeber"): NameComponents(
        given_name="M.",
        surname="Ravonel Salzgeber",
    ),
    ("j", "", "robles barba"): NameComponents(
        given_name="J.",
        surname="Robles Barba",
    ),
    ("d.", "", "paredes hernandez"): NameComponents(
        given_name="D.",
        surname="Paredes Hernandez",
    ),
    ("v", "", "sánchez margalet"): NameComponents(
        given_name="V.",
        surname="Sánchez Margalet",
    ),
    ("r", "k", "chew"): NameComponents(
        given_name="R.-K.",
        surname="Chew",
    ),
    ("\u589e\u5174", "", "\u6e38"): NameComponents(
        given_name="Zeng-Xing",
        surname="You",
    ),
    ("\u5149", "", "\u5f69\u4e43"): NameComponents(
        given_name="\u5f69\u4e43",
        surname="\u5149",
    ),
    ("\u5742\u53e3", "", "\u5e73"): NameComponents(
        given_name="\u5e73",
        surname="\u5742\u53e3",
    ),
}

# Every occurrence of these compact surname-first tuples was inspected in the
# 691,008,961-author source corpus. Endpoint exchange is safe, but expanding a
# compact source form such as ``Sh`` into invented initials is not.
REVIEWED_EXACT_EMPTY_SUFFIX_ENDPOINT_REORDERS = frozenset(
    {
        ("chen", "", "mh"),
        ("chen", "", "pj"),
        ("chen", "", "xr"),
        ("chen", "", "yj"),
        ("chen", "", "zb"),
        ("chang", "", "jn"),
        ("chan", "", "mc"),
        ("chan", "", "nk"),
        ("guo", "", "wh"),
        ("kazantsev", "", "a.v."),
        ("kang", "", "s-c"),
        ("lee", "", "b"),
        ("lee", "", "ds"),
        ("lee", "", "kh"),
        ("lee", "", "ls"),
        ("lee", "", "sh"),
        ("lee", "", "sl"),
        ("lee", "", "wr"),
        ("li", "", "cl"),
        ("li", "", "sh"),
        ("li", "", "zh"),
        ("liu", "", "pt"),
        ("liu", "", "zh"),
        ("sun", "", "cy"),
        ("sun", "", "hq"),
        ("wang", "", "hl"),
        ("wang", "", "jp"),
        ("wang", "", "kq"),
        ("wang", "", "sl"),
        ("wang", "", "yy"),
        ("wang", "", "zm"),
        ("wu", "", "ch"),
        ("wu", "", "wn"),
        ("yang", "", "dx"),
        ("zhang", "", "ds"),
        ("zhang", "", "jz"),
        ("zhou", "", "xg"),
        ("xu", "", "br"),
        ("xu", "", "lz"),
    },
)

# These reviewed assignments require exact component text. They must bypass
# ordinary source-assignment initial formatting, which would add punctuation
# to semantic bare initials or repartition a deliberately grouped given name.
REVIEWED_EXACT_BARE_PERSONAL_ASSIGNMENTS = {
    ("i", "", "gedenyomanfajaranugrahwinartaputra"): NameComponents(
        given_name="I",
        surname="GedeNyomanFajarAnugrahWinartaPutra",
    ),
    ("i", "ketut", "junitha"): NameComponents(
        given_name="I",
        middle_name="Ketut",
        surname="Junitha",
    ),
    ("i", "made", "kamiana"): NameComponents(
        given_name="I",
        middle_name="Made",
        surname="Kamiana",
    ),
    ("i", "dewa putu", "pramantara"): NameComponents(
        given_name="I",
        middle_name="Dewa Putu",
        surname="Pramantara",
    ),
    ("i d g a", "", "subagia"): NameComponents(
        given_name="I",
        middle_name="D G A",
        surname="Subagia",
    ),
    ("a", "", "fernandez ajó"): NameComponents(
        given_name="A",
        surname="Fernandez Ajó",
    ),
    ("j.", "i.", "yi"): NameComponents(
        given_name="J. I.",
        surname="Yi",
    ),
}

# Exact Korean tuples manually adjudicated from frozen audit and paper-roster
# evidence. Each value is the count of leading source-middle tokens belonging
# to the given name. Zero preserves a fused source given name. Assignments take
# all spelling, case, and separators from the source.
REVIEWED_EXACT_KOREAN_GIVEN_PREFIX_PACKS = {
    ("bong", "soo", "cha"): 1,
    ("boo", "young", "ko"): 1,
    ("byoung", "yoon", "kim"): 1,
    ("chang", "hee", "lee"): 1,
    ("chang", "mo", "yang"): 1,
    ("changwoo", "", "lee"): 0,
    ("dong", "soo", "han"): 1,
    ("eun", "kee", "jeong"): 1,
    ("han", "jin", "jung"): 1,
    ("heung", "soo", "lee"): 1,
    ("hoe", "joon", "kim"): 1,
    ("hyoung", "sub", "kim"): 1,
    ("jae", "moon", "lee"): 1,
    ("jae", "won", "chung"): 1,
    ("jeong", "seon", "yeo"): 1,
    ("ji", "hyun", "moon"): 1,
    ("ji", "soo", "lee"): 1,
    ("ji", "woon", "ha"): 1,
    ("jin", "cheul", "kim"): 1,
    ("jong", "hak", "kim"): 1,
    ("jong", "ho", "kim"): 1,
    ("jong", "hoon", "kang"): 1,
    ("jong", "soo", "woo"): 1,
    ("joon", "young", "choi"): 1,
    ("kang", "ju", "kim"): 1,
    ("keum", "seok", "bae"): 1,
    ("kyeong", "ah", "kim"): 1,
    ("min", "young", "lee"): 1,
    ("minwoo", "", "lee"): 0,
    ("sang", "hoon", "han"): 1,
    ("sang", "hyub", "lee"): 1,
    ("sang", "min", "yoon"): 1,
    ("sang", "yong", "shin"): 1,
    ("sang", "yun", "han"): 1,
    ("sanghun", "", "lee"): 0,
    ("sangji", "", "lee"): 0,
    ("seok", "yong", "kang"): 1,
    ("seung", "jun", "lee"): 1,
    ("soo", "ick", "cho"): 1,
    ("su", "ja", "kim"): 1,
    ("su", "jin", "hwang"): 1,
    ("su", "jung", "choi"): 1,
    ("sumin", "", "lee"): 0,
    ("sung", "hoon", "chung"): 1,
    ("sung", "ik", "lee"): 1,
    ("sunghak", "", "lee"): 0,
    ("suji", "", "choi"): 0,
    ("weon", "ju", "lee"): 1,
    ("won", "hyung a.", "ryu"): 1,
    ("woo", "sung", "jeon"): 1,
    ("ye", "hun", "choi"): 1,
    ("yi", "ho", "lee"): 1,
    ("youme", "", "ko"): 0,
    ("young", "hee", "choi"): 1,
    ("young", "in", "shin"): 1,
    ("young", "mo", "sung"): 1,
    ("youn", "sik", "kim"): 1,
    ("yoon", "kyung", "choi"): 1,
    ("yunjin", "", "lee"): 0,
}
REVIEWED_EXACT_SOURCE_REORDER_VETOES = frozenset(
    {
        ("han", "w.", "tun"),
        ("kai", "", "zenger"),
        ("masaki", "", "morishige"),
        ("miki", "", "toyota"),
        ("shi", "(tracy)", "xu"),
        ("shinsei", "", "ryu"),
    },
)


def _source_comparison_text(value: str | None) -> str:
    """Trim source boundaries for comparisons without changing retained lineage."""
    return (value or "").strip()


def _structural_roman_hyphen_key(value: str | None) -> str:
    """Build an exact-set key using only approved structural Roman hyphens."""
    return _source_comparison_text(value).translate(_STRUCTURAL_ROMAN_HYPHEN_TRANSLATION).casefold()


def _source_component_key(source: SourceAuthorFields) -> tuple[str, str, str]:
    """Normalize one source tuple exactly as the reviewed rule inventory does."""
    return (
        " ".join((source.first_name or "").split()).casefold(),
        " ".join((source.middle_names or "").split()).casefold(),
        " ".join((source.last_name or "").split()).casefold(),
    )


def _alnum_key(value: str) -> str:
    """Return the case-insensitive letters and digits in one endpoint.

    Unlike ``east_asian_name_order._endpoint_key`` this deliberately applies no
    NFKC or accent folding: the reviewed tables were frozen against raw surfaces.
    """
    return "".join(character.casefold() for character in value if character.isalnum())


def _has_reviewed_cleanup_surname_prefix(tokens: list[str]) -> bool:
    """Return whether cleanup put initials or a credential before a surname."""
    return any(
        _MULTI_INITIAL_TOKEN_RE.fullmatch(token) or _alnum_key(token) in _REVIEWED_CLEANUP_PREFIX_CREDENTIAL_KEYS
        for token in tokens
    )


def reviewed_cleanup_surname_expansion_prefers_scalar(
    cleanup: NameComponents,
    scalar: NameComponents,
) -> bool:
    """Keep scalar's narrower surname for the reviewed cleanup expansion shape."""
    cleanup_tokens = cleanup.surname.split()
    scalar_tokens = scalar.surname.split()
    if not scalar_tokens or len(cleanup_tokens) <= len(scalar_tokens):
        return False
    if [_alnum_key(token) for token in cleanup_tokens[-len(scalar_tokens) :]] != [_alnum_key(token) for token in scalar_tokens]:
        return False
    return _has_reviewed_cleanup_surname_prefix(cleanup_tokens[: -len(scalar_tokens)])


def _compatibility_component_key(value: str) -> str:
    """Fold width, case, and punctuation for a structured component match."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _contains_katakana(value: str) -> bool:
    """Return whether a component contains Katakana after width folding."""
    return any(
        "\u30a0" <= character <= "\u30ff" and unicodedata.category(character).startswith("L")
        for character in unicodedata.normalize("NFKC", value)
    )


def _ends_in_catalog_comma(value: str) -> bool:
    """Return whether a component ends in a catalog comma."""
    return unicodedata.normalize("NFKC", value).rstrip().endswith((",", "、"))


def _contains_katakana_generation_suffix(value: str) -> bool:
    """Return whether a component contains a complete generation-suffix token."""
    normalized = unicodedata.normalize("NFKC", value)
    return any(
        _compatibility_component_key(token) in _KATAKANA_GENERATION_SUFFIXES
        for token in re.split(r"[\s,\u3001\u30fb\uff0c]+", normalized)
    )


def _alnum_token_sequence(value: str) -> tuple[str, ...]:
    """Return NFKC/case-folded maximal alphanumeric runs."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    tokens: list[str] = []
    current: list[str] = []
    for character in normalized:
        if character.isalnum():
            current.append(character)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tuple(tokens)


def _is_katakana_catalog_surface(value: str) -> bool:
    """Match the reviewed Katakana catalog grammar on one flattened name."""
    normalized = unicodedata.normalize("NFKC", value)
    segments = tuple(segment.strip() for segment in re.split(r"[,\u3001]", normalized) if segment.strip())
    if len(segments) < 2 or not any(separator in normalized for separator in _CATALOG_SEPARATORS):  # noqa: PLR2004
        return False
    allowed_punctuation = _CATALOG_SEPARATORS | frozenset({".", "-", "\u30fb", "(", ")", "[", "]"})
    return _contains_katakana(normalized) and all(
        _contains_katakana(character)
        or character.isspace()
        or unicodedata.category(character).startswith("M")
        or character in allowed_punctuation
        for character in normalized
    )


def reviewed_leading_jr_peer_assignment(
    source: SourceAuthorFields,
    paper_authors: list[SourceAuthorFields],
    focal_index: int,
) -> NameComponents | None:
    """Copy one exact structured peer and move a reviewed leading Jr to suffix."""
    leading_jr = _source_comparison_text(source.first_name)
    if leading_jr not in _REVIEWED_JR_FORMS or source.suffix:
        return None
    remainder = f"{source.middle_names or ''} {source.last_name or ''}".strip()
    organization_surface = unicodedata.normalize("NFKC", remainder)
    organization_words = {match.group().casefold() for match in _UNICODE_LETTER_WORD_RE.finditer(organization_surface)}
    if organization_words & _REVIEWED_JR_ORGANIZATION_WORDS:
        return None

    remainder_tokens = _alnum_token_sequence(remainder)
    matches = [
        peer
        for index, peer in enumerate(paper_authors)
        if index != focal_index
        and (peer.first_name or "").strip()
        and (peer.last_name or "").strip()
        and _alnum_token_sequence(peer.full_name()) == remainder_tokens
        and not (
            not (peer.middle_names or "").strip()
            and unicodedata.normalize("NFKC", (peer.first_name or "").strip()).casefold() in _REVIEWED_JR_FAMILY_PARTICLES
        )
    ]
    if len(matches) != 1:
        return None
    peer = matches[0]
    return NameComponents(
        given_name=peer.first_name or "",
        middle_name=peer.middle_names or "",
        surname=peer.last_name or "",
        suffix=leading_jr,
    )


def reviewed_fullwidth_katakana_alias_assignment(
    source: SourceAuthorFields,
    paper_names: list[str],
    focal_index: int,
) -> NameComponents | None:
    """Split one reviewed family-first Katakana catalog alias shape."""
    if (source.first_name or "").strip() or (source.middle_names or "").strip() or source.suffix:
        return None
    raw = source.last_name or ""
    if raw.count("\uff0c") != 1 or "," in raw or "\u3001" in raw or not _is_katakana_catalog_surface(raw):
        return None
    family, given = (segment.strip() for segment in raw.split("\uff0c"))
    normalized_segments = tuple(unicodedata.normalize("NFKC", segment) for segment in (family, given))
    if not family or not given or any(marker in segment for segment in normalized_segments for marker in ("\u30fb", ".")):
        return None

    normalized_given = normalized_segments[1]
    open_index = normalized_given.find("(")
    alias = normalized_given[open_index + 1 : -1].strip()
    if not (
        normalized_given[:open_index].strip()
        and alias
        and normalized_given.endswith(")")
        and normalized_given.count("(") == normalized_given.count(")") == 1
    ):
        return None
    if any(marker in _compatibility_component_key(raw) for marker in _KATAKANA_GENERATION_SUFFIXES) or not any(
        index != focal_index and _is_katakana_catalog_surface(name) for index, name in enumerate(paper_names)
    ):
        return None
    return NameComponents(given_name=given, surname=family)


def _is_middle_dot_person_segment(segment: str) -> bool:
    """Accept one conservative authored component around U+00B7."""
    has_letter = False
    for character in segment:
        category = unicodedata.category(character)
        if category.startswith("L"):
            has_letter = True
            continue
        if category.startswith("M") or character.isspace() or character in _MIDDLE_DOT_PERSON_PUNCTUATION:
            continue
        return False
    if not has_letter:
        return False

    words = {match.group().casefold() for match in _UNICODE_LETTER_WORD_RE.finditer(segment)}
    if words & _REVIEWED_JR_ORGANIZATION_WORDS:
        return False
    organization_markers = (*CJK_NON_PERSON_SUFFIX_MARKERS, *REVIEWED_HANGUL_ORGANIZATION_MARKERS)
    return not any(marker in segment for marker in organization_markers)


def _punctuate_bare_latin_initial(segment: str) -> str:
    """Add the canonical period to a single authored Latin letter."""
    return f"{segment}." if len(segment) == 1 and _is_latin_letter(segment) else segment


def reviewed_middle_dot_packed_transliteration_assignment(
    source: SourceAuthorFields,
) -> NameComponents | None:
    """Split a guarded last-field-only CJK transliteration at U+00B7.

    A middle dot is not globally a personal-name delimiter. This rule claims
    only the reviewed two- or three-component transliteration shape. It keeps
    authored component boundaries and scripts while canonicalizing bare Latin
    initials.
    """
    if (source.first_name or "").strip() or (source.middle_names or "").strip() or (source.suffix or "").strip():
        return None

    raw = source.last_name or ""
    segments = tuple(segment.strip() for segment in raw.split(_MIDDLE_DOT))
    if (
        len(segments) not in {2, 3}
        or any(not segment for segment in segments)
        or any(not _is_middle_dot_person_segment(segment) for segment in segments)
        or not any(_is_cjk_letter(character) for segment in segments for character in segment)
    ):
        return None

    given, *remainder = (_punctuate_bare_latin_initial(segment) for segment in segments)
    if len(remainder) == 1:
        return NameComponents(given_name=given, surname=remainder[0])
    return NameComponents(given_name=given, middle_name=remainder[0], surname=remainder[1])


def _endpoint_reorder_assignment(source: SourceAuthorFields) -> NameComponents:
    """Exchange reviewed endpoints and remove an authored catalog comma."""
    return NameComponents(
        given_name=source.last_name or "",
        surname=(source.first_name or "").rstrip(",\u060c"),
    )


def reviewed_exact_source_assignment(source: SourceAuthorFields) -> NameComponents | None:
    """Apply one manually reviewed assignment keyed to normalized source fields."""
    source_key = _source_component_key(source)
    if not (source.suffix or "").strip():
        empty_suffix_assignment = REVIEWED_EXACT_BARE_PERSONAL_ASSIGNMENTS.get(
            source_key,
        ) or REVIEWED_EXACT_EMPTY_SUFFIX_ASSIGNMENTS.get(source_key)
        if empty_suffix_assignment is None and source_key in REVIEWED_EXACT_EMPTY_SUFFIX_ENDPOINT_REORDERS:
            empty_suffix_assignment = _endpoint_reorder_assignment(source)
        if empty_suffix_assignment is not None:
            return empty_suffix_assignment
    literal_assignment = REVIEWED_EXACT_SOURCE_LITERAL_ASSIGNMENTS.get(source_key)
    if literal_assignment is not None:
        return literal_assignment
    korean_given_prefix_length = REVIEWED_EXACT_KOREAN_GIVEN_PREFIX_PACKS.get(source_key)
    if korean_given_prefix_length is not None and not (source.suffix or "").strip():
        source_first = " ".join((source.first_name or "").split())
        source_middle_tokens = (source.middle_names or "").split()
        given_name = " ".join((source_first, *source_middle_tokens[:korean_given_prefix_length]))
        middle_name = " ".join(source_middle_tokens[korean_given_prefix_length:])
        surname = " ".join((source.last_name or "").split())
        return NameComponents(
            given_name=given_name,
            middle_name=middle_name,
            surname=surname,
        )
    if source_key in REVIEWED_EXACT_SOURCE_ENDPOINT_REORDERS:
        return _endpoint_reorder_assignment(source)

    selectors = REVIEWED_EXACT_SOURCE_ROLE_ASSIGNMENTS.get(source_key)
    if selectors is None:
        return None
    given_field, middle_field, surname_field = selectors
    return NameComponents(
        given_name=getattr(source, given_field) or "",
        middle_name=getattr(source, middle_field) or "",
        surname=getattr(source, surname_field) or "",
    )


def reviewed_cyrillic_surname_given_patronymic_assignment(
    source: SourceAuthorFields,
) -> NameComponents | None:
    """Rotate a reviewed long-suffix Cyrillic surname-given-patronymic shape."""
    surname = (source.first_name or "").strip()
    given = (source.middle_names or "").strip()
    patronymic = (source.last_name or "").strip()
    if not all(_CYRILLIC_NAME_TOKEN_RE.fullmatch(part) for part in (surname, given, patronymic)):
        return None

    surname_key = surname.casefold()
    given_key = given.casefold()
    patronymic_key = patronymic.casefold()
    if given_key.endswith(_CYRILLIC_PATRONYMIC_SUFFIXES):
        return None
    gender_concordant = (
        surname_key.endswith(_FEMININE_CYRILLIC_SURNAME_SUFFIXES)
        and patronymic_key.endswith(_FEMININE_CYRILLIC_PATRONYMIC_SUFFIXES)
    ) or (
        surname_key.endswith(_MASCULINE_CYRILLIC_SURNAME_SUFFIXES)
        and patronymic_key.endswith(_MASCULINE_CYRILLIC_PATRONYMIC_SUFFIXES)
    )
    if not gender_concordant:
        return None
    return NameComponents(given_name=given, middle_name=patronymic, surname=surname)


def reviewed_hangul_affiliation_person_assignment(source: SourceAuthorFields) -> NameComponents | None:
    """Recover a Latin author after the complete reviewed MD/Hangul affiliation shape."""
    middle = (source.middle_names or "").strip()
    last_tokens = (source.last_name or "").split()
    if (
        (source.first_name or "").strip() != "MD"
        or (source.suffix or "").strip()
        or not middle.startswith("PhD ")
        or not any(marker in middle for marker in REVIEWED_HANGUL_ORGANIZATION_MARKERS)
        or not 2 <= len(last_tokens) <= 3  # noqa: PLR2004 - complete four-row corpus shape.
        or not all(token.isascii() and token.isalpha() for token in last_tokens)
    ):
        return None
    return NameComponents(
        given_name=" ".join(last_tokens[:-1]),
        surname=last_tokens[-1],
    )


def reviewed_katakana_middle_period_cyclic_reversal(
    source: SourceAuthorFields,
    selected: NameComponents,
) -> bool:
    """Match the reviewed catalog shape that promotes a middle token to given."""
    source_parts = (source.first_name or "", source.middle_names or "", source.last_name or "")
    middle = unicodedata.normalize("NFKC", source_parts[1]).strip()
    source_keys = tuple(_compatibility_component_key(part) for part in source_parts)
    selected_keys = tuple(
        _compatibility_component_key(part) for part in (selected.given_name, selected.middle_name, selected.surname)
    )
    return (
        all(source_keys)
        and all(_contains_katakana(part) for part in source_parts)
        and _ends_in_catalog_comma(source_parts[0])
        and _ends_in_catalog_comma(source_parts[2])
        and middle.endswith(".")
        and middle.count(".") == 1
        and not _contains_katakana_generation_suffix(source_parts[1])
        and selected_keys == (source_keys[1], source_keys[2], source_keys[0])
    )


def reviewed_exact_source_reversal(
    source: SourceAuthorFields,
    selected: NameComponents,
) -> bool:
    """Return whether a candidate reverses one exact reviewed given-first tuple."""
    if _source_component_key(source) not in REVIEWED_EXACT_SOURCE_REORDER_VETOES:
        return False
    return _alnum_key(source.first_name or "") == _alnum_key(selected.surname) and _alnum_key(
        source.last_name or "",
    ) == _alnum_key(selected.given_name)


def reviewed_initials_comma_reversal(
    source: SourceAuthorFields,
    selected: NameComponents,
) -> bool:
    """Return whether a candidate reverses a reviewed initials-comma source shape."""
    source_first = (source.first_name or "").rstrip()
    source_last = (source.last_name or "").strip()
    letter_runs = re.findall(r"[A-Za-z]+", source_first)
    if (
        _source_comparison_text(source.middle_names)
        or not source_first.isascii()
        or not source_first.endswith(",")
        or source_first.count(",") != 1
        or not 1 <= sum(map(len, letter_runs)) <= 3  # noqa: PLR2004 - reviewed initials cap.
        or any(len(run) != 1 for run in letter_runs)
        or not source_last
    ):
        return False

    first_key = _alnum_key(source_first)
    last_key = _alnum_key(source_last)
    return bool(
        first_key
        and last_key
        and first_key != last_key
        and first_key == _alnum_key(selected.surname)
        and last_key == _alnum_key(selected.given_name),
    )


def _preserve_reviewed_atomic_korean_source_given(
    source: SourceAuthorFields,
    selected: NameComponents,
) -> NameComponents:
    """Keep source spacing/case after the shared formatter preserves atomic tokens."""
    source_given = " ".join((source.first_name or "").split())
    source_tokens = tuple(source_given.split())
    if not any(token.casefold() in REVIEWED_ATOMIC_KOREAN_GIVEN_FORMS for token in source_tokens):
        return selected
    if _alnum_key(source_given) != _alnum_key(selected.given_name):
        return selected
    return replace(selected, given_name=source_given, given_tokens=source_tokens)


def routed_components_leak_cjk(parsed) -> bool:
    """Return whether a routed answer retains an unsegmented CJK letter."""
    joined = " ".join(part for part in (parsed.given_name, parsed.middle_name, parsed.surname) if part)
    return any(_is_cjk_letter(character) for character in joined)


def _is_cjk_letter(character: str) -> bool:
    """Return whether one Unicode letter belongs to a routed CJK script."""
    if not unicodedata.category(character).startswith("L"):
        return False
    name = unicodedata.name(character, "")
    return any(marker in name for marker in _CJK_LETTER_NAME_MARKERS)


def _is_latin_letter(character: str) -> bool:
    """Return whether one Unicode letter belongs to the Latin script."""
    return unicodedata.category(character).startswith("L") and "LATIN" in unicodedata.name(character, "")


def _pp_only_native_abstain_prefers_scalar(
    surface: str,
    japanese_probability: Callable[[str], float],
) -> bool:
    """Return whether spaced native-script evidence overrides a PP-only abstain parse."""
    if not any(character.isspace() for character in surface):
        return False
    if not surface or any(not character.isspace() and not _is_cjk_letter(character) for character in surface):
        return False

    probability = japanese_probability(surface)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        message = f"Japanese classifier returned invalid probability {probability!r}"
        raise EvidenceFailure(message)
    return probability >= JAPANESE_ML_THRESHOLD


def _pp_parse_uses_contextual_han_surname_reading(surface: str, result: ParseResult) -> bool:
    """Return whether PP applied a reviewed reading to its assigned Han surname."""
    original = result.parsed_original_order if result.success else None
    if original is None or not original.order:
        return False

    groups = surface.split()
    if original.order[0] == "surname":
        source_surname = groups[0] if groups else ""
    elif original.order[-1] == "surname":
        source_surname = groups[-1] if groups else ""
    else:
        return False
    if len(source_surname) != len(original.surname_tokens):
        return False
    for source_character, surname_token in zip(source_surname, original.surname_tokens, strict=True):
        if any(
            mapped_character == source_character and target.casefold() == surname_token.casefold()
            for (mapped_character, _source_reading), target in HAN_SURNAME_POSITION_READINGS.items()
        ):
            return True
    return False


def canonical_name_leaks_mixed_script(canonical_name) -> bool:
    """Return whether a normalized surname retains CJK beside Latin components."""
    if canonical_name is None:
        return False
    normalized = canonical_name.normalized
    components = (normalized.given_name, normalized.middle_name, normalized.surname, normalized.suffix)
    return any(_is_cjk_letter(character) for character in normalized.surname) and any(
        _is_latin_letter(character) for part in components for character in part
    )


def serialize_enum_values(value):
    """Recursively convert enum instances to plain values for `.dict()` output."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, list):
        return [serialize_enum_values(item) for item in value]
    if isinstance(value, tuple):
        return tuple(serialize_enum_values(item) for item in value)
    if isinstance(value, dict):
        return {key: serialize_enum_values(item) for key, item in value.items()}
    return value


class _Model(BaseModel):
    """Strict base model for the TIMO request and response boundary."""

    class Config:
        extra = "forbid"

        @staticmethod
        def schema_extra(schema: dict[str, Any], model: type[BaseModel]) -> None:
            """Expose Pydantic-v1 nullable fields as nullable JSON Schema types."""
            properties = schema["properties"]
            for field_name, field in model.__fields__.items():
                if field.allow_none:
                    property_name = field.alias if field.alias in properties else field_name
                    property_schema = properties[property_name]
                    if "type" in property_schema:
                        property_schema["type"] = [property_schema["type"], "null"]
                    else:
                        properties[property_name] = {
                            "anyOf": [property_schema, {"type": "null"}],
                        }

    def dict(self, *args, **kwargs):
        """Return plain Python serialization values for enum fields."""
        return serialize_enum_values(super().dict(*args, **kwargs))


class SourceAuthorFields(_Model):
    """Original fields retained losslessly for lineage and reviewed shape rules."""

    first_name: StrictStr | None = Field(default=None)
    middle_names: StrictStr | None = Field(default=None)
    last_name: StrictStr | None = Field(default=None)
    suffix: StrictStr | None = Field(default=None)

    def full_name(self) -> str:
        """Derive the current scalar/PP input; suffix is intentionally excluded."""
        return " ".join(part for part in (self.first_name or "", self.middle_names or "", self.last_name or "") if part).strip()


def _normalized_exact_endpoints(
    normalizer: PersonNameNormalizationService,
    *,
    given_name: str,
    surname: str,
) -> NameComponents | None:
    """Normalize two proven endpoints without permitting a role rewrite."""
    normalized = normalizer.normalize_components(first_name=given_name, last_name=surname)
    if normalized.outcome is not PersonNameOutcome.PERSON or normalized.canonical_name is None:
        return None
    selected = normalized.canonical_name.normalized
    if (
        selected.middle_name
        or selected.suffix
        or selected.given_name.casefold() != given_name.casefold()
        or selected.surname.casefold() != surname.casefold()
    ):
        return None
    return selected


def _reviewed_repeated_full_name_row(source: SourceAuthorFields) -> NameComponents | None:
    """Remove a source-last duplicate of the two retained atomic fields."""
    first = (source.first_name or "").strip()
    surname = (source.middle_names or "").strip()
    repeated = " ".join((source.last_name or "").split())
    if (
        (source.suffix or "").strip()
        or not first.isascii()
        or not surname.isascii()
        or not first.isalpha()
        or not surname.isalpha()
        or first.casefold() == surname.casefold()
        or repeated != f"{first} {surname}"
    ):
        return None
    return NameComponents(given_name=first, surname=surname)


def _reviewed_repeated_full_name_assignment(
    source: SourceAuthorFields,
    paper_authors: list[SourceAuthorFields],
) -> NameComponents | None:
    """Repair a duplicated full-name column only under complete-paper consensus."""
    selected = _reviewed_repeated_full_name_row(source)
    if selected is None or len(paper_authors) < 2:  # noqa: PLR2004
        return None
    return selected if all(_reviewed_repeated_full_name_row(peer) is not None for peer in paper_authors) else None


def _reviewed_titled_given_row(
    source: SourceAuthorFields,
    normalizer: PersonNameNormalizationService,
) -> NameComponents | None:
    """Flip one exact source-family plus dotted-title/given shape."""
    family = (source.first_name or "").strip()
    middle = (source.middle_names or "").strip()
    source_last = (source.last_name or "").strip()
    match = _TITLED_GIVEN_RE.fullmatch(source_last)
    if (
        middle
        or (source.suffix or "").strip()
        or not family
        or len(family.split()) != 1
        or family in {"Dr.", "Prof."}
        or match is None
    ):
        return None
    return _normalized_exact_endpoints(normalizer, given_name=match.group(1), surname=family)


def _reviewed_titled_given_assignment(
    source: SourceAuthorFields,
    paper_authors: list[SourceAuthorFields],
    normalizer: PersonNameNormalizationService,
) -> NameComponents | None:
    """Apply titled-given inversion only when every multi-author row agrees."""
    selected = _reviewed_titled_given_row(source, normalizer)
    if selected is None or len(paper_authors) < 2:  # noqa: PLR2004
        return None
    return selected if all(_reviewed_titled_given_row(peer, normalizer) is not None for peer in paper_authors) else None


def _reviewed_dotted_surname_initial_row(source: SourceAuthorFields) -> NameComponents | None:
    """Rotate one source-family plus two-or-three separately dotted initials row."""
    family = (source.first_name or "").strip()
    initials = (source.last_name or "").split()
    if (
        (source.middle_names or "").strip()
        or (source.suffix or "").strip()
        or len(family) <= 1
        or not family.isalpha()
        or not _MIN_DOTTED_SOURCE_INITIALS <= len(initials) <= _MAX_DOTTED_SOURCE_INITIALS
        or not all(_DOTTED_INITIAL_RE.fullmatch(token) for token in initials)
    ):
        return None
    return NameComponents(
        given_name=initials[0],
        middle_name=" ".join(initials[1:]),
        surname=family,
    )


def _reviewed_paper_wide_dotted_surname_initial_assignment(
    source: SourceAuthorFields,
    paper_authors: list[SourceAuthorFields],
) -> NameComponents | None:
    """Rotate dotted citation initials only under a uniform paper schema."""
    selected = _reviewed_dotted_surname_initial_row(source)
    if selected is None or len(paper_authors) < 2:  # noqa: PLR2004
        return None
    return selected if all(_reviewed_dotted_surname_initial_row(peer) is not None for peer in paper_authors) else None


def _reviewed_trailing_affiliation_row(
    source: SourceAuthorFields,
    normalizer: PersonNameNormalizationService,
    surname_is_recognized: Callable[[str], bool],
) -> NameComponents | None:
    """Remove one paper-wide lowercase ``a`` affiliation marker."""
    given = (source.first_name or "").strip()
    source_last = (source.last_name or "").strip()
    if (
        (source.middle_names or "").strip()
        or (source.suffix or "").strip()
        or _ASCII_HYPHENATED_WORD_RE.fullmatch(given) is None
        or _TRAILING_AFFILIATION_SURNAME_RE.fullmatch(source_last) is None
    ):
        return None
    surname = source_last[:-1]
    if not surname_is_recognized(surname):
        return None
    return _normalized_exact_endpoints(normalizer, given_name=given, surname=surname)


def _reviewed_paper_wide_trailing_affiliation_assignment(
    source: SourceAuthorFields,
    paper_authors: list[SourceAuthorFields],
    normalizer: PersonNameNormalizationService,
    surname_is_recognized: Callable[[str], bool],
) -> NameComponents | None:
    """Repair a leaked trailing ``a`` only on a fully reviewed paper roster."""
    roster_key = tuple(_source_component_key(peer) for peer in paper_authors)
    if roster_key not in _REVIEWED_TRAILING_AFFILIATION_ROSTERS:
        return None
    selected = _reviewed_trailing_affiliation_row(source, normalizer, surname_is_recognized)
    if selected is None:
        return None
    return (
        selected
        if all(_reviewed_trailing_affiliation_row(peer, normalizer, surname_is_recognized) is not None for peer in paper_authors)
        else None
    )


def _detached_diacritic_span_is_repairable(match: re.Match[str]) -> bool:
    """Return whether one detached marker span has the reviewed accent meaning."""
    head = match.group("head")
    mark = match.group("mark")
    tail = match.group("tail")
    if not tail[0].islower():
        return False
    if (head.casefold(), tail.casefold()) in _DETACHED_DIACRITIC_TRANSLITERATION_CONTROLS:
        return False
    _combining, allowed_next = _DETACHED_DIACRITIC_SPECS[mark]
    return tail[0].casefold() in allowed_next


def _repair_detached_diacritic_spans(value: str) -> tuple[str, int]:
    """Compose every detached marker span admitted by the reviewed grammar."""
    repair_count = 0

    def replace_span(match: re.Match[str]) -> str:
        nonlocal repair_count
        if not _detached_diacritic_span_is_repairable(match):
            return match.group(0)
        head = match.group("head")
        mark = match.group("mark")
        tail = match.group("tail")
        combining, _allowed_next = _DETACHED_DIACRITIC_SPECS[mark]
        accented = unicodedata.normalize("NFC", f"{tail[0]}{combining}")
        repair_count += 1
        return f"{head}{accented}{tail[1:]}"

    previous = None
    while previous != value:
        previous, value = value, _DETACHED_DIACRITIC_SPAN_RE.sub(replace_span, value)
    return value, repair_count


def _reviewed_detached_diacritic_row(
    source: SourceAuthorFields,
    normalizer: PersonNameNormalizationService,
) -> NameComponents | None:
    """Repair one intrinsically valid detached accent source row."""
    if (source.suffix or "").strip():
        return None
    middle = source.middle_names or ""
    if (
        not _DETACHED_DIACRITIC_TOKEN_RE.search(middle)
        or _DETACHED_DIACRITIC_TOKEN_RE.search(source.first_name or "")
        or _DETACHED_DIACRITIC_TOKEN_RE.search(source.last_name or "")
    ):
        return None
    raw_name = source.full_name()
    marker_count = len(_DETACHED_DIACRITIC_TOKEN_RE.findall(raw_name))
    matches = list(_DETACHED_DIACRITIC_SPAN_RE.finditer(raw_name))
    if (
        marker_count == 0
        or len(matches) != marker_count
        or not all(_detached_diacritic_span_is_repairable(match) for match in matches)
    ):
        return None
    repaired, repair_count = _repair_detached_diacritic_spans(raw_name)
    if repair_count != marker_count or _DETACHED_DIACRITIC_TOKEN_RE.search(repaired):
        return None
    normalized = normalizer.normalize_text(repaired)
    if normalized.outcome is not PersonNameOutcome.PERSON or normalized.canonical_name is None:
        return None
    selected = normalized.canonical_name.normalized
    return selected if selected.surname else None


def _reviewed_detached_diacritic_assignment(
    source: SourceAuthorFields,
    paper_authors: list[SourceAuthorFields],
    normalizer: PersonNameNormalizationService,
) -> NameComponents | None:
    """Repair detached accents with two independently valid paper rows."""
    selected = _reviewed_detached_diacritic_row(source, normalizer)
    if selected is None:
        return None
    valid_count = 0
    for peer in paper_authors:
        valid_count += _reviewed_detached_diacritic_row(peer, normalizer) is not None
        if valid_count >= 2:  # noqa: PLR2004
            return selected
    return None


def _reviewed_joined_uppercase_surname_assignment(
    source: SourceAuthorFields,
    paper_authors: list[SourceAuthorFields],
    surname_is_recognized: Callable[[str], bool],
) -> NameComponents | None:
    """Split an uppercase surname prefix with one other recognized peer row."""
    if (source.first_name or "").strip() or (source.middle_names or "").strip() or (source.suffix or "").strip():
        return None
    focal_match = _JOINED_UPPERCASE_SURNAME_RE.fullmatch(source.last_name or "")
    if focal_match is None or not surname_is_recognized(focal_match.group(1)):
        return None
    recognized_rows = 0
    for peer in paper_authors:
        if (peer.first_name or "").strip() or (peer.middle_names or "").strip() or (peer.suffix or "").strip():
            continue
        peer_match = _JOINED_UPPERCASE_SURNAME_RE.fullmatch(peer.last_name or "")
        if peer_match is not None and surname_is_recognized(peer_match.group(1)):
            recognized_rows += 1
    if recognized_rows < 2:  # noqa: PLR2004
        return None
    surname, given = focal_match.groups()
    return NameComponents(given_name=given, surname=surname.title())


def _joined_uppercase_candidate_adds_normalization(
    selected: NameComponents,
    expected: NameComponents,
) -> bool:
    """Return whether an aligned candidate adds information beyond the source split."""

    def role_key(value: str) -> str:
        return "".join(
            character.casefold() for character in value if not character.isspace() and character not in ROMAN_HYPHEN_LIKE
        )

    roles_match = role_key(f"{selected.given_name}{selected.middle_name}") == role_key(
        expected.given_name,
    ) and role_key(selected.surname) == role_key(expected.surname)
    selected_fields = (selected.given_name, selected.middle_name, selected.surname, selected.suffix)
    expected_fields = (expected.given_name, expected.middle_name, expected.surname, expected.suffix)
    return roles_match and selected_fields != expected_fields


def _south_indian_terminal_initial_row(source: SourceAuthorFields, *, focal: bool) -> bool:
    """Match one role in the reviewed three-author terminal-initial schema."""
    first = (source.first_name or "").strip()
    last_tokens = (source.last_name or "").split()
    if (
        (source.middle_names or "").strip()
        or (source.suffix or "").strip()
        or len(first) <= 1
        or _ASCII_WORD_RE.fullmatch(first) is None
    ):
        return False
    if focal:
        return (
            len(last_tokens) == 2  # noqa: PLR2004
            and len(last_tokens[0]) > 1
            and _ASCII_WORD_RE.fullmatch(last_tokens[0]) is not None
            and _ASCII_INITIAL_RE.fullmatch(last_tokens[1]) is not None
        )
    return len(last_tokens) == 1 and _ASCII_INITIAL_RE.fullmatch(last_tokens[0]) is not None


def _reviewed_south_indian_terminal_initial_context(
    source: SourceAuthorFields,
    paper_authors: list[SourceAuthorFields],
    focal_index: int,
) -> bool:
    """Identify the one closed triad where scalar should beat source repartition."""
    return (
        len(paper_authors) == 3  # noqa: PLR2004
        and _south_indian_terminal_initial_row(source, focal=True)
        and all(
            _south_indian_terminal_initial_row(peer, focal=False)
            for index, peer in enumerate(paper_authors)
            if index != focal_index
        )
    )


def _reviewed_leading_title_or_credential_assignment(  # noqa: PLR0911 - closed source grammars fail independently.
    source: SourceAuthorFields,
    normalizer: PersonNameNormalizationService,
) -> NameComponents | None:
    """Remove one exact reviewed prefix from a complete structured name."""
    first, middle, last, suffix = (
        (value or "").strip() for value in (source.first_name, source.middle_names, source.last_name, source.suffix)
    )
    if suffix:
        return None

    retained_first: str | None = None
    retained_middle: str | None = None
    if first.casefold() == "rn" and middle.casefold() == "msn":
        packed_name = last.split()
        if len(packed_name) != 2 or not all(token.isalpha() for token in packed_name):  # noqa: PLR2004
            return None
        retained_first, last = packed_name
    elif first in {"Er.", "M.Pd"} and len(middle.split()) == 1 and len(last.split()) == 1:
        retained_first = middle
    elif first.startswith("MUDr.") and first != "MUDr." and not middle and len(last.split()) == 1:
        attached_name = first.removeprefix("MUDr.")
        if len(attached_name.split()) != 1:
            return None
        retained_first = attached_name
    elif first == "Assist" and len(last.split()) == 1:
        match middle.split():
            case [".Lect.", given_name, middle_name]:
                retained_middle = f"{given_name} {middle_name}"
            case _:
                return None
    else:
        return None

    atomic = normalizer.normalize_components(
        first_name=retained_first,
        middle_name=retained_middle,
        last_name=last,
    )
    if atomic.outcome is not PersonNameOutcome.PERSON or atomic.canonical_name is None:
        return None
    selected = atomic.canonical_name.normalized
    return selected if selected.given_name and selected.surname else None


def _reviewed_closed_comma_credential_assignment(
    source: SourceAuthorFields,
    normalizer: PersonNameNormalizationService,
) -> NameComponents | None:
    """Normalize one full-corpus-reviewed closed credential tail atomically."""
    if reviewed_closed_comma_credential_tail_head(source.last_name, source.suffix) is None:
        return None
    normalized = normalizer.normalize_components(
        first_name=source.first_name,
        middle_name=source.middle_names,
        last_name=source.last_name,
        suffix=source.suffix,
    )
    if normalized.outcome is not PersonNameOutcome.PERSON or normalized.canonical_name is None:
        return None

    selected = normalized.canonical_name.normalized
    return selected if selected.given_name and selected.surname else None


class Instance(_Model):
    """One paper and only the non-focal portion of its optional VYS pool.

    Focal names are derived from ``pp_authors`` and prepended internally.  This
    makes it impossible to supply a second focal slice that disagrees with the
    source authors.  All alignment is positional; duplicate text is valid and
    must never be joined back to authors by name.

    An omitted or empty ``vys_other_names`` list selects PP-only routing. A
    nonempty list enables PP/VYS routing.
    """

    pp_authors: list[SourceAuthorFields] = Field(
        description="paper authors; output remains aligned to this exact order",
    )
    vys_other_names: list[StrictStr] = Field(
        default_factory=list,
        description="non-focal VYS names only; an empty list selects PP-only routing",
    )

    def dict(self, *args, **kwargs):
        """Serialize the request without nullable source components."""
        kwargs.setdefault("exclude_none", True)
        return super().dict(*args, **kwargs)

    def json(self, *args, **kwargs):
        """Serialize request JSON without nullable source components."""
        kwargs.setdefault("exclude_none", True)
        return super().json(*args, **kwargs)


def merge_resolved_suffix(source_suffix: str | None, selected_suffix: str | None) -> str | None:
    """Apply the complete current fill-only suffix policy.

    A nonempty source suffix wins unchanged.  Otherwise a nonempty suffix from
    the selected semantic result fills it.  If neither is available, the exact
    source value (``None`` or ``""``) is retained.  Whitespace is nonempty and
    is therefore preserved rather than silently cleaned at this boundary.
    """
    if source_suffix:
        return source_suffix
    if selected_suffix:
        return selected_suffix
    return source_suffix


class ResolvedAuthorFields(_Model):
    """One terminal, directly writable author-field result.

    ``PRESERVE_INPUT`` means the selected policy did not flip the derived input
    order. ``SUPPRESS`` tells the writer not to emit the author while retaining
    this aligned diagnostic slot. ``SOURCE`` plus either action preserves source
    boundary text, mapping absent name components to required empty strings;
    ``SOURCE`` plus ``ASSIGN`` applies roles supported by reviewed source
    structure or one exact reviewed source tuple.
    PP/VYS materialization may also assign source tokens to output fields. The
    suffix is already final, so an application must not merge it again.
    """

    first_name: StrictStr
    middle_names: StrictStr
    last_name: StrictStr
    suffix: StrictStr | None = None
    resolution_provenance: ResolutionProvenance
    resolution_action: ResolutionAction
    resolution_reason: ResolutionReason

    @root_validator
    def _validate_legal_decision(cls, values):  # noqa: N805
        """Reject reason/provenance/action combinations outside the closed table."""
        reason = values.get("resolution_reason")
        provenance = values.get("resolution_provenance")
        action = values.get("resolution_action")
        if reason is None or provenance is None or action is None:
            return values
        expected = resolution_decision_spec(reason)
        if provenance is not expected.provenance or action is not expected.action:
            message = (
                f"{reason.value} requires "
                f"({expected.provenance.value}, {expected.action.value}), got "
                f"({provenance.value}, {action.value})"
            )
            raise ValueError(message)
        return values

    @classmethod
    def from_selected_components(
        cls,
        *,
        source: SourceAuthorFields,
        selected: NameComponents,
        reason: ResolutionReason,
    ) -> ResolvedAuthorFields:
        """Materialize one selected result, including the final suffix and metadata."""
        decision = resolution_decision_spec(reason)
        if decision.provenance is ResolutionProvenance.SOURCE and decision.action is not ResolutionAction.ASSIGN:
            message = f"{reason.value} requires exact source materialization"
            raise ValueError(message)
        return cls(
            first_name=selected.given_name,
            middle_names=selected.middle_name,
            last_name=selected.surname,
            suffix=merge_resolved_suffix(source.suffix, selected.suffix),
            resolution_provenance=decision.provenance,
            resolution_action=decision.action,
            resolution_reason=reason,
        )

    @classmethod
    def from_source(
        cls,
        source: SourceAuthorFields,
        *,
        reason: ResolutionReason,
    ) -> ResolvedAuthorFields:
        """Copy source boundaries, mapping absent name components to empty strings."""
        decision = resolution_decision_spec(reason)
        if decision.provenance is not ResolutionProvenance.SOURCE or decision.action is ResolutionAction.ASSIGN:
            message = f"{reason.value} is not a SOURCE non-assignment reason"
            raise ValueError(message)
        return cls(
            first_name=source.first_name or "",
            middle_names=source.middle_names or "",
            last_name=source.last_name or "",
            suffix=source.suffix,
            resolution_provenance=decision.provenance,
            resolution_action=decision.action,
            resolution_reason=reason,
        )


class _Resolver:
    """Apply the terminal author policy to scalar and batch candidates."""

    def __init__(self, detector: ChineseNameDetector):
        self._detector = detector
        self._source_normalizer = PersonNameNormalizationService()

    @staticmethod
    def _source_resolution(
        source: SourceAuthorFields,
        reason: ResolutionReason,
    ) -> ResolvedAuthorFields:
        return ResolvedAuthorFields.from_source(source, reason=reason)

    @staticmethod
    def _selected_has_personal_initials(selected: NameComponents) -> bool:
        """Return whether a reviewed semantic assignment needs initial canonicalization."""

        for value in (selected.given_name, selected.middle_name):
            for token in value.split():
                parts = [part for part in re.split(r"[.-]+", token) if part]
                if parts and all(len(part) == 1 and unicodedata.name(part, "").startswith("LATIN ") for part in parts):
                    return True
        return False

    def _source_assignment_resolution(
        self,
        *,
        source: SourceAuthorFields,
        selected: NameComponents,
        reason: ResolutionReason,
    ) -> ResolvedAuthorFields:
        """Canonicalize initials in a SOURCE assignment.

        SOURCE non-assignment resolutions remain byte-exact. These assignments
        already have proven semantic roles, but their source-derived components
        bypass the ordinary formatters.
        """

        if self._selected_has_personal_initials(selected):
            normalized = self._source_normalizer.normalize_components(
                first_name=selected.given_name,
                middle_name=selected.middle_name,
                last_name=selected.surname,
                suffix=selected.suffix,
            )
            if normalized.outcome is not PersonNameOutcome.PERSON or normalized.canonical_name is None:
                message = f"source assignment could not be normalized: {selected!r}"
                raise RuntimeError(message)
            selected = normalized.canonical_name.normalized
        return ResolvedAuthorFields.from_selected_components(
            source=source,
            selected=selected,
            reason=reason,
        )

    def _hard_scalar_resolution(
        self,
        source: SourceAuthorFields,
        constraint: HardScalarConstraint,
    ) -> ResolvedAuthorFields:
        """Materialize the one canonical value carried by a hard constraint."""
        canonical = constraint.canonical_name
        if canonical_name_leaks_mixed_script(canonical) or not canonical.normalized.surname:
            return self._source_resolution(
                source,
                ResolutionReason.HARD_SCALAR_MATERIALIZATION_FAILED,
            )
        return ResolvedAuthorFields.from_selected_components(
            source=source,
            selected=canonical.normalized,
            reason=constraint.reason,
        )

    def _scalar_repartitions_reviewed_compound_surname(
        self,
        source: SourceAuthorFields,
        selected: NameComponents,
    ) -> bool:
        """Return whether scalar parsing only breaks apart a compound surname."""
        surname = (source.last_name or "").split()
        if len(surname) <= 1:
            return False
        source_is_reviewed = _source_component_key(source) in REVIEWED_SCALAR_COMPOUND_SURNAMES
        source_is_curated = self._detector.is_curated_compound_surname(source.last_name or "")
        if not source_is_reviewed and not source_is_curated:
            return False

        given = (source.first_name or "").split()
        middle = (source.middle_names or "").split()
        selected_personal = [*selected.given_name.split(), *selected.middle_name.split()]
        source_personal = [*given, *middle, *surname[:-1]]

        def surface_key(tokens: Sequence[str]) -> str:
            return _alnum_key(" ".join(tokens))

        if (
            source_is_reviewed
            and surface_key(selected_personal) == surface_key([*given, *middle])
            and surface_key(selected.surname.split()) == surface_key(surname)
        ):
            return True

        return surface_key(selected_personal) == surface_key(source_personal) and surface_key(
            selected.surname.split(),
        ) == surface_key(surname[-1:])

    def _structured_surname_initial_tail_candidate(
        self,
        source: SourceAuthorFields,
        selected: NameComponents,
    ) -> NameComponents | None:
        """Recover surname-first initials proven by a last-name-only source field."""
        if (source.first_name or "").strip() or (source.middle_names or "").strip():
            return None
        normalized = self._source_normalizer.normalize_components(
            last_name=source.last_name,
            suffix=source.suffix,
        )
        if normalized.outcome is not PersonNameOutcome.PERSON or normalized.canonical_name is None:
            return None
        candidate = normalized.canonical_name.normalized
        initials = [*candidate.given_name.split(), *candidate.middle_name.split()]
        is_initial = PersonNameNormalizationService._is_initial  # noqa: SLF001
        if len(initials) <= 1 or not all(is_initial(token) for token in initials):
            return None

        def folded(value: str) -> list[str]:
            return [token.casefold() for token in value.split()]

        return (
            candidate
            if folded(candidate.surname) == folded(selected.given_name)
            and [*folded(candidate.given_name), *folded(candidate.middle_name)]
            == [*folded(selected.middle_name), *folded(selected.surname)]
            and (bool((source.suffix or "").strip()) or folded(candidate.suffix) == folded(selected.suffix))
            else None
        )

    def _scalar_clean_source_surname_repartition_candidate(  # noqa: C901, PLR0911
        self,
        source: SourceAuthorFields,
        selected: NameComponents,
    ) -> NameComponents | None:
        """Return a clean structured-surname alternative to a scalar repartition."""
        if len((source.last_name or "").split()) < 2:  # noqa: PLR2004
            return None
        normalized = self._source_normalizer.normalize_components(
            first_name=source.first_name,
            middle_name=source.middle_names,
            last_name=source.last_name,
            suffix=source.suffix,
        )
        if normalized.outcome is not PersonNameOutcome.PERSON or normalized.canonical_name is None:
            return None
        candidate = normalized.canonical_name.normalized

        def folded(value: str) -> list[str]:
            return [token.casefold() for token in value.split()]

        selected_surname = selected.surname.split()
        candidate_surname = candidate.surname.split()
        if not selected_surname or len(selected_surname) >= len(candidate_surname):
            return None
        if folded(selected.surname) != [token.casefold() for token in candidate_surname[-len(selected_surname) :]]:
            return None
        peeled = candidate_surname[: -len(selected_surname)]
        if folded(selected.given_name) != folded(candidate.given_name):
            return None
        if folded(selected.middle_name) != [*folded(candidate.middle_name), *[token.casefold() for token in peeled]]:
            return None
        if folded(selected.suffix) != folded(candidate.suffix):
            return None
        is_initial = PersonNameNormalizationService._is_initial  # noqa: SLF001
        if any(is_initial(token) for token in (source.first_name or "").split()):
            return None
        if any(is_initial(token) for token in peeled):
            return None
        return candidate

    def _reviewed_source_rules_resolution(
        self,
        source: SourceAuthorFields,
        paper_authors: list[SourceAuthorFields],
        focal_index: int,
    ) -> ResolvedAuthorFields | None:
        """Apply the closed reviewed-source rules in precedence order.

        Rules are evaluated lazily so a later rule never runs (or raises) on a
        row a higher-precedence rule already claims.
        """
        pattern_reason = ResolutionReason.REVIEWED_SOURCE_PATTERN_ASSIGNMENT
        source_key = _source_component_key(source)
        if not (source.suffix or "").strip():
            bare_personal_assignment = REVIEWED_EXACT_BARE_PERSONAL_ASSIGNMENTS.get(source_key)
            if bare_personal_assignment is not None:
                return ResolvedAuthorFields.from_selected_components(
                    source=source,
                    selected=bare_personal_assignment,
                    reason=ResolutionReason.REVIEWED_EXACT_SOURCE_ASSIGNMENT,
                )
        hangul_assignment = reviewed_hangul_affiliation_person_assignment(source)
        if hangul_assignment is not None:
            return self._source_assignment_resolution(
                source=source,
                selected=hangul_assignment,
                reason=pattern_reason,
            )
        middle_dot_assignment = reviewed_middle_dot_packed_transliteration_assignment(source)
        if middle_dot_assignment is not None:
            return ResolvedAuthorFields.from_selected_components(
                source=source,
                selected=middle_dot_assignment,
                reason=pattern_reason,
            )
        if (
            reviewed_non_person_source_pattern(source.first_name, source.middle_names, source.last_name, source.suffix)
            is not None
        ):
            return self._source_resolution(source, ResolutionReason.REVIEWED_NON_PERSON_PATTERN)
        assignment_rules: tuple[tuple[Callable[[], NameComponents | None], ResolutionReason], ...] = (
            (lambda: reviewed_exact_source_assignment(source), ResolutionReason.REVIEWED_EXACT_SOURCE_ASSIGNMENT),
            (
                lambda: _reviewed_detached_diacritic_assignment(source, paper_authors, self._source_normalizer),
                pattern_reason,
            ),
            (lambda: _reviewed_repeated_full_name_assignment(source, paper_authors), pattern_reason),
            (
                lambda: _reviewed_titled_given_assignment(source, paper_authors, self._source_normalizer),
                pattern_reason,
            ),
            (
                lambda: _reviewed_paper_wide_dotted_surname_initial_assignment(source, paper_authors),
                ResolutionReason.STRUCTURED_SURNAME_INITIAL_TAIL_ASSIGNMENT,
            ),
            (
                lambda: _reviewed_paper_wide_trailing_affiliation_assignment(
                    source,
                    paper_authors,
                    self._source_normalizer,
                    lambda surname: self._detector._require_surname_resolver().parser_is_surname((surname,)),  # noqa: SLF001
                ),
                pattern_reason,
            ),
            (
                lambda: _reviewed_leading_title_or_credential_assignment(source, self._source_normalizer),
                pattern_reason,
            ),
            (lambda: reviewed_cyrillic_surname_given_patronymic_assignment(source), pattern_reason),
            (lambda: reviewed_leading_jr_peer_assignment(source, paper_authors, focal_index), pattern_reason),
            (lambda: _reviewed_closed_comma_credential_assignment(source, self._source_normalizer), pattern_reason),
        )
        for rule, reason in assignment_rules:
            selected = rule()
            if selected is not None:
                return self._source_assignment_resolution(source=source, selected=selected, reason=reason)
        return None

    def _reviewed_cleanup_resolution(
        self,
        source: SourceAuthorFields,
    ) -> tuple[ResolvedAuthorFields | None, NameComponents | None]:
        """Resolve the reviewed cleanup pattern, or defer it to a scalar tiebreak.

        Returns ``(resolved, None)`` when cleanup is terminal, ``(None, selected)``
        when cleanup expanded the surname and must be weighed against the scalar
        result, and ``(None, None)`` when the pattern does not apply.
        """
        cleanup_pattern = reviewed_source_cleanup_pattern(
            source.first_name,
            source.middle_names,
            source.last_name,
            source.suffix,
        )
        if cleanup_pattern is None:
            return None, None
        normalized = self._source_normalizer.normalize_components(
            first_name=source.first_name,
            middle_name=source.middle_names,
            last_name=source.last_name,
            suffix=source.suffix,
        )
        if normalized.outcome is not PersonNameOutcome.PERSON or normalized.canonical_name is None:
            return None, None
        selected = normalized.canonical_name.normalized
        source_suffix = (source.suffix or "").strip()
        if not normalized.dropped_tokens and selected.suffix == source_suffix:
            return None, None
        surname_tokens = selected.surname.split()
        if len(surname_tokens) <= 1 or not _has_reviewed_cleanup_surname_prefix(surname_tokens[:-1]):
            resolved = self._source_assignment_resolution(
                source=source,
                selected=selected,
                reason=ResolutionReason.REVIEWED_SOURCE_PATTERN_ASSIGNMENT,
            )
            return resolved, None
        return None, selected

    def terminal_resolution(  # noqa: C901, PLR0911, PLR0912, PLR0913
        self,
        *,
        source: SourceAuthorFields,
        paper_authors: list[SourceAuthorFields],
        raw_name: str,
        paper_names: list[str],
        focal_index: int,
        parsed,
        batch_reason: ResolutionReason,
        router_not_person: bool,
    ) -> ResolvedAuthorFields:
        """Choose and materialize exactly one operational author result."""
        reviewed_resolution = self._reviewed_source_rules_resolution(source, paper_authors, focal_index)
        if reviewed_resolution is not None:
            return reviewed_resolution
        cleanup_resolution, cleanup_selected = self._reviewed_cleanup_resolution(source)
        if cleanup_resolution is not None:
            return cleanup_resolution
        try:
            scalar_resolution = self._detector.routing_scalar_resolution(raw_name)
        except EvidenceFailure as error:
            LOGGER.warning(
                "TIMO resolution preserved source fields after evidence failure for %r: %s",
                raw_name,
                error,
            )
            return self._source_resolution(
                source,
                ResolutionReason.HANDLED_EVIDENCE_FAILURE,
            )
        except HardScalarMaterializationFailure as error:
            LOGGER.warning(
                "TIMO resolution preserved source fields after hard scalar materialization failed for %r: %s",
                raw_name,
                error,
            )
            return self._source_resolution(
                source,
                ResolutionReason.HARD_SCALAR_MATERIALIZATION_FAILED,
            )

        if cleanup_selected is not None:
            if (
                scalar_resolution is not None
                and not isinstance(scalar_resolution, ApplyAssignment | PreserveBaseline)
                and not canonical_name_leaks_mixed_script(scalar_resolution)
                and reviewed_cleanup_surname_expansion_prefers_scalar(
                    cleanup_selected,
                    scalar_resolution.normalized,
                )
            ):
                return self._materialize_selected_candidate(
                    source=source,
                    paper_authors=paper_authors,
                    raw_name=raw_name,
                    paper_names=paper_names,
                    focal_index=focal_index,
                    selected=scalar_resolution.normalized,
                    reason=ResolutionReason.SCALAR_BASELINE,
                )
            return self._source_assignment_resolution(
                source=source,
                selected=cleanup_selected,
                reason=ResolutionReason.REVIEWED_SOURCE_PATTERN_ASSIGNMENT,
            )

        if isinstance(scalar_resolution, ApplyAssignment | PreserveBaseline):
            return self._hard_scalar_resolution(source, scalar_resolution)

        scalar_canonical = scalar_resolution
        scalar_is_unsafe = canonical_name_leaks_mixed_script(scalar_canonical)
        if (
            not scalar_is_unsafe
            and scalar_canonical is not None
            and scalar_canonical.normalized.surname
            and self._scalar_repartitions_reviewed_compound_surname(source, scalar_canonical.normalized)
        ):
            return self._source_resolution(
                source,
                ResolutionReason.SCALAR_KNOWN_COMPOUND_SURNAME_PRESERVE_INPUT,
            )

        if parsed is not None and routed_components_leak_cjk(parsed):
            return self._source_resolution(
                source,
                ResolutionReason.ROUTED_CJK_SAFETY_SUPPRESSION,
            )
        if batch_reason in {
            ResolutionReason.PP_ONLY_ABSTAIN_REVIEWED_COMPOUND_SURNAME,
            ResolutionReason.BATCH_ABSTAIN_MATERIALIZATION_FAILED,
        }:
            return self._source_resolution(source, batch_reason)

        selected_suffix = scalar_canonical.normalized.suffix if scalar_canonical is not None and not scalar_is_unsafe else None
        if parsed is not None and parsed.surname:
            return self._materialize_selected_candidate(
                source=source,
                paper_authors=paper_authors,
                raw_name=raw_name,
                paper_names=paper_names,
                focal_index=focal_index,
                selected=NameComponents(
                    given_name=parsed.given_name,
                    middle_name=parsed.middle_name,
                    surname=parsed.surname,
                    suffix=selected_suffix or "",
                ),
                reason=batch_reason,
            )

        if scalar_is_unsafe:
            return self._source_resolution(
                source,
                ResolutionReason.MIXED_SCRIPT_SAFETY_SUPPRESSION,
            )
        if scalar_canonical is not None and scalar_canonical.normalized.surname:
            structured_initial_tail = self._structured_surname_initial_tail_candidate(
                source,
                scalar_canonical.normalized,
            )
            if structured_initial_tail is not None:
                return self._source_assignment_resolution(
                    source=source,
                    selected=structured_initial_tail,
                    reason=ResolutionReason.STRUCTURED_SURNAME_INITIAL_TAIL_ASSIGNMENT,
                )
            source_surname_candidate = self._scalar_clean_source_surname_repartition_candidate(
                source,
                scalar_canonical.normalized,
            )
            if source_surname_candidate is not None and not _reviewed_south_indian_terminal_initial_context(
                source,
                paper_authors,
                focal_index,
            ):
                return self._source_assignment_resolution(
                    source=source,
                    selected=source_surname_candidate,
                    reason=ResolutionReason.SCALAR_CLEAN_SOURCE_SURNAME_REPARTITION_ASSIGNMENT,
                )
            return self._materialize_selected_candidate(
                source=source,
                paper_authors=paper_authors,
                raw_name=raw_name,
                paper_names=paper_names,
                focal_index=focal_index,
                selected=scalar_canonical.normalized,
                reason=ResolutionReason.SCALAR_BASELINE,
            )

        if router_not_person:
            katakana_assignment = reviewed_fullwidth_katakana_alias_assignment(source, paper_names, focal_index)
            if katakana_assignment is not None:
                return self._source_assignment_resolution(
                    source=source,
                    selected=katakana_assignment,
                    reason=ResolutionReason.REVIEWED_SOURCE_PATTERN_ASSIGNMENT,
                )
        source_reason = (
            ResolutionReason.NON_PERSON_SOURCE_PASSTHROUGH if router_not_person else ResolutionReason.NO_USABLE_SEMANTIC_RESULT
        )
        return self._source_resolution(source, source_reason)

    def _materialize_selected_candidate(  # noqa: PLR0913
        self,
        *,
        source: SourceAuthorFields,
        paper_authors: list[SourceAuthorFields],
        raw_name: str,
        paper_names: list[str],
        focal_index: int,
        selected: NameComponents,
        reason: ResolutionReason,
    ) -> ResolvedAuthorFields:
        """Apply candidate-aware reorder vetoes, then materialize once."""
        if reviewed_initials_comma_reversal(source, selected) or reviewed_katakana_middle_period_cyclic_reversal(
            source,
            selected,
        ):
            return self._source_resolution(
                source,
                ResolutionReason.INITIALS_COMMA_REORDER_VETO_PRESERVE_INPUT,
            )
        if reviewed_exact_source_reversal(source, selected):
            return self._source_resolution(
                source,
                ResolutionReason.REVIEWED_EXACT_SOURCE_REORDER_VETO_PRESERVE_INPUT,
            )
        selected = _preserve_reviewed_atomic_korean_source_given(source, selected)
        joined_uppercase_assignment = _reviewed_joined_uppercase_surname_assignment(
            source,
            paper_authors,
            lambda surname: self._detector._require_surname_resolver().parser_is_surname((surname,)),  # noqa: SLF001
        )
        if joined_uppercase_assignment is not None and not _joined_uppercase_candidate_adds_normalization(
            selected,
            joined_uppercase_assignment,
        ):
            return self._source_assignment_resolution(
                source=source,
                selected=joined_uppercase_assignment,
                reason=ResolutionReason.REVIEWED_SOURCE_PATTERN_ASSIGNMENT,
            )
        selected, conflict_reason = self._detector.routing_reorder_veto(
            raw_name,
            selected,
            paper_names=paper_names,
            focal_index=focal_index,
        )
        return ResolvedAuthorFields.from_selected_components(
            source=source,
            selected=selected,
            reason=conflict_reason or reason,
        )

    def resolve_pp_vys_author(  # noqa: PLR0913
        self,
        *,
        source: SourceAuthorFields,
        paper_authors: list[SourceAuthorFields],
        raw_name: str,
        paper_names: list[str],
        focal_index: int,
        row,
        pp_result,
        vys_result,
    ) -> ResolvedAuthorFields:
        """Resolve one author from a PP/VYS routing row."""
        decision = row["router_prediction"]
        input_order_candidate = row.get("input_order_candidate", "unknown")
        if decision == "pp":
            chosen = pp_result
            batch_reason = ResolutionReason.PP_SELECTED
        elif decision == "vys":
            chosen = vys_result
            batch_reason = ResolutionReason.VYS_SELECTED
        elif decision == "abstain":
            if input_order_candidate == "pp":
                chosen = pp_result
                batch_reason = ResolutionReason.PP_VYS_ABSTAIN_PP_INPUT
            elif input_order_candidate == "vys":
                chosen = vys_result
                batch_reason = ResolutionReason.PP_VYS_ABSTAIN_VYS_INPUT
            else:
                message = f"abstain with unexpected input_order_candidate={input_order_candidate!r} (expected 'pp'/'vys')"
                raise ValueError(message)
        elif decision == "not_person":
            chosen = None
            batch_reason = ResolutionReason.PP_SELECTED
        else:
            message = f"pp-vys router returned unexpected router_prediction={decision!r}"
            raise ValueError(message)

        parsed = chosen.parsed if chosen is not None and chosen.success else None
        if decision == "abstain" and (parsed is None or not parsed.surname):
            batch_reason = ResolutionReason.BATCH_ABSTAIN_MATERIALIZATION_FAILED
        return self.terminal_resolution(
            source=source,
            paper_authors=paper_authors,
            raw_name=raw_name,
            paper_names=paper_names,
            focal_index=focal_index,
            parsed=parsed,
            batch_reason=batch_reason,
            router_not_person=decision == "not_person",
        )

    def resolve_pp_author(  # noqa: PLR0913
        self,
        *,
        source: SourceAuthorFields,
        paper_authors: list[SourceAuthorFields],
        raw_name: str,
        paper_names: list[str],
        focal_index: int,
        row,
        result,
    ) -> ResolvedAuthorFields:
        """Resolve one author from a PP-only routing row."""
        decision = row["router_prediction"]
        if decision == "pp":
            parsed = result.parsed if result.success else None
            batch_reason = ResolutionReason.PP_SELECTED
        elif decision == "abstain":
            if _structural_roman_hyphen_key(source.last_name) in REVIEWED_TRAILING_HYPHENATED_COMPOUND_SURNAMES:
                parsed = None
                batch_reason = ResolutionReason.PP_ONLY_ABSTAIN_REVIEWED_COMPOUND_SURNAME
            elif (
                self._detector._ethnicity_service is not None  # noqa: SLF001
                and not _pp_parse_uses_contextual_han_surname_reading(raw_name, result)
                and _pp_only_native_abstain_prefers_scalar(
                    raw_name,
                    self._detector._ethnicity_service.japanese_probability,  # noqa: SLF001
                )
            ):
                parsed = None
                batch_reason = ResolutionReason.PP_ONLY_ABSTAIN_INPUT
            else:
                parsed = pp_abstain_parsed(result, row)
                batch_reason = ResolutionReason.PP_ONLY_ABSTAIN_INPUT
                if parsed is None or not parsed.surname:
                    batch_reason = ResolutionReason.BATCH_ABSTAIN_MATERIALIZATION_FAILED
        elif decision == "not_person":
            parsed = None
            batch_reason = ResolutionReason.PP_SELECTED
        else:
            message = f"pp-abstain router returned unexpected router_prediction={decision!r}"
            raise ValueError(message)

        return self.terminal_resolution(
            source=source,
            paper_authors=paper_authors,
            raw_name=raw_name,
            paper_names=paper_names,
            focal_index=focal_index,
            parsed=parsed,
            batch_reason=batch_reason,
            router_not_person=decision == "not_person",
        )
