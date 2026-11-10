"""Shared Unicode punctuation policy for personal names.

The apostrophe oracle follows Unicode 17.0 ``Quotation_Mark`` entries for
single quotes plus the spacing modifier characters that Unicode's NamesList
cross-references to apostrophe, prime, or single-quote forms. Double quotes,
combining marks, invisible tag characters, and letters whose names merely
mention an apostrophe are intentionally excluded.

The structural Roman hyphen fold is deliberately narrower than Unicode
``Dash``. It contains semantic/compatibility hyphens plus metadata
substitutions supported by the August 2026 author-corpus review. A second,
post-preprocessing safety set keeps ambiguous separators visible to downstream
ethnicity gates without claiming that they are semantically hyphens. The
generic person normalizer's older Unicode-name oracle remains separate so
sharing this module does not change that path's deployed behavior.
"""

import re
import unicodedata
from collections.abc import Mapping

APOSTROPHE_LIKE = frozenset(
    {
        "'",
        "\u02b9",  # modifier letter prime
        "\u02bb",  # modifier letter turned comma
        "\u02bc",  # modifier letter apostrophe
        "\u02bd",  # modifier letter reversed comma
        "\u02be",  # modifier letter right half ring
        "\u02bf",  # modifier letter left half ring
        "\u02c0",  # modifier letter glottal stop
        "\u02c1",  # modifier letter reversed glottal stop
        "\u02c8",  # modifier letter vertical line
        "\u02ca",  # modifier letter acute accent
        "\u02cb",  # modifier letter grave accent
        "\u02ee",  # modifier letter double apostrophe
        "\u0559",  # Armenian modifier letter left half ring
        "\u055a",  # Armenian apostrophe
        "\u05f3",  # Hebrew punctuation geresh
        "\u07f4",  # NKo high tone apostrophe
        "\u07f5",  # NKo low tone apostrophe
        "\u2018",  # left single quotation mark
        "\u2019",  # right single quotation mark
        "\u201a",  # single low-9 quotation mark
        "\u201b",  # single high-reversed-9 quotation mark
        "\u2032",  # prime
        "\u2035",  # reversed prime
        "\u2039",  # single left-pointing angle quotation mark
        "\u203a",  # single right-pointing angle quotation mark
        "\u275b",  # heavy single turned comma quotation mark ornament
        "\u275c",  # heavy single comma quotation mark ornament
        "\u275f",  # heavy low single comma quotation mark ornament
        "\u276e",  # heavy left-pointing angle quotation mark ornament
        "\u276f",  # heavy right-pointing angle quotation mark ornament
        "\ua78b",  # Latin capital letter saltillo
        "\ua78c",  # Latin small letter saltillo
        "\uff07",  # fullwidth apostrophe
        "`",  # grave accent used as an apostrophe
        "\uff40",  # fullwidth grave accent
        "\u00b4",  # acute accent used as an apostrophe
    },
)

_CANONICAL_ROMAN_HYPHENS = frozenset(
    {
        "-",
        "\u2010",  # hyphen
        "\u2011",  # non-breaking hyphen
        "\ufe63",  # small hyphen-minus
        "\uff0d",  # fullwidth hyphen-minus
    },
)

_REVIEWED_METADATA_HYPHENS = frozenset(
    {
        "\u2012",  # figure dash
        "\u2013",  # en dash
        "\u2014",  # em dash
        "\u2043",  # hyphen bullet
        "\u2212",  # minus sign
    },
)

ROMAN_HYPHEN_LIKE = _CANONICAL_ROMAN_HYPHENS | _REVIEWED_METADATA_HYPHENS

# These marks must not be promoted to structural ASCII parity. Folding them
# after preprocessing preserves the boundary evidence used by the ethnicity
# gates and prevents sep_pattern from turning them into spaces first.
_POST_PREPROCESSING_SAFETY_HYPHENS = frozenset(
    {
        "\u00ad",  # soft hyphen: boundary evidence in author metadata
        "\u2015",  # horizontal bar
        "\u2027",  # hyphenation point
        "\u208b",  # subscript minus
        "\ufe58",  # small em dash
    },
)

# All approved structural Roman hyphens are equivalent in comparison keys.
# Preserve the pre-review deletion behavior for these validation-only marks;
# U+2027 deliberately remains visible to direct token normalization.
_LEGACY_COMPARISON_ONLY_HYPHENS = frozenset("\u00ad\u2015\u208b\ufe58")
_NORMALIZATION_HYPHEN_DELETE = ROMAN_HYPHEN_LIKE | _LEGACY_COMPARISON_ONLY_HYPHENS
# Exact Unicode 14 expansion of the generic person's pre-consolidation
# category/name predicate, plus the three explicit legacy extras. Keeping it a
# table avoids a Unicode lookup and nested function call for every input char.
PERSON_HYPHEN_LIKE = frozenset(
    "-\u00ad\u058a\u05be\u1400\u1806\u2010\u2011\u2012\u2013\u2014\u2015"
    "\u2027\u2043\u208b\u2212\u2e17\u2e1a\u2e3a\u2e3b\u2e40\u2e5d"
    "\u301c\u3030\u30a0\ufe31\ufe32\ufe58\ufe63\uff0d\U00010ead"
    "\u00b1\u02d7\u0320\u2052\u2796\u2a29\u2a2a\u2a2b\u2a2c\u2a3a\u2a41\U000e002d",
)


# Invisible markup that carries no name content. These survive `clean_roman_pattern` (they are
# neither Roman letters nor punctuation), leave the token unmatched by the tokenizer, and the
# name then resolves as NON_PERSON_SOURCE_PASSTHROUGH -- so a real author is marked a non-person.
# 24,933 rows of `paper_authors_full` carry one; a LEADING one also becomes the first initial in
# scholar's `cluster_block_key`, because that key slices the raw first name.
#
# Deliberately EXCLUDED, per the measured split:
#   U+200C ZERO WIDTH NON-JOINER -- orthographically required in Persian and Devanagari, so
#       folding it changes real names. Left alone.
#   U+00AD, U+0088, U+007F, U+FFFD -- symptoms of mojibake (UTF-8 read as Latin-1); stripping
#       them yields a plausible-looking WRONG name (`MarÃ­a` stays wrong), so the record should
#       be flagged upstream rather than silently cleaned here. U+00AD is also already handled as
#       a hyphen by _LEGACY_COMPARISON_ONLY_HYPHENS.
#   U+200B ZERO WIDTH SPACE -- the word separator in Thai, Khmer, Lao and Myanmar, which do not
#       put spaces between words, so deleting it merges two words into one token. It would have
#       to become a space there rather than vanish; left alone.
#   U+200D ZERO WIDTH JOINER -- the counterpart to U+200C: requests the conjunct / half-form in
#       Devanagari and Bengali, as ZWNJ requests the non-joined form. Same argument as ZWNJ, so
#       the same treatment.
#   Private Use Area (Co) -- font-specific glyphs surviving PDF extraction, no recoverable text.
INVISIBLE_MARKUP = frozenset(
    "\u200e"      # LEFT-TO-RIGHT MARK
    "\u200f"      # RIGHT-TO-LEFT MARK
    "\u202a"      # LEFT-TO-RIGHT EMBEDDING
    "\u202b"      # RIGHT-TO-LEFT EMBEDDING
    "\u202c"      # POP DIRECTIONAL FORMATTING
    "\u202d"      # LEFT-TO-RIGHT OVERRIDE
    "\u202e"      # RIGHT-TO-LEFT OVERRIDE
    "\u2060"      # WORD JOINER
    "\u2066\u2067\u2068\u2069"  # LRI / RLI / FSI / PDI
    "\ufeff",     # ZERO WIDTH NO-BREAK SPACE (BOM)
)
INVISIBLE_MARKUP_DELETE_TRANSLATION = str.maketrans(dict.fromkeys(INVISIBLE_MARKUP, None))

APOSTROPHE_FOLD_TRANSLATION = str.maketrans(dict.fromkeys(APOSTROPHE_LIKE, "'"))
HYPHEN_FOLD_TRANSLATION = str.maketrans(
    dict.fromkeys(ROMAN_HYPHEN_LIKE | _POST_PREPROCESSING_SAFETY_HYPHENS, "-"),
)
PERSON_JOINER_FOLD_TRANSLATION = APOSTROPHE_FOLD_TRANSLATION | str.maketrans(
    dict.fromkeys(PERSON_HYPHEN_LIKE, "-"),
)
NAME_JOINER_DELETE_TRANSLATION = str.maketrans(
    dict.fromkeys(APOSTROPHE_LIKE | _NORMALIZATION_HYPHEN_DELETE, None),
)

_TRANSLITERATION_APOSTROPHE_RE = re.compile(
    r"(?<![^\W\d_])(?P<prefix>[^\W\d_]+)\s*'\s*(?=(?P<tail>[^\W\d_]+)(?![^\W\d_]))",
    re.UNICODE,
)
_TRANSLITERATION_APOSTROPHE_PAIRS = frozenset(
    {
        ("cui", "e"),
        ("ma", "ayan"),
        ("o", "Brart"),
        ("o", "connor"),
        ("p", "eng"),
        ("sa", "di"),
        ("ts", "ai"),
    },
)
_DUTCH_T_SPACING_RE = re.compile(r"(?<=\S)(?:\s+'\s*|'\s+)t(?=\s+\S)", re.UNICODE)


def fold_internal_name_joiners(value: str, translation: Mapping[int, str | None]) -> str:
    """Fold configured joiners while removing paired outer delimiter runs.

    A lone leading hyphen or terminal transliteration apostrophe can carry name
    semantics, so an outer run is a delimiter only when the opposite boundary
    has a run that folds to the same ASCII mark.
    """
    first_text = len(value) - len(value.lstrip())
    last_text = len(value.rstrip()) - 1
    if first_text <= last_text:
        boundary = translation.get(ord(value[first_text]))
        if boundary is not None and boundary == translation.get(ord(value[last_text])):
            content_start = first_text
            while content_start <= last_text and translation.get(ord(value[content_start])) == boundary:
                content_start += 1
            content_end = last_text
            while content_end >= content_start and translation.get(ord(value[content_end])) == boundary:
                content_end -= 1
            value = f"{value[:first_text]} {value[content_start : content_end + 1]} {value[last_text + 1 :]}"
    return value.translate(translation)


def fold_spaced_transliteration_apostrophes(value: str) -> str:
    """Repair spacing around reviewed transliteration apostrophes.

    Only audited prefix/tail pairs are joined. Separated variants of the Dutch
    ``'t`` particle retain its token boundary. The caller must first fold
    supported apostrophe variants to ASCII.
    """

    def join(match: re.Match[str]) -> str:
        prefix = match.group("prefix")
        tail = match.group("tail")
        if (prefix.casefold(), tail) not in _TRANSLITERATION_APOSTROPHE_PAIRS:
            return match.group(0)
        return f"{prefix}'"

    value = _TRANSLITERATION_APOSTROPHE_RE.sub(join, value)
    return _DUTCH_T_SPACING_RE.sub(" 't", value)


# spacing diacritics; ASCII ^ and backtick are not marks here (a caret between a vowel and a consonant is handled
# on its own in ``_reattach_in_token``)
_SPACING_TO_COMBINING = {
    "\u00a8": "\u0308", "\u02da": "\u030a", "\u00b4": "\u0301",
    "\u02dc": "\u0303", "\u00b8": "\u0327", "\u02c7": "\u030c", "\u00af": "\u0304", "\u02d8": "\u0306",
    "\u02dd": "\u030b", "\u02d9": "\u0307", "\u02db": "\u0328",
}
_APOSTROPHE_LIKE_MARKS = frozenset({"\u00b4"})
_COMBINING_TO_SPACING = {combining: spacing for spacing, combining in _SPACING_TO_COMBINING.items()}
# a combining mark right after a letter or another mark is ordinary NFD text (``Iba\u0301n\u0303ez``, ``İ\u0307\u0307``) and
# is never moved; only spacing marks and combining marks that follow no letter are detached
_DETACHED_MARK = "(?:[" + "".join(_SPACING_TO_COMBINING) + "]|(?<![^\\W\\d_]|[\\u0300-\\u036f])[\\u0300-\\u036f])"
_ANY_DETACHED_MARK_RE = re.compile(_DETACHED_MARK)
_TOKEN_MARK_RE = re.compile(
    "(?P<prev>[^\\W\\d_])?(?P<mark>" + _DETACHED_MARK + ")(?P<next>[^\\W\\d_])?",
    re.UNICODE,
)


def _base(ch: str) -> str:
    return unicodedata.normalize("NFD", ch)[:1] if ch else ""


def _compose(base: str, combining: str) -> str | None:
    if not base:
        return None
    out = unicodedata.normalize("NFC", base + combining)
    return out if len(out) == 1 else None


_CARON_BASES = frozenset("cdelnrstzCDELNRSTZ")


def _attach_caron(prev: str, nxt: str, *, mark_before_letter: bool = False) -> str | None:
    """Join a detached caron to a Central European consonant or e (č š ž ř ň ď ť ľ ě); on g it is Turkish ğ.

    The previous letter goes first, except d/n/t before e (Czech writes ně, dě, tě: ``Zdenˇek`` → ``Zdeněk``), r/s/z/c
    after another caron letter and s/z/c after r (ř š ž č are far more common there: ``Ondˇrej`` → ``Ondřej``,
    ``Beneˇs`` → ``Beneš``, ``Jurˇca`` → ``Jurča``; 99.2% right on the corpus's 3,555 detached-caron mentions with one
    real reading, was 93.4%), and tokens whose extractor put the caron before its letter (``Kaˇcmarˇcik``). Vowel
    carons (ǎ ǐ ǒ ǔ) are pinyin tone marks, never a detached accent.
    """
    next_first = (
        mark_before_letter
        or (prev in "dntDNT" and nxt in "eE")
        or (nxt in "rszcRSZC" and prev in _CARON_BASES and prev not in "rszcRSZC")
        or (prev in "rR" and nxt in "szcSZC")
    )
    order = ((nxt, True), (prev, False)) if next_first else ((prev, False), (nxt, True))
    for base, is_next in order:
        if base in _CARON_BASES:
            out = _compose(base, "\u030c")
        elif base in ("g", "G"):
            out = "\u011f" if base == "g" else "\u011e"
        else:
            continue
        if out is not None:
            return prev + out if is_next else out + nxt
    return None


# letters and clusters that mark a Polish word (Turkish and Romanian names, the other cedilla users, have none):
# a cedilla after a or e there is the ogonek of ą ę (ie¸ is ę in 15 of 15 corpus words, a¸/e¸ elsewhere 73 of 276)
# (case-sensitive and anchored, so ``Ta¸skıran`` with its Turkish ı does not look Polish)
_POLISH_WORD_RE = re.compile("[łśżźćńŁŚŻŹĆŃwW]|[cC]z|[sS]z|[rR]z|[sScC]k[ia]$|ie\u00b8|i\u02dbe")
_ROMANIAN_WORD_RE = re.compile("escu$|eanu$|[şţșțâî]|^Dr[aeiou]\u02d8?g", re.IGNORECASE)
# a Czech or Slovak word: a caron letter or a detached caron, or a -sky/-cky ending (ý is extracted as y´ there)
_CZECH_WORD_RE = re.compile("[řšČčžěůďťňŘŠŽĚŮĎŤŇ]|\u02c7|[sScC][kK][yY]\u00b4$")
_MAX_REPAIRED_TOKEN = 200
_STRONG_POLISH_WORD_RE = re.compile("[łśżźćńŁŚŻŹĆŃ]|[cC]z|[sS]z|[rR]z|[sScC]k[ia]$|ie\u00b8|i\u02dbe")
_CARET_CIRCUMFLEX_RE = re.compile("([aeiouAEIOU])\\^(?=[b-df-hj-np-tv-z])")
_MARKUP_BEFORE_MARK_RE = re.compile("[" + "".join(sorted(INVISIBLE_MARKUP)) + "]+(?=[\\u0300-\\u036f])")
# Ã and Â before a spacing mark are UTF-8 read as Latin-1 (``AntÃ´nio``, ``SÃ¸ren``); such a word is decoded when it
# round-trips exactly
_MOJIBAKE_BEFORE_MARK_RE = re.compile("[ÃÂ][" + "".join(_SPACING_TO_COMBINING) + "]")
# Ä and Å before a spacing mark are other mojibake (cp1250 ``Ä˝ubica`` for Ľubica): such a word is left as written
_OTHER_MOJIBAKE_BEFORE_MARK_RE = re.compile("[ÄÅ][" + "".join(_SPACING_TO_COMBINING) + "]")
# a mid-word ´ after a letter with one of these marks is the next letter's acute (``Dvoř´ak``, ``Sagü´es``, ``Szczȩ´sniak``);
# after an acute, macron, tilde, circumflex or breve it is an apostrophe (``Sā´id``, ``Marí´a``, ``Iñ´aki``, ``Quô´c``)
_ACUTE_AFTER_MARKS = frozenset({"\u030c", "\u0308", "\u0327", "\u0328"})


def _reattach_in_token(token: str) -> str:
    """Repair the detached marks of one token; a mark that composes with neither neighbour leaves the token as it was."""
    if "`" in token:
        return token  # a backtick there makes ´ an apostrophe variant (``Le Bloc´`'h``), never an accent
    if "^" in token:
        # a caret between a vowel and a consonant is that vowel's circumflex (``Lemai^tre`` → ``Lemaître``, ``Jéro^me``);
        # between two vowels it is ambiguous (``Beno^it`` is Benoît), and elsewhere (``Smith^1``, ``^hiyi``,
        # ``Gomes^Neto``) it is a separator or footnote, so any other caret leaves the word alone
        fixed = _CARET_CIRCUMFLEX_RE.sub(lambda m: unicodedata.normalize("NFC", m.group(1) + "\u0302"), token)
        return token if "^" in fixed else _reattach_in_token(fixed)
    if (len(token) > 1 and token[0] == token[-1] == "\u00b4") or "\u00b4\u00b4" in token:
        return token  # ``´Iñaki´`` and ``Ángel´´`` use ´ as quote marks, not accents
    if _MOJIBAKE_BEFORE_MARK_RE.search(token):
        # UTF-8 read as Latin-1 (``SÃ¸ren``, ``AntÃ´nio``): decode it only when that round-trips exactly, else leave it
        try:
            decoded = token.encode("latin-1").decode("utf-8")
        except UnicodeError:
            return token
        return _reattach_in_token(decoded)  # ``STOJANOVICÂ´`` → ``STOJANOVIC´`` → ``STOJANOVIĆ``
    if _OTHER_MOJIBAKE_BEFORE_MARK_RE.search(token):
        return token
    if len(token) > _MAX_REPAIRED_TOKEN:
        return token  # no name word is this long, and the per-mark repair is quadratic in the token length
    original = token
    # a spacing mark doubled by the extractor is one mark (``Mu¨¨ller``); a doubled ´ is a quote mark (``Ángel´´``)
    token = re.sub("([" + "".join(m for m in _SPACING_TO_COMBINING if m != "\u00b4") + "])\\1+", "\\1", token)
    failed = False
    caron_before = 0
    for m in _TOKEN_MARK_RE.finditer(token):
        if _SPACING_TO_COMBINING.get(m.group("mark"), m.group("mark")) == "\u030c":
            p, n = m.group("prev") or "", m.group("next") or ""
            p_ok, n_ok = p in _CARON_BASES or p in "gG", n in _CARON_BASES or n in "gG"
            caron_before += (n_ok and not p_ok) - (p_ok and not n_ok)

    def _fail(match):
        nonlocal failed
        failed = True
        return match.group(0)

    current = token
    all_caps = token.upper() == token
    polish_word = _POLISH_WORD_RE.search(token) is not None
    strong_polish_word = _STRONG_POLISH_WORD_RE.search(token) is not None
    czech_word = _CZECH_WORD_RE.search(token) is not None
    romanian_word = _ROMANIAN_WORD_RE.search(token) is not None

    def fix(match):
        nonlocal failed
        prev, nxt, mark = match.group("prev") or "", match.group("next") or "", match.group("mark")
        combining = _SPACING_TO_COMBINING.get(mark, mark)
        start = match.start("mark")
        if mark in _APOSTROPHE_LIKE_MARKS and not prev and nxt and start and unicodedata.combining(current[start - 1]):
            # NFD text: the letter before ´ carries combining marks of its own (``Mari\u0301´a``); judge those marks as below
            base_at = start
            while base_at and unicodedata.combining(current[base_at - 1]):
                base_at -= 1
            if base_at and set(current[base_at:start]) & _ACUTE_AFTER_MARKS and (nxt in "aeiou" or nxt in "cnsz"):
                out = _compose(nxt, combining)
                if out is not None:
                    return out
            return match.group(0)
        if mark in _APOSTROPHE_LIKE_MARKS and prev and nxt:
            # a mid-word ´ is an acute or an apostrophe (José, Maz'ya, Ken'ichi, Ćosić), so it is left as written, except
            # right after a letter that already carries a mark, where it is the next vowel's acute (``Dvoř´ak``, ``Sagü´es``)
            vowel_next = nxt in "aeiou" or (all_caps and nxt in "AEIOU")
            prev_marks = set(unicodedata.normalize("NFD", prev)[1:])
            # a letter takes one accent, so after a marked vowel the acute is the consonant's (``Szczȩ´sniak``, ``ˇS´CETA``)
            consonant_next = nxt in "cnsz" or (all_caps and nxt in "CNSZ")
            if prev_marks & _ACUTE_AFTER_MARKS and (vowel_next or consonant_next):
                out = _compose(nxt, combining)
                if out is not None:
                    return prev + out
            return match.group(0)
        prev_base = _base(prev).lower()
        polish_ending = current.lower().endswith(("ien\u00b4", "yn\u00b4"))  # Bień, Kwiecień, Horyń
        # a bare w is too weak a sign here (``Brown´``, ``Owen´``)
        polish_letter = (prev_base in "nsz" and strong_polish_word) or (prev_base == "n" and polish_ending)
        czech_y = prev_base == "y" and czech_word
        if mark == "\u00b4" and not nxt and prev_base not in "aeiouc" and not polish_letter and not czech_y:
            # a word-final ´ is an accent only on a vowel or c (``Jose´``, ``Ružic´``), on y in a Czech or Slovak word
            # (``Sechovsky´``), or on n/s/z in a Polish word (``Grzes´``); otherwise it is an elision, a soft sign or stray
            # left as written (``Dell´``, ``Igor´``, ``Vasyl´``, ``Fernandes´``, ``Nancy´``, ``Istvan´``)
            return match.group(0)
        after_next = current[match.end() : match.end() + 1]
        polish_s_cluster = nxt in "sS" and after_next in "wlrcmnpWLRCMNP"  # Św Śl Śr Śc (``´Swia¸tek``, ``´Sliwa``)
        if mark == "\u00b4" and not prev and nxt and not (
            nxt in "aeiouAEIOU" or (nxt in "cnszCNSZ" and polish_word) or polish_s_cluster
        ):
            # a word-initial ´ goes on a vowel, on ś before a consonant, or on ć ń ś ź in a Polish word; before any other
            # consonant it is stray (``´Pedraza``, ``´Castro``, ``´Silva``)
            return match.group(0)
        if mark == combining and start and current[start - 1] not in "-\u2010":
            # a combining mark after any other mark (``Mu\u1dc4\u0301ller``) or after punctuation (``O.\u0301scar``) is not a
            # detached accent we can place; only one at a word start or after a hyphen is
            return match.group(0)
        if combining == "\u030c":
            return _attach_caron(prev, nxt, mark_before_letter=caron_before > 0) or _fail(match)
        # a tilde next to n is the Spanish ñ on either side (``Ya˜nez`` → ``Yañez``, ``Mun˜oz`` → ``Muñoz``): of the
        # corpus's 1,795 such mentions only ``N˜ao-Linear`` (not a name) wanted ã; Portuguese ã/õ sit between vowels
        tilde = combining == "\u0303"
        after = current[match.end() : match.end() + 1].lower()
        dot = combining == "\u0307"
        # next to a lowercase i, a z or e on the other side takes the dot (``Gi˙zewska``); a capital I is always İ (``I˙zzet``)
        dot_base_other = ((nxt and nxt in "zeZE") or (prev and prev in "zeZE")) and polish_word  # Turkish Fi˙li˙z keeps i's dot
        capital_i = prev == "I" or nxt == "I"
        if dot and ((prev and prev in "iI") or (nxt and nxt in "iI")) and (capital_i or not dot_base_other):
            # a dot next to i is the i's own dot: dropped on a lowercase i (``Meli˙ha``), İ on a capital (``I˙nci``)
            if prev and prev in "iI":
                return ("\u0130" if prev == "I" else prev) + nxt
            return prev + ("\u0130" if nxt == "I" else nxt)
        g_before = prev and prev in "gG" and match.start("prev") > 0  # Turkish words never start with ğ (``G˘ain˘a``)
        g_after = nxt and nxt in "gG" and prev and prev.lower() in "aeiou"
        romanian = romanian_word  # ă before g in Drăgan, Călugăreanu
        if combining == "\u0306" and not prev and nxt and nxt in "gG":
            failed = True  # a breve before a word-initial g is not ğ (Turkish words never start with ğ)
            return match.group(0)
        if combining == "\u0306" and (g_after or g_before) and not romanian:
            # a breve next to g is the Turkish ğ (``Do˘gan``, ``Ça˘glar``: 343 corpus mentions, none on the vowel)
            out = _compose(nxt if nxt in "gG" else prev, combining)
            if out is not None:
                return prev + out if nxt in "gG" else out + nxt
        if tilde and prev and prev in "oO" and nxt and nxt in "nN" and after in ("u", "i"):
            out = _compose(prev, combining)  # Estonian õ before n (``To˜nu`` → ``Tõnu``), not Spanish ñ
            if out is not None:
                return out + nxt
        portuguese = ("ao", "ae", "oe")
        if tilde and nxt and (nxt + after).lower() in portuguese and (prev + nxt).lower() not in portuguese:
            order = ((nxt, True), (prev, False))  # the tilde opens an ão/ãe/õe diphthong (``Jo˜ao`` → ``João``)
        elif tilde and prev and nxt and prev in "nN" and nxt.lower() in "aeiou":
            order = ((prev, False), (nxt, True))
        elif tilde and prev and nxt and nxt in "nN" and prev.lower() in "aeiou":
            order = ((nxt, True), (prev, False))
        else:
            vowel_next = nxt.lower() in "aeiouy" and prev.lower() not in "aeiouy"
            order = ((nxt, True), (prev, False)) if vowel_next or not prev else ((prev, False), (nxt, True))
        next_is_accented = match.end() < len(current) and "\u0300" <= current[match.end()] <= "\u036f"
        if dot:
            # names take a dot above only on ż ė (``Z˙ylicz`` → ``Żylicz``); ċ ġ are rare, and a dot by c or g is more often a
            # misread dot below (``Ngo˙c`` is Ngọc), so it is left
            order = tuple((base, is_next) for base, is_next in order if base and base in "zeZE")
        elif combining == "\u0327":
            if prev and prev in "aeAE" and polish_word:
                # in a Polish word a cedilla after a or e is a misread ogonek (``Wie¸cek`` → ``Więcek``, ``Wa¸s`` → ``Wąs``)
                out = _compose(prev, "\u0328")
                if out is not None:
                    return out + nxt
            # otherwise it is ç ş ţ far more often than ȩ or a Latvian ņ (``Güne¸s`` → ``Güneş``, ``Du¨zgu¨n¸cınar``)
            order = tuple(sorted(order, key=lambda o: o[0] not in "cstCST"))
        elif combining == "\u0328" and not prev:
            failed = True  # no Polish word starts with ą or ę (``B | ˛aczkowska`` lost its B to another field)
            return match.group(0)
        elif combining == "\u0328":
            # an ogonek goes only on a or e, the Polish ą ę (``Pi˛ekos`` → ``Piękos``); with neither neighbour the word is
            # left as it was rather than composing ǫ į ų onto a misread cedilla (``Franc˛ois``)
            polish = polish_word  # before its vowel only in a Polish word (``S˛ahin`` is Şahin)
            order = tuple(
                (b, n) for b, n in sorted(order, key=lambda o: not o[1]) if b and b in "aeAE" and ((not n and nxt) or polish)
            )  # after its vowel at a word end also only in a Polish word (``SUKHORUKOVA˛``)
        for base, is_next in order:
            if is_next and next_is_accented:
                continue  # a letter that already carries a combining mark takes no second one (``Oz˙óg`` → ``Ożóg``)
            out = _compose(base, combining)
            if out is not None:
                return prev + out if is_next else out + nxt
        failed = True
        return match.group(0)

    # one mark at a time on the text repaired so far, so a letter between two marks is seen by both
    # (``Conceic¸a˜o`` → ``Conceição``; a single substitution pass handed the a to the cedilla match and put the tilde on o)
    pos = 0
    while (match := _TOKEN_MARK_RE.search(current, pos)) is not None:
        replacement = fix(match)
        if failed:
            return original
        nxt = match.group("next") or ""
        if replacement == match.group(0):
            pos = match.end() - len(nxt)
            continue
        current = current[: match.start()] + replacement + current[match.end() :]
        pos = match.start() + max(len(replacement) - 1, 0)
    return current


def strip_soft_hyphens(text: str) -> str:
    """Delete U+00AD on the person-name path.

    The Chinese path keeps it as a reviewed syllable-boundary cue (``Guo\u00ade Li`` →
    ``Guo-E Li``). For every other name it is a line-break artifact or half of a mojibake ``í``
    — in the ORCID corpus 0 of 100 classifiable cases were a boundary — and reading it as a
    hyphen wrote ``Zaval-niuk`` for Завальнюк.
    """
    return text.replace("\u00ad", "") if "\u00ad" in text else text


def attach_detached_diacritics(text: str) -> str:
    """Reattach a diacritic that PDF extraction left as its own character.

    ``Arenh¨ovel`` carries U+00A8, whose NFKC form is a space plus a combining diaeresis, so
    every later step sees two tokens. Within a token the mark joins the neighbouring letter it
    composes with, the vowel first (``¨ovel`` → ``övel``, ``Jose´`` → ``José``); a bare
    combining mark after a space is the NFKC rendering of such a mark and joins the next
    token. A token with a mark that composes with nothing is left exactly as it was
    (``S˘c˘epin``). A combining mark after a letter or another mark is ordinary NFD text and is never moved.
    """
    # invisible markup between a letter and its accent would make the accent look detached (``Iba\u0301n\u200e\u0303ez``);
    # only that markup is removed here, so names without a detached-looking mark are untouched
    text = _MARKUP_BEFORE_MARK_RE.sub("", text)
    if not _ANY_DETACHED_MARK_RE.search(text) and "^" not in text:
        return text
    # a combining mark after a space is the NFKC remnant of a spacing mark: restore the spacing form so the
    # token repair may move it to the letter it composes with (an attached combining mark is never moved); between
    # spaces it stays a spacing mark (``O ́ connor``), which later reads as an apostrophe
    text = re.sub("(?<=[^\\W\\d_]) ([\\u0300-\\u036f])", lambda m: " " + _COMBINING_TO_SPACING.get(m.group(1), m.group(1)), text)
    text = re.sub("(?<=\\s)([\\u0300-\\u036f])(?=[^\\W\\d_])", lambda m: _COMBINING_TO_SPACING.get(m.group(1), m.group(1)), text)
    parts = re.split(r"(\s+)", text)
    return "".join(part if part.isspace() or not part else _reattach_in_token(part) for part in parts)
