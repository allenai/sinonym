"""Shared Unicode punctuation definitions for personal names.

The apostrophe oracle follows Unicode 17.0 ``Quotation_Mark`` entries for
single quotes plus the spacing modifier characters that Unicode's NamesList
cross-references to apostrophe, prime, or single-quote forms. Double quotes,
combining marks, invisible tag characters, and letters whose names merely
mention an apostrophe are intentionally excluded.
"""

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

HYPHEN_LIKE = frozenset(
    {
        "-",
        "\u00ad",  # soft hyphen
        "\u058a",  # Armenian hyphen
        "\u05be",  # Hebrew punctuation maqaf
        "\u1400",  # Canadian syllabics hyphen
        "\u1806",  # Mongolian todo soft hyphen
        "\u2010",  # hyphen
        "\u2011",  # non-breaking hyphen
        "\u2012",  # figure dash
        "\u2013",  # en dash
        "\u2014",  # em dash
        "\u2015",  # horizontal bar
        "\u2043",  # hyphen bullet
        "\u2027",  # hyphenation point
        "\u208b",  # subscript minus
        "\u2212",  # minus sign
        "\u2e17",  # double oblique hyphen
        "\u2e1a",  # hyphen with diaeresis
        "\u2e3a",  # two-em dash
        "\u2e3b",  # three-em dash
        "\u2e40",  # double hyphen
        "\u2e5d",  # oblique hyphen
        "\u301c",  # wave dash
        "\u3030",  # wavy dash
        "\u30a0",  # katakana-hiragana double hyphen
        "\ufe31",  # vertical em dash presentation form
        "\ufe32",  # vertical en dash presentation form
        "\ufe58",  # small em dash
        "\ufe63",  # small hyphen-minus
        "\uff0d",  # fullwidth hyphen-minus
        "\U00010ead",  # Yezidi hyphenation mark
    },
)

APOSTROPHE_FOLD_TRANSLATION = str.maketrans(dict.fromkeys(APOSTROPHE_LIKE, "'"))
HYPHEN_FOLD_TRANSLATION = str.maketrans(dict.fromkeys(HYPHEN_LIKE, "-"))
NAME_JOINER_DELETE_TRANSLATION = str.maketrans(dict.fromkeys(APOSTROPHE_LIKE | HYPHEN_LIKE, None))
