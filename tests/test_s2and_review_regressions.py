"""Regressions from the S2AND 2026-09 name-order review (revert_list.csv, s2and_exports_20260925).

Each case is a structured source exactly as the author-refresh backfill sent it and the
reviewer-confirmed correct fields.
"""

import unicodedata

import pytest

from sinonym.coretypes.routing_resolution import ResolutionProvenance, ResolutionReason
from sinonym.timo.interface import Instance, Predictor, PredictorConfig, SourceAuthorFields


@pytest.fixture(scope="module")
def predictor() -> Predictor:
    return Predictor(PredictorConfig(parallel="never"), ".")


def resolve(predictor: Predictor, first, middle, last):
    instance = Instance(pp_authors=[SourceAuthorFields(first_name=first, middle_names=middle, last_name=last)])
    author = predictor.predict_batch([instance])[0].authors[0]
    return author.first_name, author.middle_names, author.last_name


DOTTED_INITIALS_CASES = [
    # zbmath: Yakov Sinai; ``Ya.`` was taken as a Chinese surname and Sinai cut into Si-Nai
    (("Ya.", "G.", "Sinai"), ("Ya.", "G.", "Sinai")),
    (("Ya. G.", None, "Sinai"), ("Ya.", "G.", "Sinai")),
    (("Yu.", "G.", "Sinai"), ("Yu.", "G.", "Sinai")),
    (("Ya.", None, "Sinai"), ("Ya.", "", "Sinai")),
    # unchanged behaviour for ordinary initials
    (("J.", "A.", "Minahan"), ("J.", "A.", "Minahan")),
    (("Y.", None, "Zhang"), ("Y.", "", "Zhang")),
    (("J.", "J.", "Zhu"), ("J.-J.", "", "Zhu")),
    (("Yu.", "V.", "Shubin"), ("Yu.", "V.", "Shubin")),
    (("Kh.", None, "GANIYEV"), ("Kh.", "", "Ganiyev")),
]


@pytest.mark.parametrize("source", [("Dr.", None, "Reshmi"), ("Ms.", None, "Bithi"), ("L.", "St.", "Marie")])
def test_titles_are_not_dotted_initials(predictor, source):
    first, middle, last = resolve(predictor, *source)
    assert first not in ("Dr.", "Ms.") and (middle, last) != ("St.", "Marie")


@pytest.mark.parametrize(("source", "expected"), DOTTED_INITIALS_CASES)
def test_dotted_initials_plus_one_surname_token_keep_their_roles(predictor, source, expected):
    assert resolve(predictor, *source) == expected


def test_glued_caps_surname_accepts_a_capital_after_the_hyphen(predictor):
    # aminer BOTH_WRONG rows: a MAG byline of glued ``SURNAMEGiven-Name`` tokens
    instance = Instance(pp_authors=[
        SourceAuthorFields(last_name="ZHANGWei-Jun"),
        SourceAuthorFields(last_name="JIANGZong-Hui"),
        SourceAuthorFields(last_name="LIUXiao-Wei"),
    ])
    rows = predictor.predict_batch([instance])[0].authors
    assert [(a.first_name, a.middle_names, a.last_name) for a in rows] == [
        ("Wei-Jun", "", "Zhang"),
        ("Zong-Hui", "", "Jiang"),
        ("Xiao-Wei", "", "Liu"),
    ]


def test_scalar_baseline_still_canonicalizes_initials_without_moving_boundary(predictor):
    assert resolve(predictor, "J", None, "Smith") == ("J.", "", "Smith")


DECOMPOSED_ACCENT_CASES = [
    # names written in NFD (each accent a combining mark after its letter); adjacent accented letters and stacked
    # marks used to be moved onto the neighbouring letter (Dvořaḱ, Núnẽz, Nguyêñ)
    (("Antonín", None, "Dvořák"), ("Antonín", "", "Dvořák")),
    (("José", None, "Núñez"), ("José", "", "Núñez")),
    (("Jiří", None, "Kořínek"), ("Jiří", "", "Kořínek")),
    (("Aracelis", None, "Calzadilla-Núñez"), ("Aracelis", "", "Calzadilla-Núñez")),
    (("Văn", None, "Nguyễn"), ("Văn", "", "Nguyễn")),
]


@pytest.mark.parametrize(("source", "expected"), DECOMPOSED_ACCENT_CASES)
def test_decomposed_accents_stay_on_their_letters(predictor, source, expected):
    decomposed = tuple(unicodedata.normalize("NFD", part) if part else part for part in source)
    assert tuple(unicodedata.normalize("NFC", part) for part in resolve(predictor, *decomposed)) == expected


def test_stacked_dot_marks_are_kept_as_written(predictor):
    # Turkish names with a dotted capital plus combining dots used to have a dot moved onto the next letter (İ̇ṡmail)
    assert resolve(predictor, "İ̇̇smail", None, "Sağir") == ("İ̇̇smail", "", "Sağir")


@pytest.mark.parametrize(("source", "expected"), [
    (("John", None, "Smith^1"), ("John", "", "Smith^1")),
    (("Mazen`", None, "Abdalla"), ("Mazen'", "", "Abdalla")),
])
def test_ascii_caret_and_backtick_are_not_accents(predictor, source, expected):
    # a caret footnote or a backtick used as an apostrophe used to become an accent (Smitĥ, Mazeǹ)
    assert resolve(predictor, *source) == expected


def _byline(predictor, *authors):
    instance = Instance(pp_authors=[SourceAuthorFields(first_name=f, middle_names=m, last_name=l) for f, m, l in authors])
    return [(a.first_name, a.middle_names, a.last_name) for a in predictor.predict_batch([instance])[0].authors]


def test_reviewed_japanese_surname_vetoes_the_flip_in_a_given_first_byline(predictor):
    # inspire:718787 — Asaka was absent from every surname list, so
    # the one-sided Japanese vote flipped him; with the reviewed surname the given-first byline wins
    rows = _byline(predictor, ("Takao", None, "Asaka"), ("Jonathan", None, "Mayer"), ("Michael", None, "Smith"))
    assert rows[0] == ("Takao", "", "Asaka")
    rows = _byline(predictor, ("Masaki", None, "Shigemori"), ("Edward", None, "Witten"), ("Juan", None, "Maldacena"))
    assert rows[0] == ("Masaki", "", "Shigemori")


def test_korean_surname_as_last_token_is_not_moved_to_the_front(predictor):
    # inspire:953808 / :909467 — Myung and Hyun
    assert resolve(predictor, "Yun", "Soo", "Myung") == ("Yun", "Soo", "Myung")
    assert resolve(predictor, "Chang", "Ho", "Hyun") == ("Chang", "Ho", "Hyun")


@pytest.mark.parametrize("source", [("Gong-Bo", None, "Zhao"), ("Da", None, "Xing")])
def test_lone_two_token_name_with_a_far_rarer_surname_keeps_source_order(predictor, source):
    # aminer: accepted surname-first alone although the other token is the 20x likelier
    # surname; with no batch to lean on the source order stands
    assert resolve(predictor, *source) == (source[0], "", source[2])


def test_lone_two_token_name_with_the_likelier_surname_still_flips(predictor):
    assert resolve(predictor, "Zhang", None, "Wei") == ("Wei", "", "Zhang")


def test_token_with_an_uncomposable_mark_is_left_whole():
    # inspire: carons rendered as breves; composing only the second one would split the surname
    from sinonym.name_punctuation import attach_detached_diacritics

    assert attach_detached_diacritics("Evgenij V. S˘c˘epin") == "Evgenij V. S˘c˘epin"
    assert attach_detached_diacritics("Mu¨ller-Preussker") == "Müller-Preussker"


@pytest.mark.parametrize(("source", "expected"), [
    ("Buresˇová", "Burešová"),
    ("Frantisˇek", "František"),
    ("Sonˇa", "Soňa"),
    ("Soˇna", "Soňa"),
    ("Šovˇcíková", "Šovčíková"),
    ("Zdenˇek", "Zdeněk"),
    ("Gražulevicˇiene˙", "Gražulevičienė"),
    ("Karacaogˇlu", "Karacaoğlu"),
    ("Kaˇcmarˇcik", "Kačmarčik"),
    ("Bezouˇka", "Bezouˇka"),
    ("Ondˇrej", "Ondřej"),
    ("Petˇrík", "Petřík"),
    ("Beneˇs", "Beneš"),
    ("Mareˇs", "Mareš"),
    ("Doleˇzal", "Doležal"),
    ("Peˇsek", "Pešek"),
    ("Mlˇcoch", "Mlčoch"),
    ("Koklicˇ", "Koklič"),
    ("Kovácˇová", "Kováčová"),
    ("Sˇtefánik", "Štefánik"),
    ("Lˇubomír", "Ľubomír"),
    ("Jiˇrí Hoˇrejší", "Jiří Hořejší"),
    ("Jurˇca", "Jurča"),
    ("Gorˇse", "Gorše"),
    ("Bednarˇsek", "Bednaršek"),
])
def test_detached_caron_joins_a_consonant_or_e(source, expected):
    # focus set: a detached caron joined the next vowel (Buresǒvá, Sonǎ) and Turkish ğ came out ǧ;
    # vowel carons are pinyin tone marks, never a detached accent
    from sinonym.name_punctuation import attach_detached_diacritics

    assert attach_detached_diacritics(source) == expected


def test_soft_hyphen_artifact_is_deleted_but_a_syllable_boundary_is_kept(predictor):
    # ORCID corpus: a soft hyphen inside a non-Chinese name is a line-break artifact or mojibake;
    # on the Chinese path it stays the reviewed syllable-boundary cue
    assert resolve(predictor, "O.", None, "Zaval\u00adniuk") == ("O.", "", "Zavalniuk")
    assert resolve(predictor, "Jan", None, "Haščí\u00adk") == ("Jan", "", "Haščík")
    assert resolve(predictor, "Guo\u00ade", None, "Li")[0] == "Guo-E"


GLUED_SINGLE_INITIAL_CASES = [
    # inspire:57264 / 67927 — one initial written against the surname with no first name
    (("", None, "M.Nguyen"), ("M.", "", "Nguyen")),
    (("", None, "K.Krueger"), ("K.", "", "Krueger")),
    (("J", None, "A.Minahan"), ("J.", "A.", "Minahan")),
    # the Chinese path's own handling is unchanged
    (("", None, "Z.Li"), ("Z.", "", "Li")),
    (("", None, "Z.Y.Li"), ("Z.-Y.", "", "Li")),
]


@pytest.mark.parametrize(("source", "expected"), GLUED_SINGLE_INITIAL_CASES)
def test_one_glued_initial_is_split_off_the_surname(predictor, source, expected):
    assert resolve(predictor, *source) == expected


GLUED_GIVEN_INITIAL_CASES = [
    # an initial glued to the next given word: the word becomes a middle name
    (("M.Hesham", None, "El Naggar"), ("M.", "Hesham", "El Naggar")),
    (("M.Cristina", "L.", "Martins"), ("M.", "Cristina L.", "Martins")),
    # ... and a listed compound surname takes it
    (("J.Dias", "de", "Deus"), ("J.", "", "Dias de Deus")),
    # a word repeating the surname stays a middle name
    (("A.Nag", None, "Nag"), ("A.", "Nag", "Nag")),
]


@pytest.mark.parametrize(("source", "expected"), GLUED_GIVEN_INITIAL_CASES)
def test_glued_given_initial_is_split_off(predictor, source, expected):
    assert resolve(predictor, *source) == expected


VIETNAMESE_INITIALLED_CASES = [
    # paper 122189999's author: the spelled-out mentions already flip; the initialled ones did not,
    # splitting one person between ``p hung`` and ``h pham``
    (("Pham", "Q.", "Hung"), ("Hung", "Q.", "Pham")),
    (("Pham", "Quang", "Hung"), ("Hung", "Quang", "Pham")),
    (("Nguyen", "V.", "Hieu"), ("Hieu", "V.", "Nguyen")),
    # no surname to work from
    (("P.", None, "Hung"), ("P.", "", "Hung")),
]


@pytest.mark.parametrize(("source", "expected"), VIETNAMESE_INITIALLED_CASES)
def test_vietnamese_family_name_is_recognized_past_a_middle_initial(predictor, source, expected):
    assert resolve(predictor, *source) == expected


ABSTAIN_ECHO_CASES = [
    # footnote digits on an authored word never reach the echoed surname
    (("Song", None, "Chen1"), ("Song", "", "Chen")),
    # a parenthetical nickname after the surname never becomes the surname
    (("Fei", None, "Gao (Phoebe)"), ("Fei", "", "Gao")),
    # the abstain path promised "trailing authored word is the surname" but rebuilt the name from the parser's
    # re-split syllables (issue #35): a fragment of an authored word became the surname
    ((None, None, "Ya. G. Sinai"), ("Ya.", "G.", "Sinai")),
    (("Tzung", "K.", "Hsiai"), ("Tzung", "K.", "Hsiai")),
    (("Son", "A.", "Hoang"), ("Son", "A.", "Hoang")),
    # two authored given words pinyinized into one four-syllable chain
    # a structured source keeps its own fields when the echo fires (middle stays middle)
    (("Xiaojun", "Lance", "Lian"), ("Xiaojun", "Lance", "Lian")),
    (("Leni", "Hiu Tong", "Cheung"), ("Leni", "Hiu Tong", "Cheung")),
    # unchanged: resyllabifying one authored word, joining two short given words, joined initials
    ((None, None, "JihengLi"), ("Ji-Heng", "", "Li")),
    (("Weijun", None, "Li"), ("Wei-Jun", "", "Li")),
    (("Hiu Tong", None, "Cheung"), ("Hiu-Tong", "", "Cheung")),
    (("J.", "J.", "Zhu"), ("J.-J.", "", "Zhu")),
    (("Zhao-Rui", None, "Feng"), ("Zhao-Rui", "", "Feng")),
]


@pytest.mark.parametrize(("source", "expected"), ABSTAIN_ECHO_CASES)
def test_abstain_reading_keeps_authored_word_boundaries(predictor, source, expected):
    assert resolve(predictor, *source) == expected


CYRILLIC_CASES = [
    # Ukrainian and Kazakh letters outside the Russian block leaked past the Roman filter and the residue was
    # parsed as pinyin ("Ok Pa", "K. Ah"); Russian already failed the Chinese parse and was preserved
    (("Віра", None, "Поліщук"), ("Віра", "", "Поліщук")),
    (("Мілан", None, "Бєлік"), ("Мілан", "", "Бєлік")),
    (("Світлана", None, "Гладьо"), ("Світлана", "", "Гладьо")),
    (("Иван", None, "Петров"), ("Иван", "", "Петров")),
    (("Γιώργος", None, "Παπαδόπουλος"), ("Γιώργος", "", "Παπαδόπουλος")),
]


@pytest.mark.parametrize(("source", "expected"), CYRILLIC_CASES)
def test_cyrillic_and_greek_names_never_parse_as_chinese(predictor, source, expected):
    instance = Instance(pp_authors=[SourceAuthorFields(first_name=source[0], middle_names=source[1], last_name=source[2])])
    author = predictor.predict_batch([instance])[0].authors[0]
    assert (author.first_name, author.middle_names, author.last_name) == expected
    assert author.chinese_detected is False


SMALL_CLASS_CASES = [
    # a dotted-initials run before a multi-token surname is initials, not the M.A. degree
    (("M.A.", None, "Medina Alcaide"), ("M.", "A.", "Medina Alcaide")),
    (("M.A.", None, "Di Vito"), ("M.", "A.", "Di Vito")),
    # bare Hon is a Cantonese given name; only the dotted title is dropped
    (("Hon", "Lon", "Tam"), ("Hon", "Lon", "Tam")),
    (("Hon.", "John", "Smith"), ("John", "", "Smith")),
    # an all-caps credential lookalike alone in the surname field is the surname
    (("Hue", "Thi", "DO"), ("Hue", "Thi", "Do")),
    (("John", "Smith", "III"), ("John", "", "Smith")),
    # pinyin ü written as v is a syllable, not two initials
    (("Xv", None, "Wu"), ("Xv", "", "Wu")),
    # a trailing comma on a full given word before a real surname is punctuation; before initials it is the
    # surname-first marker it always was
    (("Thomas,", "W.", "Crowther"), ("Thomas", "W.", "Crowther")),
    (("Adamu,", None, "D. A."), ("D.", "A.", "Adamu")),
    # a last-only "Surname I. I." reads surname-first (unchanged); the single-initial form is left to the
    # scalar parse, since South-Indian "Given I." sources make it a coin flip (judged 12 for / 6 against)
    ((None, None, "Petunin A.V."), ("A.", "V.", "Petunin")),
]


@pytest.mark.parametrize(("source", "expected"), SMALL_CLASS_CASES)
def test_small_deterministic_classes_from_the_corpus_scan(predictor, source, expected):
    assert resolve(predictor, *source) == expected


ORDER_PRIOR_CASES = [
    # the corpus writes these pairs one way >= 80% of the time (pre-normalization names, 691M mentions): when the
    # router picks the PP parse, the prior settles the orientation
    (("Jie", None, "Shan"), ("Jie", "", "Shan")),
    (("Ye", None, "Zhang"), ("Ye", "", "Zhang")),
    (("Huang", None, "Xi"), ("Xi", "", "Huang")),
    (("Gu", None, "Xin"), ("Xin", "", "Gu")),
    (("Chen", None, "Hong"), ("Hong", "", "Chen")),
    (("Lu", None, "Rong"), ("Rong", "", "Lu")),
    # on abstain the source order stands even against the corpus majority: there the prior broke 19,574 registered
    # surnames and fixed 2,275 (ORCID registry)
    (("Lu", None, "Xi"), ("Lu", "", "Xi")),
    (("Gong", None, "Lei"), ("Gong", "", "Lei")),
    (("Liu", None, "Yang"), ("Liu", "", "Yang")),
    # already in the corpus orientation: untouched
    (("Xi", None, "Lu"), ("Xi", "", "Lu")),
    (("Fang", None, "Du"), ("Fang", "", "Du")),
    # outside the Chinese path or hyphenated: the prior does not apply
    (("Thomas", None, "Martin"), ("Thomas", "", "Martin")),
    (("Martin", None, "Thomas"), ("Martin", "", "Thomas")),
    (("Jin-Young", None, "Lee"), ("Jin-Young", "", "Lee")),
]


@pytest.mark.parametrize(("source", "expected"), ORDER_PRIOR_CASES)
def test_corpus_order_prior_settles_ambiguous_two_token_names(predictor, source, expected):
    assert resolve(predictor, *source) == expected


def test_corpus_order_prior_reason_is_reported(predictor):
    instance = Instance(pp_authors=[SourceAuthorFields(first_name="Jie", last_name="Shan")])
    author = predictor.predict_batch([instance])[0].authors[0]
    assert author.resolution_reason.value == "corpus_order_prior_assignment"


def test_corpus_order_prior_applies_without_a_pool_below_the_old_pool_bar(predictor):
    # the corpus writes Jing Yu 82% of the time: below the 86% bar 0.6.0 used with a pool, above the 80% no-pool bar
    instance = Instance(pp_authors=[SourceAuthorFields(first_name="Jing", last_name="Yu")])
    author = predictor.predict_batch([instance])[0].authors[0]
    assert author.resolution_reason == ResolutionReason.CORPUS_ORDER_PRIOR_ASSIGNMENT
    assert (author.first_name, author.last_name) == ("Jing", "Yu")


@pytest.mark.parametrize("surname", ["Efendi̇oğlu", "Çi̇ner", "Yerlikaya Aydemi̇r"])
def test_attached_combining_marks_are_not_moved(predictor, surname):
    # Turkish İ lower-cases to i + U+0307, which has no precomposed form; the detached-diacritic repair
    # moved that dot onto the next letter (Efendi̇oğlu -> Efendiȯğlu). An attached mark is never moved.
    assert resolve(predictor, "Attila", None, surname) == ("Attila", "", surname)


COMPOUND_CASES = [
    # corpus-backed compound surnames (data/COMPOUND_SURNAMES.md) end up whole in `last` from either source shape
    (("Jean", "du", "Plessis"), ("Jean", "", "du Plessis")),
    (("Jean", None, "du Plessis"), ("Jean", "", "du Plessis")),
    (("A.", "Gonzalez", "Lopez"), ("A.", "", "Gonzalez Lopez")),
    # vocabulary tier: a frequent Hispanic pair the corpus splits too often to be listed
    (("Maria", "Garcia", "Garcia"), ("Maria", "", "Garcia Garcia")),
    (("Maria", "Jose Gonzalez", "Lopez"), ("Maria", "Jose", "Gonzalez Lopez")),
    (("Ana", "Paula da", "Silva"), ("Ana", "Paula", "da Silva")),
    # a given-name middle before a plain surname is untouched
    (("John", "Michael", "de Vries"), ("John", "Michael", "de Vries")),
    # the rule only extends a surname: a longer parsed surname is never cut back to a listed suffix
    (("Rilbson", "Henrique Silva", "dos Santos"), ("Rilbson", "Henrique", "Silva dos Santos")),
    (("Erica", None, "van de Waal"), ("Erica", "", "van de Waal")),
    (("Maximillian", None, "Van Wyk de Vries"), ("Maximillian", "", "Van Wyk de Vries")),
    # a parse that reordered the fields keeps its given name and surname
    (("Luong", "Van", "Viet"), ("Viet", "Van", "Luong")),
    # a surname the source wrote whole and longer than any listed surface is kept whole
    (("Sergio", None, "Vazquez Montes de Oca"), ("Sergio", "", "Vazquez Montes de Oca")),
    (("José", "Boris", "Bellido Santa María"), ("José", "Boris", "Bellido Santa María")),
    (("Mayara", None, "Secco Torres Silva"), ("Mayara", "", "Secco Torres Silva")),
    # a surname the source wrote whole is restored when the parse had shortened it
    (("Ana", None, "Garcia Lopez"), ("Ana", "", "Garcia Lopez")),
    # hyphen tier: other people's sources write the surname hyphenated (Cohen-Tanugi)
    (("J.", "Cohen", "Tanugi"), ("J.", "", "Cohen Tanugi")),
    (("Johann", None, "Cohen-Tanugi"), ("Johann", "", "Cohen-Tanugi")),
    (("Hakim", "AL", "Garalleh"), ("Hakim", "", "AL Garalleh")),
    # ... unless the corpus overwhelmingly writes the head word as a middle name before that tail
    (("Rajesh", "Kumar", "Gupta"), ("Rajesh", "Kumar", "Gupta")),
]


EDGE_PARTICLE_CASES = [
    # a particle glued to the end of the middle field opens a one-word surname; a given name before it stays in the
    # middle, a surname word joins the surname (how the other sources write Gehrmann-De Ridder)
    (("Ana", "Maria-da", "Silva"), ("Ana", "Maria", "da Silva")),
    (("Ana", "Maria-de-la", "Cruz"), ("Ana", "Maria", "de la Cruz")),
    (("A.", "Gehrmann-De", "Ridder"), ("A.", "", "Gehrmann-De Ridder")),
    (("Uta", "Fritze-von", "Alvensleben"), ("Uta", "", "Fritze-von Alvensleben")),
    (("Ana", "de-la", "Cruz"), ("Ana", "", "de la Cruz")),
    (("Hans", "Peter-von", "Berg"), ("Hans", "Peter", "von Berg")),
    (("Joao", "Carlos do", "Nascimento"), ("Joao", "Carlos", "do Nascimento")),
    # the shape the other sources already resolve to is unchanged
    (("Daniel", "Martín de", "Blas"), ("Daniel", "Martín", "de Blas")),
    # initials, all-caps tokens and East Asian given syllables before the particle are left alone
    (("Ana", "M.-de", "Souza"), ("Ana", "M.-de", "Souza")),
    (("Jose", "MARTIN-DE", "BLAS"), ("Jose", "Martin-De", "Blas")),
    (("Thi", "Do", "Nguyen"), ("Thi", "Do", "Nguyen")),
    (("Zhi-De", None, "Wang"), ("Zhi-De", "", "Wang")),
    # the given field is never cut: hyphenated East Asian given names end in particle-like syllables
    (("Yeong-Der", None, "Yao"), ("Yeong-Der", "", "Yao")),
    (("Byung-Do", None, "Rhee"), ("Byung-Do", "", "Rhee")),
    # a word the corpus uses both as a given name and a surname is left where the source put it
    (("Daniel", "Martín-de", "Blas"), ("Daniel", "Martín-de", "Blas")),
    # a particle that opens a given word, or a surname longer than one word, is not this shape
    (("Luis", "del-Carmen", "Silva"), ("Luis", "del-Carmen", "Silva")),
    (("Ana", "Maria-da", "Silva Santos"), ("Ana", "Maria-da", "Silva Santos")),
    # Arabic / South Asian connectives are not moved
    (("Ali", "Abdul-Al", "Rahman"), ("Ali", "Abdul-Al", "Rahman")),
]


@pytest.mark.parametrize(("source", "expected"), EDGE_PARTICLE_CASES)
def test_edge_particles_open_the_surname(predictor, source, expected):
    assert resolve(predictor, *source) == expected


TOKEN_ORDER_CASES = [
    # a two-token Latin name whose first token is a corpus surname word and whose last is a corpus given name
    (("Ferrari", None, "Giulia"), ("Giulia", "", "Ferrari")),
    (("Kushnir", None, "Yishai"), ("Yishai", "", "Kushnir")),
    (("Smith", None, "John"), ("John", "", "Smith")),
    # given-first names and names with a middle, hyphen or weak evidence stay as written
    (("John", None, "Smith"), ("John", "", "Smith")),
    (("Maria", None, "Jose"), ("Maria", "", "Jose")),
    (("Smith", "A.", "John"), ("Smith", "A.", "John")),
    (("Prakash", None, "Sumit"), ("Prakash", "", "Sumit")),
    (("VV", None, "Robin"), ("VV", "", "Robin")),
]


CHAIN_CAP_CASES = [
    # a selected Chinese-path parse that chains more than one given name's worth of syllables keeps the authored words
    (("Zhen-Yin", "Annie", "Chen"), ("Zhen-Yin", "Annie", "Chen")),
    (("Ting-Wei", "Ernie", "Liao"), ("Ting-Wei", "Ernie", "Liao")),
    (("Leni", "Hiu Tong", "Cheung"), ("Leni", "Hiu Tong", "Cheung")),
    # two-syllable given names still hyphenate as before
    (("Weijun", None, "Zhang"), ("Wei-Jun", "", "Zhang")),
]


PARTICLE_GUARD_CASES = [
    # A: a two-letter all-caps token between mixed-case names is initials, not a particle (AL/LA/EL/DA/ST/TE/LE)
    (("Michelle", "LA", "Nelson"), ("Michelle", "LA", "Nelson")),
    (("Coen", "DA", "Stehouwer"), ("Coen", "DA", "Stehouwer")),
    # DE / VAN / VON / DI stay particles: sources that capitalize the surname capitalize the particle too
    (("Peter", "VAN", "OOSTEROM"), ("Peter", "", "van Oosterom")),
    (("Eduardo", "De", "Loures"), ("Eduardo", "", "De Loures")),
    # B: after a Korean / Cantonese / Vietnamese given syllable, bin / van / da is a given syllable
    (("Young", "Bin", "Kim"), ("Young", "Bin", "Kim")),
    (("Quan", "Van", "Dau"), ("Quan", "Van", "Dau")),
    (("Maria", "da", "Silva"), ("Maria", "", "da Silva")),
]


@pytest.mark.parametrize(("source", "expected"), PARTICLE_GUARD_CASES)
def test_particle_guards(predictor, source, expected):
    assert resolve(predictor, *source) == expected


@pytest.mark.parametrize(("source", "expected"), CHAIN_CAP_CASES)
def test_given_chain_cap_keeps_authored_words(predictor, source, expected):
    assert resolve(predictor, *source) == expected


@pytest.mark.parametrize(("source", "expected"), TOKEN_ORDER_CASES)
def test_token_order_prior_swaps_surname_first_latin_pairs(predictor, source, expected):
    assert resolve(predictor, *source) == expected


@pytest.mark.parametrize(("source", "expected"), COMPOUND_CASES)
def test_corpus_compound_surnames_are_joined_or_kept_whole(predictor, source, expected):
    assert resolve(predictor, *source) == expected


CORPUS_SCALE_CASES = [
    # a comma closing the first field before two full words is the surname-first marker (Crossref bylines from
    # Nigeria) unless the corpus knows the closed word as a given name; before an initial it is punctuation
    (("Anih,", "David", "Chinonso"), ("David", "Chinonso", "Anih")),
    (("Alharbi,", "Ghadah", "Lafi"), ("Ghadah", "Lafi", "Alharbi")),
    (("Jennifer,", "Alyson", "Carter"), ("Jennifer", "Alyson", "Carter")),
    (("Jayne,", "L", "Astbury"), ("Jayne", "L.", "Astbury")),
    (("Thomas,", "W.", "Crowther"), ("Thomas", "W.", "Crowther")),
    # with no middle the comma is the marker too, unless the last word is a known surname
    (("Opoku,", None, "Albert"), ("Albert", "", "Opoku")),
    (("Jennifer,", None, "Carter"), ("Jennifer", "", "Carter")),
    (("Roshini,", None, "Pillay"), ("Roshini", "", "Pillay")),
    # an all-caps MR / DR alone before a mixed-case surname, and hyphenated initials, are initials, not titles
    (("MR", None, "Islam"), ("MR", "", "Islam")),
    (("DR", None, "Patel"), ("DR", "", "Patel")),
    (("M-R", None, "Christiaens"), ("M.-R.", "", "Christiaens")),
    (("M.-R.", None, "Taskinen"), ("M.-R.", "", "Taskinen")),
    (("Dr.", None, "Smith"), ("", "", "Smith")),
    (("Mr", None, "Islam"), ("", "", "Islam")),
    (("DR", "JOHN", "SMITH"), ("John", "", "Smith")),
    # construct prefixes open a family name even though each is a frequent given name (registry: 73-100% registered whole)
    (("Mariani", "Abdul", "Hamid"), ("Mariani", "", "Abdul Hamid")),
    (("Nadia", "Md", "Nor"), ("Nadia", "", "Md Nor")),
    (("Adiratna", "Mat", "Ripen"), ("Adiratna", "", "Mat Ripen")),
    (("Lahsen", "Ait", "Brahim"), ("Lahsen", "", "Ait Brahim")),
    # `ul` is not one: half the people split on it registered the tail alone
    (("Ehtesham", "Ul", "Haq"), ("Ehtesham", "Ul", "Haq")),
    # Hyun is a surname only as the last token; leading it is a given syllable
    (("Hyun", "Ho", "Sung"), ("Hyun", "Ho", "Sung")),
    (("Hyun", "Sik", "Jun"), ("Hyun", "Sik", "Jun")),
    (("Chang", "Ho", "Hyun"), ("Chang", "Ho", "Hyun")),
    # a name written only in Chinese characters is surname-first; the romanized-pair order prior must not flip it,
    # but it still applies when the source also carries a romanized form
    ((None, None, "杨柳"), ("Liu", "", "Yang")),
    ((None, None, "董江"), ("Jiang", "", "Dong")),
    (("庞", "璐", "Pang Lu"), ("Lu", "", "Pang")),
]


@pytest.mark.parametrize(("source", "expected"), CORPUS_SCALE_CASES)
def test_corpus_scale_regressions(predictor, source, expected):
    assert resolve(predictor, *source) == expected


SURNAME_FIRST_INITIALS_CASES = [
    # a surname written first, dotted given-name initials after it; the corpus uses the word as a surname
    (("Dmitrieva", None, "N.M."), ("N.", "M.", "Dmitrieva")),
    (("Kazachenko", None, "A. V."), ("A.", "V.", "Kazachenko")),
    (("Назарова", None, "А.Р."), ("А.", "Р.", "Назарова")),
    # the word is a given name and the initials the family name: unchanged
    (("Sowmya", None, "K. R."), ("Sowmya", "K.", "R.")),
    # a single bare initial is not this shape
    (("Priya", None, "S"), ("Priya", "", "S")),
]


@pytest.mark.parametrize(("source", "expected"), SURNAME_FIRST_INITIALS_CASES)
def test_surname_first_initials(predictor, source, expected):
    assert resolve(predictor, *source) == expected


EXCLUDED_COMPOUND_HEAD_CASES = [
    # a given name or patronymic heading the tail is not part of the family name
    (("Ahmed", "Mohamed", "Ali"), ("Ahmed", "Mohamed", "Ali")),
    (("Ali", "Mohammad", "Hussein"), ("Ali", "Mohammad", "Hussein")),
    # other construct prefixes still open a family name
    (("Siti", "Abdul", "Rahman"), ("Siti", "", "Abdul Rahman")),
    (("Ahmad", "Mohd", "Noor"), ("Ahmad", "", "Mohd Noor")),
]


@pytest.mark.parametrize(("source", "expected"), EXCLUDED_COMPOUND_HEAD_CASES)
def test_excluded_compound_heads(predictor, source, expected):
    assert resolve(predictor, *source) == expected


SOURCE_DOUBLE_SURNAME_CASES = [
    # a two-word last name the source wrote whole stays whole when the given name is an initial or two given names
    (("S.", None, "Rivera Maza"), ("S.", "", "Rivera Maza")),
    (("L", None, "Vicioso Recio"), ("L.", "", "Vicioso Recio")),
    (("Fredy Alberto", None, "Velásquez Barón"), ("Fredy Alberto", "", "Velásquez Barón")),
    # the peeled word must be a known surname: sources also put a middle given name there (``P. / Jyoteeshkumar Reddy``)
    (("P.", None, "Jyoteeshkumar Reddy"), ("P.", "Jyoteeshkumar", "Reddy")),
    (("Martha Leonor", None, "Ayala Rengifo"), ("Martha", "Leonor Ayala", "Rengifo")),
    # an East Asian two-word last name after an initial is given name + surname
    (("T.", None, "Ngoc Mai"), ("T.", "Ngoc", "Mai")),
    # the first word is a given name the source put in the last-name field
    (("J.", None, "Alberto Garcia"), ("J.", "Alberto", "Garcia")),
]


@pytest.mark.parametrize(("source", "expected"), SOURCE_DOUBLE_SURNAME_CASES)
def test_source_double_surname_kept(predictor, source, expected):
    assert resolve(predictor, *source) == expected


GIVEN_NAME_PHRASE_CASES = [
    # a Spanish / Portuguese given-name phrase stays with the given name it follows, never in the surname
    (("Sofía", "del Carmen", "Jaramillo Quiroz"), ("Sofía del Carmen", "", "Jaramillo Quiroz")),
    (("María del Carmen", None, "Zárate Hernández"), ("María del Carmen", "", "Zárate Hernández")),
    (("Maria", "Daniela del Pilar", "Zambrano Arenas"), ("Maria", "Daniela del Pilar", "Zambrano Arenas")),
    (("José", "de Jesús", "Pérez Bueno"), ("José de Jesús", "", "Pérez Bueno")),
    (("Mª", "de los Ángeles", "Nogales Naharro"), ("Mª de los Ángeles", "", "Nogales Naharro")),
    # written at the start of last_name: only the clearly given-name phrases move
    (("Diana", None, "del Carmen Torres Corrales"), ("Diana del Carmen", "", "Torres Corrales")),
    (("Irinéa", None, "de Lourdes Batista"), ("Irinéa de Lourdes", "", "Batista")),
    (("Victória", None, "de Jesus Souza"), ("Victória", "", "de Jesus Souza")),
    # phrases that are as often part of the family name are left to the parse
    (("Ana", "de los Santos", "Gomez"), ("Ana", "", "de los Santos Gomez")),
    # a surname particle phrase is not a given-name phrase
    (("J.", "M.", "García del Moral"), ("J.", "M.", "García del Moral")),
]


@pytest.mark.parametrize(("source", "expected"), GIVEN_NAME_PHRASE_CASES)
def test_given_name_phrase_stays_out_of_the_surname(predictor, source, expected):
    assert resolve(predictor, *source) == expected


# ORCID index paper 232113405 (Development, 2021, Highwire): the byline and its production-style venue pool from the
# 2026-10-21 snapshot. The router picks VYS for "Yang / Tao" and the ORCID registry family name is Tao; the corpus order
# prior used to flip it to "Tao / Yang".
VYS_PRIOR_BYLINE = [
    ("Zhiyuan", "He"),
    ("Ting", "Zou"),
    ("Qiao", "Xiao"),
    ("Guoqiang", "Yuan"),
    ("Miaomiao", "Liu"),
    ("Yang", "Tao"),
    ("Dan", "Zhou"),
    ("Xu", "Zhang"),
    ("Qiming", "Deng"),
    ("Shiquan", "Wang"),
    ("Aiping", "Zheng"),
    ("Jun", "Zhu"),
    ("Yueyang", "Liang"),
    ("Xiumei", "Yu"),
    ("Aijun", "Wang"),
    ("Huainian", "Liu"),
    ("Lingxia", "Wang"),
    ("Ping", "Li"),
    ("Shuangcheng", "Li"),
]
VYS_PRIOR_POOL = [
    "Yongchun Zhang",
    "Jianwen Que",
    "Jianzhen Li",
    "Guoqiang Gu",
    "Li He",
    "Weiguo He",
    "Jiayu Wen",
    "Qi Dai",
    "Hao Luo",
    "Chun-Wei Feng",
    "Guangqin Wang",
    "Chao Li",
    "Shunji He",
    "Zhiyong Liu",
    "Jianshi Yu",
    "Tian Liu",
    "Xiujuan Yang",
    "Mengyun Wang",
    "Yue Du",
    "Song Gao",
    "Zheng Wang",
    "Pengxiang Qu",
    "Yang Gao",
    "Jingyi Wang",
    "Zhengqi Liu",
    "Jingcheng Zhang",
    "Yong Zhang",
    "Suzhu Qing",
    "Yongsheng Wang",
    "Huijuan Wu",
    "Nan Tang",
    "Yuanbin Xie",
    "Ching H. Chen",
]


def test_order_prior_does_not_override_a_router_decision_made_with_a_venue_pool(predictor):
    authors = [SourceAuthorFields(first_name=first, last_name=last) for first, last in VYS_PRIOR_BYLINE]
    with_pool = predictor.predict_batch([Instance(pp_authors=authors, vys_other_names=VYS_PRIOR_POOL)])[0].authors[5]
    assert (with_pool.first_name, with_pool.last_name) == ("Yang", "Tao")
    assert with_pool.resolution_reason != ResolutionReason.CORPUS_ORDER_PRIOR_ASSIGNMENT


def test_order_prior_does_not_rewrite_a_vys_pick(predictor):
    # with a venue pool the router's pick stands against the corpus majority; this source wrote `Zhou YI`, so Yi is the
    # surname, and 0.6.0's prior flipped it
    byline = list(VYS_PRIOR_BYLINE)
    byline[5] = ("Zhou", "Yi")
    authors = [SourceAuthorFields(first_name=first, last_name=last) for first, last in byline]
    author = predictor.predict_batch([Instance(pp_authors=authors, vys_other_names=VYS_PRIOR_POOL)])[0].authors[5]
    assert (author.first_name, author.last_name) == ("Zhou", "Yi")
    assert author.resolution_reason == ResolutionReason.VYS_SELECTED
    assert author.resolution_provenance == ResolutionProvenance.VYS


def test_order_prior_does_not_rewrite_a_pp_pick_made_with_a_venue_pool(predictor):
    # as above for a PP pick: this source wrote `Sun XU`, so Xu is the surname, and 0.6.0's prior flipped it
    byline = list(VYS_PRIOR_BYLINE)
    byline[5] = ("Sun", "Xu")
    authors = [SourceAuthorFields(first_name=first, last_name=last) for first, last in byline]
    author = predictor.predict_batch([Instance(pp_authors=authors, vys_other_names=VYS_PRIOR_POOL)])[0].authors[5]
    assert (author.first_name, author.last_name) == ("Sun", "Xu")
    assert author.resolution_reason == ResolutionReason.PP_SELECTED


def test_rejected_source_assignment_keeps_the_paper(predictor):
    # a source put collaboration text in one surname; that author is passed through as written and the paper still resolves
    authors = [
        SourceAuthorFields(first_name="Heba", last_name="Bintalib"),
        SourceAuthorFields(first_name="S", middle_names="Hamza", last_name="Abbas"),
        SourceAuthorFields(first_name="Klaus", last_name="Warnatz"),
        SourceAuthorFields(first_name="John", middle_names="R", last_name="Hurst (and On Behalf Of The Eglildnet Consort"),
    ]
    rows = predictor.predict_batch([Instance(pp_authors=authors)])[0].authors
    assert len(rows) == 4
    assert [(r.first_name, r.last_name) for r in rows[:3]] == [("Heba", "Bintalib"), ("S.", "Abbas"), ("Klaus", "Warnatz")]
    assert (rows[3].first_name, rows[3].middle_names, rows[3].last_name) == (
        "John",
        "R",
        "Hurst (and On Behalf Of The Eglildnet Consort",
    )
    assert rows[3].resolution_reason == ResolutionReason.NON_PERSON_SOURCE_PASSTHROUGH


def test_rejected_given_phrase_assignment_is_passed_through(predictor):
    author = predictor.predict_batch(
        [Instance(pp_authors=[SourceAuthorFields(first_name="M.", middle_names="del Carmen", last_name="Garcia Lopez 1,2")])],
    )[0].authors[0]
    assert (author.first_name, author.middle_names, author.last_name) == ("M.", "del Carmen", "Garcia Lopez 1,2")
    assert author.resolution_reason == ResolutionReason.NON_PERSON_SOURCE_PASSTHROUGH


def test_rejected_reviewed_rule_pick_keeps_the_author(predictor):
    # a reviewed pattern matches but the normalizer rejects its pick: the rule is skipped and the author is still written
    instance = Instance(pp_authors=[SourceAuthorFields(first_name="E.A.", last_name="and Solomon")])
    author = predictor.predict_batch([instance])[0].authors[0]
    assert (author.first_name, author.middle_names, author.last_name) == ("E.A.", "", "and Solomon")
    assert author.resolution_reason == ResolutionReason.NON_PERSON_SOURCE_PASSTHROUGH


def test_rejected_hangul_affiliation_pick_does_not_fail_the_paper(predictor):
    source = SourceAuthorFields(first_name="MD", middle_names="PhD 서울대학교", last_name="J University")
    author = predictor.predict_batch([Instance(pp_authors=[source])])[0].authors[0]
    assert (author.first_name, author.middle_names, author.last_name) == ("MD", "PhD 서울대학교", "J University")
    assert author.resolution_reason == ResolutionReason.NON_PERSON_SOURCE_PASSTHROUGH


def test_reviewed_roster_check_is_skipped_for_papers_of_other_sizes(predictor, monkeypatch):
    # the roster rule runs once per author; building its roster key on every author of a large paper is quadratic
    from sinonym.timo import _resolution  # noqa: PLC0415

    key_calls = []
    original_key = _resolution._source_component_key  # noqa: SLF001
    monkeypatch.setattr(_resolution, "_source_component_key", lambda s: key_calls.append(s) or original_key(s))
    keys_per_rule_call = []
    original = _resolution._reviewed_paper_wide_trailing_affiliation_assignment  # noqa: SLF001

    def counting(source, paper_authors, normalizer, surname_is_recognized):
        before = len(key_calls)
        result = original(source, paper_authors, normalizer, surname_is_recognized)
        keys_per_rule_call.append(len(key_calls) - before)
        return result

    monkeypatch.setattr(_resolution, "_reviewed_paper_wide_trailing_affiliation_assignment", counting)
    authors = [SourceAuthorFields(first_name=f"Given{i}", last_name=f"Surname{i}") for i in range(50)]
    predictor.predict_batch([Instance(pp_authors=authors)])
    assert len(keys_per_rule_call) == 50
    assert sum(keys_per_rule_call) == 0
