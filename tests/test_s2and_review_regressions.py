"""Regressions from the S2AND 2026-09 name-order review (revert_list.csv, s2and_exports_20260925).

Each case is a structured source exactly as the author-refresh backfill sent it and the
reviewer-confirmed correct fields.
"""

import pytest

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


DETACHED_DIACRITIC_CASES = [
    # inspire: a spacing diaeresis/ring left by PDF extraction split the surname
    (("H", None, "Arenh¨ovel"), ("H.", "", "Arenhövel")),
    (("M.", None, "B¨oge"), ("M.", "", "Böge")),
    (("H.", None, "Pru˚chová"), ("H.", "", "Průchová")),
    # a mark that composes with nothing is left alone
    (("Nicolas", None, "D´ Amelio"), ("Nicolas", "", "D' Amelio")),
]


@pytest.mark.parametrize(("source", "expected"), DETACHED_DIACRITIC_CASES)
def test_detached_diacritic_rejoins_its_letter(predictor, source, expected):
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
    # the corpus writes these pairs one way >= 80% of the time (pre-normalization names, 691M mentions): the
    # reviewed S2AND labels agreed with the corpus on every one of them, the byline-only readings did not
    (("Lu", None, "Xi"), ("Xi", "", "Lu")),
    (("Gong", None, "Lei"), ("Lei", "", "Gong")),
    (("Huang", None, "Xi"), ("Xi", "", "Huang")),
    (("Gu", None, "Xin"), ("Xin", "", "Gu")),
    (("Chen", None, "Hong"), ("Hong", "", "Chen")),
    (("Lu", None, "Rong"), ("Rong", "", "Lu")),
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
    instance = Instance(pp_authors=[SourceAuthorFields(first_name="Lu", last_name="Xi")])
    author = predictor.predict_batch([instance])[0].authors[0]
    assert author.resolution_reason.value == "corpus_order_prior_assignment"


def test_leading_combining_mark_does_not_glue_words(predictor):
    # ``Mithat ̈unsal`` (NFKC remnant of a spacing diaeresis before the surname) stays two words
    assert resolve(predictor, "Mithat", None, "\u0308unsal") == ("Mithat", "", "Ünsal")


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
]


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
    (("Hakim", "AL", "Garalleh"), ("Hakim", "AL", "Garalleh")),
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
