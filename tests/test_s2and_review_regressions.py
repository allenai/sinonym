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
