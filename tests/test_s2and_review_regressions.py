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


SOURCE_BOUNDARY_CASES = [
    # inspire:28657 — multi-token surname sliced into middle + surname
    (("C", None, "Garcia Recio"), ("C", "", "Garcia Recio")),
    (("A.", None, "Sabio Vera"), ("A.", "", "Sabio Vera")),
    (("W.", None, "Haj Ahmad"), ("W.", "", "Haj Ahmad")),
]


@pytest.mark.parametrize(("source", "expected"), SOURCE_BOUNDARY_CASES)
def test_scalar_baseline_keeps_source_surname_boundary(predictor, source, expected):
    assert resolve(predictor, *source) == expected


SURNAME_MAY_STILL_GROW_OR_BE_SPLIT = [
    # growing the surname from a structured middle is a reviewed correct fix
    (("C.", "H. Llewellyn", "Smith"), ("C.", "H.", "Llewellyn Smith")),
    (("A.", "De", "Rújula"), ("A.", "", "De Rújula")),
    # a last-name-only source made no partition claim
    (("", None, "A.A.Andrianov"), ("A.", "A.", "Andrianov")),
    (("", None, "Hwa-Yaw Tam"), ("Hwa-Yaw", "", "Tam")),
]


@pytest.mark.parametrize(("source", "expected"), SURNAME_MAY_STILL_GROW_OR_BE_SPLIT)
def test_source_boundary_guard_is_shrink_only(predictor, source, expected):
    assert resolve(predictor, *source) == expected


CAPS_SURNAME_CUE_CASES = [
    # MAG marks the surname in caps; the delivered names kept input order and title-cased it
    (("LI", None, "Hui"), ("Hui", "", "Li")),
    (("LIU", None, "Lian"), ("Lian", "", "Liu")),
    (("Lian", None, "LIU"), ("Lian", "", "Liu")),
    (("LI", None, "Huihui"), ("Hui-Hui", "", "Li")),
    # a particle before the caps surname stays with it
    (("A. S.", None, "de CASTRO"), ("A.", "S.", "de Castro")),
    (("J. W.", None, "van HOLTEN"), ("J.", "W.", "van Holten")),
]


@pytest.mark.parametrize(("source", "expected"), CAPS_SURNAME_CUE_CASES)
def test_all_caps_token_marks_the_surname(predictor, source, expected):
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


@pytest.mark.parametrize("source", [("JH", None, "Smith"), ("JOHN", None, "SMITH"), ("John", None, "Smith III")])
def test_caps_cue_ignores_bare_initials_and_all_caps_names(predictor, source):
    assert resolve(predictor, *source)[2].casefold().startswith("smith")


def test_scalar_baseline_still_canonicalizes_initials_without_moving_boundary(predictor):
    assert resolve(predictor, "J", None, "Smith") == ("J.", "", "Smith")
