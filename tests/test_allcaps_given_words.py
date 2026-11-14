"""A three-letter given name in capitals stays in capitals as initials unless registry people use it as a given name."""

import pytest

from sinonym.timo.interface import Instance, Predictor, PredictorConfig, SourceAuthorFields
from sinonym.utils.allcaps_given_words import is_capitalized_initials

INITIALS = [
    (("ESL", None, "Pedersen"), ("ESL", "", "Pedersen")),
    (("NAR", None, "Rao"), ("NAR", "", "Rao")),
    (("TPI", None, "Pembroke"), ("TPI", "", "Pembroke")),
    (("BRC", None, "Saraiva"), ("BRC", "", "Saraiva")),
    (("RJM", None, "Lammers"), ("RJM", "", "Lammers")),
    # a word that is also a name, but registry people more often have it as initials
    (("ART", None, "Figueiredo"), ("ART", "", "Figueiredo")),
    (("ESL", "J", "Pedersen"), ("ESL", "J.", "Pedersen")),
    (("Pedersen, ESL", None, None), ("ESL", "", "Pedersen")),
    (("ESL", None, "Pedersen Jr"), ("ESL", "", "Pedersen")),
]

NAMES = [
    (("ANA", None, "Dias"), ("Ana", "", "Dias")),
    (("EWA", None, "Kowalska"), ("Ewa", "", "Kowalska")),
    (("HUI", None, "Smith"), ("Hui", "", "Smith")),
    (("ANA MARIA", None, "Dias"), ("Ana", "Maria", "Dias")),
    # the Chinese path
    (("LEI", None, "WANG"), ("Lei", "", "Wang")),
    (("WEI", None, "Zhang"), ("Wei", "", "Zhang")),
    # a surname in capitals too: the whole name is in capitals, which S2AND does not read as initials
    (("ESL", None, "PEDERSEN"), ("Esl", "", "Pedersen")),
    (("EDO", "BAGUS", "PRASETYO"), ("Edo", "Bagus", "Prasetyo")),
    (("PEDERSEN, ESL", None, None), ("Esl", "", "Pedersen")),
]

UNCHANGED = [
    (("MI", None, "Williams"), ("MI", "", "Williams")),
    (("YU", None, "Wang"), ("Yu", "", "Wang")),
    (("JOHN", None, "Smith"), ("John", "", "Smith")),
    (("Esl", None, "Pedersen"), ("Esl", "", "Pedersen")),
    ((None, None, "ESL"), ("", "", "Esl")),
    (("ÖZE", None, "Smith"), ("Öze", "", "Smith")),
    # a middle name in capitals is cased as before: S2AND compares middle names without case
    (("Eva", "ESL", "Pedersen"), ("Eva", "Esl", "Pedersen")),
]

# Known limits, pinned so a change is noticed.
KNOWN_LIMITS = [
    # a listed name that is this person's initials (Melissa A. E.)
    (("MAE", None, "Lawson"), ("Mae", "", "Lawson")),
]


@pytest.fixture(scope="module")
def predictor() -> Predictor:
    return Predictor(PredictorConfig(parallel="never"), ".")


def resolve(predictor: Predictor, first, middle, last):
    source = SourceAuthorFields(first_name=first, middle_names=middle, last_name=last)
    author = predictor.predict_batch([Instance(pp_authors=[source])])[0].authors[0]
    return author.first_name, author.middle_names, author.last_name


@pytest.mark.parametrize(("source", "expected"), INITIALS + NAMES + UNCHANGED + KNOWN_LIMITS)
def test_allcaps_given_name(predictor, source, expected):
    assert resolve(predictor, *source) == expected


def test_is_capitalized_initials():
    assert is_capitalized_initials("ESL")
    assert not is_capitalized_initials("ANA")
    assert not is_capitalized_initials("Esl")
    assert not is_capitalized_initials("MI")
    assert not is_capitalized_initials("ESLP")
    assert not is_capitalized_initials("ÖZE")
