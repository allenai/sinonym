"""0.7.2: a generational suffix closing the middle field goes to the suffix field."""

import pytest

from sinonym.timo.interface import Instance, Predictor, SourceAuthorFields

CASES = [
    (("J.", "Jr", "Bradley", None), ("J.", "", "Bradley", "Jr")),
    (("Joseph", "Thomas Iii", "Bradley", None), ("Joseph", "Thomas", "Bradley", "III")),
    (("José", "Gonçalves Jr", "Franco", None), ("José", "Gonçalves", "Franco", "Jr")),
    (("Thomas", "E. IV", "Cleveland", None), ("Thomas", "E.", "Cleveland", "IV")),
    (("Ramés", "II", "Salcedo Jiménez", None), ("Ramés", "", "Salcedo Jiménez", "II")),
    (("Hendrik", "Hg III", "Hansen", None), ("Hendrik", "H. G.", "Hansen", "III")),
    (("Fracisco", "Emmanuel Jr., III", "Munsayac", None), ("Fracisco", "Emmanuel Jr.", "Munsayac", "III")),
    (("John", "Paul, III", "Jones", None), ("John", "Paul", "Jones", "III")),
    (("Mervyn", "Jr", "Lim", None), ("Mervyn", "", "Lim", "Jr")),
]
CONTROLS = [
    ("Claire", "SR", "Holmes", None),
    ("Fiona", "JR", "Eccles", None),
    ("Héctor", "Junior", "Peche Cruz", None),
    ("Nick", "Sr", "Lan", None),
    ("Yun", "Ii", "Go", None),
    ("Soo", "II", "Shin", None),
    ("Strahil", "Iv", "Pastuhov", None),
    ("Byung", "II", "Kim", None),
    ("Yu", "Jr", "Lin", None),
    ("Chung", "II", "Wi", None),
    ("ARISE", "III", "Consortium", None),
    ("Wei", "Jr", "Lin", None),
]


def _resolve(predictor: Predictor, fields: tuple) -> tuple:
    first, middle, last, suffix = fields
    source = SourceAuthorFields(first_name=first, middle_names=middle, last_name=last, suffix=suffix)
    author = predictor.predict_batch([Instance(pp_authors=[source])])[0].authors[0]
    return author.first_name, author.middle_names, author.last_name, author.suffix


@pytest.mark.parametrize(("source", "expected"), CASES)
def test_middle_suffix_moves_to_suffix(predictor: Predictor, source: tuple, expected: tuple) -> None:
    assert _resolve(predictor, source) == expected


@pytest.mark.parametrize("source", CONTROLS)
def test_ambiguous_middle_is_not_a_suffix(predictor: Predictor, source: tuple) -> None:
    first, middle, last, _ = source
    assert _resolve(predictor, source) == (first, middle, last, None)


@pytest.mark.parametrize(("first", "last"), [("Fuji", "Ren"), ("FUJI", "REN")])
def test_reviewed_given_first_name_keeps_source_order(predictor: Predictor, first: str, last: str) -> None:
    assert _resolve(predictor, (first, None, last, None)) == ("Fuji", "", "Ren", None)


def test_reviewed_given_first_name_keeps_source_order_beside_japanese_names(predictor: Predictor) -> None:
    sources = [("Kazuyuki", "Matsumoto"), ("Fuji", "Ren"), ("Shingo", "Kuroiwa")]
    authors = predictor.predict_batch(
        [Instance(pp_authors=[SourceAuthorFields(first_name=first, last_name=last) for first, last in sources])],
    )[0].authors
    assert [(author.first_name, author.last_name) for author in authors] == sources


@pytest.mark.parametrize(
    "sources",
    [
        [("Ren", "Fuji")],
        [("Kazuyuki", "Matsumoto"), ("Ren", "Fuji"), ("Shingo", "Kuroiwa")],
        [("Sun", "Xiao"), ("Ye", "Jiaqi"), ("Ren", "Fuji")],
    ],
)
def test_reviewed_surname_first_name_is_swapped(predictor: Predictor, sources: list) -> None:
    authors = predictor.predict_batch(
        [Instance(pp_authors=[SourceAuthorFields(first_name=first, last_name=last) for first, last in sources])],
    )[0].authors
    focal = authors[[first for first, _ in sources].index("Ren")]
    assert (focal.first_name, focal.last_name) == ("Fuji", "Ren")
