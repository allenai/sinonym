"""0.7.2: a generational suffix closing the middle field goes to the suffix field."""

import pytest

from sinonym.services.source_repairs import split_joined_initials
from sinonym.timo.interface import Instance, Predictor, SourceAuthorFields, _is_source_passthrough, _korean_given

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


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (("RONALD", "C.", "MA", None), ("Ronald", "C.", "Ma", None)),
        (("RONALD", "C.W.", "MA", None), ("Ronald", "C. W.", "Ma", None)),
        (("JOHN", "O.", "MA", None), ("John", "O.", "Ma", None)),
        (("GEORGE", "S.-K.", "MA", None), ("George", "S.-K.", "Ma", None)),
        (("NYUK", "LING", "MA", None), ("Nyuk", "Ling", "Ma", None)),
        (("THỊ", "THANH HIẾU", "MA", None), ("Thị", "Thanh Hiếu", "Ma", None)),
    ],
)
def test_capitals_ma_stays_the_surname(predictor: Predictor, source: tuple, expected: tuple) -> None:
    assert _resolve(predictor, source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (("John", "II.", "Smith", None), ("John", "", "Smith", "II")),
        (("John", "III.", "Smith", None), ("John", "", "Smith", "III")),
        (("John", "IV.", "Smith", None), ("John", "", "Smith", "IV")),
        (("John", "IV.", "de Vries", None), ("John", "", "de Vries", "IV")),
        (("John", "II", "Smith", None), ("John", "", "Smith", "II")),
        (("Soo", "II.", "Shin", None), ("Soo", "II.", "Shin", None)),
        (("John", "II. de", "Vries", None), ("John", "II.", "de Vries", None)),
        (("John", "VI.", "Smith", None), ("John", "VI.", "Smith", None)),
        (("John", "IX.", "Smith", None), ("John", "IX.", "Smith", None)),
        (("John", "XX.", "Smith", None), ("John", "XX.", "Smith", None)),
        (("Johanna", "IP", "de Vries", None), ("Johanna", "I. P.", "de Vries", None)),
        (("John", "XL.", "Smith", None), ("John", "X. L.", "Smith", None)),
    ],
)
def test_dotted_roman_numeral_is_a_suffix_or_stays_unsplit(predictor: Predictor, source: tuple, expected: tuple) -> None:
    assert _resolve(predictor, source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (("DG", "III", "Association", None), ("DG", "III", "Association", None)),
        (("ARISE", "III", "Consortium", None), ("ARISE", "III", "Consortium", None)),
        (("DG", "III", "Downey", None), ("D.", "G.", "Downey", "III")),
        (("DG", None, "Downey", None), ("D.", "G.", "Downey", None)),
    ],
)
def test_undone_suffix_move_restores_the_given_field(predictor: Predictor, source: tuple, expected: tuple) -> None:
    assert _resolve(predictor, source) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Yu-Lina Wang", "Yu-Lina Wang"),
        ("Yü-Lina Wang", "Yü-Lina Wang"),
        ("Lu-Lina Wang", "Lu-Lina Wang"),
        ("Lü-Lina Wang", "Lü-Lina Wang"),
        ("Wei-Lina Wang", "Wei-Lina Wang"),
        ("Wēi-Lina Wang", "Wei-Lina Wang"),
        ("Le-Lina Wang", None),
        ("YuLina Wang", None),
        ("Lina Wang", None),
        ("Hansen-Lange Wang", None),
    ],
)
def test_accented_bound_given_part_counts_as_chinese(detector, name: str, expected: str | None) -> None:
    result = detector.normalize_name(name)
    assert (result.result if result.success else None) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (("Budi", "S.T. M.Kom.", "Santoso", None), ("Budi", "", "Santoso", None)),
        (("Budi", "S.E. M.Si.", "Santoso", None), ("Budi", "", "Santoso", None)),
        (("Budi", "Ir. M.Kom.", "Santoso", None), ("Budi", "", "Santoso", None)),
        (("Budi", "ST M.Kom.", "Santoso", None), ("Budi", "", "Santoso", None)),
        (("Budi A.", "S.T. M.Kom.", "Santoso", None), ("Budi", "A.", "Santoso", None)),
        (("Budi", "M.Kom. S.T.", "Santoso", None), ("Budi", "", "Santoso", None)),
        (("S.T. M.Kom.", "Budi", "Santoso", None), ("Budi", "", "Santoso", None)),
        (("Budi", "A. M.Kom.", "Santoso", None), ("Budi", "A.", "Santoso", None)),
        (("Budi", "S. T.", "Santoso", None), ("Budi", "S. T.", "Santoso", None)),
    ],
)
def test_bachelor_then_master_degree_run_is_stripped(predictor: Predictor, source: tuple, expected: tuple) -> None:
    assert _resolve(predictor, source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (("Wei", "PhD", "Du", None), ("Wei", "", "Du", None)),
        (("Thi", "PhD", "Le", None), ("Thi", "", "Le", None)),
        (("Wei", "PhD", "Ma", None), ("Wei", "", "Ma", None)),
        (("Anna", "PhD", "Da", None), ("Anna", "", "Da", None)),
        (("Anna", "PhD", "Di", None), ("Anna", "", "Di", None)),
        (("Anna", "PhD", "La", None), ("Anna", "", "La", None)),
        (("Thi", "PhD", "Do", None), ("Thi", "", "Do", None)),
        (("Thi", "PhD", "Ba", None), ("Thi", "", "Ba", None)),
        (("Carsten", "Dr.", "Le", None), ("Carsten", "", "Le", None)),
        (("Translated", "by N.B.", "Le", None), ("N.", "B.", "Le", None)),
        (("Wei", "PhD", "Smith", None), ("Wei", "", "Smith", None)),
    ],
)
def test_listed_short_surname_does_not_block_title_and_degree_repairs(
    predictor: Predictor,
    source: tuple,
    expected: tuple,
) -> None:
    assert _resolve(predictor, source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (("Wei", "PhD", "de", None), ("Wei", "Phd", "de", None)),
        (("Wei", "PhD", "DU", None), ("Wei", "Phd", "du", None)),
        (("Anna", "PhD", "University", None), ("Anna", "PhD", "University", None)),
        (("Translated", "by N.B.", "Group", None), ("Translated", "By N. B.", "Group", None)),
        (("Anna", "Dr.", "Group", None), ("Anna", "Dr.", "Group", None)),
        (("Anna", "PhD", "Dr", None), ("Anna", "Phd", "Dr", None)),
        (("Anna", "PhD", "Md", None), ("Anna", "Phd", "Md", None)),
    ],
)
def test_unchanged_control_surname_field_without_a_surname(predictor: Predictor, source: tuple, expected: tuple) -> None:
    assert _resolve(predictor, source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (("A. B.", "Dr.", "Smith", None), ("A.", "B.", "Smith", None)),
        (("J.", "Dr.", "Smith", None), ("J.", "", "Smith", None)),
        (("J.A.", "Dr.", "Smith", None), ("J.", "A.", "Smith", None)),
        (("J.-P.", "Dr.", "Smith", None), ("J.-P.", "", "Smith", None)),
        (("Carsten", "Dr.", "Tille", None), ("Carsten", "", "Tille", None)),
        (("JA", "Dr.", "Smith", None), ("JA", "Dr.", "Smith", None)),
    ],
)
def test_dotted_initials_do_not_block_title_strip(predictor: Predictor, source: tuple, expected: tuple) -> None:
    assert _resolve(predictor, source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (("DG", None, "Xi", None), ("D.-G.", "", "Xi", None)),
        (("DG", None, "Li", None), ("D.-G.", "", "Li", None)),
        (("DG", None, "Vi", None), ("DG", "", "Vi", None)),
        (("Wg", None, "Vii", None), ("Wg", "", "Vii", None)),
    ],
)
def test_surname_xi_is_not_a_regnal_number(predictor: Predictor, source: tuple, expected: tuple) -> None:
    assert _resolve(predictor, source) == expected


def test_joined_initials_split_before_surname_xi() -> None:
    assert split_joined_initials("DG", None, "Xi", None) == ("D.", "G.")
    assert split_joined_initials("DG", None, "Vi", None) is None


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (("John", "Jr.", "Smith", "MD"), ("John", "", "Smith", "Jr.")),
        (("John", "Paul Jr", "Kim", "MD"), ("John", "Paul", "Kim", "Jr")),
        (("John", "Paul Jr", "Kim", None), ("John", "Paul", "Kim", "Jr")),
        (("John", "Jr.", "Smith", "Sr."), ("John", "Jr.", "Smith", "Sr.")),
        (("John", "Jr.", "Smith", "(UK)"), ("John", "Jr.", "Smith", "(UK)")),
        (("John", "A.", "Smith", "MD"), ("John", "A.", "Smith", None)),
    ],
)
def test_middle_suffix_moves_beside_a_degree_only_suffix_field(predictor: Predictor, source: tuple, expected: tuple) -> None:
    assert _resolve(predictor, source) == expected


def test_korean_given_form_leaves_a_constructed_source_passthrough(predictor: Predictor) -> None:
    source = SourceAuthorFields(first_name="John", middle_names="S***", last_name="Smith")
    passthrough = predictor.predict_batch([Instance(pp_authors=[source])])[0].authors[0]
    author = passthrough.copy(update={"first_name": "Yujin", "middle_names": "", "last_name": "Kim"})
    assert _is_source_passthrough(author)
    assert _korean_given(author) is author
    assert _resolve(predictor, ("Yujin", None, "Kim", None)) == ("Yu-Jin", "", "Kim", None)
    assert _resolve(predictor, ("Kim", None, "Yujin", None)) == ("Yu-Jin", "", "Kim", None)


@pytest.mark.parametrize(
    ("suffix", "expected"),
    [("1st.", "1st"), ("2nd.", "2nd"), ("3RD.", "3rd"), ("2nd", "2nd"), ("Jr.", "Jr."), ("III.", "III")],
)
def test_ordinal_suffix_drops_its_period(predictor: Predictor, suffix: str, expected: str) -> None:
    assert _resolve(predictor, ("John", None, "Smith", suffix)) == ("John", "", "Smith", expected)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (("John *", None, "Smith", None), ("John", "", "Smith", None)),
        (("John", "*", "Smith", None), ("John", "", "Smith", None)),
        (("John ∗", None, "Smith", None), ("John", "", "Smith", None)),  # noqa: RUF001
        (("John*", None, "Smith", None), ("John", "", "Smith", None)),
        (("John", "S***", "Smith", None), ("John", "S***", "Smith", None)),
        (("John **", None, "Smith", None), ("John **", "", "Smith", None)),
    ],
)
def test_standalone_footnote_star_is_dropped(predictor: Predictor, source: tuple, expected: tuple) -> None:
    assert _resolve(predictor, source) == expected
