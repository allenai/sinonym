"""Regressions for the 0.7.0 name fixes, with real-name controls. Names are made up or public figures."""

import pytest

from sinonym.timo.interface import Instance, Predictor, PredictorConfig, SourceAuthorFields


@pytest.fixture(scope="module")
def predictor() -> Predictor:
    return Predictor(PredictorConfig(parallel="never"), ".")


def resolve(predictor: Predictor, first, middle, last, suffix=None):
    source = SourceAuthorFields(first_name=first, middle_names=middle, last_name=last, suffix=suffix)
    author = predictor.predict_batch([Instance(pp_authors=[source])])[0].authors[0]
    return author.first_name, author.middle_names, author.last_name


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (("მაია", None, "კაპანაძე"), ("მაია", "", "კაპანაძე")),
        (("ნინო", "ა.", "ბერიძე"), ("ნინო", "ა.", "ბერიძე")),
        (("ᲛᲐᲘᲐ", None, "ᲙᲐᲞᲐᲜᲐᲫᲔ"), ("მაია", "", "კაპანაძე")),
        (("Nino", None, "Beridze"), ("Nino", "", "Beridze")),
    ],
)
def test_georgian_is_not_title_cased_into_mtavruli(predictor, source, expected):
    assert resolve(predictor, *source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (("Ida", "Di", "Di Mola"), ("Ida", "", "Di Mola")),
        (("Miled", "El", "El Hajji"), ("Miled", "", "El Hajji")),
        (("Pamela", "R von", "von Hurst"), ("Pamela", "R.", "von Hurst")),
        (("GRAZIELLA", "MARTINS DE", "DE MELLO"), ("Graziella", "Martins", "de Mello")),
        (("Kim", "Van", "Van"), ("Kim", "", "Van Van")),
        (("Thi", "Le", "Le"), ("Thi", "Le", "Le")),
        (("Carlos", "A. Coello", "Coello"), ("Carlos", "A.", "Coello Coello")),
    ],
)
def test_particle_written_twice_is_kept_once(predictor, source, expected):
    assert resolve(predictor, *source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (("Carsten", "Dr.", "Tille"), ("Carsten", "", "Tille")),
        (("Michael", "Dr. Med.", "Laule"), ("Michael", "", "Laule")),
        (("Jennifer", "L. Dr", "Sparr"), ("Jennifer", "L.", "Sparr")),
        (("Doç.", "Dr. Taner", "ARTAN"), ("Taner", "", "Artan")),
        (("Assoc.", "Prof. Dr. Faruk", "Gün"), ("Faruk", "", "Gün")),
        (("Translated", "by N.B.", "Zelinska"), ("N.", "B.", "Zelinska")),
        (("Reviewed", "by Anna", "Blackwell"), ("Anna", "", "Blackwell")),
        # controls: capitals that can be initials, names that are also titles, title runs the parser already strips
        (("Hans", "DR", "Hohmann"), ("Hans", "DR", "Hohmann")),
        (("Ana", "Ms", "Silva"), ("Ana", "Ms", "Silva")),
        (("Sheron", "Sir Loon", "Goh"), ("Sheron", "Sir Loon", "Goh")),
        (("PD", "Dr. Konrad", "Schwarzkopf"), ("Konrad", "", "Schwarzkopf")),
        (("Dr.", "Cunyi", "Yang"), ("Cunyi", "", "Yang")),
        (("Juergen", "Dipl Chem Dr", "Pabst"), ("Juergen", "Dipl Chem Dr", "Pabst")),
    ],
)
def test_titles_in_the_given_fields_are_dropped(predictor, source, expected):
    assert resolve(predictor, *source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (("Dr.-Ing.", "Dr. h. c. Herwart", "Opitz"), ("Herwart", "", "Opitz")),
        (("Hans", "C. Dr.", "Weber"), ("Hans", "C.", "Weber")),
        (("Hans", "Dr. H.", "Weber"), ("Hans", "H.", "Weber")),
    ],
)
def test_honoris_causa_goes_with_its_title(predictor, source, expected):
    assert resolve(predictor, *source) == expected
