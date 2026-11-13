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


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (("V.L.", None, "Babуch"), ("V.", "L.", "Babych")),
        (("E.V.", None, "Uvarovа"), ("E.", "V.", "Uvarova")),
        (("Аleksii", None, "Tsapko"), ("Aleksii", "", "Tsapko")),
        (("I.", "O.", "Tсelykh"), ("I.", "O.", "Tсelykh")),
        (("Р.", None, "Росa"), ("Р.", "", "Росa")),
        (("Bерх", None, "Ivanov"), ("Bерх", "", "Ivanov")),
        # controls: Cyrillic names, a Cyrillic ending in capitals, the middle field
        (("Иван", None, "Петров"), ("Иван", "", "Петров")),
        (("I.", "M.", "TKACHЕНКО"), ("I.", "M.", "Tkachенко")),
        (("Vera", "Аlexandrovna", "Belova"), ("Vera", "Аlexandrovna", "Belova")),
    ],
)
def test_cyrillic_lookalikes_in_a_latin_word(predictor, source, expected):
    assert resolve(predictor, *source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (("*Emmanuel", "Ifeanyi", "Obeagu"), ("Emmanuel", "Ifeanyi", "Obeagu")),
        (("Jungkil", None, "Kim∗"), ("Jungkil", "", "Kim")),
        (("ZoltÃ¡n", None, "GÃ¡lik"), ("Zoltán", "", "Gálik")),
        (("Esen", "GÃ¶kÃ§e", "Ã–zdamar"), ("Esen", "Gökçe", "Özdamar")),
        (("E.P.", "https://orcid.org/0000-0000-0000-0000", "Buga"), ("E.", "P.", "Buga")),
        (("K^|^ocirc;z^|^ocirc;", None, "SATO"), ("Kôzô", "", "Sato")),
        # controls: real letters that look like mojibake, a surname that is only a link
        (("JOÃO", None, "SILVA"), ("João", "", "Silva")),
        (("Åsa", None, "Ångström"), ("Åsa", "", "Ångström")),
        (
            ("Pedro", "César", "https://orcid.org/0000-0000-0000-0000"),
            ("Pedro", "César", "https://orcid.org/0000-0000-0000-0000"),
        ),
        (("Irène", None, "S***"), ("Irène", "", "S***")),
        (("Httpa", None, "Smith"), ("Httpa", "", "Smith")),
    ],
)
def test_junk_in_name_fields_is_cleaned(predictor, source, expected):
    assert resolve(predictor, *source) == expected


@pytest.mark.parametrize(
    "source",
    [("https://orcid.org/0000-0000-0000-0000", None, "Smith"), ("orcid", "0000-0000-0000-0000", "Smith")],
)
def test_junk_cleanup_keeps_a_name_without_a_given_name(source):
    from sinonym.services.source_repairs import strip_junk

    assert strip_junk(*source) is None


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (("Luis", "C. 子山", "Ho 何"), ("Luis", "C.", "Ho")),
        (("Kevin", "M.", "Wong 黃浩文"), ("Kevin", "M.", "Wong")),
        (("Alex", None, "Green (文浩航)"), ("Alex", "", "Green")),
        # control: a romanized Chinese name keeps the Han form the parser reads for the surname
        (("Su", "苏", "Zhang 张"), ("Su", "", "Zhang")),
    ],
)
def test_han_form_beside_a_latin_name_is_dropped(predictor, source, expected):
    assert resolve(predictor, *source) == expected


@pytest.mark.parametrize(
    "source",
    [("简月", "Jian", "Yue"), ("Junjie", "俊杰", "Qiang 强"), ("Hongan", "红安", "Ma 马"), ("Chak", "On 鄒澤安", "Chow")],
)
def test_romanized_chinese_name_keeps_its_han_form(source):
    from sinonym.services.source_repairs import drop_han_beside_latin

    assert drop_han_beside_latin(*source) is None


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (("S.Kom.", "M.Eng.", "Waskitho Wibisono"), ("Waskitho", "", "Wibisono")),
        (("S.Si", "M.Eng", "Ardhi Wicaksono"), ("Ardhi", "", "Wicaksono")),
        (("S.", "T. M.Eng.", "Diananta Pramitasari"), ("Diananta", "", "Pramitasari")),
        (("St", "M.Eng.", "Maria Anityasari"), ("Maria", "", "Anityasari")),
        (("S.Fis.", "S.Pd. M.Ph", "Totok Budi Santoso"), ("Totok", "Budi", "Santoso")),
        (("S.", "T. M.Eng. Bimo Sunarfri", "Hantono"), ("Bimo", "Sunarfri", "Hantono")),
        # controls: initials that spell a degree, a lone degree before a one-word name, a byline word left over
        (("S.", "T.", "Wibowo"), ("S.", "T.", "Wibowo")),
        (("Simon", "S.M.", "Ng"), ("Simon", "S. M.", "Ng")),
        (("M.Pd", None, "Sutama"), ("M.Pd", "", "Sutama")),
        (("M.Sn", "Pembimbing", "Muksin"), ("M.Sn", "Pembimbing", "Muksin")),
    ],
)
def test_indonesian_degrees_in_the_given_fields(predictor, source, expected):
    assert resolve(predictor, *source) == expected
