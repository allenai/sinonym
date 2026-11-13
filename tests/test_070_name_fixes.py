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
        (("E.P.", "https://orcid.org/0000-0002-1825-0097", "Buga"), ("E.", "P.", "Buga")),
        (("K^|^ocirc;z^|^ocirc;", None, "SATO"), ("Kôzô", "", "Sato")),
        # controls: real letters that look like mojibake, masked letters, a word that only starts like a link
        (("JOÃO", None, "SILVA"), ("João", "", "Silva")),
        (("Åsa", None, "Ångström"), ("Åsa", "", "Ångström")),
        (("Irène", None, "S***"), ("Irène", "", "S***")),
        (("Httpa", None, "Smith"), ("Httpa", "", "Smith")),
    ],
)
def test_junk_in_name_fields_is_cleaned(predictor, source, expected):
    assert resolve(predictor, *source) == expected


@pytest.mark.parametrize(
    "source",
    [("https://orcid.org/0000-0002-1825-0097", None, "Smith"), ("orcid", "0000-0002-1825-0097", "Smith")],
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


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (("Q.", "M. Jonathan", "Wu"), ("Q.", "M. Jonathan", "Wu")),
        (("Y", "C Gary", "Lee"), ("Y.", "C. Gary", "Lee")),
        (("J.", "A. Gareth", "Williams"), ("J.", "A. Gareth", "Williams")),
        (("M", "G Myriam", "Hunink"), ("M.", "G. Myriam", "Hunink")),
        (("C.", "Janneke", "van der Woude"), ("C.", "Janneke", "van der Woude")),
        # controls: a particle phrase, a word that is also a surname, a source surname field of two words
        (("Luis", "Miguel de Diego", "Pareja"), ("Luis", "Miguel", "de Diego Pareja")),
        (("Maria", "Teresa", "Garcia Lopez"), ("Maria", "Teresa", "Garcia Lopez")),
        # given-name words that also open double surnames (Blanca Mena, Andrés Santos)
        (("M.", "J. Blanca", "Mena"), ("M.", "J.", "Blanca Mena")),
        (("Francisco", "J. Andrés", "Santos"), ("Francisco", "J.", "Andrés Santos")),
        # before a common Chinese surname the bars are looser (Philip, James, Paul; David despite the guard)
        (("C.", "L. Philip", "Chen"), ("C.", "L. Philip", "Chen")),
        (("C.", "P. James", "Chen"), ("C.", "P. James", "Chen")),
        (("R. V. Paul", None, "Chan"), ("R.", "V. Paul", "Chan")),
        (("X.", "J. David", "Lu"), ("X.", "J. David", "Lu")),
        (("C.", "L. PHILIP", "CHEN"), ("C.", "L. Philip", "Chen")),
        # accepted trade-off: one ORCID person registers the family name John Low
        (("C.", "T. John", "Low"), ("C.", "T. John", "Low")),
        # controls: a non-Chinese surname, a word that is also a surname (Simon), a South Indian name
        (("A.", "B. Philip", "Gara"), ("A.", "B.", "Philip Gara")),
        (("M.Y.", "Simon", "Lee"), ("M.", "Y.", "Simon Lee")),
        (("B.", "V. Srinivasa", "Rao"), ("B.", "V.", "Srinivasa Rao")),
    ],
)
def test_given_name_word_stays_out_of_the_surname(predictor, source, expected):
    assert resolve(predictor, *source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (("Walter", None, "Faria-Neto"), ("Walter", "", "Faria", "Neto")),
        (("Edward", None, "Araujo-Júnior"), ("Edward", "", "Araujo", "Júnior")),
        (("João", None, "Silva-Filho"), ("João", "", "Silva", "Filho")),
        # controls: a particle before the hyphen, the word as the whole surname
        (("A.", None, "Di-Neto"), ("A.", "", "Di-Neto", None)),
        (("A.", None, "Del-Neto"), ("A.", "", "Del-Neto", None)),
        (("Ana", None, "Neto"), ("Ana", "", "Neto", None)),
        ((None, None, "Campos-Neto"), ("", "", "Campos-Neto", None)),
    ],
)
def test_hyphenated_generation_word_goes_to_the_suffix(predictor, source, expected):
    source_fields = SourceAuthorFields(first_name=source[0], middle_names=source[1], last_name=source[2])
    author = predictor.predict_batch([Instance(pp_authors=[source_fields])])[0].authors[0]
    assert (author.first_name, author.middle_names, author.last_name, author.suffix) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (("Kumar", None, "P"), ("P.", "", "Kumar")),
        (("Popov", None, "V."), ("V.", "", "Popov")),
        (("Kumar", None, "RS"), ("RS", "", "Kumar")),
        (("Kumar", None, "R S"), ("R.", "S.", "Kumar")),
        (("Kumar", None, "O."), ("O.", "", "Kumar")),
        # controls: a given-name word before initials, a one-letter Korean surname
        (("Natrayan", None, "L"), ("Natrayan", "", "L")),
        (("Kim", None, "O"), ("Kim", "", "O")),
    ],
)
def test_surname_before_initials_in_the_surname_field(predictor, source, expected):
    assert resolve(predictor, *source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (("M.Pd", "Simson", "Tarigan"), "reviewed_source_pattern_assignment"),
        ((None, None, "Anthony C. Laborte, Marissa C. Hitalia*"), "reviewed_non_person_pattern"),
    ],
)
def test_reviewed_source_shapes_keep_their_rule(predictor, source, expected):
    fields = SourceAuthorFields(first_name=source[0], middle_names=source[1], last_name=source[2])
    author = predictor.predict_batch([Instance(pp_authors=[fields])])[0].authors[0]
    assert author.resolution_reason.value == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (("Maria", None, "Lopez (ORCID: 0000-0002-1825-0097)"), ("Maria", "", "Lopez")),
        (("Maria", None, "Lopez [0000-0002-1825-0097]"), ("Maria", "", "Lopez")),
        (("John", "A.", "Smith ORCID iD https://orcid.org/0000-0002-1825-0097"), ("John", "A.", "Smith")),
        (("Pedro", "César", "https://orcid.org/0000-0002-1825-0097"), ("Pedro", "", "César")),
        (("Nemanja", "M. idorcid 0000-0002-1825-0097", "Barać"), ("Nemanja", "M.", "Barać")),
        (("M.G.", "d https://orcid.org/", "Lapteva"), ("M.", "G. D.", "Lapteva")),
        (("Joseph", None, "httporcid.org0000-0002-1825-0097 Glauber"), ("Joseph", "", "Glauber")),
        (("Priscila", None, "Alves Balista0000-0002-1825-0097"), ("Priscila", "", "Alves Balista")),
        (("Aidos", None, "Aitmaganbet (ORCID: 0000-0002-1825-009)"), ("Aidos", "", "Aitmaganbet")),
        # nothing real left for the surname: left as written
        (("Ana", None, "0000-0002-1825-0097"), ("Ana", "", "0000-0002-1825-0097")),
        (("J.", None, "https://orcid.org/0000-0002-1825-0097"), ("J.", "", "https://orcid.org/0000-0002-1825-0097")),
    ],
)
def test_orcid_ids_are_deleted_from_name_fields(predictor, source, expected):
    assert resolve(predictor, *source) == expected


@pytest.mark.parametrize(
    "source",
    [
        ("Fidel", "Torcida", "Fernández-Baldor"),
        ("ID", None, "ORCID"),
        (None, None, "ORCID"),
        ("ORCID", None, "iD"),
        ("Orcid", None, "https"),
        ("ORCID®", "iDs", "Parinaz"),
    ],
)
def test_orcid_inside_a_word_or_a_logo_author_is_not_a_name_to_clean(source):
    from sinonym.services.source_repairs import strip_junk

    assert strip_junk(*source) is None


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (("Vitor", "Engr cia", "Valenti"), ("Vitor", "Engr Cia", "Valenti")),
        (("Hans", "Dr med", "Weber"), ("Hans", "", "Weber")),
        (("Publisher", "Dr sachin", "patil"), ("Publisher", "Sachin", "Patil")),
        (("Assoc", "prof Tony", "Walker"), ("Tony", "", "Walker")),
    ],
)
def test_undotted_title_before_a_lowercase_fragment_is_a_broken_word(predictor, source, expected):
    assert resolve(predictor, *source) == expected


def test_palochka_and_komi_de_are_not_read_as_latin(predictor):
    assert resolve(predictor, "Taşpӏnar", None, "Ali")[0] == "Taşpӏnar"


@pytest.mark.parametrize(
    ("value", "expected"),
    [("AlaÑµa", "AlaÑµa"), ("Âº", "Âº"), ("KÐ°rmakulova", "Karmakulova"), ("MÂª", "Mª")],
)
def test_mojibake_decoding_stays_latin(value, expected):
    from sinonym.services.source_repairs import _unmojibake

    assert _unmojibake(value) == expected


def resolve_with_suffix(predictor: Predictor, first, middle, last, suffix=None):
    source = SourceAuthorFields(first_name=first, middle_names=middle, last_name=last, suffix=suffix)
    author = predictor.predict_batch([Instance(pp_authors=[source])])[0].authors[0]
    return author.first_name, author.middle_names, author.last_name, author.suffix


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (("Francisco", None, "Aquino Júnior"), ("Francisco", "", "Aquino", "Júnior")),
        (("Francisco", None, "Aquino JÚNIOR"), ("Francisco", "", "Aquino", "Júnior")),
        (("Paulo", None, "Souza Junior"), ("Paulo", "", "Souza", "Junior")),
        (("John", None, "Smith Jr"), ("John", "", "Smith", "Jr")),
        (("John A.", None, "Smith, Jr"), ("John", "A.", "Smith", "Jr")),
        (("John", None, "Smith", "JR."), ("John", "", "Smith", "Jr.")),
        (("John", None, "Smith", "Junior"), ("John", "", "Smith", "Junior")),
        (("John", None, "Smith Senior"), ("John", "", "Smith", "Senior")),
        (("John", None, "Smith iii"), ("John", "", "Smith", "III")),
        (("John", None, "Smith", "ii"), ("John", "", "Smith", "II")),
        (("John", None, "Smith-Jr."), ("John", "", "Smith", "Jr.")),
        # controls: already the written form, a surname that only looks like a suffix
        (("John", None, "Smith Jr."), ("John", "", "Smith", "Jr.")),
        (("José", "Ribamar", "Santos Neto"), ("José", "Ribamar", "Santos", "Neto")),
        (("Roxy", None, "Senior"), ("Roxy", "", "Senior", None)),
    ],
)
def test_generational_suffix_keeps_its_source_spelling(predictor, source, expected):
    assert resolve_with_suffix(predictor, *source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (("John", None, "Smith", "PhD"), ("John", "", "Smith", None)),
        (("John", None, "Smith", "MD PhD"), ("John", "", "Smith", None)),
        (("John", None, "Smith", "Dr"), ("John", "", "Smith", None)),
        (("John", None, "Smith", "Jr., PhD"), ("John", "", "Smith", "Jr.")),
        (("John", None, "Smith Jr", "PhD"), ("John", "", "Smith", "Jr")),
        # controls: a suffix field that is not only credentials stays as written
        (("John", None, "Smith", "(UK)"), ("John", "", "Smith", "(UK)")),
        (("John", None, "Smith", "Jr Foo"), ("John", "", "Smith", "Jr Foo")),
        (("John", None, "Smith", "Ma"), ("John", "", "Smith", "Ma")),
        (("John", None, "Smith", "   "), ("John", "", "Smith", "   ")),
        (("John", None, "Smith", "Column Editor"), ("John", "", "Smith", "Column Editor")),
    ],
)
def test_credentials_and_titles_leave_the_suffix_field(predictor, source, expected):
    assert resolve_with_suffix(predictor, *source) == expected


def test_source_passthrough_keeps_its_suffix(predictor):
    source = SourceAuthorFields(first_name="Consortium", middle_names="The ATLAS", last_name="", suffix="PhD")
    author = predictor.predict_batch([Instance(pp_authors=[source])])[0].authors[0]
    assert author.resolution_provenance.value == "source"
    assert author.resolution_action.value != "assign"
    assert author.suffix == "PhD"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (("CHAE", "WOO", "MA"), ("Chae", "Woo", "Ma")),
        (("MICHAEL KIN GAY", None, "MA"), ("Michael", "Kin Gay", "Ma")),
        (("NYUK", "LING", "MA"), ("Nyuk", "Ling", "Ma")),
        # controls: a mixed-case source keeps reading it as the degree; other degrees; one word before it
        (("Elizabeth", "Pearce", "MA"), ("Elizabeth", "", "Pearce")),
        (("JOHN", "SMITH", "MD"), ("John", "", "Smith")),
        (("Baxter", None, "MA"), ("MA", "", "Baxter")),
        (("BAXTER", "-", "MA"), ("", "", "Baxter-Ma")),
        (("李", "明", "MA"), ("Li-Ming", "", "Ma")),
        (("THU", "THI", "DO"), ("THU", "THI", "Do")),
    ],
)
def test_capitals_surname_that_is_also_a_degree(predictor, source, expected):
    assert resolve(predictor, *source) == expected


def test_capitals_ma_with_a_degree_in_the_suffix_field_stays_the_degree(predictor):
    assert resolve(predictor, "JOHN", "SMITH", "MA", "PhD") == ("John", "", "Smith")
