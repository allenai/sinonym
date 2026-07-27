"""A split must never invent a single-letter component that the input did not contain.

Cases below are real corpus names. The bilingual rows are ground truth: the Han text names the
surname, so they settle which reading is correct where the romanisation alone is ambiguous.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tests._case_assertions import assert_person_normalized_name

HAN = re.compile(r"[㐀-䶿一-鿿豈-﫿]")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Han text names the surname; the hyphenated single letter belongs to the given name.
        ("王阿川 WANG A-chuan", "A-Chuan Wang"),
        ("曹阿秀 CAO A-xiu", "A-Xiu Cao"),
        ("刘阿鹏 LIU A-peng", "A-Peng Liu"),
        # Same shape, romanisation only.
        ("Fu A-Long", "A-Long Fu"),
        ("Sun A-ping", "A-Ping Sun"),
        ("Li Guo-e", "Guo-E Li"),
        ("Guo-e Li", "Guo-E Li"),
        ("Xiu-e Sun", "Xiu-E Sun"),
        ("Ping-E Huang", "Ping-E Huang"),
        ("Gui-e Xu", "Gui-E Xu"),
        ("Wen Yue-e", "Yue-E Wen"),
        ("A-Li Wang", "A-Li Wang"),
        ("Wang E-sheng", "E-Sheng Wang"),
        ("Zhao Yun-e", "Yun-E Zhao"),
        # camelCase is an author-supplied boundary too, so the lone letter is a syllable
        ("Zhang WenE", "Wen-E Zhang"),
        ("ChangE Liu", "Chang-E Liu"),
        ("QingE Wu", "Qing-E Wu"),
        ("XiangE Sun", "Xiang-E Sun"),
        ("Han XiuE", "Xiu-E Han"),
        # an apostrophe is the same author-supplied boundary as a hyphen
        ("Zheng Cui'e", "Cui-E Zheng"),
        ("Wu Yue'e", "Yue-E Wu"),
        ("Xiu'e Zheng", "Xiu-E Zheng"),
    ],
)
def test_hyphenated_single_letter_stays_in_the_given_name(detector, raw, expected):
    result = detector.normalize_name(raw)

    assert result.success, f"expected a Chinese parse, got {result.error_message}"
    assert result.result == expected
    assert result.parsed.middle_tokens == []


@pytest.mark.parametrize(
    ("raw", "given", "middle", "surname"),
    [
        ("Arun Rao", "Arun", "", "Rao"),  # was "Run" + middle "A"
        ("Fu Ali", "Fu", "", "Ali"),  # was "Fu" + middle "A" + surname "Li"
        ("Liana Lau", "Liana", "", "Lau"),
        ("Ewing Pa", "Ewing", "", "Pa"),  # was "Wing" + middle "E"
        ("Ewen Y. Tseng", "Ewen", "Y.", "Tseng"),  # was "Wen" + middle "E Y"
        ("Moua Yang", "Moua", "", "Yang"),  # was "Mou" + middle "A"
        ("Tsze Tsang", "Tsze", "", "Tsang"),
        # Malay/Indonesian names whose first syllable was being split off, leaving a Chinese
        # surname behind: `Awang` -> surname `wang`, `Atay` -> `tay`, `Ayu` -> `yu`.
        ("IP Awang", "IP", "", "Awang"),
        ("IM Atay", "IM", "", "Atay"),
        ("Ni Putu Ayu", "Ni", "Putu", "Ayu"),
    ],
)
def test_leading_letter_stays_in_the_token(detector, raw, given, middle, surname):
    result = detector.normalize_person_name(raw)

    assert result is not None
    assert result.normalized.given_name == given
    assert result.normalized.middle_name == middle
    assert result.normalized.surname == surname


@pytest.mark.parametrize(
    ("raw", "expected", "middle"),
    [
        ("Wei Q. Zhang", "Wei Q Zhang", ["Q"]),  # standalone initial token: still a middle initial
        ("Wei Q Zhang", "Wei Q Zhang", ["Q"]),
        ("Li Wei A.", "Wei A Li", ["A"]),
        ("Li Xiaoming", "Xiao-Ming Li", []),
        ("Li Guo-er", "Guo-Er Li", []),
        ("Ka-Fai Chan", "Ka-Fai Chan", []),
    ],
)
def test_standalone_initials_and_multi_letter_splits_are_unchanged(detector, raw, expected, middle):
    result = detector.normalize_name(raw)

    assert result.success
    assert result.result == expected
    assert result.parsed.middle_tokens == middle


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Liang Alei", "Alei Liang"),  # surname is Liang; 阿蕾 is a given name
    ],
)
def test_surname_first_concatenated_forms_keep_their_surname(detector, raw, expected):
    """Expected failure, in the check_test_status baseline.

    Refusing the LEADING split costs Chinese detection, so the person path reads these
    given-first and the surname lands in the wrong field. 阿 is a prefix, so `Alei` has no
    boundary to split at and no positional evidence to admit one. The trailing shape
    (`Zhao Liane`) keeps its surname — see
    test_bare_trailing_letter_is_reported_as_a_middle_initial.
    """
    person = detector.normalize_person_name(raw)

    assert_person_normalized_name(person, raw, expected)


@pytest.mark.parametrize(
    ("raw", "given", "middle", "surname"),
    [
        # Bilingual rows are ground truth: the Han text names the surname. Refusing a trailing
        # split strands these on the all-person path, which has no Han handling and reports the
        # Han string itself as a component (`Xiaoe 小娥 LI 李` -> given `Xiaoe`, middle `小娥 LI`,
        # surname `李`). Taking the split keeps them on the Chinese path, where the Han resolves.
        ("Xiaoe 小娥 LI 李", "Xiao", ["E"], "Li"),
        ("Yuee 月娥 Xie 谢", "Yue", ["E"], "Xie"),
        ("朱慧娥 Zhu Huie", "Hui", ["E"], "Zhu"),
        ("李梅娥 Li Meie", "Mei", ["E"], "Li"),
        ("王美娥 WANG Meie", "Mei", ["E"], "Wang"),
        ("罗月娥 LUO Yuee", "Yue", ["E"], "Luo"),
        ("赵培娥 Zhao Peie", "Pei", ["E"], "Zhao"),
        ("鲁秀娥 LU Xiue", "Xiu", ["E"], "Lu"),
        # An author-supplied boundary reaches the same place without the invented initial.
        ("Cui-E 翠娥 Hu 胡", "Cui-E", [], "Hu"),
        ("Lian-E 连娥 Lu 芦", "Lian-E", [], "Lu"),
        ("Zhi-E 志娥 Liu 刘", "Zhi-E", [], "Liu"),
        ("Qing-E 庆娥 ZHANG 张", "Qing-E", [], "Zhang"),
        ("Chang-e Jin 金常娥", "Chang-E", [], "Jin"),
        ("Wei-e Zhao 赵伟娥", "Wei-E", [], "Zhao"),
        ("任洪娥 REN Hong-e", "Hong-E", [], "Ren"),
        ("A-Hui 阿慧 Zhao 赵", "A-Hui", [], "Zhao"),
    ],
)
def test_mixed_script_rows_resolve_to_the_han_surname(detector, raw, given, middle, surname):
    result = detector.normalize_name(raw)

    assert result.success, f"expected a Chinese parse, got {result.error_message}"
    assert result.parsed.given_name == given
    assert result.parsed.middle_tokens == middle
    assert result.parsed.surname == surname
    assert not HAN.search(result.result), f"Han text leaked into a component: {result.result}"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Expected failures, in the check_test_status baseline. Leading 阿/鄂 has no boundary to
        # split at, so these stay on the all-person path, which has no Han handling: the Han
        # string itself lands in a component. The Han text names the correct surname.
        ("Alei 阿蕾 Li 李", "Alei Li"),
        ("Axin 阿鑫 Guo 郭", "Axin Guo"),
        ("侯阿慧 Hou Ahui", "Ahui Hou"),
        ("张阿龙 ZHANG Along", "Along Zhang"),
        ("樊阿馨 Fan Axin", "Axin Fan"),
        ("杨阿坤 Yang Akun", "Akun Yang"),
        ("Eyou Wang 王鄂友", "Eyou Wang"),
    ],
)
def test_leading_letter_bilingual_rows_resolve_to_the_han_surname(detector, raw, expected):
    person = detector.normalize_person_name(raw)

    assert_person_normalized_name(person, raw, expected)


def _cohorts():
    path = Path(__file__).resolve().parent / "data" / "manufactured_initials_cohorts.json"
    return json.loads(path.read_text(encoding="utf-8"))


COHORTS = _cohorts()


@pytest.mark.parametrize("case", COHORTS["given_first_preserved"], ids=lambda c: c["raw"])
def test_given_first_names_keep_correct_components_without_the_split(detector, case):
    """Corpus names that lose Chinese detection and keep source token order.

    `case["was"]` is the discarded parse, which reached its surname only via the invented initial
    (`Ajun Wan` -> given `Jun`, middle `A`). Order preservation is not correctness: it is right
    where only the trailing token is a known surname, and wrong for surname-first input, which
    needs a judge pass rather than a lexicon test to adjudicate.
    """
    result = detector.normalize_person_name(case["raw"])

    assert result is not None
    assert result.normalized.given_name == case["given"]
    assert result.normalized.middle_name == case["middle"]
    assert result.normalized.surname == case["surname"]


@pytest.mark.parametrize("case", COHORTS["components_not_recovered"], ids=lambda c: c["raw"])
def test_components_after_a_refused_split_are_recorded(detector, case):
    """Names whose components are not recovered in source order after the split is refused.

    Three kinds, all recorded rather than fixed:
    - bilingual Han+Latin rows (`张阿龙 ZHANG Along`, `樊阿馨 Fan Axin`): wrong either way — splitting
      read the given syllable as the surname, the person path keeps the Han text as a component.
    - trailing academic credentials (`CHAE WOO MA`, `Alam Hannan MA`, `Tzee Luai MENG`): the person
      path strips `MA`/`MEng`, so the real surname is lost — a person-path weakness these rows
      reach only because the Chinese parse is refused.
    - genuine order misreads (`Chae Bin Lee` -> surname `Bin Lee`).
    Several entries are improvements (`Asai Ren`, `Kai Etsuo` are correct Japanese readings here).
    """
    result = detector.normalize_person_name(case["raw"])

    assert result is not None
    assert result.normalized.given_name == case["given"]
    assert result.normalized.middle_name == case["middle"]
    assert result.normalized.surname == case["surname"]


@pytest.mark.parametrize(
    ("raw", "given", "surname"),
    [
        # Rejecting the leading letter must refuse the token outright, not scan on: positions
        # after the indicated boundary are weaker readings, and falling through to them produces
        # `Er`+`an` / `Ah`+`Ao`.
        ("Eran Kot", "Eran", "Kot"),
        ("Eren Chu", "Eren", "Chu"),
        ("Ahao Wu", "Ahao", "Wu"),
        ("Ahsi Lo", "Ahsi", "Lo"),
        ("Anen He", "Anen", "He"),
    ],
)
def test_leading_letter_does_not_fall_through_to_a_weaker_boundary(detector, raw, given, surname):
    result = detector.normalize_name(raw)
    assert not result.success

    person = detector.normalize_person_name(raw)
    assert person is not None
    assert person.normalized.given_name == given
    assert person.normalized.surname == surname
    assert person.normalized.middle_name == ""


@pytest.mark.parametrize(
    ("raw", "given", "middle", "surname"),
    [
        # No author-supplied boundary, so the reading is undecidable — `Yuee` is 月娥 and `Duane`
        # is not, and nothing in the token says which. The split is taken and the letter reported
        # as a middle initial: wrong on the 娥 names, but the SURNAME is right in every row and
        # the given name keeps its first initial, so the cluster block key is unaffected.
        # Refusing the split moves the surname into the wrong field for surname-first input
        # (`Chen Meie` -> surname `Meie`); admitting the letter as a syllable invents
        # `Duan-E`/`Gai-A` on names that are not Chinese at all.
        ("Duane Lee", "Duan", "E", "Lee"),
        ("Gaia Wang", "Gai", "A", "Wang"),
        ("Maia Wu", "Mai", "A", "Wu"),
        ("Rana Li", "Ran", "A", "Li"),
        ("Yuee Dai", "Yue", "E", "Dai"),
        ("Cuie Wen", "Cui", "E", "Wen"),
        ("Chen Meie", "Mei", "E", "Chen"),
        ("Liang Weia", "Wei", "A", "Liang"),
    ],
)
def test_bare_trailing_letter_is_reported_as_a_middle_initial(detector, raw, given, middle, surname):
    result = detector.normalize_name(raw)

    assert result.success, f"expected a Chinese parse, got {result.error_message}"
    assert result.parsed.given_name == given
    assert result.parsed.middle_tokens == [middle]
    assert result.parsed.surname == surname


@pytest.mark.parametrize(
    "raw",
    [
        # A lone letter against a rest longer than MAX_UNBALANCED_SPLIT_REST_LENGTH is not a
        # split at all: `Cheung`/`Leung`/`Chuan` are whole syllables.
        "Cheunga Wang",
        "Leunga Li",
        "Chuane Li",
        "Qionge Wu",
        "Hsiaoa Chen",
    ],
)
def test_a_lone_letter_against_a_long_rest_is_not_a_split(detector, raw):
    assert not detector.normalize_name(raw).success


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # An all-caps prefix makes the camel regex return a one-character first part
        # (`LIshan` -> ['L', 'Ishan']). That is an artifact, not an author boundary, so the
        # positional scan still finds the real split.
        ("LIshan Wang", "Li-Shan Wang"),
        ("ZIjiang Chen", "Zi-Jiang Chen"),
        ("MIndi Li", "Min-Di Li"),
        ("CHuanxue Wu", "Chuan-Xue Wu"),
        ("JIanghan Liu", "Jiang-Han Liu"),
    ],
)
def test_all_caps_prefix_is_not_treated_as_a_camel_boundary(detector, raw, expected):
    result = detector.normalize_name(raw)

    assert result.success, f"expected a Chinese parse, got {result.error_message}"
    assert result.result == expected


@pytest.mark.parametrize(
    ("raw", "given", "surname"),
    [
        ("Ana-Maria O'Neill", "Ana-Maria", "O'Neill"),
        ("Mohd Ma'ruf", "Mohd", "Ma'ruf"),
    ],
)
def test_apostrophe_handling_is_scoped_to_a_trailing_lone_letter(detector, raw, given, surname):
    person = detector.normalize_person_name(raw)

    assert person is not None
    assert person.normalized.given_name == given
    assert person.normalized.surname == surname


def test_split_decision_does_not_depend_on_a_previously_seen_lowercase_token(detector):
    """The unsplittable cache is keyed on the lowercased token, but this decision depends on case.

    Processing `weia` first must not poison `WeiA`, whose camelCase boundary makes the trailing
    letter a syllable rather than the middle initial the bare form reports.
    """
    assert detector.normalize_name("weia Zhang").result == "Wei A Zhang"

    result = detector.normalize_name("WeiA Zhang")

    assert result.success
    assert result.result == "Wei-A Zhang"
