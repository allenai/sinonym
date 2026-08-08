# ruff: noqa: RUF001

"""A split must never invent a single-letter component that the input did not contain.

Cases below are real corpus names. The bilingual rows are ground truth: the Han text names the
surname, so they settle which reading is correct where the romanisation alone is ambiguous.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from sinonym.timo.routing_v3 import RoutingInstanceV3, SourceAuthorFields
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
        # the curly forms are what real metadata carries, and they must fold to the ASCII
        # boundary rather than being deleted
        ("Zheng Cui’e", "Cui-E Zheng"),
        ("Wu Yue’e", "Yue-E Wu"),
        ("Xiu’e Zheng", "Xiu-E Zheng"),
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
        ("Wei Q. Zhang", "Wei Q. Zhang", ["Q."]),  # standalone initial token: still a middle initial
        ("Wei Q Zhang", "Wei Q. Zhang", ["Q."]),
        ("Li Wei A.", "Wei A. Li", ["A."]),
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
    test_bare_trailing_letter_is_not_manufactured_as_an_initial.
    """
    person = detector.normalize_person_name(raw)

    assert_person_normalized_name(person, raw, expected)


@pytest.mark.parametrize(
    ("raw", "given", "middle", "surname"),
    [
        # The aligned Han character proves that the final letter is a full syllable, not an
        # initial. Both Roman-first and Han-first layouts therefore bind it into the given name.
        ("Xiaoe 小娥 LI 李", "Xiao-E", [], "Li"),
        ("Yuee 月娥 Xie 谢", "Yue-E", [], "Xie"),
        ("朱慧娥 Zhu Huie", "Hui-E", [], "Zhu"),
        ("李梅娥 Li Meie", "Mei-E", [], "Li"),
        ("王美娥 WANG Meie", "Mei-E", [], "Wang"),
        ("罗月娥 LUO Yuee", "Yue-E", [], "Luo"),
        ("赵培娥 Zhao Peie", "Pei-E", [], "Zhao"),
        ("鲁秀娥 LU Xiue", "Xiu-E", [], "Lu"),
        # An author-supplied boundary reaches the same binding without native inference.
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
        # Native alignment proves the leading A/E is a complete given-name syllable.
        ("Alei 阿蕾 Li 李", "A-Lei Li"),
        ("Axin 阿鑫 Guo 郭", "A-Xin Guo"),
        ("侯阿慧 Hou Ahui", "A-Hui Hou"),
        ("张阿龙 ZHANG Along", "A-Long Zhang"),
        ("樊阿馨 Fan Axin", "A-Xin Fan"),
        ("杨阿坤 Yang Akun", "A-Kun Yang"),
        ("Eyou Wang 王鄂友", "E-You Wang"),
    ],
)
def test_leading_letter_bilingual_rows_resolve_to_the_han_surname(detector, raw, expected):
    person = detector.normalize_person_name(raw)

    assert_person_normalized_name(person, raw, expected)


def _cohorts():
    path = Path(__file__).resolve().parent / "data" / "manufactured_initials_cohorts.json"
    return json.loads(path.read_text(encoding="utf-8"))


COHORTS = _cohorts()


_BOUND_GIVEN_EXPECTATIONS = {
    "Bao'e Yang": "Bao-E",
    "Cai'E Cui": "Cai-E",
    "Cui'e Wang": "Cui-E",
    "Cui'e Wei": "Cui-E",
    "Cui'e Wu": "Cui-E",
    "Hua'e Xu": "Hua-E",
    "Long'e Dai": "Long-E",
    "Qiu'e Yao": "Qiu-E",
    "ShuE Ji": "Shu-E",
    "Tian'e Zhou": "Tian-E",
    "WeiE Wang": "Wei-E",
    "XiuE Yan": "Xiu-E",
    "You'e He": "You-E",
    "Yue'e Fang": "Yue-E",
    "Yue'e Liu": "Yue-E",
    "Yue‐e Ma": "Yue-E",
}

_RECOVERED_COMPONENT_EXPECTATIONS = {
    "Alei 阿蕾 Li 李": ("A-Lei", "", "Li"),
    "Ali 阿理 Luo 罗": ("A-Li", "", "Luo"),
    "Aming 啊鸣 Lin 林": ("A-Ming", "", "Lin"),
    "Axin 阿新 Xie 谢": ("A-Xin", "", "Xie"),
    "Axin 阿鑫 Guo 郭": ("A-Xin", "", "Guo"),
    "Ejun Peng 彭鄂军": ("E-Jun", "", "Peng"),
    "Eyou Wang 王鄂友": ("E-You", "", "Wang"),
    "Xiaoe 小娥 LI 李": ("Xiao-E", "", "Li"),
    "Yuee 月娥 Xie 谢": ("Yue-E", "", "Xie"),
    "侯阿慧 Hou Ahui": ("A-Hui", "", "Hou"),
    "孙阿辉 Sun Ahui": ("A-Hui", "", "Sun"),
    "屈阿雪 Qu Axue": ("A-Xue", "", "Qu"),
    "张阿娟 ZHANG Ajuan": ("A-Juan", "", "Zhang"),
    "张阿龙 ZHANG Along": ("A-Long", "", "Zhang"),
    "杜阿朋 Du A'peng": ("A-Peng", "", "Du"),
    "樊阿馨 Fan Axin": ("A-Xin", "", "Fan"),
    "杨阿坤 Yang Akun": ("A-Kun", "", "Yang"),
    "陈阿君 Chen Ajun": ("A-Jun", "", "Chen"),
}


def _expected_initial_punctuation(value: str) -> str:
    """Add canonical punctuation only to an expected standalone initial."""
    return f"{value.upper()}." if len(value) == 1 and value.isalpha() else value


@pytest.mark.parametrize("case", COHORTS["given_first_preserved"], ids=lambda c: c["raw"])
def test_given_first_names_keep_correct_components_without_the_split(detector, case):
    """Corpus names must not acquire manufactured initial components.

    Explicit apostrophe/camel boundaries canonicalize as Chinese given-name hyphens. Unbound
    names retain source token order, and genuine standalone initials receive periods.
    """
    result = detector.normalize_person_name(case["raw"])

    assert result is not None
    expected = {
        "Liang Alei": ("Alei", "", "Liang"),
    }.get(
        case["raw"],
        (
            _BOUND_GIVEN_EXPECTATIONS.get(
                case["raw"],
                _expected_initial_punctuation(case["given"]),
            ),
            _expected_initial_punctuation(case["middle"]),
            case["surname"],
        ),
    )
    assert (
        result.normalized.given_name,
        result.normalized.middle_name,
        result.normalized.surname,
    ) == expected


@pytest.mark.parametrize("case", COHORTS["components_not_recovered"], ids=lambda c: c["raw"])
def test_components_after_a_refused_split_are_recorded(detector, case):
    """Retain reviewed components while applying the canonical initial/native policy.

    Aligned bilingual rows now recover the native-bound given name. Other historical rows retain
    their reviewed assignment, with standalone initials canonically dotted.
    """
    result = detector.normalize_person_name(case["raw"])

    assert result is not None
    expected = _RECOVERED_COMPONENT_EXPECTATIONS.get(
        case["raw"],
        (
            _expected_initial_punctuation(case["given"]),
            _expected_initial_punctuation(case["middle"]),
            case["surname"],
        ),
    )
    assert (
        result.normalized.given_name,
        result.normalized.middle_name,
        result.normalized.surname,
    ) == expected


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
    ("raw", "given", "surname"),
    [
        # With no standalone token, author boundary, or native alignment, a final letter cannot
        # be called either an initial or a Chinese syllable. Preserve the whole given token.
        ("Duane Lee", "Duane", "Lee"),
        ("Gaia Wang", "Gaia", "Wang"),
        ("Maia Wu", "Maia", "Wu"),
        ("Rana Li", "Rana", "Li"),
        ("Yuee Dai", "Yuee", "Dai"),
        ("Cuie Wen", "Cuie", "Wen"),
        ("Chen Meie", "Meie", "Chen"),
        ("Liang Weia", "Weia", "Liang"),
    ],
)
def test_bare_trailing_letter_is_not_manufactured_as_an_initial(detector, raw, given, surname):
    person = detector.normalize_person_name(raw)

    assert person is not None
    assert person.normalized.given_name == given
    assert person.normalized.middle_name == ""
    assert person.normalized.surname == surname

    chinese = detector.normalize_name(raw)
    if chinese.success:
        assert chinese.parsed is not None
        assert chinese.parsed.given_name == given
        assert chinese.parsed.middle_tokens == []
        assert chinese.parsed.surname == surname


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
    ("raw", "expected", "given_tokens"),
    [
        ("Xiang'an Yan", "Xiang-An Yan", ["Xiang", "An"]),
        ("Xiang\u2018an Yan", "Xiang-An Yan", ["Xiang", "An"]),
        ("Xiang\u2019an Yan", "Xiang-An Yan", ["Xiang", "An"]),
        ("Xiang\u02bcan Yan", "Xiang-An Yan", ["Xiang", "An"]),
        ("Xiang\uff07an Yan", "Xiang-An Yan", ["Xiang", "An"]),
        ("Xian'gan Yan", "Xian-Gan Yan", ["Xian", "Gan"]),
    ],
)
def test_apostrophe_preserves_explicit_multiletter_given_boundary(detector, raw, expected, given_tokens):
    result = detector.normalize_name(raw)

    assert result.success, f"expected a Chinese parse, got {result.error_message}"
    assert result.result == expected
    assert result.parsed is not None
    assert result.parsed.given_tokens == given_tokens
    assert result.parsed.middle_tokens == []
    assert result.parsed.surname == "Yan"

    person = detector.normalize_person_name(raw)
    assert person is not None
    assert person.text == expected


def test_routed_v3_preserves_explicit_multiletter_given_boundary(routing_predictor_v3):
    source = SourceAuthorFields(first_name="Xiang'an", last_name="Yan")

    (paper,) = routing_predictor_v3.predict_batch([RoutingInstanceV3(pp_authors=[source])])
    resolved = paper.authors[0].resolved_fields

    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == ("Xiang-An", "", "Yan")


def test_wade_giles_aspiration_apostrophe_is_not_forced_to_be_a_boundary(detector):
    result = detector.normalize_name("Ch'inghua Wang")

    assert result.success, f"expected a Chinese parse, got {result.error_message}"
    assert result.result == "Ching-Hua Wang"
    assert result.parsed is not None
    assert result.parsed.given_tokens == ["Ching", "Hua"]


@pytest.mark.parametrize(
    ("raw", "given", "surname"),
    [
        ("Ana-Maria O'Neill", "Ana-Maria", "O'Neill"),
        ("Mohd Ma'ruf", "Mohd", "Ma'ruf"),
    ],
)
def test_apostrophe_given_boundary_does_not_reinterpret_person_surnames(detector, raw, given, surname):
    person = detector.normalize_person_name(raw)

    assert person is not None
    assert person.normalized.given_name == given
    assert person.normalized.surname == surname


def test_split_decision_does_not_depend_on_a_previously_seen_lowercase_token(detector):
    """The unsplittable cache is keyed on the lowercased token, but this decision depends on case.

    The unbound lowercase form remains whole. Processing it first must not poison `WeiA`, whose
    camelCase boundary proves that the trailing letter is a syllable.
    """
    lower = detector.normalize_person_name("weia Zhang")

    assert lower is not None
    assert lower.normalized.given_name == "Weia"
    assert lower.normalized.middle_name == ""
    assert lower.normalized.surname == "Zhang"

    result = detector.normalize_name("WeiA Zhang")

    assert result.success
    assert result.result == "Wei-A Zhang"
