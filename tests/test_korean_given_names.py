"""A two-syllable Korean given name gets one written form, ``Jin-Ho``, whatever form the source used."""
import pytest

from sinonym.timo.interface import Instance, Predictor, PredictorConfig, SourceAuthorFields
from sinonym.utils.korean_given_names import korean_given_form

UNITED = [
    # every source form of one name gives the same output
    (("Jinho", None, "Park"), ("Jin-Ho", "", "Park")),
    (("Jin-ho", None, "Park"), ("Jin-Ho", "", "Park")),
    (("Jin", "Ho", "Park"), ("Jin-Ho", "", "Park")),
    (("Jin Ho", None, "Park"), ("Jin-Ho", "", "Park")),
    (("JinHo", None, "Park"), ("Jin-Ho", "", "Park")),
    (("Park", None, "Jinho"), ("Jin-Ho", "", "Park")),
    (("KIM", None, "JINHO"), ("Jin-Ho", "", "Kim")),
    (("Sujin", None, "Lee"), ("Su-Jin", "", "Lee")),
    (("Su", "Jin", "Lee"), ("Su-Jin", "", "Lee")),
    (("Sujin", None, "Kim"), ("Su-Jin", "", "Kim")),
    (("Seungho", None, "Shin"), ("Seung-Ho", "", "Shin")),
    (("Seung", "Ho", "Jeon"), ("Seung-Ho", "", "Jeon")),
    # Korean names that also parse as Chinese
    (("Young", "Hee", "Choi"), ("Young-Hee", "", "Choi")),
    (("Younghee", None, "Choi"), ("Young-Hee", "", "Choi")),
    (("Ji Young", None, "Lee"), ("Ji-Young", "", "Lee")),
    (("Ji", "Young", "Lee"), ("Ji-Young", "", "Lee")),
    (("Jiyoung", None, "Lee"), ("Ji-Young", "", "Lee")),
    # real ORCID people whose mentions used two forms
    (("Tae", "Wook", "Kim"), ("Tae-Wook", "", "Kim")),
    (("Sanghyeon", None, "Lee"), ("Sang-Hyeon", "", "Lee")),
    (("Minjeong", None, "Lee"), ("Min-Jeong", "", "Lee")),
    (("Eunjung", None, "Kim"), ("Eun-Jung", "", "Kim")),
    (("Jung", "Yeon", "Heo"), ("Jung-Yeon", "", "Heo")),
    (("Jongmin", None, "Jeon"), ("Jong-Min", "", "Jeon")),
    # the boundary comes from Korean syllables, not the pinyin splitter
    (("Jihye", None, "Kim"), ("Ji-Hye", "", "Kim")),
    (("Seungwoo", None, "Kim"), ("Seung-Woo", "", "Kim")),
    (("Eunji", None, "Lee"), ("Eun-Ji", "", "Lee")),
    (("Hyunwoo", None, "Park"), ("Hyun-Woo", "", "Park")),
]

UNCHANGED = [
    # Chinese names keep the Chinese path's output
    (("Xiaoming", None, "Li"), ("Xiao-Ming", "", "Li")),
    (("Weiran", None, "Lee"), ("Wei-Ran", "", "Lee")),
    (("Lina", None, "Tang"), ("Lina", "", "Tang")),
    # non-Korean given names
    (("Annie", None, "Chen"), ("Annie", "", "Chen")),
    (("Niran", None, "Chen"), ("Niran", "", "Chen")),
    (("Keren", None, "Lee"), ("Keren", "", "Lee")),
    (("Juan", None, "Kim"), ("Juan", "", "Kim")),
    # initials and longer given names stay as written
    (("J.", "H.", "Park"), ("J.", "H.", "Park")),
    (("Jin-Ho", "M.", "Park"), ("Jin-Ho", "M.", "Park")),
    # a word Korean registry people use too rarely, or a source boundary that disagrees with the listed one
    (("Minjun", None, "Kim"), ("Minjun", "", "Kim")),
    (("Jinh", "Oh", "Park"), ("Jinh", "Oh", "Park")),
]

AMBIGUOUS_CUT = [
    # a compact single-case word with a second valid cut that Korean registry people do not almost all cut one way:
    # a wrong cut (Sung-Il for Sun Gil) would make S2AND veto the person's other mentions, so it stays as written
    (("Sungil", None, "Kim"), ("Sungil", "", "Kim")),
    (("SUNGIL", None, "KIM"), ("Sungil", "", "Kim")),
    (("sungil", None, "kim"), ("Sungil", "", "Kim")),
    (("Hyungil", None, "Park"), ("Hyungil", "", "Park")),
    (("Daeun", None, "Lee"), ("Daeun", "", "Lee")),
    (("Kim", None, "Sungil"), ("Sungil", "", "Kim")),
    (("KIM", None, "SUNGIL"), ("Sungil", "", "Kim")),
    # the source marks the cut: the listed form, as for any other listed word
    (("Sung", "Il", "Kim"), ("Sung-Il", "", "Kim")),
    (("SUNG IL", None, "KIM"), ("Sung-Il", "", "Kim")),
    (("Sung-il", None, "Kim"), ("Sung-Il", "", "Kim")),
    (("SungIl", None, "Kim"), ("Sung-Il", "", "Kim")),
    (("DA EUN", None, "LEE"), ("Da-Eun", "", "Lee")),
    (("Kim", None, "Jung Il"), ("Jung-Il", "", "Kim")),
    (("SeongEun", None, "Choi"), ("Seong-Eun", "", "Choi")),
    (("HyungIl", None, "Kim"), ("Hyung-Il", "", "Kim")),
    (("SungIL", None, "Kim"), ("Sung-Il", "", "Kim")),
    (("SUNGIl", None, "Kim"), ("Sung-Il", "", "Kim")),
    # a source cut at the other place is kept
    (("Sun Gil", None, "Kim"), ("Sun", "Gil", "Kim")),
    (("SUNGil", None, "Kim"), ("SUNGil", "", "Kim")),
]

# Known limits, pinned so a change is noticed.
KNOWN_LIMITS = [
    (("Youngil", None, "Jeong"), ("Young-Il", "", "Jeong")),
    # a listed word some non-Korean people also write (most registry people writing it are in South Korea)
    (("Dain", None, "Lee"), ("Da-In", "", "Lee")),
    # a word mostly Korean in the registry, also written by a few people in a Chinese region
    (("Sujin", None, "Yang"), ("Su-Jin", "", "Yang")),
    (("Junho", None, "Lee"), ("Jun-Ho", "", "Lee")),
]


@pytest.fixture(scope="module")
def predictor() -> Predictor:
    return Predictor(PredictorConfig(parallel="never"), ".")


def resolve(predictor: Predictor, first, middle, last):
    source = SourceAuthorFields(first_name=first or None, middle_names=middle or None, last_name=last)
    author = predictor.predict_batch([Instance(pp_authors=[source])])[0].authors[0]
    return author.first_name, author.middle_names, author.last_name


@pytest.mark.parametrize(("source", "expected"), UNITED + UNCHANGED + AMBIGUOUS_CUT + KNOWN_LIMITS)
def test_korean_given_name(predictor, source, expected):
    assert resolve(predictor, *source) == expected


def test_korean_given_form_needs_a_korean_surname():
    assert korean_given_form("Jinho", "", "Park") == "Jin-Ho"
    assert korean_given_form("Jinho", "", "Smith") is None
    assert korean_given_form("Jin", "Ho", "Park") == "Jin-Ho"
    assert korean_given_form("Jin", "Ho Min", "Park") is None


def test_korean_given_form_ambiguous_word_needs_a_marked_cut():
    assert korean_given_form("Sungil", "", "Kim") is None
    assert korean_given_form("Sung", "Il", "Kim") == "Sung-Il"
    assert korean_given_form("SungIl", "", "Kim") == "Sung-Il"
    assert korean_given_form("Daeun", "", "Lee") is None
    assert korean_given_form("Da-eun", "", "Lee") == "Da-Eun"
