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

# Known limits, pinned so a change is noticed.
KNOWN_LIMITS = [
    # an ng + vowel boundary is ambiguous; the list follows most Korean registry people (one registers "Sun Gil")
    (("Sungil", None, "Kim"), ("Sung-Il", "", "Kim")),
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


@pytest.mark.parametrize(("source", "expected"), UNITED + UNCHANGED + KNOWN_LIMITS)
def test_korean_given_name(predictor, source, expected):
    assert resolve(predictor, *source) == expected


def test_korean_given_form_needs_a_korean_surname():
    assert korean_given_form("Jinho", "", "Park") == "Jin-Ho"
    assert korean_given_form("Jinho", "", "Smith") is None
    assert korean_given_form("Jin", "Ho", "Park") == "Jin-Ho"
    assert korean_given_form("Jin", "Ho Min", "Park") is None
