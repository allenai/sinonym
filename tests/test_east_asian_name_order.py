"""Regression tests for conservative East Asian family-first routing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sinonym.services.east_asian_name_order import EastAsianNameOrderService

if TYPE_CHECKING:
    from sinonym import ChineseNameDetector


def test_japanese_romanized_routes_only_unambiguous_dictionary_direction(
    detector: ChineseNameDetector,
) -> None:
    routed = detector.normalize_person_name("Shirakawa Hideki")
    ambiguous = detector.normalize_person_name("Motoki Kouzaki")

    assert routed is not None
    assert routed.text == "Hideki Shirakawa"
    assert routed.normalized.given_name == "Hideki"
    assert routed.normalized.surname == "Shirakawa"
    assert routed.source.order == ("surname", "given")
    assert ambiguous is not None
    assert ambiguous.text == "Motoki Kouzaki"
    assert ambiguous.source.order == ("given", "surname")


def test_structured_japanese_romanized_matches_raw_directional_route(
    detector: ChineseNameDetector,
) -> None:
    raw = detector.normalize_person_name("Kumagai Jin")
    structured = detector.normalize_person_name_components(first_name="Kumagai", last_name="Jin")

    assert raw is not None
    assert structured is not None
    assert structured.text == raw.text == "Jin Kumagai"
    assert structured.normalized == raw.normalized
    assert structured.source_text == "Kumagai Jin"
    assert structured.source.given_name == "Kumagai"
    assert structured.source.surname == "Jin"
    assert structured.source.given_tokens == ("Kumagai",)
    assert structured.source.surname_tokens == ("Jin",)
    assert structured.source.order == ("given", "surname")


def test_structured_japanese_romanized_preserves_ambiguous_or_given_first_pairs(
    detector: ChineseNameDetector,
) -> None:
    for first_name, last_name in (("Motoki", "Kouzaki"), ("Akira", "Kurosawa")):
        surface = f"{first_name} {last_name}"
        raw = detector.normalize_person_name(surface)
        structured = detector.normalize_person_name_components(first_name=first_name, last_name=last_name)

        assert raw is not None
        assert structured is not None
        assert structured.text == raw.text == surface
        assert structured.normalized == raw.normalized
        assert structured.source.given_name == first_name
        assert structured.source.surname == last_name
        assert structured.source.order == ("given", "surname")


def test_japanese_native_uses_classifier_then_component_boundary() -> None:
    decision = EastAsianNameOrderService().infer(
        "中田英寿",
        japanese_probability=lambda _name: 1.0,
    )

    assert decision is not None
    assert decision.first_name == "英寿"
    assert decision.last_name == "中田"
    assert decision.source_order == ("surname", "given")


def test_japanese_native_preserves_when_classifier_abstains() -> None:
    decision = EastAsianNameOrderService().infer(
        "中田英寿",
        japanese_probability=lambda _name: 0.79,
    )

    assert decision is None


def test_korean_routes_strict_shapes_and_preserves_ambiguous_romanization(
    detector: ChineseNameDetector,
) -> None:
    romanized = detector.normalize_person_name("Kim Min-jun")
    ambiguous = detector.normalize_person_name("Kim Yuna")
    native = detector.normalize_person_name("김민수")

    assert romanized is not None
    assert romanized.text == "Min-jun Kim"
    assert romanized.source.order == ("surname", "given")
    assert ambiguous is not None
    assert ambiguous.text == "Kim Yuna"
    assert native is not None
    assert native.text == "민수 김"
    assert native.source.order == ("surname", "given")


def test_vietnamese_requires_unicode_evidence_and_preserves_given_span(
    detector: ChineseNameDetector,
) -> None:
    unicode_name = detector.normalize_person_name("Nguyễn Văn An")
    ascii_name = detector.normalize_person_name("Nguyen Van An")

    assert unicode_name is not None
    assert unicode_name.text == "An Văn Nguyễn"
    assert unicode_name.normalized.given_name == "An"
    assert unicode_name.normalized.middle_name == "Văn"
    assert unicode_name.normalized.surname == "Nguyễn"
    assert unicode_name.source.order == ("surname", "middle", "given")
    assert ascii_name is not None
    assert ascii_name.text == "Nguyen Van An"


def test_comma_order_remains_authoritative(detector: ChineseNameDetector) -> None:
    canonical = detector.normalize_person_name("Kim, Min-jun")

    assert canonical is not None
    assert canonical.text == "Min-jun Kim"
    assert canonical.source.order == ("surname", "given")


def test_spaced_kanji_surname_first_is_routed_given_first(
    detector: ChineseNameDetector,
) -> None:
    # Spaced native kanji was only handled in the compact form, so "佐藤 優" fell through
    # to the generic given-first assumption and swapped the roles (given "佐藤"). It is now
    # routed family-first like the compact "佐藤優", matching the given-first canonical.
    for surface, expected_text, given, surname in (
        ("佐藤 優", "優 佐藤", "優", "佐藤"),
        ("高橋 洋一", "洋一 高橋", "洋一", "高橋"),
        ("田中 太郎", "太郎 田中", "太郎", "田中"),
        ("中村 修二", "修二 中村", "修二", "中村"),
    ):
        routed = detector.normalize_person_name(surface)
        assert routed is not None, surface
        assert routed.text == expected_text
        assert routed.normalized.given_name == given
        assert routed.normalized.surname == surname
        assert routed.source.order == ("surname", "given")
        # matches the compact form's routing
        compact = detector.normalize_person_name(surface.replace(" ", ""))
        assert compact is not None
        assert compact.text == expected_text


def test_spaced_kanji_given_first_or_ambiguous_is_not_double_swapped(
    detector: ChineseNameDetector,
) -> None:
    # A spaced kanji name that is already given-first (or where the reverse is also
    # dictionary-plausible) must be left as-is, not swapped back.
    for surface in ("優 佐藤", "太郎 田中"):
        result = detector.normalize_person_name(surface)
        assert result is not None
        assert result.text == surface
        assert result.source.order == ("given", "surname")


def test_spaced_han_chinese_name_not_routed_as_japanese(
    detector: ChineseNameDetector,
) -> None:
    # Chinese spaced-han names must not be swapped by the Japanese spaced-kanji handler
    # (ML-Japanese gated + native-dictionary evidence). The Chinese pipeline handles them.
    for surface in ("王 伟", "李 明", "陈 明"):
        result = detector.normalize_person_name(surface)
        assert result is not None
        assert result.text == surface
        assert result.source.order == ("given", "surname")


def test_european_diacritic_partner_not_swapped_to_east_asian(
    detector: ChineseNameDetector,
) -> None:
    # A Western given that is a CJK-surname homograph ("Kim") was swapped to the
    # family name whenever the partner carried a non-ASCII char — but Nordic/German
    # diacritics (ø/å/ö/ü) are European, not East-Asian evidence. "Kim" is a very common
    # Scandinavian GIVEN name, so these must stay given-first (given=Kim).
    for surface, given, surname in (
        ("Kim Brøsen", "Kim", "Brøsen"),
        ("Kim Møller", "Kim", "Møller"),
        ("Kim Haugbølle", "Kim", "Haugbølle"),
        ("Kim Hørslev-Petersen", "Kim", "Hørslev-Petersen"),  # was Korean-path via hyphen
        ("Kim Jørgensen", "Kim", "Jørgensen"),
        ("Kim Müller", "Kim", "Müller"),
        ("Kim Nygård", "Kim", "Nygård"),
    ):
        routed = detector.normalize_person_name(surface)
        assert routed is not None, surface
        assert routed.normalized.given_name == given, surface
        assert routed.normalized.surname == surname, surface


def test_mccune_reischauer_korean_still_routes(
    detector: ChineseNameDetector,
) -> None:
    # The European-diacritic gate must NOT block McCune-Reischauer romanized Korean, whose
    # breve vowels (ŏ/ŭ) are inside the East-Asian repertoire — these must still swap
    # family-first (surname = the Korean-surname token).
    for surface, surname in (
        ("Hwang Chŏl-su", "Hwang"),
        ("Kim Sŏ-yŏn", "Kim"),
    ):
        routed = detector.normalize_person_name(surface)
        assert routed is not None, surface
        assert routed.normalized.surname == surname, surface
        assert routed.source.order == ("surname", "given"), surface


def test_vietnamese_repertoire_preserved(
    detector: ChineseNameDetector,
) -> None:
    # Well-formed Vietnamese (horn/breve/tone marks are all in-repertoire) must still route
    # surname-first; the gate only excludes European-exclusive diacritics.
    for surface, surname in (
        ("Nguyễn Văn Anh", "Nguyễn"),
        ("Trần Thị Mai", "Trần"),
        ("Nguyễn Duy Cường", "Nguyễn"),  # ư = u+horn, in repertoire
    ):
        routed = detector.normalize_person_name(surface)
        assert routed is not None, surface
        assert routed.normalized.surname == surname, surface
        assert routed.source.order[0] == "surname", surface


def test_non_european_noise_glyphs_do_not_block_east_asian_routing(
    detector: ChineseNameDetector,
) -> None:
    # Turkish İ, Cyrillic homoglyphs, and Hepburn/McCune macrons are NOT European-exclusive
    # diacritics, so an otherwise East-Asian name carrying one still routes family-first
    # (these were regressions under the earlier Vietnamese-repertoire whitelist).
    tin = detector.normalize_person_name("Nguyễn Khac Tin")  # clean Vietnamese control
    assert tin is not None
    assert tin.normalized.surname == "Nguyễn"
    cyrillic = detector.normalize_person_name("Nguyễn Lam Аnh")  # 'А' is Cyrillic U+0410
    assert cyrillic is not None
    assert cyrillic.normalized.surname == "Nguyễn"


def test_east_asian_order_hard_and_ambiguous_cases_characterization(
    detector: ChineseNameDetector,
) -> None:
    """HARD / AMBIGUOUS cases — documented, deliberately NOT fixed.

    A diacritic identifies a name as European but does NOT determine its order; the fix
    only declines the *forced* East-Asian swap for European-exclusive diacritics and lets
    the Western given-first default apply. The cases below have no clean linguistic
    invariant and would need an ORCID/Western-surname signal + an LLM judge (see PR19).
    """
    # (A) Pure-ASCII Western hyphenated partner: indistinguishable from a hyphenated Korean
    # given by any character rule (the Korean given-syllable lexicon is incomplete, so
    # requiring known syllables would break real Korean like "Kim Bong-Whan"). Still swaps.
    for surface in ("Kim Dam-Johansen", "Jo Leonardi-Bee", "Jo Rycroft-Malone"):
        routed = detector.normalize_person_name(surface)
        assert routed is not None, surface
        assert routed.source.order == ("surname", "given"), surface  # unchanged (abstain)

    # (B) A shared diacritic (é is both French and Vietnamese) cannot disambiguate ethnicity,
    # so "Kim André" still routes as Vietnamese (surname = Kim). Documented limitation.
    andre = detector.normalize_person_name("Kim André")
    assert andre is not None
    assert andre.normalized.surname == "Kim"

    # (C) Mojibake recovered: "Cƣờng" uses U+01A3 (a corrupted "ư"). The blocklist gate
    # only declines the swap on European-EXCLUSIVE diacritics, and U+01A3 is not one, so a
    # name that is otherwise plainly Vietnamese still routes surname-first (was a casualty
    # of the earlier repertoire whitelist).
    mojibake = detector.normalize_person_name("Nguyễn Duy Cƣờng")
    assert mojibake is not None
    assert mojibake.normalized.surname == "Nguyễn"
