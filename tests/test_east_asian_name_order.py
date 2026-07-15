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
