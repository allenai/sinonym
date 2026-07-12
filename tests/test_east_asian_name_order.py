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
