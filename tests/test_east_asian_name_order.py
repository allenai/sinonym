"""Regression tests for conservative East Asian family-first routing."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from sinonym.coretypes import CanonicalName, NameComponents
from sinonym.coretypes.routing_resolution import EastAsianEvidenceReason, EvidenceFailure, ResolutionReason
from sinonym.services import east_asian_name_order
from sinonym.services.east_asian_name_order import (
    CROSS_CULTURAL_CONFLICT_HEADS,
    EastAsianNameOrderDecision,
    EastAsianNameOrderPreservation,
    EastAsianNameOrderService,
)

if TYPE_CHECKING:
    from sinonym import ChineseNameDetector


def test_cross_cultural_reorder_conflict_heads_are_a_closed_positive_set() -> None:
    assert {"mai", "to"} == CROSS_CULTURAL_CONFLICT_HEADS


def test_roman_lexicon_rejects_non_four_top_vietnamese_surname_asset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "japanese_surnames": ["sato"],
        "japanese_possible_surnames": ["sato"],
        "japanese_given_first_exact_surfaces": [],
        "japanese_given_names": ["mai"],
        "korean_surnames": ["kim"],
        "vietnamese_surnames": ["le", "nguyen", "pham", "tran"],
        "vietnamese_top4_surnames": ["le", "nguyen", "pham"],
    }
    monkeypatch.setattr(east_asian_name_order, "_load_payload", lambda _name: payload)
    east_asian_name_order._roman_lexicons.cache_clear()  # noqa: SLF001
    try:
        with pytest.raises(ValueError, match="exactly 4 entries"):
            east_asian_name_order._roman_lexicons()  # noqa: SLF001
    finally:
        east_asian_name_order._roman_lexicons.cache_clear()  # noqa: SLF001


def test_roman_lexicon_requires_directional_surnames_to_be_possible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "japanese_surnames": ["sato"],
        "japanese_possible_surnames": ["abe"],
        "japanese_given_first_exact_surfaces": [],
        "japanese_given_names": ["mai"],
        "korean_surnames": ["kim"],
        "vietnamese_surnames": ["le", "nguyen", "pham", "tran"],
        "vietnamese_top4_surnames": ["le", "nguyen", "pham", "tran"],
    }
    monkeypatch.setattr(east_asian_name_order, "_load_payload", lambda _name: payload)
    east_asian_name_order._roman_lexicons.cache_clear()  # noqa: SLF001
    try:
        with pytest.raises(ValueError, match="must be a subset"):
            east_asian_name_order._roman_lexicons()  # noqa: SLF001
    finally:
        east_asian_name_order._roman_lexicons.cache_clear()  # noqa: SLF001


@pytest.mark.parametrize(
    ("raw_name", "selected", "paper_names", "focal_index", "expected_reason"),
    [
        (
            "Mai Hata",
            NameComponents(given_name="Ha-Ta", surname="Mai"),
            ["Akira Suzuki", "Mai Hata"],
            1,
            ResolutionReason.JAPANESE_GIVEN_FIRST_REORDER_VETO_PRESERVE_INPUT,
        ),
        (
            "To Keku",
            NameComponents(given_name="Ke-Ku", surname="To"),
            ["To Keku", "Yuki Tanaka"],
            0,
            ResolutionReason.CONTEXT_SUPPORTED_REORDER_VETO_PRESERVE_INPUT,
        ),
        (
            "Nguyen Khanh Pham",
            NameComponents(given_name="Pham", middle_name="Khanh", surname="Nguyen"),
            ["Nguyen Khanh Pham", "Minh Nguyen"],
            0,
            ResolutionReason.CONTEXT_SUPPORTED_REORDER_VETO_PRESERVE_INPUT,
        ),
        (
            "Nguyen Van Nhi Tran",
            NameComponents(given_name="Tran", middle_name="Van Nhi", surname="Nguyen"),
            ["Anh Pham", "Nguyen Van Nhi Tran"],
            1,
            ResolutionReason.CONTEXT_SUPPORTED_REORDER_VETO_PRESERVE_INPUT,
        ),
        (
            "Tran Duc Le",
            NameComponents(given_name="Le", middle_name="Duc", surname="Tran"),
            ["Tran Duc Le", "Minh Nguyen", "Anh Pham", "Tran Minh"],
            0,
            ResolutionReason.CONTEXT_SUPPORTED_REORDER_VETO_PRESERVE_INPUT,
        ),
        (
            "Bui Anh Tran",
            NameComponents(given_name="Tran", middle_name="Anh", surname="Bui"),
            ["Bui Anh Tran", "Minh Nguyen"],
            0,
            ResolutionReason.CONTEXT_SUPPORTED_REORDER_VETO_PRESERVE_INPUT,
        ),
    ],
)
def test_candidate_reorder_conflicts_are_typed(
    raw_name: str,
    selected: NameComponents,
    paper_names: list[str],
    focal_index: int,
    expected_reason: ResolutionReason,
) -> None:
    service = EastAsianNameOrderService()

    assert (
        service.reorder_conflict_reason(
            raw_name,
            selected,
            paper_names=paper_names,
            focal_index=focal_index,
        )
        is expected_reason
    )


@pytest.mark.parametrize(
    ("raw_name", "selected"),
    [
        ("Ren Sugai", NameComponents(given_name="Su-Gai", surname="Ren")),
        ("Ma Kai", NameComponents(given_name="Kai", surname="Ma")),
        ("Mai Muto", NameComponents(given_name="Muto", surname="Mai")),
        ("Ran Nakai", NameComponents(given_name="Nakai", surname="Ran")),
    ],
)
def test_strict_japanese_given_first_reversal_veto_needs_no_paper_context(
    raw_name: str,
    selected: NameComponents,
) -> None:
    service = EastAsianNameOrderService()

    assert (
        service.reorder_conflict_reason(
            raw_name,
            selected,
            paper_names=[raw_name],
            focal_index=0,
        )
        is ResolutionReason.JAPANESE_GIVEN_FIRST_REORDER_VETO_PRESERVE_INPUT
    )


@pytest.mark.parametrize(
    "raw_name",
    [
        "Haruki Kadono",
        "Kou Hiroya",
        "Masaki Takamoto",
        "Masaki Tomonaga",
        "Shoji Kagami",
        "Takaya Miwa",
    ],
)
def test_reviewed_exact_japanese_surface_vetoes_endpoint_reversal(raw_name: str) -> None:
    service = EastAsianNameOrderService()
    first, last = raw_name.split()

    assert (
        service.reorder_conflict_reason(
            raw_name,
            NameComponents(given_name=last, surname=first),
            paper_names=[raw_name],
            focal_index=0,
        )
        is ResolutionReason.JAPANESE_GIVEN_FIRST_REORDER_VETO_PRESERVE_INPUT
    )


@pytest.mark.parametrize(
    "raw_name",
    [
        "  KOU\tHIROYA ",
        "\uff2b\uff4f\uff55\u3000\uff28\uff49\uff52\uff4f\uff59\uff41",
    ],
)
def test_reviewed_exact_japanese_surface_normalizes_case_width_and_whitespace(raw_name: str) -> None:
    assert EastAsianNameOrderService()._is_reviewed_japanese_given_first_exact_surface(raw_name)  # noqa: SLF001


@pytest.mark.parametrize("raw_name", ["Kōu Hiroya", "Takayā Miwa", "Kou-Hiroya"])
def test_reviewed_exact_japanese_surface_preserves_accents_and_punctuation(raw_name: str) -> None:
    assert not EastAsianNameOrderService()._is_reviewed_japanese_given_first_exact_surface(raw_name)  # noqa: SLF001


@pytest.mark.parametrize(
    ("raw_name", "expected_text"),
    [
        ("Kōu Hiroya", "Hiroya Kōu"),
        ("Takayā Miwa", "Miwa Takayā"),
    ],
)
def test_accent_variants_do_not_activate_ascii_exact_preselection(
    detector: ChineseNameDetector,
    raw_name: str,
    expected_text: str,
) -> None:
    normalized = detector.normalize_person_name(raw_name)

    assert normalized is not None
    assert normalized.text == expected_text
    assert normalized.source.order == ("surname", "given")


@pytest.mark.parametrize(
    "raw_name",
    [
        "Haruki Kadono",
        "Kou Hiroya",
        "Masaki Takamoto",
        "Masaki Tomonaga",
        "Shoji Kagami",
        "Takaya Miwa",
    ],
)
def test_reviewed_exact_japanese_surface_materializes_input_order_at_detector_boundary(
    detector: ChineseNameDetector,
    raw_name: str,
) -> None:
    first, last = raw_name.split()

    selected, reason = detector.routing_reorder_veto(
        raw_name,
        NameComponents(given_name=last, surname=first),
        paper_names=[raw_name],
        focal_index=0,
    )

    assert (selected.given_name, selected.middle_name, selected.surname) == (first, "", last)
    assert reason is ResolutionReason.JAPANESE_GIVEN_FIRST_REORDER_VETO_PRESERVE_INPUT


@pytest.mark.parametrize(
    "raw_name",
    [
        "Haruki Kadono",
        "Masaki Takamoto",
        "Masaki Tomonaga",
        "Shoji Kagami",
    ],
)
def test_reviewed_postselection_exact_surface_preserves_public_scalar_order(
    detector: ChineseNameDetector,
    raw_name: str,
) -> None:
    first, last = raw_name.split()

    normalized = detector.normalize_person_name(raw_name)

    assert normalized is not None
    assert normalized.text == raw_name
    assert (normalized.normalized.given_name, normalized.normalized.surname) == (first, last)
    assert normalized.source.order == ("given", "surname")


@pytest.mark.parametrize(
    "raw_name",
    [
        "Haruki Kadono",
        "Masaki Takamoto",
        "Masaki Tomonaga",
        "Shoji Kagami",
    ],
)
def test_reviewed_postselection_exact_surface_has_raw_structured_parity(
    detector: ChineseNameDetector,
    raw_name: str,
) -> None:
    first, last = raw_name.split()

    raw = detector.normalize_person_name(raw_name)
    structured = detector.normalize_person_name_components(first_name=first, last_name=last)

    assert raw is not None
    assert structured is not None
    assert structured.text == raw.text == raw_name
    assert structured.normalized == raw.normalized
    assert structured.source.order == ("given", "surname")


@pytest.mark.parametrize("raw_name", ["Gan Kai", "Mao Kai", "Shi Kai", "Yu Mi", "Yuan Tai"])
def test_ambiguous_exact_surface_does_not_bypass_japanese_reversal_veto_without_context(raw_name: str) -> None:
    service = EastAsianNameOrderService()
    first, last = raw_name.split()

    assert (
        service.reorder_conflict_reason(
            raw_name,
            NameComponents(given_name=last, surname=first),
            paper_names=[raw_name],
            focal_index=0,
        )
        is ResolutionReason.JAPANESE_GIVEN_FIRST_REORDER_VETO_PRESERVE_INPUT
    )


@pytest.mark.parametrize("raw_name", ["Gan Kai", "Shi Kai", "Yu Mi"])
def test_reviewed_family_first_surface_can_bypass_veto_with_complete_peer(raw_name: str) -> None:
    service = EastAsianNameOrderService()
    first, last = raw_name.split()

    assert (
        service.reorder_conflict_reason(
            raw_name,
            NameComponents(given_name=last, surname=first),
            paper_names=[raw_name, "Zhang Changzheng"],
            focal_index=0,
        )
        is None
    )


@pytest.mark.parametrize("context_name", ["Zhao", "Std Control"])
def test_reviewed_family_first_surface_requires_a_surname_bearing_peer(context_name: str) -> None:
    service = EastAsianNameOrderService()

    assert (
        service.reorder_conflict_reason(
            "Shi Kai",
            NameComponents(given_name="Kai", surname="Shi"),
            paper_names=["Shi Kai", context_name],
            focal_index=0,
        )
        is ResolutionReason.JAPANESE_GIVEN_FIRST_REORDER_VETO_PRESERVE_INPUT
    )


def test_yuan_tai_does_not_bypass_veto_with_context() -> None:
    service = EastAsianNameOrderService()

    assert (
        service.reorder_conflict_reason(
            "Yuan Tai",
            NameComponents(given_name="Tai", surname="Yuan"),
            paper_names=["Yuan Tai", "Pei Yan"],
            focal_index=0,
        )
        is ResolutionReason.JAPANESE_GIVEN_FIRST_REORDER_VETO_PRESERVE_INPUT
    )


def test_reviewed_japanese_given_first_surface_vetoes_only_a_reversal() -> None:
    service = EastAsianNameOrderService()
    raw_name = "智幸 小枝"

    assert (
        service.reorder_conflict_reason(
            raw_name,
            NameComponents(given_name="小枝", surname="智幸"),
            paper_names=[raw_name],
            focal_index=0,
        )
        is ResolutionReason.JAPANESE_GIVEN_FIRST_REORDER_VETO_PRESERVE_INPUT
    )
    assert (
        service.reorder_conflict_reason(
            raw_name,
            NameComponents(given_name="智幸", surname="小枝"),
            paper_names=[raw_name],
            focal_index=0,
        )
        is None
    )


def test_reviewed_vietnamese_given_first_surface_vetoes_without_context() -> None:
    service = EastAsianNameOrderService()

    assert (
        service.reorder_conflict_reason(
            "Tuan Le",
            NameComponents(given_name="Le", surname="Tuan"),
            paper_names=["Tuan Le"],
            focal_index=0,
        )
        is ResolutionReason.VIETNAMESE_GIVEN_FIRST_REORDER_VETO_PRESERVE_INPUT
    )


def test_reviewed_cross_cultural_given_first_surface_vetoes_without_context() -> None:
    service = EastAsianNameOrderService()

    assert (
        service.reorder_conflict_reason(
            "To Keku",
            NameComponents(given_name="Ke-Ku", surname="To"),
            paper_names=["To Keku"],
            focal_index=0,
        )
        is ResolutionReason.CONTEXT_SUPPORTED_REORDER_VETO_PRESERVE_INPUT
    )


@pytest.mark.parametrize(
    ("raw_name", "selected"),
    [
        ("Mai Hata", NameComponents(given_name="Mai", surname="Hata")),
        ("Mai Hata Jr.", NameComponents(given_name="Hata", surname="Mai")),
        ("do Dom\u00ednguez-Carrillo", NameComponents(given_name="Dom\u00ednguez-Carrillo", surname="do")),
        ("DO Nascimento Cm", NameComponents(given_name="Nascimento", surname="DO")),
        ("Do Hyun \uc131\ub3c4\ud604Sung", NameComponents(given_name="\uc131\ub3c4\ud604Sung", surname="Do")),
        ("do Elvio Fern\u00e1ndez Toledo", NameComponents(given_name="Elvio Fern\u00e1ndez Toledo", surname="do")),
        ("Pham Thanh Nguyen-Ba", NameComponents(given_name="Nguyen-Ba", middle_name="Thanh", surname="Pham")),
        ("Bui Hai Hoang", NameComponents(given_name="Hoang", middle_name="Hai", surname="Bui")),
        ("Yu Chih-Chen", NameComponents(given_name="Chih-Chen", surname="Yu")),
        ("Sato Haruto", NameComponents(given_name="Haruto", surname="Sato")),
        ("Ren Sugai", NameComponents(given_name="Ren", surname="Sugai")),
        ("Tuan Le", NameComponents(given_name="Le Extra", surname="Tuan")),
        ("Wang Le", NameComponents(given_name="Le", surname="Wang")),
        ("Sato Le", NameComponents(given_name="Le", surname="Sato")),
    ],
)
def test_reorder_veto_declines_unproven_or_same_order_candidates(
    raw_name: str,
    selected: NameComponents,
) -> None:
    service = EastAsianNameOrderService()

    assert (
        service.reorder_conflict_reason(
            raw_name,
            selected,
            paper_names=[raw_name],
            focal_index=0,
        )
        is None
    )


@pytest.mark.parametrize(
    "raw_name",
    [
        "Đinh Thị Hiền Lê",
        "Bùi Văn Lễ",
    ],
)
def test_lower_prior_vietnamese_context_veto_is_limited_to_bare_ascii(
    raw_name: str,
) -> None:
    tokens = raw_name.split()
    service = EastAsianNameOrderService()

    assert (
        service.reorder_conflict_reason(
            raw_name,
            NameComponents(
                given_name=tokens[-1],
                middle_name=" ".join(tokens[1:-1]),
                surname=tokens[0],
            ),
            paper_names=[raw_name, "Minh Nguyen"],
            focal_index=0,
        )
        is None
    )


@pytest.mark.parametrize(
    ("raw_name", "selected", "context_name"),
    [
        ("Nguyen Khanh Pham", NameComponents(given_name="Pham", surname="Nguyen"), "Tran Minh"),
        ("Bui Anh Tran", NameComponents(given_name="Tran", surname="Bui"), "Tran Minh"),
    ],
)
def test_reorder_veto_requires_given_first_context_majority(
    raw_name: str,
    selected: NameComponents,
    context_name: str,
) -> None:
    service = EastAsianNameOrderService()

    assert (
        service.reorder_conflict_reason(
            raw_name,
            selected,
            paper_names=[raw_name, context_name],
            focal_index=0,
        )
        is None
    )


def test_reorder_veto_declines_tied_context() -> None:
    service = EastAsianNameOrderService()

    assert (
        service.reorder_conflict_reason(
            "Nguyen Khanh Pham",
            NameComponents(given_name="Pham", surname="Nguyen"),
            paper_names=["Minh Nguyen", "Nguyen Khanh Pham", "Tran Minh"],
            focal_index=1,
        )
        is None
    )


def test_reorder_veto_excludes_only_the_focal_duplicate_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = EastAsianNameOrderService()
    voted_names: list[str] = []

    def vote_given_first(name: str) -> str:
        voted_names.append(name)
        return "given_first"

    monkeypatch.setattr(service, "_vietnamese_order_vote", vote_given_first)

    assert (
        service.reorder_conflict_reason(
            "Nguyen Khanh Pham",
            NameComponents(given_name="Pham", surname="Nguyen"),
            paper_names=["Nguyen Khanh Pham", "Nguyen Khanh Pham"],
            focal_index=0,
        )
        is ResolutionReason.CONTEXT_SUPPORTED_REORDER_VETO_PRESERVE_INPUT
    )
    assert voted_names == ["Nguyen Khanh Pham"]


@pytest.mark.parametrize(
    ("raw_name", "selected"),
    [
        (
            "Nguyen Khanh Pham",
            NameComponents(given_name="Pham Extra", surname="Nguyen"),
        ),
        (
            "Nguyen Khanh Bui",
            NameComponents(given_name="Bui", surname="Nguyen"),
        ),
    ],
)
def test_reorder_veto_requires_exact_reversal_and_two_top4_endpoints(
    raw_name: str,
    selected: NameComponents,
) -> None:
    service = EastAsianNameOrderService()

    assert (
        service.reorder_conflict_reason(
            raw_name,
            selected,
            paper_names=[raw_name, "Minh Nguyen"],
            focal_index=0,
        )
        is None
    )


def test_reorder_veto_alignment_is_positional() -> None:
    service = EastAsianNameOrderService()
    selected = NameComponents(given_name="Pham", surname="Nguyen")

    with pytest.raises(IndexError, match="outside"):
        service.reorder_conflict_reason(
            "Nguyen Khanh Pham",
            selected,
            paper_names=["Nguyen Khanh Pham"],
            focal_index=1,
        )
    with pytest.raises(ValueError, match="positional"):
        service.reorder_conflict_reason(
            "Nguyen Khanh Pham",
            selected,
            paper_names=["Minh Nguyen"],
            focal_index=0,
        )


@pytest.mark.parametrize(
    ("raw_name", "expected_given", "expected_surname"),
    [
        ("Kim Amsley-Camp", "Kim", "Amsley-Camp"),
        ("Hong Mu-Mosley", "Hong", "Mu-Mosley"),
    ],
)
def test_western_surname_tail_veto_keeps_generic_source_order(
    detector: ChineseNameDetector,
    raw_name: str,
    expected_given: str,
    expected_surname: str,
) -> None:
    canonical = detector.normalize_person_name(raw_name)

    assert canonical is not None
    assert canonical.normalized.given_name == expected_given
    assert canonical.normalized.surname == expected_surname


def test_korean_western_conflict_is_typed_as_preservation_not_a_decision() -> None:
    service = EastAsianNameOrderService()

    resolution = service.infer_resolution(
        "Kim Stene-Larsen",
        japanese_probability=lambda _name: 0.0,
    )

    assert isinstance(resolution, EastAsianNameOrderPreservation)
    assert resolution.reason is EastAsianEvidenceReason.KOREAN_WESTERN_SUFFIX_CONFLICT


def test_western_suffix_veto_preserves_korean_breve_romanization(
    detector: ChineseNameDetector,
) -> None:
    canonical = detector.normalize_person_name("Kim Hŭisŏn")

    assert canonical is not None
    assert canonical.normalized.given_name == "Hŭisŏn"
    assert canonical.normalized.surname == "Kim"


@pytest.mark.parametrize(
    ("raw_name", "expected_reason", "expected_given", "expected_surname"),
    [
        (
            "Hồ Anderson",
            "vietnamese_unicode_surname_first",
            "Anderson",
            "Hồ",
        ),
        (
            "Cho Arison",
            "japanese_romanized_directional_dictionary",
            "Arison",
            "Cho",
        ),
    ],
)
def test_western_suffix_veto_does_not_suppress_other_family_first_routes(
    raw_name: str,
    expected_reason: str,
    expected_given: str,
    expected_surname: str,
) -> None:
    service = EastAsianNameOrderService()

    decision = service.infer_resolution(raw_name, japanese_probability=lambda _name: 0.0)

    assert isinstance(decision, EastAsianNameOrderDecision)
    assert decision.reason == expected_reason
    assert decision.first_name == expected_given
    assert decision.last_name == expected_surname
    assert decision.source_order == ("surname", "given")
    assert service.family_first_conflict_reason(raw_name) is None


@pytest.mark.parametrize(
    ("raw_name", "expected_given", "expected_surname"),
    [
        ("kim Jiyoung", "Jiyoung", "Kim"),
        ("Park Seonae", "Seonae", "Park"),
        ("Jang Jisoo", "Jisoo", "Jang"),
    ],
)
def test_unique_compact_korean_given_routes_family_first_without_respelled_token(
    detector: ChineseNameDetector,
    raw_name: str,
    expected_given: str,
    expected_surname: str,
) -> None:
    canonical = detector.normalize_person_name(raw_name)

    assert canonical is not None
    assert canonical.normalized.given_name == expected_given
    assert canonical.normalized.surname == expected_surname
    assert "-" not in canonical.normalized.given_name


def test_unique_compact_korean_rule_does_not_override_chinese_parser_success(
    detector: ChineseNameDetector,
) -> None:
    result = detector.normalize_name("Son SeungHun")

    assert result.success
    assert result.parsed is not None
    assert result.parsed.given_name == "Seung-Hun"
    assert result.parsed.surname == "Son"


def test_unique_compact_korean_rule_leaves_given_first_input_unchanged(
    detector: ChineseNameDetector,
) -> None:
    canonical = detector.normalize_person_name("Jiyoung Kim")

    assert canonical is not None
    assert canonical.normalized.given_name == "Jiyoung"
    assert canonical.normalized.surname == "Kim"


@pytest.mark.parametrize(
    ("raw_name", "expected_given", "expected_surname", "expected_order"),
    [
        ("克典 佐々木", "克典", "佐々木", ("given", "surname")),
        ("佐々 政人", "政人", "佐々", ("surname", "given")),
        ("佐々木克典", "克典", "佐々木", ("surname", "given")),
    ],
)
def test_iteration_mark_resolver_requires_one_complete_dual_exclusive_assignment(
    raw_name: str,
    expected_given: str,
    expected_surname: str,
    expected_order: tuple[str, str],
) -> None:
    decision = EastAsianNameOrderService().infer_iteration_mark(
        raw_name,
        japanese_probability=lambda _name: 1.0,
    )

    assert decision is not None
    assert decision.given_tokens == (expected_given,)
    assert decision.surname_tokens == (expected_surname,)
    assert decision.source_order == expected_order
    assert decision.reason is EastAsianEvidenceReason.JAPANESE_ITERATION_MARK_DUAL_EXCLUSIVE


@pytest.mark.parametrize("selector", ["\ufe00", "\ufe0f", "\U000e0100", "\U000e01ef"])
@pytest.mark.parametrize(
    ("raw_template", "surname_template", "expected_order"),
    [
        ("克典 佐{selector}々木", "佐{selector}々木", ("given", "surname")),
        ("佐{selector}々木 克典", "佐{selector}々木", ("surname", "given")),
        ("佐{selector}々木克典", "佐{selector}々木", ("surname", "given")),
        ("佐々木{selector}克典", "佐々木{selector}", ("surname", "given")),
    ],
)
def test_iteration_mark_variation_selectors_are_lookup_only(
    selector: str,
    raw_template: str,
    surname_template: str,
    expected_order: tuple[str, str],
) -> None:
    raw_name = raw_template.format(selector=selector)
    classifier_inputs: list[str] = []

    decision = EastAsianNameOrderService().infer_iteration_mark(
        raw_name,
        japanese_probability=lambda value: classifier_inputs.append(value) or 1.0,
    )

    assert classifier_inputs == [raw_name.replace(selector, "")]
    assert decision is not None
    assert decision.first_name == "克典"
    assert decision.last_name == surname_template.format(selector=selector)
    assert decision.source_order == expected_order
    assert decision.reason is EastAsianEvidenceReason.JAPANESE_ITERATION_MARK_DUAL_EXCLUSIVE


@pytest.mark.parametrize("selector", ["\ufe00", "\U000e0100"])
def test_iteration_mark_variation_selector_canonical_preserves_authored_surname(
    detector: ChineseNameDetector,
    selector: str,
) -> None:
    raw_name = f"佐々木{selector}克典"

    result = detector.normalize_name(raw_name)

    assert result.success
    assert result.canonical_name is not None
    assert result.canonical_name.text == f"克典 佐々木{selector}"
    assert result.canonical_name.normalized.given_name == "克典"
    assert result.canonical_name.normalized.surname == f"佐々木{selector}"
    assert result.canonical_name.source.order == ("surname", "given")


@pytest.mark.parametrize(
    ("raw_name", "expected_classifier_inputs"),
    [
        ("佐\ufe00々木 X", []),
        ("三谷 奈\U000e0100々", ["三谷 奈々"]),
    ],
)
def test_iteration_mark_variation_selectors_do_not_widen_role_evidence(
    raw_name: str,
    expected_classifier_inputs: list[str],
) -> None:
    classifier_inputs: list[str] = []

    decision = EastAsianNameOrderService().infer_iteration_mark(
        raw_name,
        japanese_probability=lambda value: classifier_inputs.append(value) or 1.0,
    )

    assert decision is None
    assert classifier_inputs == expected_classifier_inputs


@pytest.mark.parametrize(
    ("raw_name", "expected_given", "expected_surname", "expected_order"),
    [
        ("純 野々崎", "純", "野々崎", ("given", "surname")),
        ("善彦 福々迫", "善彦", "福々迫", ("given", "surname")),
        ("勝 等々力", "勝", "等々力", ("given", "surname")),
        ("賢吉 目々沢", "賢吉", "目々沢", ("given", "surname")),
        ("扶実子 佐々木", "扶実子", "佐々木", ("given", "surname")),
        ("津 佐々木", "津", "佐々木", ("given", "surname")),
        ("盡 佐々木", "盡", "佐々木", ("given", "surname")),
        ("蘭子 佐々木", "蘭子", "佐々木", ("given", "surname")),
    ],
)
def test_iteration_mark_resolver_accepts_one_sided_exclusive_evidence(
    raw_name: str,
    expected_given: str,
    expected_surname: str,
    expected_order: tuple[str, str],
) -> None:
    decision = EastAsianNameOrderService().infer_iteration_mark(
        raw_name,
        japanese_probability=lambda _name: 1.0,
    )

    assert decision is not None
    assert decision.given_tokens == (expected_given,)
    assert decision.surname_tokens == (expected_surname,)
    assert decision.source_order == expected_order
    assert decision.reason is EastAsianEvidenceReason.JAPANESE_ITERATION_MARK_ONE_SIDED_EXCLUSIVE


@pytest.mark.parametrize(
    "raw_name",
    [
        "三谷 奈々",  # the iteration mark belongs to a known given name
        "純 菜々子",  # a three-character marked given name is not a surname
        "佐々木 山田",  # the complement is a known surname, not a given-name gap
        "未知 野々甲",  # neither side has positive shipped role evidence
        "純 々",  # a bare iteration mark is not a name span
        "佐々木扶実子",  # the one-sided extension does not expand compact partitions
        "純 野々崎 別",  # the one-sided extension requires exactly two source spans
        "佐々木新",  # the complement is not given-only, so the nested split is unsafe
    ],
)
def test_iteration_mark_resolver_abstains_without_frozen_role_evidence(
    raw_name: str,
) -> None:
    assert (
        EastAsianNameOrderService().infer_iteration_mark(
            raw_name,
            japanese_probability=lambda _name: 1.0,
        )
        is None
    )


def test_iteration_mark_resolver_requires_japanese_classifier_gate() -> None:
    assert (
        EastAsianNameOrderService().infer_iteration_mark(
            "佐々木克典",
            japanese_probability=lambda _name: 0.79,
        )
        is None
    )


def test_iteration_mark_canonical_overrides_chinese_sidecar_only_when_roles_are_proven(
    detector: ChineseNameDetector,
) -> None:
    result = detector.normalize_name("佐々木克典")

    assert result.success
    assert result.canonical_name is not None
    assert result.canonical_name.text == "克典 佐々木"
    assert result.canonical_name.normalized.given_name == "克典"
    assert result.canonical_name.normalized.surname == "佐々木"
    assert result.canonical_name.source.order == ("surname", "given")


@pytest.mark.parametrize("probability", [math.nan, math.inf, -0.1, 1.1])
def test_iteration_mark_rule_rejects_invalid_classifier_probability(
    probability: float,
) -> None:
    with pytest.raises(EvidenceFailure, match="invalid probability"):
        EastAsianNameOrderService().infer_iteration_mark(
            "佐々木 克典",
            japanese_probability=lambda _name: probability,
        )


@pytest.mark.parametrize("raw_name", ["佐藤優", "佐藤 優"])
@pytest.mark.parametrize("probability", [math.nan, math.inf, -0.1, 1.1])
def test_all_japanese_native_rules_reject_invalid_classifier_probability(
    raw_name: str,
    probability: float,
) -> None:
    with pytest.raises(EvidenceFailure, match="invalid probability"):
        EastAsianNameOrderService().infer_resolution(
            raw_name,
            japanese_probability=lambda _name: probability,
        )


@pytest.mark.parametrize("raw_name", ["\u8fbb", "\u512a\ufe00"])
def test_singleton_japanese_native_surface_has_no_name_boundary(raw_name: str) -> None:
    resolution = EastAsianNameOrderService().infer_resolution(
        raw_name,
        japanese_probability=lambda _name: 0.9,
    )

    assert resolution is None


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


@pytest.mark.parametrize(
    "raw_name",
    ["Kou Hiroya", "Takaya Miwa"],
)
def test_reviewed_exact_japanese_surface_makes_scalar_route_abstain(
    detector: ChineseNameDetector,
    raw_name: str,
) -> None:
    assert EastAsianNameOrderService().infer_resolution(raw_name, japanese_probability=lambda _name: 0.0) is None

    normalized = detector.normalize_person_name(raw_name)

    assert normalized is not None
    assert normalized.text == raw_name
    assert normalized.source.order == ("given", "surname")


@pytest.mark.parametrize("raw_name", ["Hiroya Utsumi", "Hiroya Imao", "Miwa Uzuki"])
def test_possible_japanese_surname_cannot_initiate_a_flip(
    detector: ChineseNameDetector,
    raw_name: str,
) -> None:
    normalized = detector.normalize_person_name(raw_name)

    assert normalized is not None
    assert normalized.text == raw_name


def test_possible_japanese_surname_does_not_override_strong_family_first_evidence(
    detector: ChineseNameDetector,
) -> None:
    normalized = detector.normalize_person_name("Satomi Miwa")

    assert normalized is not None
    assert normalized.text == "Miwa Satomi"


def test_possible_japanese_surname_respects_strong_family_first_paper_context() -> None:
    service = EastAsianNameOrderService()

    reason = service.reorder_conflict_reason(
        "Satomi Miwa",
        NameComponents(given_name="Miwa", surname="Satomi"),
        paper_names=["Kuwahara Tsuyoshi", "Satomi Miwa", "Kono Miyuki"],
        focal_index=1,
    )

    assert reason is None


def test_possible_japanese_surname_uses_given_first_paper_context_for_conflict() -> None:
    service = EastAsianNameOrderService()

    reason = service.reorder_conflict_reason(
        "Satomi Miwa",
        NameComponents(given_name="Miwa", surname="Satomi"),
        paper_names=["Akira Suzuki", "Satomi Miwa", "Yuki Takeuchi"],
        focal_index=1,
    )

    assert reason is ResolutionReason.JAPANESE_GIVEN_FIRST_REORDER_VETO_PRESERVE_INPUT


@pytest.mark.parametrize(
    "raw_name",
    ["Daikou Shiota", "Shin Kadono", "Jun Hata", "Shin Haba"],
)
def test_reviewed_possible_surnames_veto_reversal_with_given_first_paper_context(raw_name: str) -> None:
    first, last = raw_name.split()

    reason = EastAsianNameOrderService().reorder_conflict_reason(
        raw_name,
        NameComponents(given_name=last, surname=first),
        paper_names=["Akira Suzuki", raw_name, "Yuki Takeuchi"],
        focal_index=1,
    )

    assert reason is ResolutionReason.JAPANESE_GIVEN_FIRST_REORDER_VETO_PRESERVE_INPUT


def test_reviewed_possible_surname_does_not_overrule_surname_first_paper_context() -> None:
    reason = EastAsianNameOrderService().reorder_conflict_reason(
        "Daikou Shiota",
        NameComponents(given_name="Shiota", surname="Daikou"),
        paper_names=["Kuwahara Tsuyoshi", "Daikou Shiota", "Kono Miyuki"],
        focal_index=1,
    )

    assert reason is None


@pytest.mark.parametrize("raw_name", ["Shiota Daikou", "Hirano Taro"])
def test_new_possible_surnames_cannot_initiate_scalar_reorder(
    detector: ChineseNameDetector,
    raw_name: str,
) -> None:
    normalized = detector.normalize_person_name(raw_name)

    assert normalized is not None
    assert normalized.text == raw_name


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
    decision = EastAsianNameOrderService().infer_resolution(
        "中田英寿",
        japanese_probability=lambda _name: 1.0,
    )

    assert isinstance(decision, EastAsianNameOrderDecision)
    assert decision.first_name == "英寿"
    assert decision.last_name == "中田"
    assert decision.source_order == ("surname", "given")


@pytest.mark.parametrize("selector", ["\ufe00", "\ufe0f", "\U000e0100", "\U000e01ef"])
def test_japanese_native_variation_selectors_are_lookup_only(selector: str) -> None:
    classifier_inputs: list[str] = []
    raw_name = f"\u5c71\u7530{selector}\u592a\u90ce"

    decision = EastAsianNameOrderService().infer_resolution(
        raw_name,
        japanese_probability=lambda value: classifier_inputs.append(value) or 1.0,
    )

    assert classifier_inputs == ["\u5c71\u7530\u592a\u90ce"]
    assert isinstance(decision, EastAsianNameOrderDecision)
    assert decision.first_name == "\u592a\u90ce"
    assert decision.last_name == f"\u5c71\u7530{selector}"
    assert decision.source_order == ("surname", "given")


@pytest.mark.parametrize("selector", ["\ufe00", "\ufe0f", "\U000e0100", "\U000e01ef"])
def test_variation_selector_japanese_matches_base_classification(
    detector: ChineseNameDetector,
    selector: str,
) -> None:
    raw_name = f"\u5c71\u7530{selector}\u592a\u90ce"
    base = detector.normalize_name("\u5c71\u7530\u592a\u90ce")
    variant = detector.normalize_name(raw_name)

    assert not base.success
    assert variant.success is base.success
    assert variant.error_message == base.error_message == "Japanese name detected by ML classifier"
    assert variant.canonical_name is not None
    assert variant.canonical_name.source_text == raw_name
    assert variant.canonical_name.text == f"\u592a\u90ce \u5c71\u7530{selector}"
    assert variant.canonical_name.normalized.given_name == "\u592a\u90ce"
    assert variant.canonical_name.normalized.surname == f"\u5c71\u7530{selector}"


@pytest.mark.parametrize("selector", ["\ufe00", "\ufe0f", "\U000e0100", "\U000e01ef"])
@pytest.mark.parametrize("base_name", ["\u738b\u4f1f", "\u738b\u5c0f\u660e", "\u6b27\u9633\u4f1f"])
def test_variation_selector_chinese_matches_base_semantics(
    detector: ChineseNameDetector,
    selector: str,
    base_name: str,
) -> None:
    raw_name = f"{base_name[0]}{selector}{base_name[1:]}"
    base = detector.normalize_name(base_name)
    variant = detector.normalize_name(raw_name)

    assert base.success
    assert variant.success
    assert variant.result == base.result
    assert variant.parsed == base.parsed
    assert variant.parsed_original_order == base.parsed_original_order
    assert variant.canonical_name is not None
    assert base.canonical_name is not None
    assert variant.canonical_name.source_text == raw_name
    assert variant.canonical_name.normalized == base.canonical_name.normalized


def test_japanese_native_preserves_when_classifier_abstains() -> None:
    decision = EastAsianNameOrderService().infer_resolution(
        "中田英寿",
        japanese_probability=lambda _name: 0.79,
    )

    assert decision is None


def test_spaced_japanese_route_declines_valid_probability_below_threshold() -> None:
    decision = EastAsianNameOrderService().infer_resolution(
        "佐藤 優",
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


@pytest.mark.parametrize(
    ("raw_name", "clean_name"),
    [
        ("Kim Min\u2010Jun", "Kim Min-Jun"),
        ("Dr. Kim Min-Jun", "Kim Min-Jun"),
        ("Kim Min-Jun, MD", "Kim Min-Jun"),
        ("Nguy\u1ec5n V\u0103n An MD", "Nguy\u1ec5n V\u0103n An"),
    ],
)
def test_east_asian_routing_uses_cleaned_person_tokens(
    detector: ChineseNameDetector,
    raw_name: str,
    clean_name: str,
) -> None:
    expected = detector.normalize_person_name(clean_name)
    actual = detector.normalize_person_name(raw_name)
    scalar = detector.routing_scalar_resolution(raw_name)

    assert expected is not None
    assert actual is not None
    assert actual.source_text == raw_name
    assert actual.text == expected.text
    assert actual.normalized == expected.normalized
    assert isinstance(scalar, CanonicalName)
    assert scalar.text == expected.text
    assert scalar.normalized == expected.normalized


@pytest.mark.parametrize(
    ("raw_name", "expected_text", "expected_source_order"),
    [
        ("Dr. Kim Min-Jun", "Min-Jun Kim", ("given", "surname", "given")),
        ("MD Kim Min-Jun", "Min-Jun Kim", ("suffix", "surname", "given")),
        ("Dr. Kim Min-Jun IV", "Min-Jun Kim IV", ("given", "surname", "given", "suffix")),
    ],
)
def test_routed_source_relabels_only_cleaned_name_occurrences(
    detector: ChineseNameDetector,
    raw_name: str,
    expected_text: str,
    expected_source_order: tuple[str, ...],
) -> None:
    canonical = detector.normalize_person_name(raw_name)

    assert canonical is not None
    assert canonical.source_text == raw_name
    assert canonical.text == expected_text
    assert detector._ordered_component_tokens(canonical.source) == raw_name.split()  # noqa: SLF001
    assert canonical.source.order == expected_source_order
    assert canonical.source.surname_tokens == ("Kim",)
    assert canonical.source.given_tokens[-1] == "Min-Jun"


def test_routed_source_splits_compact_native_occurrence_without_dropping_title(
    detector: ChineseNameDetector,
) -> None:
    raw_name = "Dr. \u5c71\u7530\u592a\u90ce"

    canonical = detector.normalize_person_name(raw_name)

    assert canonical is not None
    assert canonical.source_text == raw_name
    assert canonical.text == "\u592a\u90ce \u5c71\u7530"
    assert detector._ordered_component_tokens(canonical.source) == ["Dr.", "\u5c71\u7530", "\u592a\u90ce"]  # noqa: SLF001
    assert canonical.source.order == ("given", "surname", "given")
    assert canonical.source.given_tokens == ("Dr.", "\u592a\u90ce")
    assert canonical.source.surname_tokens == ("\u5c71\u7530",)


def test_east_asian_routing_preserves_normalized_and_source_suffixes(
    detector: ChineseNameDetector,
) -> None:
    raw = detector.normalize_person_name("Kim Min-Jun IV")
    scalar = detector.routing_scalar_resolution("Kim Min-Jun IV")
    structured = detector.normalize_person_name_components(
        first_name="Kim",
        last_name="Min-Jun",
        suffix="IV",
    )

    assert raw is not None
    assert raw.text == "Min-Jun Kim IV"
    assert (raw.normalized.given_name, raw.normalized.surname, raw.normalized.suffix) == (
        "Min-Jun",
        "Kim",
        "IV",
    )
    assert raw.source.suffix == "IV"
    assert raw.source.suffix_tokens == ("IV",)
    assert raw.source.order == ("surname", "given", "suffix")
    assert isinstance(scalar, CanonicalName)
    assert scalar == raw
    assert structured is not None
    assert structured.text == raw.text
    assert structured.normalized == raw.normalized
    assert structured.source.order == ("given", "surname", "suffix")


def test_vietnamese_requires_unicode_evidence_and_preserves_given_span(
    detector: ChineseNameDetector,
) -> None:
    # A diacritic is what identifies a bare surname match as Vietnamese, because most of the
    # surname list is short and doubles as Korean/Chinese/Western syllables. `Le Van Thanh` stays
    # given-first; the admitted exceptions are listed in
    # ASCII_ROUTABLE_VIETNAMESE_SURNAMES and have their own tests below.
    unicode_name = detector.normalize_person_name("Nguyễn Văn An")
    ascii_name = detector.normalize_person_name("Le Van Thanh")

    assert unicode_name is not None
    assert unicode_name.text == "An Văn Nguyễn"
    assert unicode_name.normalized.given_name == "An"
    assert unicode_name.normalized.middle_name == "Văn"
    assert unicode_name.normalized.surname == "Nguyễn"
    assert unicode_name.source.order == ("surname", "middle", "given")
    assert ascii_name is not None
    assert ascii_name.text == "Le Van Thanh"


def test_bare_ascii_admitted_vietnamese_surnames_route_family_first(
    detector: ChineseNameDetector,
) -> None:
    # Each admitted token is in no other lexicon, so it cannot preempt the Korean or Japanese
    # routes, and each is near-exclusively a surname. Blind labelling put the leading token as the
    # surname in 99.3-99.8% of sampled rows.
    for surface, surname, given, middle in (
        ("Nguyen Van Hieu", "Nguyen", "Hieu", "Van"),
        ("Nguyen Minh Duc", "Nguyen", "Duc", "Minh"),
        ("Nguyen Thi Oanh", "Nguyen", "Oanh", "Thi"),
        ("NGUYEN XUAN THO", "Nguyen", "Tho", "Xuan"),
        ("Tran Quoc Khanh", "Tran", "Khanh", "Quoc"),
        ("Tran Tinh Hien", "Tran", "Hien", "Tinh"),
        ("Pham Huu Tiep", "Pham", "Tiep", "Huu"),
        ("Pham Hai Yen", "Pham", "Yen", "Hai"),
    ):
        routed = detector.normalize_person_name(surface)
        assert routed is not None, surface
        assert routed.normalized.surname == surname, surface
        assert routed.normalized.given_name == given, surface
        assert routed.normalized.middle_name == middle, surface
        assert routed.source.order == ("surname", "middle", "given"), surface


def test_hoang_is_not_admitted_where_both_tokens_are_surnames(
    detector: ChineseNameDetector,
) -> None:
    # `hoang` clears the no-other-lexicon test but fails the surname-purity one in ONE shape: where
    # both tokens of a two-token name are Vietnamese surnames it is the GIVEN name two times in
    # three, so "Hoang Nguyen" and "Hoang Pham" would be flipped backwards. That shape stays
    # declined for every head admitted on the without-surname-partner list, which is what makes the
    # rest of the list safe to admit.
    for surface in ("Hoang Nguyen", "Hoang Pham", "Hoang Tran", "Vu Nguyen", "Ngo Le"):
        routed = detector.normalize_person_name(surface)
        assert routed is not None, surface
        assert routed.normalized.surname != surface.split()[0], surface


def test_unlisted_bare_ascii_vietnamese_surnames_still_need_a_diacritic(
    detector: ChineseNameDetector,
) -> None:
    # The short entries stay gated. `Mai`, `Le`, `Do`, `Kim`, `Ha`, `Ho` and `Ly` are ordinary given
    # names elsewhere, and `kim`/`ha`/`ho` are Korean surnames too — admitting them here would
    # preempt the Korean route, since Vietnamese is tried first. Relaxing the whole 46-entry list
    # would move 550,469 mentions, of which `kim` alone is 181,705 ("Kim Overvad", "Kim Krisberg").
    for surface in ("Le Van Thanh", "Do Van Hung", "Mai Smith", "Ly Van Nam", "Ha Van Tien"):
        routed = detector.normalize_person_name(surface)
        assert routed is not None, surface
        assert routed.source.order != ("surname", "middle", "given"), surface
        assert routed.normalized.surname != surface.split()[0], surface


def test_nguyen_relaxation_does_not_preempt_the_korean_route(
    detector: ChineseNameDetector,
) -> None:
    # Regression guard for the routing order: _infer_vietnamese runs before _infer_korean, so a
    # laxer Vietnamese gate could swallow names the Korean guards are meant to decide. These are
    # the exact cases those guards own.
    for surface, surname in (
        ("Kim Rudolph-Lund", "Rudolph-Lund"),
        ("Ha van den Hout", "van den Hout"),
        ("Ha Jae-Sung", "Ha"),
        ("Oh Young-Jin", "Oh"),
        ("Kim Ji Hoon", "Kim"),
    ):
        routed = detector.normalize_person_name(surface)
        assert routed is not None, surface
        assert routed.normalized.surname == surname, surface


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
    # (ML-Japanese gated + native-dictionary evidence). The affirmative Chinese
    # pipeline owns their canonical Romanization.
    for surface, expected in (
        ("王 伟", "Wei Wang"),
        ("李 明", "Ming Li"),
        ("陈 明", "Ming Chen"),
    ):
        result = detector.normalize_person_name(surface)
        assert result is not None
        assert result.text == expected
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
    cyrillic = detector.normalize_person_name("Nguyễn Lam Аnh")  # noqa: RUF001 - confusable fixture
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
    # (A) is gone. The pure-ASCII Western hyphenated partner used to abstain and swap here on
    # the grounds that no character rule separates it from a hyphenated Korean given name. Two
    # rules now separate most of it without a lexicon: a two-letter surname with nothing Korean
    # on the given side (labelled 34.3% wrong), and a given part longer than a romanized Korean
    # syllable (92% wrong). "Jo Leonardi-Bee", "Jo Rycroft-Malone" and "Kim Dam-Johansen" all
    # moved to test_two_letter_korean_surname_needs_a_korean_given_part and
    # test_overlong_given_part_is_a_western_surname_not_a_korean_syllable. What remains hard is
    # a short unlisted Korean syllable, still unfixed and still costing names like "Jo Hea-Soog".

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


def test_single_letter_leading_token_is_an_initial_not_a_korean_surname(
    detector: ChineseNameDetector,
) -> None:
    # "O" is the only single-letter Korean surname romanization, so every "O <hyphenated>"
    # row used to route as the surname 오 — but in bibliographic data a lone leading letter
    # is an initial, and these are Western names ("O Braun-Falco" is Otto Braun-Falco).
    for surface in (
        "O Braun-Falco",
        "O Guntinas-Lichius",
        "O Siggaard-Andersen",
        "O Lyon-Caen",
    ):
        routed = detector.normalize_person_name(surface)
        assert routed is not None, surface
        assert routed.normalized.surname != "O", surface
        assert routed.source.order != ("surname", "given"), surface


def test_multi_letter_korean_surnames_still_route_family_first(
    detector: ChineseNameDetector,
) -> None:
    # The single-letter refusal must not touch the ordinary Korean routes.
    for surface, surname in (
        ("Kim Dong-il", "Kim"),
        ("Park Chan-Wook", "Park"),
        ("Lee Sang-Ho", "Lee"),
        ("Oh Young-Jin", "Oh"),
        ("Ahn Chang-Jun", "Ahn"),
    ):
        routed = detector.normalize_person_name(surface)
        assert routed is not None, surface
        assert routed.normalized.surname == surname, surface
        assert routed.source.order == ("surname", "given"), surface


def test_two_letter_korean_surname_needs_a_korean_given_part(
    detector: ChineseNameDetector,
) -> None:
    # Two-letter Korean surnames double as Western given names, and the surname lexicon alone
    # cannot separate "Jo Leonardi-Bee" (Jo is English) from "Jo Jae-Yoon" (Jo is Korean). With
    # nothing Korean on the given side these are Western people, so the leading token stays given.
    for surface, surname in (
        ("Jo Leonardi-Bee", "Leonardi-Bee"),
        ("Jo Rycroft-Malone", "Rycroft-Malone"),
        ("An Dooms-Goossens", "Dooms-Goossens"),
        ("Yu Deuerling-Zheng", "Deuerling-Zheng"),
        ("Ra Sanchez-Gomez", "Sanchez-Gomez"),
        ("Na Rodriguez-Perez", "Rodriguez-Perez"),
    ):
        routed = detector.normalize_person_name(surface)
        assert routed is not None, surface
        assert routed.normalized.surname == surname, surface
        assert routed.source.order != ("surname", "given"), surface


def test_two_letter_korean_surname_routes_when_a_given_part_is_korean(
    detector: ChineseNameDetector,
) -> None:
    # One recognised Korean given syllable is enough corroboration to keep the family-first read.
    for surface, surname in (
        ("Ha Jae-Sung", "Ha"),
        ("Jo Jae-Yoon", "Jo"),
        ("Oh Kwang-Soo", "Oh"),
        ("Yi Seon-ung", "Yi"),
        ("Ji Won Suk", "Ji"),
        ("Ho Kyung Sung", "Ho"),
    ):
        routed = detector.normalize_person_name(surface)
        assert routed is not None, surface
        assert routed.normalized.surname == surname, surface
        assert routed.source.order[0] == "surname", surface


def test_overlong_given_part_is_a_western_surname_not_a_korean_syllable(
    detector: ChineseNameDetector,
) -> None:
    # Romanized Korean syllables top out near six characters, so a longer hyphen part is a
    # Western surname element and the leading Korean-surname homograph is really a given name.
    # This reaches the lengths the two-letter rule cannot ("Kim", "Lee", "Hwang").
    for surface, surname in (
        ("Kim Rudolph-Lund", "Rudolph-Lund"),
        ("Lee Gillespie-White", "Gillespie-White"),
        ("Lee Laurent-Applegate", "Laurent-Applegate"),
        ("Kim Theilgaard-Monch", "Theilgaard-Monch"),
        ("Kim Padgett-Clarke", "Padgett-Clarke"),
        ("Min Chen-Gaddini", "Chen-Gaddini"),
        ("Kim Dam-Johansen", "Dam-Johansen"),
    ):
        routed = detector.normalize_person_name(surface)
        assert routed is not None, surface
        assert routed.normalized.surname == surname, surface
        assert routed.source.order != ("surname", "given"), surface


def test_unlisted_korean_syllables_still_route_family_first(
    detector: ChineseNameDetector,
) -> None:
    # None of these has a single part in the syllable lexicon — "jeong", "sook", "kyoung",
    # "myong", "byeong" are all missing from it. They keep routing because every part is within
    # syllable length, which is the whole point of testing length rather than lexicon membership:
    # it protects the Korean names the incomplete lexicon cannot vouch for.
    for surface, surname in (
        ("Kim Kyoung-Duck", "Kim"),
        ("Lee Byeong-Do", "Lee"),
        ("Han Myong-Sook", "Han"),
        ("Hwang Jenn-Kang", "Hwang"),
        ("Park Jeong-sook", "Park"),
    ):
        routed = detector.normalize_person_name(surface)
        assert routed is not None, surface
        assert routed.normalized.surname == surname, surface
        assert routed.source.order[0] == "surname", surface


def test_spaced_kanji_routes_on_one_sided_dictionary_evidence(
    detector: ChineseNameDetector,
) -> None:
    # The surname asset holds 2,000 entries against 69,002 given names, so most real spaced
    # kanji names match on exactly one side. Requiring both left 615,837 names / 3.74M occ
    # reordered wrongly; blind judges called that class family-first 187/187.
    for surface, surname in (
        ("三浦 耕吉郎", "三浦"),  # leading token is a known surname, 耕吉郎 unknown
        ("小川 福次郎", "小川"),
        ("山瀬 豊", "山瀬"),  # trailing token is a known given name, 山瀬 unknown
        ("松中 成浩", "松中"),
        ("梅川 尚嗣", "梅川"),
    ):
        routed = detector.normalize_person_name(surface)
        assert routed is not None, surface
        assert routed.normalized.surname == surname, surface
        assert routed.source.order == ("surname", "given"), surface


def test_spaced_kanji_abstains_when_the_evidence_is_two_sided(
    detector: ChineseNameDetector,
) -> None:
    # Only ONE two-sided shape is real ambiguity: reverse-plausible, a known given name in
    # front of a known surname. That is positive evidence of an inverted byline, and blind
    # labelling agrees on 150 of 150. Names with no evidence on either side keep the
    # given-first default too, because it is right on ~75% of that class.
    for surface, surname in (
        ("優 佐藤", "佐藤"),  # reverse plausible: given + surname
        ("和則 西﨑", "西﨑"),
        ("吉行 水畑", "水畑"),  # 吉行 is a known given name too, so this is reverse-plausible
        ("歩 中野渡", "中野渡"),  # neither token is in either asset
        ("ジョン スミス", "スミス"),  # katakana Western name
    ):
        routed = detector.normalize_person_name(surface)
        assert routed is not None, surface
        assert routed.normalized.surname == surname, surface
        assert routed.source.order != ("surname", "given"), surface


def test_spaced_kanji_routes_when_both_tokens_are_given_plausible(
    detector: ChineseNameDetector,
) -> None:
    # Both tokens in the given-name asset used to abstain, on the theory that two signals
    # cancel. They do not: the given asset is 35x the surname asset, so a leading token that
    # appears in both is usually a surname the given list also happens to carry. Blind
    # labelling puts the family name first on 91.6% of this class's occ (297 PPS-sampled
    # names) and 84.0% by name (486 names, two independent rounds).
    for surface, surname, given in (
        ("智幸 小枝", "智幸", "小枝"),
        ("秋光 純", "秋光", "純"),
        ("弥永 真生", "弥永", "真生"),
        ("江里 健輔", "江里", "健輔"),
        ("霞 三郎", "霞", "三郎"),
    ):
        routed = detector.normalize_person_name(surface)
        assert routed is not None, surface
        assert routed.normalized.surname == surname, surface
        assert routed.normalized.given_name == given, surface
        assert routed.source.order == ("surname", "given"), surface


def test_spaced_kanji_routes_when_both_tokens_are_surnames(
    detector: ChineseNameDetector,
) -> None:
    # Same correction for the both-surname class: the trailing token is normally a given name
    # that the 2,000-entry surname list happens to list as well (穂積 Hozumi, 末広 Suehiro,
    # 真木 Maki, 牧 Maki all behave that way). Blind labelling puts the family name first on
    # 89.3% of this class's occ and 93.9% by name; 58% of the whole class carries a label.
    for surface, surname, given in (
        ("北口 末広", "北口", "末広"),
        ("田中 穂積", "田中", "穂積"),
        ("三橋 牧", "三橋", "牧"),
        ("内藤 林", "内藤", "林"),
        ("中田 真木", "中田", "真木"),
    ):
        routed = detector.normalize_person_name(surface)
        assert routed is not None, surface
        assert routed.normalized.surname == surname, surface
        assert routed.normalized.given_name == given, surface
        assert routed.source.order == ("surname", "given"), surface


def test_hangul_compound_surnames_take_the_second_syllable(
    detector: ChineseNameDetector,
) -> None:
    # The seven two-syllable Korean surnames: the default 1+2 split lands inside the surname, so
    # 남궁원 shipped as surname 남 (a different, much commoner surname) with given 궁원.
    for surface, surname, given in (
        ("남궁원", "남궁", "원"),
        ("황보관", "황보", "관"),
        ("제갈돈", "제갈", "돈"),
        ("사공준", "사공", "준"),
        ("선우영", "선우", "영"),
        ("서문희", "서문", "희"),
        ("독고석", "독고", "석"),
    ):
        routed = detector.normalize_person_name(surface)
        assert routed is not None, surface
        assert routed.normalized.surname == surname, surface
        assert routed.normalized.given_name == given, surface
        assert routed.source.order == ("surname", "given"), surface


def test_hangul_single_syllable_surnames_are_unchanged(detector: ChineseNameDetector) -> None:
    # 강원실 has a compound-looking head (강원 is a province) but 강 is the surname, and 남 / 황 / 서 /
    # 선 are commoner surnames than the compounds that start with them, so only an exact match moves.
    for surface, surname in (
        ("김민수", "김"),
        ("이상훈", "이"),
        ("강원실", "강"),
        ("남기웅", "남"),
        ("황영조", "황"),
        ("서정원", "서"),
        ("선동열", "선"),
    ):
        routed = detector.normalize_person_name(surface)
        assert routed is not None, surface
        assert routed.normalized.surname == surname, surface


def test_bare_ascii_vietnamese_routes_outside_the_two_surname_shape(
    detector: ChineseNameDetector,
) -> None:
    # Distinctive heads route family-first, so "Van Minh" and "Huu Tai" stop being surnames.
    for surface, surname, given in (
        ("Vu Van Quang", "Vu", "Quang"),
        ("Hoang Van Minh", "Hoang", "Minh"),
        ("Phan Van Kiem", "Phan", "Kiem"),
        ("Bui Huu Tai", "Bui", "Tai"),
        ("Huynh Quang Huy", "Huynh", "Huy"),
    ):
        routed = detector.normalize_person_name(surface)
        assert routed is not None, surface
        assert routed.normalized.surname == surname, surface
        assert routed.normalized.given_name == given, surface


def test_bare_ascii_vietnamese_declines_a_pair_of_surnames_and_foreign_homographs(
    detector: ChineseNameDetector,
) -> None:
    # A trailing Vietnamese surname marks an inverted byline at any length, so the shape is declined
    # whether or not a middle token sits between the two surnames; the excluded heads stay excluded
    # because they are ordinary Korean surnames, Japanese given names or Western names.
    for surface, surname in (
        ("Vu Nguyen", "Nguyen"),
        ("Hoang Pham", "Pham"),
        ("Truong Khang Nguyen", "Nguyen"),
        ("Hoang Xuan Tran", "Tran"),
        ("Dinh Chau Phan", "Phan"),
        ("Kim Overvad", "Overvad"),
        ("Le Corbusier", "Corbusier"),
        ("Ho Jin Kim", "Kim"),
        ("Mai Sato", "Sato"),
    ):
        routed = detector.normalize_person_name(surface)
        assert routed is not None, surface
        assert routed.normalized.surname == surname, surface


def test_bare_ascii_vietnamese_declines_a_hyphenated_trailing_surname(
    detector: ChineseNameDetector,
) -> None:
    # The inverted-byline marker survives hyphenation: the trailing token's first half is the family
    # name, so the whole compound is, and blind labelling put the family name last on 8 of 8 such
    # rows. A hyphen whose first half is not a surname is an ordinary given name.
    for surface, surname in (
        ("Thuong Le-Tien", "Le-Tien"),
        ("Vu Thuy Khanh Le-Trilling", "Le-Trilling"),
        ("Truong Nguyen-Ba", "Nguyen-Ba"),
        ("Hoang Le-Huu", "Le-Huu"),
        ("Dinh Vo-Ngoc", "Vo-Ngoc"),
    ):
        routed = detector.normalize_person_name(surface)
        assert routed is not None, surface
        assert routed.normalized.surname == surname, surface

    given_first_hyphen = detector.normalize_person_name("Ngo Si-Huy")
    assert given_first_hyphen is not None
    assert given_first_hyphen.normalized.surname == "Ngo"


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("Nguyen Van Nhi Tran", ("Nguyen", "Van Nhi", "Tran")),
        ("Tran Duc Le", ("Tran", "Duc", "Le")),
    ],
)
def test_exact_vietnamese_given_first_surfaces_keep_endpoint_roles(
    detector: ChineseNameDetector,
    raw_name: str,
    expected: tuple[str, str, str],
) -> None:
    canonical = detector.normalize_person_name(raw_name)

    assert canonical is not None
    assert (
        canonical.normalized.given_name,
        canonical.normalized.middle_name,
        canonical.normalized.surname,
    ) == expected


@pytest.mark.parametrize(
    ("raw_name", "expected", "source_order"),
    [
        (
            "Ming Hsien Ou Yang",
            ("Ming", "Hsien", "Ou Yang"),
            ("given", "middle", "surname", "surname"),
        ),
        (
            "Shiu Lun Au Yeung",
            ("Shiu", "Lun", "Au Yeung"),
            ("given", "middle", "surname", "surname"),
        ),
        ("Thuong Le Thi", ("Thuong", "Thi", "Le"), ("given", "surname", "middle")),
        ("Trinh Nguyen Duy", ("Duy", "Trinh", "Nguyen"), ("middle", "surname", "given")),
    ],
)
def test_identity_backed_exact_surfaces_assign_all_roles(
    detector: ChineseNameDetector,
    raw_name: str,
    expected: tuple[str, str, str],
    source_order: tuple[str, ...],
) -> None:
    canonical = detector.normalize_person_name(raw_name)

    assert canonical is not None
    assert (
        canonical.normalized.given_name,
        canonical.normalized.middle_name,
        canonical.normalized.surname,
    ) == expected
    assert canonical.source.order == source_order


def test_exact_vietnamese_surface_matching_remains_accent_sensitive(
    detector: ChineseNameDetector,
) -> None:
    canonical = detector.normalize_person_name("Bùi Hoàng Thảo Trân")

    assert canonical is not None
    assert canonical.normalized.surname == "Bùi"
