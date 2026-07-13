"""Regression tests for narrow cross-cultural Chinese classification evidence."""

import pytest

from sinonym import ChineseNameDetector
from sinonym.services.person_name_normalization import DropReason, PersonNameNormalizationService, PersonNameOutcome


@pytest.fixture(scope="module")
def detector() -> ChineseNameDetector:
    """Return one initialized detector for the evidence-gate cases."""
    return ChineseNameDetector()


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("Jun-Fun Horng", "Jun-Fun Horng"),
        ("Shugi Hsien", "Shu-Gi Hsien"),
        ("Zhou Df", "Df Zhou"),
    ],
)
def test_reviewed_surname_alias_and_compact_initial_evidence_accepts_chinese(
    detector: ChineseNameDetector,
    raw_name: str,
    expected: str,
) -> None:
    result = detector.normalize_name(raw_name)

    assert result.success
    assert result.result == expected


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("Yi-xin Han", "Yi-Xin Han"),
        ("Peng-qian Han", "Peng-Qian Han"),
    ],
)
def test_broad_korean_shape_is_not_used(
    detector: ChineseNameDetector,
    raw_name: str,
    expected: str,
) -> None:
    result = detector.normalize_name(raw_name)

    assert result.success
    assert result.result == expected


@pytest.mark.parametrize(
    "raw_name",
    [
        "In-sun Yu",
        "Jin Uk Ha",
        "Ok Jeung",
    ],
)
def test_directional_korean_surname_given_evidence_precedes_chinese_shortcuts(
    detector: ChineseNameDetector,
    raw_name: str,
) -> None:
    result = detector.normalize_name(raw_name)

    assert not result.success
    assert result.error_message == "Korean structural patterns detected"


@pytest.mark.parametrize(
    "raw_name",
    [
        "Aimin Yang",
        "Bing Han",
        "Boyi Kang",
        "Chen-Yu Lee",
        "Chia-Yuan Chang",
        "Ching-Wen Yang",
        "Chien-Yao Wang",
    ],
)
def test_directional_korean_gate_preserves_reviewed_chinese_controls(
    detector: ChineseNameDetector,
    raw_name: str,
) -> None:
    assert detector.normalize_name(raw_name).success


@pytest.mark.parametrize("raw_name", ["Zhang Thi", "Wang Thi", "Thi Zhang", "Thi Wang"])
def test_vietnamese_thi_veto_precedes_chinese_surname_evidence(
    detector: ChineseNameDetector,
    raw_name: str,
) -> None:
    result = detector.normalize_name(raw_name)

    assert not result.success
    assert result.error_message == "appears to be Vietnamese name"


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("Jungting Yu", "Jung-Ting Yu"),
        ("Tsung-Jr Chen", "Tsung-Jr Chen"),
    ],
)
def test_contextual_taiwan_romanization_is_admitted_and_preserved(
    detector: ChineseNameDetector,
    raw_name: str,
    expected: str,
) -> None:
    result = detector.normalize_name(raw_name)

    assert result.success
    assert result.result == expected
    assert result.parsed_original_order is not None
    assert result.parsed_original_order.order == ["given", "surname"]


@pytest.mark.parametrize(
    "raw_name",
    [
        "Jungting Kim",
        "Tsung-Jr Kim",
        "Jungting Nguyen",
        "Tsung-Jr Tran",
        "Robert Chen Jr",
        "Jung Kim",
        "Wade Wang",
        "Sina Huang",
    ],
)
def test_contextual_taiwan_gate_rejects_collision_controls(
    detector: ChineseNameDetector,
    raw_name: str,
) -> None:
    assert not detector.normalize_name(raw_name).success


@pytest.mark.parametrize("raw_name", ["Jung Yu", "J.-W. Koo"])
def test_bare_jung_and_initials_are_not_contextual_taiwan_evidence(
    detector: ChineseNameDetector,
    raw_name: str,
) -> None:
    normalized = detector._normalizer.apply(raw_name)  # noqa: SLF001

    assert detector._ethnicity_service.contextual_taiwan_given_parts(normalized.roman_tokens) is None  # noqa: SLF001


def test_ordinary_short_non_chinese_name_is_not_treated_as_initial_bundle(
    detector: ChineseNameDetector,
) -> None:
    assert not detector.normalize_name("Sun Kim").success


def test_leading_et_al_contamination_is_removed_before_chinese_classification(
    detector: ChineseNameDetector,
) -> None:
    result = detector.normalize_name("Et al. Biquan Mo")

    assert result.success
    assert result.result == "Bi-Quan Mo"
    assert result.parsed is not None
    assert (result.parsed.given_name, result.parsed.surname) == ("Bi-Quan", "Mo")


def test_leading_et_al_contamination_has_dropped_token_audit() -> None:
    normalized = PersonNameNormalizationService().normalize_text("Et al. Biquan Mo")
    structured = PersonNameNormalizationService().normalize_components(
        first_name="Et",
        middle_name="al.",
        last_name="Biquan Mo",
    )

    assert normalized.outcome is PersonNameOutcome.PERSON
    assert normalized.canonical_name is not None
    assert normalized.canonical_name.text == "Biquan Mo"
    assert [(item.text, item.source_role, item.reason) for item in normalized.dropped_tokens] == [
        ("Et", "given", DropReason.CONNECTOR),
        ("al.", "given", DropReason.CONNECTOR),
    ]
    assert structured.outcome is PersonNameOutcome.PERSON
    assert structured.canonical_name is not None
    assert structured.canonical_name.text == "Biquan Mo"
    assert [(item.text, item.source_role, item.reason) for item in structured.dropped_tokens] == [
        ("Et", "given", DropReason.CONNECTOR),
        ("al.", "middle", DropReason.CONNECTOR),
    ]


@pytest.mark.parametrize(
    "raw_name",
    [
        "Etienne Al",
        "Et Al Biquan Mo",
        "Biquan et al. Mo",
        "Biquan Mo et al.",
        "Et Albright Mo",
    ],
)
def test_et_al_cleanup_requires_exact_leading_citation_prefix(
    detector: ChineseNameDetector,
    raw_name: str,
) -> None:
    normalized = PersonNameNormalizationService().normalize_text(raw_name)

    assert not normalized.dropped_tokens
    assert not detector.normalize_name(raw_name).success


@pytest.mark.parametrize(
    ("raw_name", "expected", "expected_surname"),
    [
        ("Peter Ch'en", "Peter Ch'en", "Ch'en"),
        ("Ch'en Peter", "Peter Ch'en", "Ch'en"),
        ("Peter Ts'ai", "Peter Ts'ai", "Ts'ai"),
    ],
)
def test_wade_giles_apostrophe_surname_is_decisive_chinese_evidence(
    detector: ChineseNameDetector,
    raw_name: str,
    expected: str,
    expected_surname: str,
) -> None:
    result = detector.normalize_name(raw_name)

    assert result.success
    assert result.result == expected
    assert result.parsed is not None
    assert result.parsed.given_name == "Peter"
    assert result.parsed.surname == expected_surname


@pytest.mark.parametrize(
    "raw_name",
    [
        "Peter O'Brien",
        "D'Angelo Marie",
        "Peter Ch'energy",
        "Ch'energy Peter",
        "Peter Ch'en-Smith",
        "O'Ch'en Peter",
    ],
)
def test_apostrophe_surname_evidence_requires_exact_wade_giles_token(
    detector: ChineseNameDetector,
    raw_name: str,
) -> None:
    assert not detector.normalize_name(raw_name).success
