"""
Name Formatting Test Suite

This module contains tests for various name formatting patterns including:
- Hyphenated names
- Comma-separated format ("Last, First")
- Names with periods/dots
- Whitespace handling
- Different capitalization patterns
"""

import pytest

from sinonym.coretypes import NameFormat, ParsedName, ParseResult
from tests._case_assertions import assert_normalized_name

# Test cases for name formatting and separators
CHINESE_NAME_TEST_CASES = [
    ("  Zhang  ,  Wei  ", (True, "Wei Zhang")),
    (". X.F.Han", (False, "initial-only name has an ambiguous cross-cultural surname")),
    ("A. I. Lee", (False, "initial-only name has an ambiguous cross-cultural surname")),
    ("Ch'en Wei", (True, "Wei Ch'en")),
    ("Chan Tai-Man", (True, "Tai-Man Chan")),
    ("Chen J.-M.", (True, "J.-M. Chen")),
    ("Chen,Mei Ling", (True, "Mei-Ling Chen")),
    ("D. W. Wang", (True, "D.-W. Wang")),
    ("Dan-dan Zhang", (True, "Dan-Dan Zhang")),
    ("JinHua", (True, "Hua Jin")),
    ("L. Han", (False, "initial-only name has an ambiguous cross-cultural surname")),
    ("LI Xiao-juan", (True, "Xiao-Juan Li")),
    ("Li.Wei.Zhang", (True, "Li-Wei Zhang")),
    ("Liu X.Y.", (True, "X.-Y. Liu")),
    ("Liu, Xiao-ming", (True, "Xiao-Ming Liu")),
    ("LuWANG", (True, "Lu Wang")),
    ("Min-Hung Lee", (True, "Min-Hung Lee")),
    ("OuMing", (True, "Ming Ou")),
    ("P.Y. Huang", (True, "P.-Y. Huang")),
    ("R. Han", (False, "initial-only name has an ambiguous cross-cultural surname")),
    ("Ts'ao Ming", (True, "Ming Ts'ao")),
    ("Wang B.", (True, "B. Wang")),
    ("Wei,   Yu-Zhong", (True, "Yu-Zhong Wei")),
    ("Wei, Yu-Zhong", (True, "Yu-Zhong Wei")),
    ("Wu M.J.", (True, "M.-J. Wu")),
    ("Wu,Yu Fei", (True, "Yu-Fei Wu")),
    ("X. -F. Han", (False, "initial-only name has an ambiguous cross-cultural surname")),
    ("X. F. Han", (False, "initial-only name has an ambiguous cross-cultural surname")),
    ("X. Han", (False, "initial-only name has an ambiguous cross-cultural surname")),
    ("X.-H. Li", (True, "X.-H. Li")),
    ("XIAO-JUAN LI", (True, "Xiao-Juan Li")),
    ("XIAOChen", (True, "Xiao Chen")),
    ("Xiao Ming-hui Li", (True, "Xiao-Ming-Hui Li")),
    ("Y. Z. Wei", (True, "Y.-Z. Wei")),
    ("Yuan, Li-Ming", (True, "Li-Ming Yuan")),
    ("YuanLi", (True, "Yuan Li")),
    ("Zeng, Wei", (True, "Wei Zeng")),
    ("ZengWei", (True, "Wei Zeng")),
    ("Zhang W.", (True, "W. Zhang")),
    (". X.F.Han", (False, "initial-only name has an ambiguous cross-cultural surname")),
    ("A. I. Lee", (False, "initial-only name has an ambiguous cross-cultural surname")),
    ("Au-Yeung Ka-Ming", (True, "Ka-Ming Au-Yeung")),
    ("Au-Yeung, Ka-Ming", (True, "Ka-Ming Au-Yeung")),
    ("Chan Tai-Man", (True, "Tai-Man Chan")),
    ("Chan, Tai Man", (True, "Tai-Man Chan")),
    ("Chen, Mei Ling", (True, "Mei-Ling Chen")),
    ("Chen, Yu", (True, "Yu Chen")),
    ("Chen-Hung Huang", (True, "Chen-Hung Huang")),
    ("Cheng-Hung Huang", (True, "Cheng-Hung Huang")),
    ("Chia-Ming Chang", (True, "Chia-Ming Chang")),
    ("Chine-Feng Wu", (True, "Chine-Feng Wu")),
    ("Choi, Suk-Zan", (True, "Suk-Zan Choi")),
    ("D. W. Wang", (True, "D.-W. Wang")),
    ("Dan-Dan Zhang", (True, "Dan-Dan Zhang")),
    ("Dan-dan Zhang", (True, "Dan-Dan Zhang")),
    ("He Jian-guo", (True, "Jian-Guo He")),
    ("L. Han", (False, "initial-only name has an ambiguous cross-cultural surname")),
    ("LI Xiao-juan", (True, "Xiao-Juan Li")),
    ("Li Xiao-Juan", (True, "Xiao-Juan Li")),
    ("Li Xiao-juan", (True, "Xiao-Juan Li")),
    ("Li.Wei.Zhang", (True, "Li-Wei Zhang")),
    ("Liu Zhi-guo", (True, "Zhi-Guo Liu")),
    ("Liu, Dehua", (True, "De-Hua Liu")),
    ("Ouyang, Xiaoming", (True, "Xiao-Ming Ouyang")),
    ("P.Y. Huang", (True, "P.-Y. Huang")),
    ("R. Han", (False, "initial-only name has an ambiguous cross-cultural surname")),
    ("Shi-Juan Li", (True, "Shi-Juan Li")),
    ("Shu-Juan Li", (True, "Shu-Juan Li")),
    ("Wang B.", (True, "B. Wang")),
    ("Wang, Li Ming", (True, "Li-Ming Wang")),
    ("Wei Min Zhang", (True, "Wei-Min Zhang")),
    ("Wei,   Yu-Zhong", (True, "Yu-Zhong Wei")),
    ("Wei, Yu-Zhong", (True, "Yu-Zhong Wei")),
    ("Wong, Siu Ming", (True, "Siu-Ming Wong")),
    ("Wu, Yufei", (True, "Yu-Fei Wu")),
    ("X. -F. Han", (False, "initial-only name has an ambiguous cross-cultural surname")),
    ("X. F. Han", (False, "initial-only name has an ambiguous cross-cultural surname")),
    ("X. Han", (False, "initial-only name has an ambiguous cross-cultural surname")),
    ("X.-H. Li", (True, "X.-H. Li")),
    ("XIAO-JUAN LI", (True, "Xiao-Juan Li")),
    ("Xiao Juan Li", (True, "Xiao-Juan Li")),
    ("Xiao Ming-hui Li", (True, "Xiao-Ming-Hui Li")),
    ("Xiao-Hong Li", (True, "Xiao-Hong Li")),
    ("Xiao-Juan Li", (True, "Xiao-Juan Li")),
    ("Xiao-juan Li", (True, "Xiao-Juan Li")),
    ("Xiaohong Li", (True, "Xiao-Hong Li")),
    ("Y. Z. Wei", (True, "Y.-Z. Wei")),
    ("Yu Jian-guo", (True, "Jian-Guo Yu")),
    ("Yu Zhong Wei", (True, "Yu-Zhong Wei")),
    ("Yu-Zhong Wei", (True, "Yu-Zhong Wei")),
    ("Yu-zhong Wei", (True, "Yu-Zhong Wei")),
    ("YuZhong Wei", (True, "Yu-Zhong Wei")),
    ("Yuzhong Wei", (True, "Yu-Zhong Wei")),
    ("Zhang Hong-xin", (True, "Hong-Xin Zhang")),
    ("Zhang, Wei", (True, "Wei Zhang")),
    ("LinShu", (True, "Shu Lin")),
    ("Chen C", (True, "C. Chen")),
    ("Li A", (True, "A. Li")),
]


@pytest.mark.parametrize(("input_name", "expected"), CHINESE_NAME_TEST_CASES)
def test_name_formatting(detector, input_name, expected):
    """Test various name formatting patterns including hyphens, commas, periods."""
    assert_normalized_name(detector, input_name, expected)


@pytest.mark.parametrize(
    ("raw_name", "expected", "expected_given_token"),
    [
        ("Dr Li", "Dr Li", "Dr"),
        ("Li Dr", "Dr Li", "Dr"),
        ("Ms Wang", "Ms Wang", "Ms"),
        ("Wang Ms", "Ms Wang", "Ms"),
        ("Sr Li", "Sr Li", "Sr"),
        ("Li Sr", "Sr Li", "Sr"),
        ("Jr Li", "Jr Li", "Jr"),
        ("Li Jr", "Jr Li", "Jr"),
        ("PhD Li", "Phd Li", "Phd"),
        ("Li PhD", "Phd Li", "Phd"),
        ("DNP Li", "Dnp Li", "Dnp"),
    ],
)
def test_compact_initial_candidate_keeps_reviewed_person_boundary_atomic(
    detector,
    raw_name: str,
    expected: str,
    expected_given_token: str,
) -> None:
    result = detector.normalize_name(raw_name)

    assert result.success
    assert result.result == expected
    assert result.parsed is not None
    assert result.parsed.given_tokens == [expected_given_token]


@pytest.mark.parametrize(
    ("raw_name", "expected", "expected_tokens"),
    [
        ("MS Li", "M.-S. Li", ["M.", "S."]),
        ("Li MS", "M.-S. Li", ["M.", "S."]),
        ("Md Li", "M.-D. Li", ["M.", "D."]),
        ("Li Md", "M.-D. Li", ["M.", "D."]),
        ("BC Wang", "B.-C. Wang", ["B.", "C."]),
        ("Wang BC", "B.-C. Wang", ["B.", "C."]),
        ("Zhou Df", "D.-F. Zhou", ["D.", "F."]),
    ],
)
def test_compact_initial_expansion_preserves_non_boundary_collisions(
    detector,
    raw_name: str,
    expected: str,
    expected_tokens: list[str],
) -> None:
    result = detector.normalize_name(raw_name)

    assert result.success
    assert result.result == expected
    assert result.parsed is not None
    assert result.parsed.given_tokens == expected_tokens


def test_compact_initial_boundary_policy_has_individual_batch_parity(detector) -> None:
    names = ["Dr Li", "Ms Wang", "Li Sr", "Li PhD", "BC Wang", "Zhou Df"]

    individual = [detector.normalize_name(name).result for name in names]
    batch = detector.analyze_name_batch(names)

    assert [result.result for result in batch.results] == individual


@pytest.mark.parametrize(
    ("selected_format", "expected_order"),
    [
        (NameFormat.GIVEN_FIRST, ["middle", "given", "middle", "surname"]),
        (NameFormat.SURNAME_FIRST, ["surname", "middle", "given", "middle"]),
        (NameFormat.MIXED, ["surname", "middle", "given", "middle"]),
    ],
)
def test_materialize_parse_result_preserves_selected_format_semantics(
    detector,
    selected_format: NameFormat,
    expected_order: list[str],
) -> None:
    detector._ensure_initialized()  # noqa: SLF001
    formatter = detector._formatting_service  # noqa: SLF001
    assert formatter is not None

    parsed = ParsedName(
        surname="Zhang",
        given_name="Wei",
        surname_tokens=["Zhang"],
        given_tokens=["Wei"],
        middle_name="A. K.",
        middle_tokens=["A.", "K."],
        order=["given", "middle", "surname"],
    )
    expected = ParseResult.success_with_name(
        "Wei A. K. Zhang",
        parsed=parsed,
        parsed_original_order=ParsedName(
            surname=parsed.surname,
            given_name=parsed.given_name,
            surname_tokens=parsed.surname_tokens,
            given_tokens=parsed.given_tokens,
            middle_name=parsed.middle_name,
            middle_tokens=parsed.middle_tokens,
            order=expected_order,
        ),
    )

    assert formatter.materialize_parse_result(
        ["Zhang"],
        ["A", "Wei", "K"],
        selected_format,
        {},
    ) == expected


def test_materialize_parse_result_preserves_compound_source_format(detector) -> None:
    detector._ensure_initialized()  # noqa: SLF001
    formatter = detector._formatting_service  # noqa: SLF001
    assert formatter is not None

    result = formatter.materialize_parse_result(
        ["Ou", "Yang"],
        ["Xiao", "Ming"],
        NameFormat.GIVEN_FIRST,
        {},
        original_compound_format="Ouyang",
    )

    assert result.success
    assert result.result == "Xiao-Ming Ouyang"
    assert result.original_compound_surname == "Ouyang"
    assert result.parsed is not None
    assert result.parsed.surname_tokens == ["Ou", "Yang"]
    assert result.parsed_original_order is not None
    assert result.parsed_original_order.order == ["given", "surname"]


def test_materialize_parse_result_converts_only_value_errors(detector, monkeypatch) -> None:
    detector._ensure_initialized()  # noqa: SLF001
    formatter = detector._formatting_service  # noqa: SLF001
    assert formatter is not None

    value_error_message = "invalid formatted components"

    def raise_value_error(*_args, **_kwargs):
        raise ValueError(value_error_message)

    monkeypatch.setattr(formatter, "format_name_output_with_tokens", raise_value_error)
    result = formatter.materialize_parse_result(["Zhang"], ["Wei"], NameFormat.SURNAME_FIRST)
    assert result == ParseResult.failure(value_error_message)

    runtime_error_message = "formatter unavailable"

    def raise_runtime_error(*_args, **_kwargs):
        raise RuntimeError(runtime_error_message)

    monkeypatch.setattr(formatter, "format_name_output_with_tokens", raise_runtime_error)
    with pytest.raises(RuntimeError, match=runtime_error_message):
        formatter.materialize_parse_result(["Zhang"], ["Wei"], NameFormat.SURNAME_FIRST)


def test_detector_formatting_converts_preprocessing_value_errors(detector, monkeypatch) -> None:
    detector._ensure_initialized()  # noqa: SLF001
    normalized_input = detector._normalizer.apply("Zhang Wei")  # noqa: SLF001
    error_message = "invalid native-token alignment"

    def raise_value_error(*_args, **_kwargs):
        raise ValueError(error_message)

    monkeypatch.setattr(detector, "_native_bound_given_tokens", raise_value_error)
    result = detector._format_parse_result(  # noqa: SLF001
        ["Zhang"],
        ["Wei"],
        normalized_input,
        ["surname", "given"],
    )

    assert result == ParseResult.failure(error_message)
