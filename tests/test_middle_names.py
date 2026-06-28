"""
Middle Name (Initial) Tests

Validates that trailing single-letter initials in Chinese names are:
- Rendered as a separate middle component in the final string
- Exposed as separate middle_tokens in ParsedName
- Not merged into the last given token

Covers both individual and batch processing paths and integrates with the
JUnit failure counting consumed by scripts/check_test_status.py.
"""

import pytest

from tests._case_assertions import assert_middle_name_result

MIDDLE_NAME_INDIVIDUAL_CASES = [
    (
        "Chi-Ying F. Huang",
        {
            "formatted": "Chi-Ying F Huang",
            "given_tokens": ["Chi", "Ying"],
            "middle_tokens": ["F"],
            "surname": "Huang",
        },
    ),
    (
        "Chung C. Wang",
        {
            "formatted": "Chung C Wang",
            "given_tokens": ["Chung"],
            "middle_tokens": ["C"],
            "surname": "Wang",
        },
    ),
    (
        "Chung F. Wong",
        {
            "formatted": "Chung F Wong",
            "given_tokens": ["Chung"],
            "middle_tokens": ["F"],
            "surname": "Wong",
        },
    ),
    (
        "Chung-Chieng A. Lai",
        {
            "formatted": "Chung-Chieng A Lai",
            "given_tokens": ["Chung", "Chieng"],
            "middle_tokens": ["A"],
            "surname": "Lai",
        },
    ),
    (
        "Gui-Qiang G. Chen",
        {
            "formatted": "Gui-Qiang G Chen",
            "given_tokens": ["Gui", "Qiang"],
            "middle_tokens": ["G"],
            "surname": "Chen",
        },
    ),
    (
        "Chi-Ying F. K. Huang",
        {
            "formatted": "Chi-Ying F K Huang",
            "given_tokens": ["Chi", "Ying"],
            "middle_tokens": ["F", "K"],
            "surname": "Huang",
        },
    ),
    (
        "Chung-Chieng A. B. Lai",
        {
            "formatted": "Chung-Chieng A B Lai",
            "given_tokens": ["Chung", "Chieng"],
            "middle_tokens": ["A", "B"],
            "surname": "Lai",
        },
    ),
]


@pytest.fixture(scope="module")
def middle_name_batch(detector):
    names = [raw for raw, _ in MIDDLE_NAME_INDIVIDUAL_CASES]
    return detector.analyze_name_batch(names)


@pytest.mark.parametrize(("raw", "expected"), MIDDLE_NAME_INDIVIDUAL_CASES)
def test_middle_name_individual(detector, raw, expected):
    assert_middle_name_result(detector.normalize_name(raw), raw, expected)


@pytest.mark.parametrize(
    ("index", "raw", "expected"),
    [(index, raw, expected) for index, (raw, expected) in enumerate(MIDDLE_NAME_INDIVIDUAL_CASES)],
)
def test_middle_name_batch(middle_name_batch, index, raw, expected):
    assert_middle_name_result(middle_name_batch.results[index], raw, expected)


# Additional mixed-script / all-Han with Roman middle initial cases
MIDDLE_NAME_MIXED_CASES = [
    (
        "李 伟 F.",
        {
            "formatted": "Wei F Li",
            "given_tokens": ["Wei"],
            "middle_tokens": ["F"],
            "surname": "Li",
        },
    ),
    (
        "李 小明 G.",
        {
            "formatted": "Xiao-Ming G Li",
            "given_tokens": ["Xiao", "Ming"],
            "middle_tokens": ["G"],
            "surname": "Li",
        },
    ),
    (
        "Zhang 伟 F.",
        {
            "formatted": "Zhang F Wei",
            "given_tokens": ["Zhang"],
            "middle_tokens": ["F"],
            "surname": "Wei",
        },
    ),
    (
        "Li 小明 H.",
        {
            "formatted": "Ming-Li H Xiao",
            "given_tokens": ["Ming", "Li"],
            "middle_tokens": ["H"],
            "surname": "Xiao",
        },
    ),
    (
        "李 小明 H. K.",
        {
            "formatted": "Xiao-Ming H K Li",
            "given_tokens": ["Xiao", "Ming"],
            "middle_tokens": ["H", "K"],
            "surname": "Li",
        },
    ),
    (
        "Zhang 伟 F. G.",
        {
            "formatted": "Zhang F G Wei",
            "given_tokens": ["Zhang"],
            "middle_tokens": ["F", "G"],
            "surname": "Wei",
        },
    ),
]


@pytest.fixture(scope="module")
def middle_name_mixed_batch(detector):
    names = [raw for raw, _ in MIDDLE_NAME_MIXED_CASES]
    return detector.analyze_name_batch(names)


@pytest.mark.parametrize(("raw", "expected"), MIDDLE_NAME_MIXED_CASES)
def test_middle_name_mixed_individual(detector, raw, expected):
    assert_middle_name_result(detector.normalize_name(raw), raw, expected)


@pytest.mark.parametrize(
    ("index", "raw", "expected"),
    [(index, raw, expected) for index, (raw, expected) in enumerate(MIDDLE_NAME_MIXED_CASES)],
)
def test_middle_name_mixed_batch(middle_name_mixed_batch, index, raw, expected):
    assert_middle_name_result(middle_name_mixed_batch.results[index], raw, expected)


def test_middle_initial_leading_between_surname_and_given(detector):
    """Ensure a leading single-letter initial in given tokens is treated as middle and original order reflects positions."""
    raw = "Li A. Wei"
    res = detector.normalize_name(raw)

    assert res.success, f"Expected success, got error: {res.error_message}"

    # Final formatting should place middle initial between given and surname
    assert res.result == "Wei A Li"

    # Parsed normalized output order
    assert res.parsed is not None
    assert res.parsed.given_tokens == ["Wei"]
    assert res.parsed.middle_tokens == ["A"]
    assert res.parsed.surname == "Li"

    # Original-order view: preserves component labels and annotates original sequence
    por = res.parsed_original_order
    assert por is not None
    assert por.order == ["surname", "middle", "given"]
    assert por.given_name == "Wei"
    assert por.surname == "Li"
    assert por.middle_tokens == ["A"]


def test_middle_initial_trailing_after_given_preserves_original_order(detector):
    raw = "Li Wei A."
    res = detector.normalize_name(raw)

    assert res.success, f"Expected success, got error: {res.error_message}"
    assert res.result == "Wei A Li"
    assert res.parsed_original_order is not None
    assert res.parsed_original_order.order == ["surname", "given", "middle"]


def test_middle_initial_trailing_batch_preserves_original_order(detector):
    names = ["Li Wei A.", "Zhang Ming F."]
    batch = detector.analyze_name_batch(names)

    assert [result.result for result in batch.results] == ["Wei A Li", "Ming F Zhang"]
    assert [result.parsed_original_order.order for result in batch.results] == [
        ["surname", "given", "middle"],
        ["surname", "given", "middle"],
    ]
