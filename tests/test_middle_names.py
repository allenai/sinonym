"""
Middle Name (Initial) Tests

Validates that trailing single-letter initials in Chinese names are:
- Rendered as a separate middle component in the final string
- Exposed as separate middle_tokens in ParsedName
- Not merged into the last given token

Covers both individual and batch processing paths and integrates with the
shared failure log consumed by scripts/check_test_status.py.
"""

import sys
from pathlib import Path

# Ensure package import works when running tests directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from sinonym import ChineseNameDetector
from tests._fail_log import log_failure

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


def test_middle_name_individual(detector):

    passed = 0
    failed = 0

    for raw, exp in MIDDLE_NAME_INDIVIDUAL_CASES:
        res = detector.normalize_name(raw)

        if not res.success:
            failed += 1
            actual = res.error_message
            log_failure("Middle name individual (formatted)", raw, True, exp["formatted"], res.success, actual)
            continue

        # 1) Check formatted string
        if res.result != exp["formatted"]:
            failed += 1
            log_failure("Middle name individual (formatted)", raw, True, exp["formatted"], res.success, res.result)
        else:
            passed += 1

        # 2) Check parsed tokens and components
        parsed = res.parsed
        if not parsed:
            failed += 1
            log_failure("Middle name individual (parsed missing)", raw, True, "parsed present", res.success, "parsed missing")
            continue

        # given_tokens should not include the middle initial(s)
        if parsed.given_tokens != exp["given_tokens"]:
            failed += 1
            log_failure(
                "Middle name individual (given_tokens)",
                raw,
                True,
                str(exp["given_tokens"]),
                res.success,
                str(parsed.given_tokens),
            )

        # middle_tokens should contain the single-letter initial(s)
        if parsed.middle_tokens != exp["middle_tokens"]:
            failed += 1
            log_failure(
                "Middle name individual (middle_tokens)",
                raw,
                True,
                str(exp["middle_tokens"]),
                res.success,
                str(parsed.middle_tokens),
            )

        # Surname string should match expected
        if parsed.surname != exp["surname"]:
            failed += 1
            log_failure(
                "Middle name individual (surname)",
                raw,
                True,
                exp["surname"],
                res.success,
                parsed.surname,
            )

    if failed:
        print(f"Middle name individual tests: {failed} failures out of {len(MIDDLE_NAME_INDIVIDUAL_CASES)} tests")
    assert failed == 0, f"Middle name individual tests: {failed} failures out of {len(MIDDLE_NAME_INDIVIDUAL_CASES)} tests"
    print(f"Middle name individual tests: {passed} passed, {failed} failed")


def test_middle_name_batch(detector):

    names = [raw for raw, _ in MIDDLE_NAME_INDIVIDUAL_CASES]
    expected = [exp for _, exp in MIDDLE_NAME_INDIVIDUAL_CASES]

    passed = 0
    failed = 0

    batch = detector.analyze_name_batch(names)

    # Validate each result
    for i, (raw, exp) in enumerate(MIDDLE_NAME_INDIVIDUAL_CASES):
        res = batch.results[i]

        if not res.success:
            failed += 1
            actual = res.error_message
            log_failure("Middle name batch (formatted)", raw, True, exp["formatted"], res.success, actual)
            continue

        # 1) Formatted string
        if res.result != exp["formatted"]:
            failed += 1
            log_failure("Middle name batch (formatted)", raw, True, exp["formatted"], res.success, res.result)
        else:
            passed += 1

        # 2) Parsed tokens
        parsed = res.parsed
        if not parsed:
            failed += 1
            log_failure("Middle name batch (parsed missing)", raw, True, "parsed present", res.success, "parsed missing")
            continue

        if parsed.given_tokens != exp["given_tokens"]:
            failed += 1
            log_failure(
                "Middle name batch (given_tokens)",
                raw,
                True,
                str(exp["given_tokens"]),
                res.success,
                str(parsed.given_tokens),
            )

        if parsed.middle_tokens != exp["middle_tokens"]:
            failed += 1
            log_failure(
                "Middle name batch (middle_tokens)",
                raw,
                True,
                str(exp["middle_tokens"]),
                res.success,
                str(parsed.middle_tokens),
            )

        if parsed.surname != exp["surname"]:
            failed += 1
            log_failure(
                "Middle name batch (surname)",
                raw,
                True,
                exp["surname"],
                res.success,
                parsed.surname,
            )

    if failed:
        print(f"Middle name batch tests: {failed} failures out of {len(MIDDLE_NAME_INDIVIDUAL_CASES)} tests")
    assert failed == 0, f"Middle name batch tests: {failed} failures out of {len(MIDDLE_NAME_INDIVIDUAL_CASES)} tests"
    print(f"Middle name batch tests: {passed} passed, {failed} failed")


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


def test_middle_name_mixed_individual_and_batch(detector):

    # Individual
    ind_failed = 0
    for raw, exp in MIDDLE_NAME_MIXED_CASES:
        res = detector.normalize_name(raw)
        if not res.success:
            ind_failed += 1
            log_failure("Middle name mixed individual (formatted)", raw, True, exp["formatted"], res.success, res.error_message)
            continue
        if res.result != exp["formatted"]:
            ind_failed += 1
            log_failure("Middle name mixed individual (formatted)", raw, True, exp["formatted"], res.success, res.result)
            continue
        parsed = res.parsed
        if not parsed:
            ind_failed += 1
            log_failure("Middle name mixed individual (parsed missing)", raw, True, "parsed present", res.success, "parsed missing")
            continue
        if parsed.given_tokens != exp["given_tokens"]:
            ind_failed += 1
            log_failure("Middle name mixed individual (given_tokens)", raw, True, str(exp["given_tokens"]), res.success, str(parsed.given_tokens))
        if parsed.middle_tokens != exp["middle_tokens"]:
            ind_failed += 1
            log_failure("Middle name mixed individual (middle_tokens)", raw, True, str(exp["middle_tokens"]), res.success, str(parsed.middle_tokens))
        if parsed.surname != exp["surname"]:
            ind_failed += 1
            log_failure("Middle name mixed individual (surname)", raw, True, exp["surname"], res.success, parsed.surname)

    assert ind_failed == 0, f"Middle name mixed individual tests: {ind_failed} failures out of {len(MIDDLE_NAME_MIXED_CASES)} tests"

    # Batch
    names = [raw for raw, _ in MIDDLE_NAME_MIXED_CASES]
    expected = [exp for _, exp in MIDDLE_NAME_MIXED_CASES]
    batch = detector.analyze_name_batch(names)

    batch_failed = 0
    for i, (raw, exp) in enumerate(MIDDLE_NAME_MIXED_CASES):
        res = batch.results[i]
        if not res.success:
            batch_failed += 1
            log_failure("Middle name mixed batch (formatted)", raw, True, exp["formatted"], res.success, res.error_message)
            continue
        if res.result != exp["formatted"]:
            batch_failed += 1
            log_failure("Middle name mixed batch (formatted)", raw, True, exp["formatted"], res.success, res.result)
            continue
        parsed = res.parsed
        if not parsed:
            batch_failed += 1
            log_failure("Middle name mixed batch (parsed missing)", raw, True, "parsed present", res.success, "parsed missing")
            continue
        if parsed.given_tokens != exp["given_tokens"]:
            batch_failed += 1
            log_failure("Middle name mixed batch (given_tokens)", raw, True, str(exp["given_tokens"]), res.success, str(parsed.given_tokens))
        if parsed.middle_tokens != exp["middle_tokens"]:
            batch_failed += 1
            log_failure("Middle name mixed batch (middle_tokens)", raw, True, str(exp["middle_tokens"]), res.success, str(parsed.middle_tokens))
        if parsed.surname != exp["surname"]:
            batch_failed += 1
            log_failure("Middle name mixed batch (surname)", raw, True, exp["surname"], res.success, parsed.surname)

    assert batch_failed == 0, f"Middle name mixed batch tests: {batch_failed} failures out of {len(MIDDLE_NAME_MIXED_CASES)} tests"


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
