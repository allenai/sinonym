from sinonym.coretypes import ParseResult


def actual_result_text(result: ParseResult) -> str:
    """Return the comparable value from a parse result."""
    return result.result if result.success else result.error_message or ""


def assert_normalized_name(detector, raw_name: str, expected: tuple[bool, str]) -> None:
    """Assert one detector normalization case."""
    expected_success, expected_text = expected
    result = detector.normalize_name(raw_name)
    actual_text = actual_result_text(result)

    assert result.success is expected_success, (
        f"{raw_name!r}: expected success={expected_success}, got success={result.success}, value={actual_text!r}"
    )
    if expected_success:
        assert result.result == expected_text, f"{raw_name!r}: expected normalized name {expected_text!r}, got {actual_text!r}"


def assert_rejected(detector, raw_name: str) -> None:
    """Assert that one input is rejected as non-Chinese."""
    result = detector.normalize_name(raw_name)

    assert not result.success, f"{raw_name!r}: expected rejection, got {result.result!r}"


def assert_middle_name_result(result: ParseResult, raw_name: str, expected: dict) -> None:
    """Assert formatted and parsed middle-name fields for one parse result."""
    assert result.success, f"{raw_name!r}: expected success, got error {result.error_message!r}"
    assert result.result == expected["formatted"], (
        f"{raw_name!r}: expected formatted name {expected['formatted']!r}, got {result.result!r}"
    )

    parsed = result.parsed
    assert parsed is not None, f"{raw_name!r}: expected parsed name details"
    assert parsed.given_tokens == expected["given_tokens"], (
        f"{raw_name!r}: expected given_tokens {expected['given_tokens']!r}, got {parsed.given_tokens!r}"
    )
    assert parsed.middle_tokens == expected["middle_tokens"], (
        f"{raw_name!r}: expected middle_tokens {expected['middle_tokens']!r}, got {parsed.middle_tokens!r}"
    )
    assert parsed.surname == expected["surname"], (
        f"{raw_name!r}: expected surname {expected['surname']!r}, got {parsed.surname!r}"
    )
