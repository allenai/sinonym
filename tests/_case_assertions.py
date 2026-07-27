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


def person_normalized_text(person) -> str:
    """Return the space-joined normalized components of an all-person parse."""
    parts = (person.normalized.given_name, person.normalized.middle_name, person.normalized.surname)
    return " ".join(part for part in parts if part)


def assert_person_normalized_name(person, raw_name: str, expected: str) -> None:
    """Assert the normalized full text of an all-person parse."""
    assert person is not None, f"{raw_name!r}: expected a person parse, got None"
    actual = person_normalized_text(person)
    assert actual == expected, f"{raw_name!r}: expected normalized name {expected!r}, got {actual!r}"


def routed_author_text(author) -> str:
    """Return the space-joined routed components of one routed author."""
    parts = (author.given_name, author.middle_name, author.surname)
    return " ".join(part for part in parts if part)


def assert_routed_rejection(author, raw_name: str) -> None:
    """Assert that one routed input is rejected rather than parsed as Chinese."""
    actual = routed_author_text(author)
    assert not author.success, f"{raw_name!r}: expected rejection, got {actual!r}"


def assert_no_canonical_name(author, raw_name: str) -> None:
    """Assert that one routed input exposes no canonical name."""
    canonical = author.canonical_name
    if canonical is None:
        actual = ""
    else:
        parts = (canonical.normalized.given_name, canonical.normalized.middle_name, canonical.normalized.surname)
        actual = " ".join(part for part in parts if part)
    assert canonical is None, f"{raw_name!r}: expected no canonical name, got {actual!r}"
