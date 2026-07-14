"""Tests for canonical-name result metadata."""

from dataclasses import FrozenInstanceError

import pytest

from sinonym.coretypes import CanonicalName, NameComponents, ParsedName, ParseResult


def _canonical_name(text: str = "Steve Marsh Blando IV") -> CanonicalName:
    source = NameComponents(
        given_name="dr steve",
        middle_name="marsh",
        surname="blando",
        suffix="IV",
        given_tokens=("dr", "steve"),
        middle_tokens=("marsh",),
        surname_tokens=("blando",),
        suffix_tokens=("IV",),
        order=("given", "given", "middle", "surname", "suffix"),
    )
    normalized = NameComponents(
        given_name="Steve",
        middle_name="Marsh",
        surname="Blando",
        suffix="IV",
        given_tokens=("Steve",),
        middle_tokens=("Marsh",),
        surname_tokens=("Blando",),
        suffix_tokens=("IV",),
        order=("given", "middle", "surname", "suffix"),
    )
    return CanonicalName(source_text="dr steve marsh blando IV", text=text, source=source, normalized=normalized)


def test_canonical_name_and_components_are_deeply_immutable() -> None:
    canonical_name = _canonical_name()
    text_attribute = "text"
    surname_attribute = "surname"
    append_method = "append"

    with pytest.raises(FrozenInstanceError):
        setattr(canonical_name, text_attribute, "Changed")
    with pytest.raises(FrozenInstanceError):
        setattr(canonical_name.normalized, surname_attribute, "Changed")
    with pytest.raises(AttributeError):
        getattr(canonical_name.normalized.surname_tokens, append_method)("Changed")


def test_parse_result_positional_constructor_remains_compatible() -> None:
    parsed = ParsedName("Zhang", "Wei", ["Zhang"], ["Wei"])

    result = ParseResult(True, "Wei Zhang", None, None, parsed, parsed)

    assert result.parsed is parsed
    assert result.parsed_original_order is parsed
    assert result.canonical_name is None


def test_map_preserves_canonical_name_on_success_and_callback_failure() -> None:
    canonical_name = _canonical_name()
    result = ParseResult.success_with_name("Steve Marsh Blando IV", canonical_name=canonical_name)

    mapped = result.map(str.upper)
    failed = result.map(lambda _value: 1 / 0)

    assert mapped.result == "STEVE MARSH BLANDO IV"
    assert mapped.canonical_name is canonical_name
    assert not failed.success
    assert failed.canonical_name is canonical_name


def test_flat_map_inherits_canonical_name_without_replacing_callback_metadata() -> None:
    canonical_name = _canonical_name()
    replacement = _canonical_name("Stephen Blando")
    result = ParseResult.success_with_name("Steve Marsh Blando IV", canonical_name=canonical_name)

    inherited = result.flat_map(lambda _value: ParseResult.success_with_name("Steve Blando"))
    inherited_failure = result.flat_map(lambda _value: ParseResult.failure("not normalized"))
    replaced = result.flat_map(
        lambda _value: ParseResult.success_with_name("Stephen Blando", canonical_name=replacement),
    )

    assert inherited.canonical_name is canonical_name
    assert inherited_failure.canonical_name is canonical_name
    assert replaced.canonical_name is replacement
