"""Cross-API routing contracts for the culture-specific initial policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from sinonym.timo.interface import Instance, Predictor, SourceAuthorFields

if TYPE_CHECKING:
    from sinonym.coretypes import CanonicalName


def _canonical_components(canonical: CanonicalName | None) -> tuple[str, str, str]:
    """Return normalized person fields from a successful canonical result."""
    assert canonical is not None
    normalized = canonical.normalized
    return normalized.given_name, normalized.middle_name, normalized.surname


@pytest.mark.parametrize(
    "raw_name",
    ["A. S. Lee", "A.S.Lee", "A S Lee", "A. S. Lim", "A. S. Tan"],
)
def test_ambiguous_initial_only_surnames_use_generic_canonical(detector, raw_name: str) -> None:
    """Initial punctuation alone must not establish the Chinese path."""
    legacy = detector.normalize_name(raw_name)
    canonical = detector.normalize_person_name(raw_name)

    assert not legacy.success
    components = _canonical_components(canonical)
    assert components[:2] == ("A.", "S.")
    assert components[2] in {"Lee", "Lim", "Tan"}


@pytest.mark.parametrize(
    ("raw_name", "given", "surname"),
    [("L Han", "L.", "Han"), ("A. Lee", "A.", "Lee")],
)
def test_single_initial_with_ambiguous_surname_also_uses_generic_canonical(
    detector,
    raw_name: str,
    given: str,
    surname: str,
) -> None:
    """One initial is no more culturally identifying than several initials."""
    legacy = detector.normalize_name(raw_name)
    canonical = detector.normalize_person_name(raw_name)

    assert not legacy.success
    assert _canonical_components(canonical) == (given, "", surname)


def test_compact_letters_do_not_create_chinese_evidence_for_ambiguous_surname(detector) -> None:
    """The Chinese compact-bundle recognizer requires a non-ambiguous surname."""
    legacy = detector.normalize_name("BC Lee")
    canonical = detector.normalize_person_name("BC Lee")

    assert not legacy.success
    assert _canonical_components(canonical) == ("BC", "", "Lee")


@pytest.mark.parametrize(
    ("first_name", "last_name", "expected"),
    [
        ("A.S.", "Lee", ("A.", "S.", "Lee")),
        ("A S", "Lim", ("A.", "S.", "Lim")),
        ("A. S.", "Wang", ("A.-S.", "", "Wang")),
    ],
)
def test_structured_canonical_routing_matches_raw_policy(
    detector,
    first_name: str,
    last_name: str,
    expected: tuple[str, str, str],
) -> None:
    """Structured input must not change the culture gate or component policy."""
    canonical = detector.normalize_person_name_components(first_name=first_name, last_name=last_name)

    assert _canonical_components(canonical) == expected


def test_native_alignment_overrides_ambiguous_roman_surname(detector) -> None:
    """Aligned Han evidence is stronger than the Roman-only Lee fallback."""
    result = detector.normalize_name("Lee \u674e Wei \u4f1f A \u963f")

    assert result.success
    assert result.parsed is not None
    assert (result.parsed.given_name, result.parsed.middle_name, result.parsed.surname) == (
        "Wei-A",
        "",
        "Lee",
    )


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("\u7d14 \u91ce\u3005\u5d0e", ("\u7d14", "", "\u91ce\u3005\u5d0e")),
        ("Shiu Lun Au Yeung", ("Shiu", "Lun", "Au Yeung")),
    ],
)
def test_hard_identity_evidence_precedes_heuristic_chinese_canonical(
    detector,
    raw_name: str,
    expected: tuple[str, str, str],
) -> None:
    """Japanese iteration marks and reviewed exact roles outrank name shape."""
    canonical = detector.normalize_person_name(raw_name)

    assert _canonical_components(canonical) == expected


@pytest.mark.parametrize(
    ("components", "expected"),
    [
        (
            {"first_name": "\u7d14", "last_name": "\u91ce\u3005\u5d0e"},
            ("\u7d14", "", "\u91ce\u3005\u5d0e"),
        ),
        (
            {"first_name": "Shiu", "middle_name": "Lun", "last_name": "Au Yeung"},
            ("Shiu", "Lun", "Au Yeung"),
        ),
    ],
)
def test_structured_hard_identity_evidence_precedes_heuristic_chinese_canonical(
    detector,
    components: dict[str, str],
    expected: tuple[str, str, str],
) -> None:
    """Structured canonicalization uses the same evidence precedence."""
    canonical = detector.normalize_person_name_components(**components)

    assert _canonical_components(canonical) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (SourceAuthorFields(first_name="L", last_name="Han"), ("L.", "", "Han")),
        (SourceAuthorFields(first_name="BC", last_name="Lee"), ("BC", "", "Lee")),
        (SourceAuthorFields(first_name="A. S.", last_name="Lee"), ("A.", "S.", "Lee")),
        (SourceAuthorFields(first_name="A S", last_name="Lee"), ("A.", "S.", "Lee")),
        (SourceAuthorFields(first_name="A. S.", last_name="Wang"), ("A.-S.", "", "Wang")),
        (SourceAuthorFields(first_name="A S", last_name="Wang"), ("A.-S.", "", "Wang")),
        (SourceAuthorFields(first_name="Wei M.", last_name="Wang"), ("Wei", "M.", "Wang")),
    ],
)
def test_terminal_fields_use_the_same_policy(
    predictor: Predictor,
    source: SourceAuthorFields,
    expected: tuple[str, str, str],
) -> None:
    """TIMO writes canonical semantic fields after its terminal decision."""
    (paper,) = predictor.predict_batch([Instance(pp_authors=[source])])
    resolved = paper.authors[0]

    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == expected
