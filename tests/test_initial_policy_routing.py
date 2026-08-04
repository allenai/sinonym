"""Cross-API routing contracts for the culture-specific initial policy."""

from __future__ import annotations

import pytest

from sinonym.timo.interface import PredictorConfig, RoutingPredictorV2, RoutingPredictorV3
from sinonym.timo.routing_v3 import RoutingInstanceV3, SourceAuthorFields


@pytest.fixture(scope="module")
def routed_v2() -> RoutingPredictorV2:
    """Return one deterministic V2 router."""
    return RoutingPredictorV2(PredictorConfig(parallel="never"), ".")


@pytest.fixture(scope="module")
def routed_v3(routing_predictor_v3: RoutingPredictorV3) -> RoutingPredictorV3:
    """Alias the shared session predictor under this module's name."""
    return routing_predictor_v3


@pytest.mark.parametrize(
    "raw_name",
    [
        "A. S. Lee",
        "A.S.Lee",
        "A S Lee",
        "A. S. Lim",
        "A. S. Tan",
    ],
)
def test_ambiguous_initial_only_surnames_use_generic_canonical(detector, raw_name: str) -> None:
    """Initial punctuation alone must not establish the Chinese path."""
    legacy = detector.normalize_name(raw_name)
    canonical = detector.normalize_person_name(raw_name)

    assert not legacy.success
    assert canonical is not None
    assert canonical.normalized.given_name == "A."
    assert canonical.normalized.middle_name == "S."
    assert canonical.normalized.surname in {"Lee", "Lim", "Tan"}


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
    assert canonical is not None
    assert canonical.normalized.given_name == given
    assert canonical.normalized.middle_name == ""
    assert canonical.normalized.surname == surname


def test_compact_letters_do_not_create_chinese_evidence_for_ambiguous_surname(detector) -> None:
    """The Chinese compact-bundle recognizer requires a non-ambiguous surname."""
    legacy = detector.normalize_name("BC Lee")
    canonical = detector.normalize_person_name("BC Lee")

    assert not legacy.success
    assert canonical is not None
    assert (canonical.normalized.given_name, canonical.normalized.middle_name, canonical.normalized.surname) == (
        "BC",
        "",
        "Lee",
    )


@pytest.mark.parametrize("raw_name", ["A. S. Wang", "A.S.Wang", "A S Wang"])
def test_affirmative_initial_only_chinese_variants_use_compound_given(detector, raw_name: str) -> None:
    """A dominant Chinese surname supplies evidence punctuation cannot."""
    legacy = detector.normalize_name(raw_name)
    canonical = detector.normalize_person_name(raw_name)

    assert legacy.success
    assert canonical is not None
    assert canonical.normalized.given_name == "A.-S."
    assert canonical.normalized.middle_name == ""
    assert canonical.normalized.surname == "Wang"


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

    assert canonical is not None
    normalized = canonical.normalized
    assert (normalized.given_name, normalized.middle_name, normalized.surname) == expected


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

    assert canonical is not None
    normalized = canonical.normalized
    assert (normalized.given_name, normalized.middle_name, normalized.surname) == expected


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

    assert canonical is not None
    normalized = canonical.normalized
    assert (normalized.given_name, normalized.middle_name, normalized.surname) == expected


@pytest.mark.parametrize(
    ("raw_name", "success", "expected"),
    [
        ("L Han", False, ("L.", "", "Han")),
        ("BC Lee", False, ("BC", "", "Lee")),
        ("A. S. Lee", False, ("A.", "S.", "Lee")),
        ("A. S. Wang", True, ("A.-S.", "", "Wang")),
        ("Wei M. Wang", True, ("Wei", "M.", "Wang")),
    ],
)
def test_v2_candidate_canonical_uses_the_same_policy(
    routed_v2: RoutingPredictorV2,
    raw_name: str,
    success: bool,
    expected: tuple[str, str, str],
) -> None:
    """V2 may decline legacy parsing, but its canonical candidate stays usable."""
    (result,) = routed_v2.route_pp([raw_name])

    assert result.success is success
    assert result.canonical_name is not None
    normalized = result.canonical_name.normalized
    assert (normalized.given_name, normalized.middle_name, normalized.surname) == expected


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
def test_v3_terminal_fields_use_the_same_policy(
    routed_v3: RoutingPredictorV3,
    source: SourceAuthorFields,
    expected: tuple[str, str, str],
) -> None:
    """V3 writes canonical semantic fields after its terminal decision."""
    (paper,) = routed_v3.predict_batch([RoutingInstanceV3(pp_authors=[source])])
    resolved = paper.authors[0].resolved_fields

    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == expected
