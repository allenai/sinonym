"""Regression tests for scalar clean source-surname repartition.

The source-only adjudication and 500-activation blind evaluation are recorded
in ``docs/scalar_source_surname_repartition_evaluation.md``. The routed V3
parity fixture separately pins the known ``production-248247190`` regression
as shipped behavior, not a desired boundary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from sinonym.coretypes import NameComponents
from sinonym.coretypes.routing_resolution import (
    ResolutionAction,
    ResolutionProvenance,
    ResolutionReason,
)
from sinonym.timo.routing_v3 import (
    RoutingInstanceV3,
    RoutingV3Resolver,
    SourceAuthorFields,
)

if TYPE_CHECKING:
    from sinonym import ChineseNameDetector
    from sinonym.timo.interface import RoutingPredictorV3

_REPARTITION_REASON = ResolutionReason.SCALAR_CLEAN_SOURCE_SURNAME_REPARTITION_ASSIGNMENT
_KNOWN_COMPOUND_REASON = ResolutionReason.SCALAR_KNOWN_COMPOUND_SURNAME_PRESERVE_INPUT

# (audit_id, source first/middle/last, expected first/middle/last) drawn from the
# locked holdout; every row is a single-author paper, so routing one author
# reproduces the production decision exactly.
_HOLDOUT_ACTIVATIONS = [
    ("237787103:0", ("Ana", "", "Vicens Poveda"), ("Ana", "", "Vicens Poveda")),
    ("243539382:0", ("Ana", "Paula", "Nunes Chaves"), ("Ana", "Paula", "Nunes Chaves")),
    ("246540162:0", ("Eduardo", "", "Souza de Cursi"), ("Eduardo", "", "Souza de Cursi")),
    ("273169208:0", ("Verónica", "", "Paz de Brenes"), ("Verónica", "", "Paz de Brenes")),
    ("179254808:0", ("Miguel", "Ángel", "Martínez Lago"), ("Miguel", "Ángel", "Martínez Lago")),
    ("69832191:0", ("Christian", "", "Tutivén Gálvez"), ("Christian", "", "Tutivén Gálvez")),
    ("227163470:0", ("Yasser", "", "Abdel Kerim"), ("Yasser", "", "Abdel Kerim")),
    ("270187072:0", ("José", "David", "López Blanco"), ("José", "David", "López Blanco")),
    ("272670292:0", ("Elena", "", "Dalla Vecchia"), ("Elena", "", "Dalla Vecchia")),
    ("277448397:0", ("Matthew", "", "Muscat Inglott"), ("Matthew", "", "Muscat Inglott")),
    (
        "270734600:0",
        ("Francisco", "Prancacio", "Araujo de Carvalho"),
        ("Francisco", "Prancacio", "Araujo de Carvalho"),
    ),
    ("254313213:0", ("Jill", "", "Coster van Voorhout"), ("Jill", "", "Coster van Voorhout")),
]


@pytest.fixture(scope="module")
def predictor(routing_predictor_v3: RoutingPredictorV3) -> RoutingPredictorV3:
    """Alias the shared session predictor under this module's name."""
    return routing_predictor_v3


@pytest.fixture(scope="module")
def resolver(detector: ChineseNameDetector) -> RoutingV3Resolver:
    """Build a resolver for direct predicate tests."""
    return RoutingV3Resolver(detector)


def _route(
    predictor: RoutingPredictorV3,
    source: SourceAuthorFields,
    *,
    vys_other_names: list[str] | None = None,
):
    (paper,) = predictor.predict_batch(
        [RoutingInstanceV3(pp_authors=[source], vys_other_names=vys_other_names)],
    )
    return paper.authors[0].resolved_fields


def _candidate(
    resolver: RoutingV3Resolver,
    source: SourceAuthorFields,
    selected: NameComponents,
) -> NameComponents | None:
    return resolver._scalar_clean_source_surname_repartition_candidate(source, selected)  # noqa: SLF001


@pytest.mark.parametrize(
    ("audit_id", "source_fields", "expected"),
    _HOLDOUT_ACTIVATIONS,
    ids=[row[0] for row in _HOLDOUT_ACTIVATIONS],
)
def test_holdout_activations_keep_the_structured_source_surname(
    predictor: RoutingPredictorV3,
    audit_id: str,
    source_fields: tuple[str, str, str],
    expected: tuple[str, str, str],
) -> None:
    first_name, middle_names, last_name = source_fields

    resolved = _route(
        predictor,
        SourceAuthorFields(first_name=first_name, middle_names=middle_names, last_name=last_name),
    )

    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == expected, audit_id
    assert resolved.resolution_reason is _REPARTITION_REASON
    assert resolved.resolution_provenance is ResolutionProvenance.SOURCE
    assert resolved.resolution_action is ResolutionAction.ASSIGN


def test_single_token_source_surname_cannot_activate(predictor: RoutingPredictorV3) -> None:
    resolved = _route(predictor, SourceAuthorFields(first_name="Ana", last_name="Poveda"))

    assert resolved.resolution_reason is not _REPARTITION_REASON


def test_initial_in_the_source_first_name_blocks_activation(predictor: RoutingPredictorV3) -> None:
    resolved = _route(predictor, SourceAuthorFields(first_name="A.", last_name="Vicens Poveda"))

    assert resolved.resolution_reason is not _REPARTITION_REASON


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            SourceAuthorFields(first_name="Ka", middle_names="Ming", last_name="Au Yeung"),
            ("Ka", "Ming", "Au Yeung"),
        ),
        (
            SourceAuthorFields(first_name="Ka Ming", last_name="Au Yeung"),
            ("Ka Ming", "", "Au Yeung"),
        ),
        (
            SourceAuthorFields(first_name="Wei", middle_names="Ming", last_name="Ou Yang"),
            ("Wei", "Ming", "Ou Yang"),
        ),
        (
            SourceAuthorFields(first_name="Li", middle_names="Ming", last_name="Zhu Ge"),
            ("Li", "Ming", "Zhu Ge"),
        ),
    ],
)
@pytest.mark.parametrize(
    "vys_other_names",
    [pytest.param(None, id="pp-only"), pytest.param(["Jane Doe"], id="pp-vys")],
)
def test_curated_spaced_compound_surname_outlives_batch_materialization(
    predictor: RoutingPredictorV3,
    source: SourceAuthorFields,
    expected: tuple[str, str, str],
    vys_other_names: list[str] | None,
) -> None:
    resolved = _route(predictor, source, vys_other_names=vys_other_names)

    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == expected
    assert resolved.resolution_reason is _KNOWN_COMPOUND_REASON
    assert resolved.resolution_provenance is ResolutionProvenance.SOURCE
    assert resolved.resolution_action is ResolutionAction.PRESERVE_INPUT


@pytest.mark.parametrize(
    "vys_other_names",
    [pytest.param(None, id="pp-only"), pytest.param(["Jane Doe"], id="pp-vys")],
)
def test_curated_compound_guard_does_not_trust_an_unreviewed_last_name(
    predictor: RoutingPredictorV3,
    vys_other_names: list[str] | None,
) -> None:
    source = SourceAuthorFields(first_name="Au", middle_names="Yeung", last_name="Ka Ming")

    resolved = _route(predictor, source, vys_other_names=vys_other_names)

    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == ("Ka-Ming", "", "Au Yeung")
    assert resolved.resolution_reason is not _KNOWN_COMPOUND_REASON


def test_predicate_accepts_a_clean_peel(resolver: RoutingV3Resolver) -> None:
    candidate = _candidate(
        resolver,
        SourceAuthorFields(first_name="Ana", last_name="Vicens Poveda"),
        NameComponents(given_name="Ana", middle_name="Vicens", surname="Poveda"),
    )

    assert candidate is not None
    assert (candidate.given_name, candidate.middle_name, candidate.surname) == ("Ana", "", "Vicens Poveda")


@pytest.mark.parametrize(
    ("source", "selected"),
    [
        pytest.param(
            SourceAuthorFields(first_name="A.", last_name="Vicens Poveda"),
            NameComponents(given_name="A.", middle_name="Vicens", surname="Poveda"),
            id="initial_in_source_first_name",
        ),
        pytest.param(
            SourceAuthorFields(first_name="Ana", last_name="M. Vicens Poveda"),
            NameComponents(given_name="Ana", middle_name="M. Vicens", surname="Poveda"),
            id="initial_in_peeled_prefix",
        ),
        pytest.param(
            SourceAuthorFields(first_name="Ana", last_name="Poveda"),
            NameComponents(given_name="Ana", middle_name="", surname="Poveda"),
            id="single_token_source_surname",
        ),
        pytest.param(
            SourceAuthorFields(first_name="Ana", last_name="Vicens Poveda"),
            NameComponents(given_name="Anna", middle_name="Vicens", surname="Poveda"),
            id="given_name_drift",
        ),
        pytest.param(
            SourceAuthorFields(first_name="Ana", last_name="Vicens Poveda"),
            NameComponents(given_name="Ana", middle_name="Vicens", surname="Poveda", suffix="Jr."),
            id="suffix_drift",
        ),
        pytest.param(
            SourceAuthorFields(first_name="Ana", last_name="Vicens Poveda"),
            NameComponents(given_name="Ana", middle_name="Poveda", surname="Vicens"),
            id="scalar_kept_the_wrong_surname_token",
        ),
        pytest.param(
            SourceAuthorFields(first_name="Ana", last_name="Vicens Poveda"),
            NameComponents(given_name="Ana", middle_name="", surname="Vicens Poveda"),
            id="scalar_did_not_repartition",
        ),
        pytest.param(
            SourceAuthorFields(first_name="Ana", last_name="Vicens Poveda"),
            NameComponents(given_name="Ana", middle_name="Reyes", surname="Poveda"),
            id="middle_name_is_not_the_peeled_prefix",
        ),
    ],
)
def test_predicate_rejects_unclean_repartitions(
    resolver: RoutingV3Resolver,
    source: SourceAuthorFields,
    selected: NameComponents,
) -> None:
    assert _candidate(resolver, source, selected) is None
