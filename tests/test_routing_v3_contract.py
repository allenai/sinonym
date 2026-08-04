"""Focused contract tests for routed V3 inputs and terminal decisions."""

import pytest
from pydantic import ValidationError

from sinonym.coretypes import CanonicalName, NameComponents
from sinonym.coretypes.routing_resolution import (
    EAST_ASIAN_EVIDENCE_RESOLUTION_REASONS,
    RESOLUTION_DECISION_TABLE,
    ApplyAssignment,
    EastAsianEvidenceReason,
    EvidenceFailure,
    PreserveBaseline,
    ResolutionAction,
    ResolutionProvenance,
    ResolutionReason,
)
from sinonym.timo.routing_v3 import (
    ResolvedAuthorFields,
    RoutingInstanceV3,
    SourceAuthorFields,
    merge_resolved_suffix,
    restore_reviewed_atomic_korean_tokens,
)


def test_v3_derives_focal_pool_positionally_without_a_second_authority() -> None:
    authors = [
        SourceAuthorFields(first_name="Juan", middle_names="Carlos", last_name="de la Cruz", suffix="Jr."),
        SourceAuthorFields(first_name="Juan Carlos", middle_names="de la", last_name="Cruz"),
    ]
    instance = RoutingInstanceV3(
        pp_authors=authors,
        vys_other_names=["Jane Doe", "Jane Doe"],
    )

    assert instance.pp_names == ["Juan Carlos de la Cruz", "Juan Carlos de la Cruz"]
    assert instance.vys_pool_names == [
        "Juan Carlos de la Cruz",
        "Juan Carlos de la Cruz",
        "Jane Doe",
        "Jane Doe",
    ]
    assert instance.pp_authors[0].last_name == "de la Cruz"
    assert instance.pp_authors[1].last_name == "Cruz"
    assert set(instance.dict()) == {"pp_authors", "vys_other_names"}


def test_v3_rejects_a_separately_supplied_focal_slice() -> None:
    with pytest.raises(ValidationError, match="extra fields not permitted"):
        RoutingInstanceV3(
            pp_authors=[SourceAuthorFields(first_name="Li", last_name="Wei")],
            pp_names=["Wei Li"],
        )


def test_v3_distinguishes_absent_vys_context_from_present_empty_others() -> None:
    authors = [SourceAuthorFields(first_name="Li", last_name="Wei")]
    present_empty = RoutingInstanceV3(pp_authors=authors, vys_other_names=[])

    assert RoutingInstanceV3(pp_authors=authors).vys_pool_names is None
    assert present_empty.vys_pool_names == ["Li Wei"]
    assert present_empty.dict()["vys_other_names"] == []
    assert RoutingInstanceV3.parse_raw(present_empty.json()) == present_empty


def test_v3_request_wire_omits_nulls_without_changing_parsed_defaults() -> None:
    instance = RoutingInstanceV3(
        pp_authors=[SourceAuthorFields(first_name="Li", last_name="Wei")],
    )

    assert instance.dict() == {"pp_authors": [{"first_name": "Li", "last_name": "Wei"}]}
    assert instance.json() == '{"pp_authors": [{"first_name": "Li", "last_name": "Wei"}]}'
    assert RoutingInstanceV3.parse_raw(instance.json()) == instance


def test_full_name_matches_current_flattening_and_excludes_suffix() -> None:
    source = SourceAuthorFields(
        first_name="  Nguyen",
        middle_names="Van   Nhi",
        last_name="Tran  ",
        suffix="III",
    )

    assert source.full_name() == "Nguyen Van   Nhi Tran"


@pytest.mark.parametrize(
    ("source", "selected", "expected"),
    [
        ("Jr.", "III", "Jr."),
        ("", "III", "III"),
        (None, "III", "III"),
        ("", None, ""),
        (None, "", None),
        ("   ", "III", "   "),
    ],
)
def test_suffix_policy_is_complete_and_fill_only(
    source: str | None,
    selected: str | None,
    expected: str | None,
) -> None:
    assert merge_resolved_suffix(source, selected) == expected


def test_every_resolution_reason_has_one_complete_decision_spec() -> None:
    expected = {
        ResolutionReason.MIXED_SCRIPT_SAFETY_SUPPRESSION: (
            ResolutionProvenance.SOURCE,
            ResolutionAction.PRESERVE_INPUT,
        ),
        ResolutionReason.ROUTED_CJK_SAFETY_SUPPRESSION: (
            ResolutionProvenance.SOURCE,
            ResolutionAction.PRESERVE_INPUT,
        ),
        ResolutionReason.HANDLED_EVIDENCE_FAILURE: (
            ResolutionProvenance.SOURCE,
            ResolutionAction.PRESERVE_INPUT,
        ),
        ResolutionReason.HARD_SCALAR_MATERIALIZATION_FAILED: (
            ResolutionProvenance.SOURCE,
            ResolutionAction.PRESERVE_INPUT,
        ),
        ResolutionReason.SCALAR_KNOWN_COMPOUND_SURNAME_PRESERVE_INPUT: (
            ResolutionProvenance.SOURCE,
            ResolutionAction.PRESERVE_INPUT,
        ),
        ResolutionReason.JAPANESE_ITERATION_MARK_ASSIGNMENT: (
            ResolutionProvenance.SCALAR,
            ResolutionAction.ASSIGN,
        ),
        ResolutionReason.IDENTITY_BACKED_EXACT_ASSIGNMENT: (
            ResolutionProvenance.SCALAR,
            ResolutionAction.ASSIGN,
        ),
        ResolutionReason.REVIEWED_EXACT_SOURCE_ASSIGNMENT: (
            ResolutionProvenance.SOURCE,
            ResolutionAction.ASSIGN,
        ),
        ResolutionReason.REVIEWED_SOURCE_PATTERN_ASSIGNMENT: (
            ResolutionProvenance.SOURCE,
            ResolutionAction.ASSIGN,
        ),
        ResolutionReason.KOREAN_WESTERN_CONFLICT_PRESERVE_INPUT: (
            ResolutionProvenance.SCALAR,
            ResolutionAction.PRESERVE_INPUT,
        ),
        ResolutionReason.JAPANESE_GIVEN_FIRST_REORDER_VETO_PRESERVE_INPUT: (
            ResolutionProvenance.SCALAR,
            ResolutionAction.PRESERVE_INPUT,
        ),
        ResolutionReason.VIETNAMESE_GIVEN_FIRST_REORDER_VETO_PRESERVE_INPUT: (
            ResolutionProvenance.SCALAR,
            ResolutionAction.PRESERVE_INPUT,
        ),
        ResolutionReason.INITIALS_COMMA_REORDER_VETO_PRESERVE_INPUT: (
            ResolutionProvenance.SOURCE,
            ResolutionAction.PRESERVE_INPUT,
        ),
        ResolutionReason.REVIEWED_EXACT_SOURCE_REORDER_VETO_PRESERVE_INPUT: (
            ResolutionProvenance.SOURCE,
            ResolutionAction.PRESERVE_INPUT,
        ),
        ResolutionReason.CONTEXT_SUPPORTED_REORDER_VETO_PRESERVE_INPUT: (
            ResolutionProvenance.SCALAR,
            ResolutionAction.PRESERVE_INPUT,
        ),
        ResolutionReason.PP_SELECTED: (ResolutionProvenance.PP, ResolutionAction.ASSIGN),
        ResolutionReason.VYS_SELECTED: (ResolutionProvenance.VYS, ResolutionAction.ASSIGN),
        ResolutionReason.PP_VYS_ABSTAIN_PP_INPUT: (
            ResolutionProvenance.PP,
            ResolutionAction.PRESERVE_INPUT,
        ),
        ResolutionReason.PP_VYS_ABSTAIN_VYS_INPUT: (
            ResolutionProvenance.VYS,
            ResolutionAction.PRESERVE_INPUT,
        ),
        ResolutionReason.PP_ONLY_ABSTAIN_INPUT: (
            ResolutionProvenance.PP,
            ResolutionAction.PRESERVE_INPUT,
        ),
        ResolutionReason.PP_ONLY_ABSTAIN_REVIEWED_COMPOUND_SURNAME: (
            ResolutionProvenance.SOURCE,
            ResolutionAction.PRESERVE_INPUT,
        ),
        ResolutionReason.BATCH_ABSTAIN_MATERIALIZATION_FAILED: (
            ResolutionProvenance.SOURCE,
            ResolutionAction.PRESERVE_INPUT,
        ),
        ResolutionReason.SCALAR_BASELINE: (ResolutionProvenance.SCALAR, ResolutionAction.ASSIGN),
        ResolutionReason.REVIEWED_NON_PERSON_PATTERN: (
            ResolutionProvenance.SOURCE,
            ResolutionAction.SUPPRESS,
        ),
        ResolutionReason.NON_PERSON_SOURCE_PASSTHROUGH: (
            ResolutionProvenance.SOURCE,
            ResolutionAction.PRESERVE_INPUT,
        ),
        ResolutionReason.NO_USABLE_SEMANTIC_RESULT: (
            ResolutionProvenance.SOURCE,
            ResolutionAction.PRESERVE_INPUT,
        ),
    }

    assert set(RESOLUTION_DECISION_TABLE) == set(ResolutionReason) == set(expected)
    assert {reason: (spec.provenance, spec.action) for reason, spec in RESOLUTION_DECISION_TABLE.items()} == expected
    assert all(isinstance(spec.provenance, ResolutionProvenance) for spec in RESOLUTION_DECISION_TABLE.values())
    assert all(isinstance(spec.action, ResolutionAction) for spec in RESOLUTION_DECISION_TABLE.values())


def test_every_east_asian_evidence_reason_has_one_hard_resolution_disposition() -> None:
    assert set(EAST_ASIAN_EVIDENCE_RESOLUTION_REASONS) == set(EastAsianEvidenceReason)
    assert {reason: resolution for reason, resolution in EAST_ASIAN_EVIDENCE_RESOLUTION_REASONS.items() if resolution} == {
        EastAsianEvidenceReason.JAPANESE_ITERATION_MARK_ONE_SIDED_EXCLUSIVE: (
            ResolutionReason.JAPANESE_ITERATION_MARK_ASSIGNMENT
        ),
        EastAsianEvidenceReason.JAPANESE_ITERATION_MARK_DUAL_EXCLUSIVE: (ResolutionReason.JAPANESE_ITERATION_MARK_ASSIGNMENT),
        EastAsianEvidenceReason.IDENTITY_BACKED_EXACT_FULL_SURFACE: ResolutionReason.IDENTITY_BACKED_EXACT_ASSIGNMENT,
        EastAsianEvidenceReason.KOREAN_WESTERN_SUFFIX_CONFLICT: (ResolutionReason.KOREAN_WESTERN_CONFLICT_PRESERVE_INPUT),
    }


def test_resolved_fields_reject_an_illegal_reason_pair() -> None:
    with pytest.raises(ValidationError, match=r"pp_selected requires \(pp, assign\)"):
        ResolvedAuthorFields(
            first_name="Wei",
            middle_names="",
            last_name="Li",
            suffix=None,
            resolution_provenance="source",
            resolution_action="preserve_input",
            resolution_reason="pp_selected",
        )


def test_selected_components_materialize_final_suffix_and_plain_enum_values() -> None:
    resolved = ResolvedAuthorFields.from_selected_components(
        source=SourceAuthorFields(first_name="Steve", last_name="Blando", suffix=""),
        selected=NameComponents(given_name="Steve", surname="Blando", suffix="IV"),
        reason=ResolutionReason.SCALAR_BASELINE,
    )

    assert resolved.dict() == {
        "first_name": "Steve",
        "middle_names": "",
        "last_name": "Blando",
        "suffix": "IV",
        "resolution_provenance": "scalar",
        "resolution_action": "assign",
        "resolution_reason": "scalar_baseline",
    }


def test_source_passthrough_preserves_boundaries_and_suffix_missingness() -> None:
    source = SourceAuthorFields(first_name="Juan Carlos", middle_names=None, last_name="de la Cruz", suffix=None)
    resolved = ResolvedAuthorFields.from_source(
        source,
        reason=ResolutionReason.NON_PERSON_SOURCE_PASSTHROUGH,
    )

    assert resolved.first_name == "Juan Carlos"
    assert resolved.middle_names == ""
    assert resolved.last_name == "de la Cruz"
    assert resolved.suffix is None


def test_source_factory_rejects_a_non_source_reason() -> None:
    with pytest.raises(ValueError, match="is not a SOURCE non-assignment reason"):
        ResolvedAuthorFields.from_source(
            SourceAuthorFields(first_name="Wei", last_name="Li"),
            reason=ResolutionReason.SCALAR_BASELINE,
        )


def test_suppression_keeps_an_aligned_source_slot_for_writer_diagnostics() -> None:
    source = SourceAuthorFields(first_name="STADT", last_name="NÜRNBERG")

    resolved = ResolvedAuthorFields.from_source(
        source,
        reason=ResolutionReason.REVIEWED_NON_PERSON_PATTERN,
    )

    assert resolved.dict() == {
        "first_name": "STADT",
        "middle_names": "",
        "last_name": "NÜRNBERG",
        "suffix": None,
        "resolution_provenance": "source",
        "resolution_action": "suppress",
        "resolution_reason": "reviewed_non_person_pattern",
    }


def test_selected_components_factory_rejects_a_source_reason() -> None:
    with pytest.raises(ValueError, match="requires exact source materialization"):
        ResolvedAuthorFields.from_selected_components(
            source=SourceAuthorFields(first_name="Wei", last_name="Li"),
            selected=NameComponents(given_name="Wei", surname="Li"),
            reason=ResolutionReason.NO_USABLE_SEMANTIC_RESULT,
        )

    with pytest.raises(ValueError, match="requires exact source materialization"):
        ResolvedAuthorFields.from_selected_components(
            source=SourceAuthorFields(first_name="STADT", last_name="NÜRNBERG"),
            selected=NameComponents(given_name="Stadt", surname="Nürnberg"),
            reason=ResolutionReason.REVIEWED_NON_PERSON_PATTERN,
        )


def _canonical(given_name: str, surname: str) -> CanonicalName:
    components = NameComponents(given_name=given_name, surname=surname)
    return CanonicalName(
        source_text=f"{given_name} {surname}",
        text=f"{given_name} {surname}",
        source=components,
        normalized=components,
    )


def test_hard_scalar_constraints_accept_the_closed_assignment_and_preservation_set() -> None:
    assignment = ApplyAssignment(
        canonical_name=_canonical("希", "佐々木"),
        evidence_reason=EastAsianEvidenceReason.JAPANESE_ITERATION_MARK_DUAL_EXCLUSIVE,
    )
    exact_assignment = ApplyAssignment(
        canonical_name=_canonical("Thuong", "Le"),
        evidence_reason=EastAsianEvidenceReason.IDENTITY_BACKED_EXACT_FULL_SURFACE,
    )
    preservation = PreserveBaseline(canonical_name=_canonical("Kim", "Stene-Larsen"))

    assert assignment.reason is ResolutionReason.JAPANESE_ITERATION_MARK_ASSIGNMENT
    assert exact_assignment.reason is ResolutionReason.IDENTITY_BACKED_EXACT_ASSIGNMENT
    assert preservation.reason is ResolutionReason.KOREAN_WESTERN_CONFLICT_PRESERVE_INPUT


def test_hard_assignment_rejects_other_east_asian_rules() -> None:
    with pytest.raises(ValueError, match="unsupported hard scalar assignment"):
        ApplyAssignment(
            canonical_name=_canonical("Min-su", "Kim"),
            evidence_reason=EastAsianEvidenceReason.KOREAN_ROMANIZED_STRICT,
        )


def test_atomic_korean_token_repair_matches_complete_hyphen_parts_and_updates_lineage() -> None:
    source = SourceAuthorFields(first_name="So Young", last_name="Kim")
    selected = NameComponents(
        given_name="So-You-Ng",
        surname="Kim",
        given_tokens=("So", "You", "Ng"),
        surname_tokens=("Kim",),
        order=("given", "surname"),
    )

    repaired = restore_reviewed_atomic_korean_tokens(source, selected)

    assert repaired.given_name == "So-Young"
    assert repaired.given_tokens == ("So", "Young")
    assert repaired.surname == "Kim"
    assert repaired.surname_tokens == ("Kim",)
    assert repaired.order == selected.order


def test_atomic_korean_token_repair_does_not_match_a_longer_hyphen_part() -> None:
    source = SourceAuthorFields(first_name="Hana", last_name="Ha-Nam")
    selected = NameComponents(
        given_name="Hana",
        surname="Ha-Nam",
        given_tokens=("Hana",),
        surname_tokens=("Ha", "Nam"),
        order=("given", "surname"),
    )

    assert restore_reviewed_atomic_korean_tokens(source, selected) == selected


def test_evidence_failure_is_typed_outside_the_hard_constraint_union() -> None:
    assert issubclass(EvidenceFailure, RuntimeError)
