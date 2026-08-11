"""Focused contracts for TIMO inputs and terminal decisions."""

import importlib.util

import pytest
from pydantic import Field, ValidationError

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
from sinonym.timo import interface
from sinonym.timo._resolution import (
    _Model,
    merge_resolved_suffix,
)
from sinonym.timo.interface import Instance, Prediction, ResolvedAuthorFields, SourceAuthorFields


def test_public_interface_is_exactly_the_six_unversioned_classes() -> None:
    expected = {
        "SourceAuthorFields",
        "Instance",
        "ResolvedAuthorFields",
        "Prediction",
        "PredictorConfig",
        "Predictor",
    }
    removed = {
        "RoutingInstanceV3",
        "RoutingPredictorV3",
        "RoutedPaperPredictionV3",
        "PredictionV2",
        "PredictorV2",
        "RoutingInstance",
        "RoutingPredictor",
        "ChineseNameDetector",
    }

    assert set(interface.__all__) == expected
    assert {name for name, value in vars(interface).items() if not name.startswith("_") and isinstance(value, type)} == expected
    assert all(not hasattr(interface, name) for name in removed)
    assert importlib.util.find_spec("sinonym.timo.routing_v3") is None


def test_request_derives_focal_pool_positionally_without_a_second_authority() -> None:
    authors = [
        SourceAuthorFields(first_name="Juan", middle_names="Carlos", last_name="de la Cruz", suffix="Jr."),
        SourceAuthorFields(first_name="Juan Carlos", middle_names="de la", last_name="Cruz"),
    ]
    instance = Instance(
        pp_authors=authors,
        vys_other_names=["Jane Doe", "Jane Doe"],
    )
    pp_names = [author.full_name() for author in instance.pp_authors]

    assert pp_names == ["Juan Carlos de la Cruz", "Juan Carlos de la Cruz"]
    assert [*pp_names, *instance.vys_other_names] == [
        "Juan Carlos de la Cruz",
        "Juan Carlos de la Cruz",
        "Jane Doe",
        "Jane Doe",
    ]
    assert instance.pp_authors[0].last_name == "de la Cruz"
    assert instance.pp_authors[1].last_name == "Cruz"
    assert set(instance.dict()) == {"pp_authors", "vys_other_names"}


def test_request_rejects_a_separately_supplied_focal_slice() -> None:
    with pytest.raises(ValidationError, match="extra fields not permitted"):
        Instance(
            pp_authors=[SourceAuthorFields(first_name="Li", last_name="Wei")],
            pp_names=["Wei Li"],
        )


def test_vys_other_names_defaults_empty_and_rejects_null() -> None:
    authors = [SourceAuthorFields(first_name="Li", last_name="Wei")]
    instance = Instance(pp_authors=authors)

    assert instance.vys_other_names == []
    assert instance.dict()["vys_other_names"] == []
    assert Instance.parse_raw(instance.json()) == instance
    with pytest.raises(ValidationError):
        Instance(pp_authors=authors, vys_other_names=None)


def test_vys_other_names_defaults_are_independent_and_strict() -> None:
    first = Instance(pp_authors=[])
    second = Instance(pp_authors=[])

    first.vys_other_names.append("Jane Doe")

    assert second.vys_other_names == []
    with pytest.raises(ValidationError):
        Instance(pp_authors=[], vys_other_names=[123])


def test_request_wire_omits_nullable_author_fields() -> None:
    instance = Instance(
        pp_authors=[SourceAuthorFields(first_name="Li", last_name="Wei")],
    )

    assert instance.dict() == {
        "pp_authors": [{"first_name": "Li", "last_name": "Wei"}],
        "vys_other_names": [],
    }
    assert Instance.parse_raw(instance.json()) == instance


def test_json_schemas_match_the_runtime_contract() -> None:
    request_schema = Instance.schema()
    source_properties = request_schema["definitions"]["SourceAuthorFields"]["properties"]

    assert {name: source_properties[name]["type"] for name in source_properties} == {
        "first_name": ["string", "null"],
        "middle_names": ["string", "null"],
        "last_name": ["string", "null"],
        "suffix": ["string", "null"],
    }
    assert request_schema["properties"]["vys_other_names"]["type"] == "array"
    assert request_schema["properties"]["pp_authors"]["type"] == "array"

    response_schema = Prediction.schema()
    assert response_schema["properties"]["authors"]["items"] == {"$ref": "#/definitions/ResolvedAuthorFields"}
    resolved_properties = response_schema["definitions"]["ResolvedAuthorFields"]["properties"]
    assert resolved_properties["suffix"]["type"] == ["string", "null"]
    assert {resolved_properties[name]["type"] for name in ("first_name", "middle_names", "last_name")} == {
        "string",
    }


def test_nullable_reference_schemas_use_any_of() -> None:
    """Nullable enum and nested-model fields generate valid reference schemas."""

    class NullableEnumModel(_Model):
        reason: ResolutionReason | None = None

    class NullableNestedModel(_Model):
        source: SourceAuthorFields | None = None

    enum_property = NullableEnumModel.schema()["properties"]["reason"]
    nested_property = NullableNestedModel.schema()["properties"]["source"]

    assert enum_property == {
        "anyOf": [
            {"$ref": "#/definitions/ResolutionReason"},
            {"type": "null"},
        ],
    }
    assert nested_property == {
        "anyOf": [
            {"$ref": "#/definitions/SourceAuthorFields"},
            {"type": "null"},
        ],
    }
    assert NullableEnumModel(reason=ResolutionReason.HANDLED_EVIDENCE_FAILURE).reason is not None
    assert NullableNestedModel(source=None).source is None


@pytest.mark.parametrize(
    ("by_alias", "property_name"),
    [
        (True, "resolutionReason"),
        (False, "reason"),
    ],
)
def test_nullable_aliased_reference_schema_uses_requested_property_name(
    by_alias: bool,
    property_name: str,
) -> None:
    """Nullable references support schemas keyed by aliases or field names."""

    class NullableAliasedModel(_Model):
        reason: ResolutionReason | None = Field(default=None, alias="resolutionReason")

    property_schema = NullableAliasedModel.schema(by_alias=by_alias)["properties"][property_name]

    assert property_schema == {
        "anyOf": [
            {"$ref": "#/definitions/ResolutionReason"},
            {"type": "null"},
        ],
    }


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
    expected_groups = {
        (ResolutionProvenance.SOURCE, ResolutionAction.PRESERVE_INPUT): {
            ResolutionReason.MIXED_SCRIPT_SAFETY_SUPPRESSION,
            ResolutionReason.ROUTED_CJK_SAFETY_SUPPRESSION,
            ResolutionReason.HANDLED_EVIDENCE_FAILURE,
            ResolutionReason.HARD_SCALAR_MATERIALIZATION_FAILED,
            ResolutionReason.SCALAR_KNOWN_COMPOUND_SURNAME_PRESERVE_INPUT,
            ResolutionReason.INITIALS_COMMA_REORDER_VETO_PRESERVE_INPUT,
            ResolutionReason.REVIEWED_EXACT_SOURCE_REORDER_VETO_PRESERVE_INPUT,
            ResolutionReason.PP_ONLY_ABSTAIN_REVIEWED_COMPOUND_SURNAME,
            ResolutionReason.BATCH_ABSTAIN_MATERIALIZATION_FAILED,
            ResolutionReason.NON_PERSON_SOURCE_PASSTHROUGH,
            ResolutionReason.NO_USABLE_SEMANTIC_RESULT,
        },
        (ResolutionProvenance.SOURCE, ResolutionAction.ASSIGN): {
            ResolutionReason.SCALAR_CLEAN_SOURCE_SURNAME_REPARTITION_ASSIGNMENT,
            ResolutionReason.STRUCTURED_SURNAME_INITIAL_TAIL_ASSIGNMENT,
            ResolutionReason.REVIEWED_EXACT_SOURCE_ASSIGNMENT,
            ResolutionReason.REVIEWED_SOURCE_PATTERN_ASSIGNMENT,
        },
        (ResolutionProvenance.SOURCE, ResolutionAction.SUPPRESS): {
            ResolutionReason.REVIEWED_NON_PERSON_PATTERN,
        },
        (ResolutionProvenance.SCALAR, ResolutionAction.PRESERVE_INPUT): {
            ResolutionReason.KOREAN_WESTERN_CONFLICT_PRESERVE_INPUT,
            ResolutionReason.JAPANESE_GIVEN_FIRST_REORDER_VETO_PRESERVE_INPUT,
            ResolutionReason.VIETNAMESE_GIVEN_FIRST_REORDER_VETO_PRESERVE_INPUT,
            ResolutionReason.CONTEXT_SUPPORTED_REORDER_VETO_PRESERVE_INPUT,
        },
        (ResolutionProvenance.SCALAR, ResolutionAction.ASSIGN): {
            ResolutionReason.JAPANESE_ITERATION_MARK_ASSIGNMENT,
            ResolutionReason.IDENTITY_BACKED_EXACT_ASSIGNMENT,
            ResolutionReason.SCALAR_BASELINE,
        },
        (ResolutionProvenance.PP, ResolutionAction.PRESERVE_INPUT): {
            ResolutionReason.PP_VYS_ABSTAIN_PP_INPUT,
            ResolutionReason.PP_ONLY_ABSTAIN_INPUT,
        },
        (ResolutionProvenance.PP, ResolutionAction.ASSIGN): {ResolutionReason.PP_SELECTED},
        (ResolutionProvenance.VYS, ResolutionAction.PRESERVE_INPUT): {
            ResolutionReason.PP_VYS_ABSTAIN_VYS_INPUT,
        },
        (ResolutionProvenance.VYS, ResolutionAction.ASSIGN): {ResolutionReason.VYS_SELECTED},
    }
    expected = {reason: disposition for disposition, reasons in expected_groups.items() for reason in reasons}

    assert list(RESOLUTION_DECISION_TABLE) == list(ResolutionReason)
    assert set(RESOLUTION_DECISION_TABLE) == set(expected)
    assert {reason: (spec.provenance, spec.action) for reason, spec in RESOLUTION_DECISION_TABLE.items()} == expected
    assert all(isinstance(spec.provenance, ResolutionProvenance) for spec in RESOLUTION_DECISION_TABLE.values())
    assert all(isinstance(spec.action, ResolutionAction) for spec in RESOLUTION_DECISION_TABLE.values())


def test_every_east_asian_evidence_reason_has_one_hard_resolution_disposition() -> None:
    assert list(EAST_ASIAN_EVIDENCE_RESOLUTION_REASONS) == list(EastAsianEvidenceReason)
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


def test_evidence_failure_is_typed_outside_the_hard_constraint_union() -> None:
    assert issubclass(EvidenceFailure, RuntimeError)
