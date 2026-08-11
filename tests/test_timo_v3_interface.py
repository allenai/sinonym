"""Compact end-to-end contract tests for the routed V3 interface."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from sinonym.coretypes import CanonicalName, NameComponents, ParseResult
from sinonym.coretypes.routing_resolution import (
    ResolutionAction,
    ResolutionProvenance,
    ResolutionReason,
)
from sinonym.name_punctuation import ROMAN_HYPHEN_LIKE
from sinonym.services.batch_analysis import RelatedBatchParseResult
from sinonym.timo.interface import PredictorConfig, RoutingPredictorV3
from sinonym.timo.routing_v3 import (
    RoutingInstanceV3,
    SourceAuthorFields,
    reviewed_initials_comma_reversal,
)


@pytest.fixture
def predictor(routing_predictor_v3: RoutingPredictorV3) -> RoutingPredictorV3:
    """Alias the shared session predictor under this module's name."""
    return routing_predictor_v3


def _route(
    predictor: RoutingPredictorV3,
    authors: list[SourceAuthorFields],
    *,
    vys_other_names: list[str] | None = None,
):
    (paper,) = predictor.predict_batch(
        [RoutingInstanceV3(pp_authors=authors, vys_other_names=vys_other_names)],
    )
    return paper.authors


def test_duplicate_names_remain_positionally_aligned(
    predictor: RoutingPredictorV3,
) -> None:
    first = SourceAuthorFields(first_name="UW", last_name="University")
    second = SourceAuthorFields(middle_names="UW", last_name="University")
    assert first.full_name() == second.full_name()

    results = _route(predictor, [first, second])

    assert [result.resolved_fields.first_name for result in results] == ["UW", ""]
    assert [result.resolved_fields.middle_names for result in results] == ["", "UW"]
    assert all(result.resolved_fields.resolution_reason is ResolutionReason.NON_PERSON_SOURCE_PASSTHROUGH for result in results)


def test_reviewed_non_person_pattern_is_a_terminal_writer_decision(
    predictor: RoutingPredictorV3,
) -> None:
    source = SourceAuthorFields(first_name="STADT", last_name="NÜRNBERG")

    (result,) = _route(predictor, [source])

    resolved = result.resolved_fields
    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == ("STADT", "", "NÜRNBERG")
    assert resolved.resolution_provenance is ResolutionProvenance.SOURCE
    assert resolved.resolution_action is ResolutionAction.SUPPRESS
    assert resolved.resolution_reason is ResolutionReason.REVIEWED_NON_PERSON_PATTERN


def test_router_not_person_still_allows_a_real_person_scalar_parse(
    predictor: RoutingPredictorV3,
) -> None:
    (result,) = _route(predictor, [SourceAuthorFields(first_name="Babak", last_name="Esmaeili")])

    resolved = result.resolved_fields
    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == ("Babak", "", "Esmaeili")
    assert resolved.resolution_action is ResolutionAction.ASSIGN
    assert resolved.resolution_reason is ResolutionReason.SCALAR_BASELINE


def test_structured_credential_only_cleanup_is_a_typed_source_passthrough(
    predictor: RoutingPredictorV3,
) -> None:
    source = SourceAuthorFields(last_name="MD, MRCP")

    (result,) = _route(predictor, [source])

    resolved = result.resolved_fields
    assert (resolved.first_name, resolved.middle_names, resolved.last_name, resolved.suffix) == (
        "",
        "",
        "MD, MRCP",
        None,
    )
    assert resolved.resolution_provenance is ResolutionProvenance.SOURCE
    assert resolved.resolution_action is ResolutionAction.PRESERVE_INPUT
    assert resolved.resolution_reason is ResolutionReason.NON_PERSON_SOURCE_PASSTHROUGH


def test_packed_credential_with_compound_initial_mononym_preserves_source(
    predictor: RoutingPredictorV3,
) -> None:
    source = SourceAuthorFields(first_name="BEng", last_name="J.-P.")

    (result,) = _route(predictor, [source])

    resolved = result.resolved_fields
    assert (resolved.first_name, resolved.middle_names, resolved.last_name, resolved.suffix) == (
        "BEng",
        "",
        "J.-P.",
        None,
    )
    assert resolved.resolution_action is ResolutionAction.ASSIGN
    assert resolved.resolution_reason is ResolutionReason.SCALAR_BASELINE


@pytest.mark.parametrize(
    ("first_name", "last_name", "expected_action", "expected_reason"),
    [
        (
            "Haruki",
            "Kadono",
            ResolutionAction.PRESERVE_INPUT,
            ResolutionReason.JAPANESE_GIVEN_FIRST_REORDER_VETO_PRESERVE_INPUT,
        ),
        ("Kou", "Hiroya", ResolutionAction.ASSIGN, ResolutionReason.SCALAR_BASELINE),
        (
            "Masaki",
            "Takamoto",
            ResolutionAction.PRESERVE_INPUT,
            ResolutionReason.JAPANESE_GIVEN_FIRST_REORDER_VETO_PRESERVE_INPUT,
        ),
        (
            "Masaki",
            "Tomonaga",
            ResolutionAction.PRESERVE_INPUT,
            ResolutionReason.JAPANESE_GIVEN_FIRST_REORDER_VETO_PRESERVE_INPUT,
        ),
        (
            "Shoji",
            "Kagami",
            ResolutionAction.PRESERVE_INPUT,
            ResolutionReason.JAPANESE_GIVEN_FIRST_REORDER_VETO_PRESERVE_INPUT,
        ),
        ("Takaya", "Miwa", ResolutionAction.ASSIGN, ResolutionReason.SCALAR_BASELINE),
    ],
)
def test_reviewed_exact_japanese_surface_preserves_v3_input_order(
    predictor: RoutingPredictorV3,
    first_name: str,
    last_name: str,
    expected_action: ResolutionAction,
    expected_reason: ResolutionReason,
) -> None:
    (result,) = _route(
        predictor,
        [SourceAuthorFields(first_name=first_name, last_name=last_name)],
    )

    resolved = result.resolved_fields
    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == (first_name, "", last_name)
    assert resolved.resolution_action is expected_action
    assert resolved.resolution_reason is expected_reason


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (SourceAuthorFields(first_name="BEng", last_name="Robert McManus"), ("Robert", "", "McManus")),
        (
            SourceAuthorFields(first_name="Dr.-Ing.", middle_names="Thomas", last_name="Schmidt"),
            ("Thomas", "", "Schmidt"),
        ),
    ],
)
def test_exact_case_credential_cleanup_reaches_v3_scalar_output(
    predictor: RoutingPredictorV3,
    source: SourceAuthorFields,
    expected: tuple[str, str, str],
) -> None:
    (result,) = _route(predictor, [source])

    resolved = result.resolved_fields
    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == expected
    assert resolved.resolution_action is ResolutionAction.ASSIGN
    assert resolved.resolution_reason is ResolutionReason.REVIEWED_SOURCE_PATTERN_ASSIGNMENT


def test_suffix_precedence_and_missingness(predictor: RoutingPredictorV3) -> None:
    results = _route(
        predictor,
        [
            SourceAuthorFields(first_name="Steve", last_name="Blando IV", suffix="Jr."),
            SourceAuthorFields(first_name="Steve", last_name="Blando IV", suffix=""),
            SourceAuthorFields(first_name="Michael", last_name="Johnson", suffix=None),
            SourceAuthorFields(first_name="Michael", last_name="Johnson", suffix=""),
        ],
    )

    assert [result.resolved_fields.suffix for result in results] == ["Jr.", "IV", None, ""]


@pytest.mark.parametrize(
    "source_text",
    [
        pytest.param("A 𠮷", id="supplementary-han-u20bb7"),
        pytest.param("É 﨑", id="non-ascii-latin"),
    ],
)
def test_mixed_script_safety_covers_unicode_han_and_latin(
    predictor: RoutingPredictorV3,
    source_text: str,
) -> None:
    source = SourceAuthorFields(first_name=source_text)

    (result,) = _route(predictor, [source])

    resolved = result.resolved_fields
    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == (source_text, "", "")
    assert resolved.resolution_provenance is ResolutionProvenance.SOURCE
    assert resolved.resolution_action is ResolutionAction.PRESERVE_INPUT
    assert resolved.resolution_reason is ResolutionReason.MIXED_SCRIPT_SAFETY_SUPPRESSION


def test_atomic_korean_token_repair_does_not_change_a_longer_hyphenated_name(
    predictor: RoutingPredictorV3,
) -> None:
    (result,) = _route(
        predictor,
        [SourceAuthorFields(first_name="Hana", last_name="Ha-Nam")],
    )

    assert (result.resolved_fields.first_name, result.resolved_fields.last_name) == ("Hana", "Ha-Nam")


def test_pp_only_abstain_is_a_terminal_input_order_decision(
    predictor: RoutingPredictorV3,
) -> None:
    (result,) = _route(
        predictor,
        [SourceAuthorFields(first_name="Wei", last_name="Wang")],
    )
    resolved = result.resolved_fields

    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == ("Wei", "", "Wang")
    assert resolved.resolution_provenance is ResolutionProvenance.PP
    assert resolved.resolution_action is ResolutionAction.PRESERVE_INPUT
    assert resolved.resolution_reason is ResolutionReason.PP_ONLY_ABSTAIN_INPUT


@pytest.mark.parametrize(
    "hyphen",
    sorted(ROMAN_HYPHEN_LIKE, key=ord),
    ids=lambda character: f"U+{ord(character):04X}",
)
def test_pp_only_compound_surname_guard_matches_structural_roman_hyphens(
    predictor: RoutingPredictorV3,
    hyphen: str,
) -> None:
    source = SourceAuthorFields(first_name="Ka Ming", last_name=f"Au{hyphen}Yeung")

    (result,) = _route(predictor, [source])

    resolved = result.resolved_fields
    assert (resolved.first_name, resolved.middle_names, resolved.last_name, resolved.suffix) == (
        "Ka Ming",
        "",
        source.last_name,
        None,
    )
    assert resolved.resolution_provenance is ResolutionProvenance.SOURCE
    assert resolved.resolution_action is ResolutionAction.PRESERVE_INPUT
    assert resolved.resolution_reason is ResolutionReason.PP_ONLY_ABSTAIN_REVIEWED_COMPOUND_SURNAME


@pytest.mark.parametrize(
    ("leading", "trailing"),
    [
        pytest.param(" ", " ", id="ascii-space"),
        pytest.param("\t", "\r\n", id="ascii-controls"),
        pytest.param("\u2003", "\u2003", id="em-space"),
    ],
)
def test_pp_only_compound_surname_guard_trims_only_outer_whitespace(
    predictor: RoutingPredictorV3,
    leading: str,
    trailing: str,
) -> None:
    source = SourceAuthorFields(first_name="Ka Ming", last_name=f"{leading}AU\u2011YEUNG{trailing}")

    (result,) = _route(predictor, [source])

    resolved = result.resolved_fields
    assert resolved.first_name == "Ka Ming"
    assert resolved.last_name == source.last_name
    assert resolved.resolution_provenance is ResolutionProvenance.SOURCE
    assert resolved.resolution_reason is ResolutionReason.PP_ONLY_ABSTAIN_REVIEWED_COMPOUND_SURNAME


@pytest.mark.parametrize(
    "hyphen",
    ["\u00ad", "\u2015", "\u2027", "\u208b", "\ufe58"],
    ids=lambda character: f"U+{ord(character):04X}",
)
def test_pp_only_compound_surname_guard_excludes_validation_only_hyphens(
    predictor: RoutingPredictorV3,
    hyphen: str,
) -> None:
    source = SourceAuthorFields(first_name="Ka Ming", last_name=f"Au{hyphen}Yeung")

    (result,) = _route(predictor, [source])

    assert result.resolved_fields.resolution_reason is not ResolutionReason.PP_ONLY_ABSTAIN_REVIEWED_COMPOUND_SURNAME


@pytest.mark.parametrize(
    "last_name",
    ["Au/Yeung", "Au \u2011 Yeung", "Au\u2011Yung", "Chan\u2011Yeung"],
)
def test_pp_only_compound_surname_guard_keeps_exact_reviewed_membership(
    predictor: RoutingPredictorV3,
    last_name: str,
) -> None:
    source = SourceAuthorFields(first_name="Ka Ming", last_name=last_name)

    (result,) = _route(predictor, [source])

    assert result.resolved_fields.resolution_reason is not ResolutionReason.PP_ONLY_ABSTAIN_REVIEWED_COMPOUND_SURNAME


@pytest.mark.parametrize(
    "middle_names",
    [
        pytest.param(None, id="missing"),
        pytest.param("", id="empty"),
        pytest.param(" ", id="ascii-space"),
        pytest.param("\t\r\n", id="ascii-controls"),
        pytest.param("\u2003", id="em-space"),
    ],
)
def test_initials_comma_veto_treats_whitespace_only_middle_as_empty(
    predictor: RoutingPredictorV3,
    middle_names: str | None,
) -> None:
    source = SourceAuthorFields(first_name="A,", middle_names=middle_names, last_name="Smith")

    (result,) = _route(predictor, [source])

    resolved = result.resolved_fields
    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == (
        "A,",
        middle_names or "",
        "Smith",
    )
    assert resolved.resolution_provenance is ResolutionProvenance.SOURCE
    assert resolved.resolution_action is ResolutionAction.PRESERVE_INPUT
    assert resolved.resolution_reason is ResolutionReason.INITIALS_COMMA_REORDER_VETO_PRESERVE_INPUT


@pytest.mark.parametrize("middle_names", ["M.", ".", " - "])
def test_initials_comma_veto_keeps_populated_middle_negative(middle_names: str) -> None:
    source = SourceAuthorFields(first_name="A,", middle_names=middle_names, last_name="Smith")
    reversed_candidate = NameComponents(given_name="Smith", surname="A")

    assert not reviewed_initials_comma_reversal(source, reversed_candidate)


def test_reorder_veto_does_not_use_vys_tail_as_paper_context(predictor: RoutingPredictorV3) -> None:
    (result,) = _route(
        predictor,
        [SourceAuthorFields(first_name="Satomi", last_name="Miwa")],
        vys_other_names=["Akira Suzuki"],
    )

    assert (result.resolved_fields.first_name, result.resolved_fields.last_name) == ("Miwa", "Satomi")
    assert result.resolved_fields.resolution_reason is ResolutionReason.SCALAR_BASELINE


def test_possible_japanese_surname_repairs_wrong_batch_reversal_without_vys_context(
    predictor: RoutingPredictorV3,
) -> None:
    (result,) = _route(
        predictor,
        [SourceAuthorFields(first_name="Mai", last_name="Hata")],
    )

    assert (result.resolved_fields.first_name, result.resolved_fields.last_name) == ("Mai", "Hata")
    assert result.resolved_fields.resolution_reason is ResolutionReason.JAPANESE_GIVEN_FIRST_REORDER_VETO_PRESERVE_INPUT


def test_failed_pp_vys_abstain_cannot_fall_through_to_scalar_reorder(
    predictor: RoutingPredictorV3,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = SourceAuthorFields(first_name="Wei", last_name="Zhang", suffix="")
    scalar_components = NameComponents(given_name="Zhang", surname="Wei")
    monkeypatch.setattr(
        predictor._detector,  # noqa: SLF001
        "routing_scalar_resolution",
        lambda _raw_name: CanonicalName(
            source_text="Wei Zhang",
            text="Zhang Wei",
            source=scalar_components,
            normalized=scalar_components,
        ),
    )

    resolved = predictor._resolver.resolve_pp_vys_author(  # noqa: SLF001
        source=source,
        paper_authors=[source],
        raw_name=source.full_name(),
        paper_names=[source.full_name()],
        focal_index=0,
        row={"router_prediction": "abstain", "input_order_candidate": "pp"},
        pp_result=ParseResult.failure("synthetic PP failure"),
        vys_result=ParseResult.failure("synthetic VYS failure"),
    )

    assert (resolved.first_name, resolved.middle_names, resolved.last_name, resolved.suffix) == ("Wei", "", "Zhang", "")
    assert (resolved.resolution_provenance, resolved.resolution_action, resolved.resolution_reason) == (
        ResolutionProvenance.SOURCE,
        ResolutionAction.PRESERVE_INPUT,
        ResolutionReason.BATCH_ABSTAIN_MATERIALIZATION_FAILED,
    )


def test_v3_runs_one_scalar_evidence_pass_per_focal_author(
    predictor: RoutingPredictorV3,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = predictor._detector._east_asian_name_order  # noqa: SLF001
    original = service.infer_resolution
    calls: list[str] = []

    def counted(raw_name: str, *, japanese_probability):
        calls.append(raw_name)
        return original(raw_name, japanese_probability=japanese_probability)

    monkeypatch.setattr(service, "infer_resolution", counted)
    _route(
        predictor,
        [SourceAuthorFields(first_name="\u4f50\u3005\u6728", last_name="\u514b\u5178")],
        vys_other_names=["Jane Doe"],
    )

    assert calls == ["\u4f50\u3005\u6728 \u514b\u5178"]


def test_batch_programming_errors_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    predictor = RoutingPredictorV3(PredictorConfig(parallel="never"), ".")
    predictor._detector._ensure_initialized()  # noqa: SLF001
    service = predictor._detector._batch_analysis_service  # noqa: SLF001
    assert service is not None

    def programming_error(*_args, **_kwargs):
        message = "synthetic batch programming error"
        raise RuntimeError(message)

    monkeypatch.setattr(service, "analyze_related_name_batches", programming_error)

    with pytest.raises(RuntimeError, match="synthetic batch programming error"):
        _route(predictor, [SourceAuthorFields(first_name="Michael", last_name="Johnson")])


def test_reordered_pp_result_is_an_invariant_failure(
    predictor: RoutingPredictorV3,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = [
        SourceAuthorFields(first_name="Michael", last_name="Johnson"),
        SourceAuthorFields(first_name="Steve", last_name="Blando"),
    ]
    names = [source.full_name() for source in sources]
    batch = predictor._detector._analyze_related_name_batches(names, None).pp_batch  # noqa: SLF001
    reordered = replace(
        batch,
        names=list(reversed(batch.names)),
        results=list(reversed(batch.results)),
        individual_analyses=list(reversed(batch.individual_analyses)),
        name_order_evidence=list(reversed(batch.name_order_evidence)),
    )
    monkeypatch.setattr(
        predictor._detector,  # noqa: SLF001
        "_analyze_related_batch_requests",
        lambda *_args, **_kwargs: [RelatedBatchParseResult(reordered, None, None)],
    )

    with pytest.raises(RuntimeError, match="names/order do not match"):
        _route(predictor, sources)


def test_reordered_vys_context_is_an_invariant_failure(
    predictor: RoutingPredictorV3,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = SourceAuthorFields(first_name="Michael", last_name="Johnson")
    pp_names = [source.full_name()]
    pool_names = [*pp_names, "Jane Doe"]
    related = predictor._detector._analyze_related_name_batches(  # noqa: SLF001
        pp_names,
        pool_names,
    )
    reordered = replace(related, vys_context_names=tuple(reversed(pool_names)))
    monkeypatch.setattr(
        predictor._detector,  # noqa: SLF001
        "_analyze_related_batch_requests",
        lambda *_args, **_kwargs: [reordered],
    )

    with pytest.raises(RuntimeError, match="batch result 1 names/order do not match"):
        _route(predictor, [source], vys_other_names=["Jane Doe"])


@pytest.mark.parametrize("returned_count", [0, 2])
def test_missing_or_extra_related_result_is_an_invariant_failure(
    predictor: RoutingPredictorV3,
    monkeypatch: pytest.MonkeyPatch,
    returned_count: int,
) -> None:
    source = SourceAuthorFields(first_name="Michael", last_name="Johnson")
    related = predictor._detector._analyze_related_name_batches(  # noqa: SLF001
        [source.full_name()],
        None,
    )
    monkeypatch.setattr(
        predictor._detector,  # noqa: SLF001
        "_analyze_related_batch_requests",
        lambda *_args, **_kwargs: [related] * returned_count,
    )

    with pytest.raises(RuntimeError, match="wrong number of batches"):
        _route(predictor, [source])


def test_misaligned_fields_are_an_invariant_failure(
    predictor: RoutingPredictorV3,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = SourceAuthorFields(first_name="Michael", last_name="Johnson")
    related = predictor._detector._analyze_related_name_batches(  # noqa: SLF001
        [source.full_name()],
        None,
    )
    misaligned = replace(related, pp_batch=replace(related.pp_batch, results=[]))
    monkeypatch.setattr(
        predictor._detector,  # noqa: SLF001
        "_analyze_related_batch_requests",
        lambda *_args, **_kwargs: [misaligned],
    )

    with pytest.raises(RuntimeError, match="misaligned fields"):
        _route(predictor, [source])


def test_present_empty_vys_cannot_change_pp_only_semantic_result(
    predictor: RoutingPredictorV3,
) -> None:
    source = SourceAuthorFields(first_name="Zhang", middle_names="Wei", last_name="Q.")

    (pp_only,) = _route(predictor, [source])
    (present_vys,) = _route(predictor, [source], vys_other_names=[])

    assert set(pp_only.dict()) == {"resolved_fields"}
    assert (
        pp_only.resolved_fields.first_name,
        pp_only.resolved_fields.middle_names,
        pp_only.resolved_fields.last_name,
    ) == ("Zhang", "Wei", "Q.")
    assert pp_only.resolved_fields == present_vys.resolved_fields


def test_v3_preserves_unicode_apostrophe_display(
    predictor: RoutingPredictorV3,
) -> None:
    source = SourceAuthorFields(first_name="Xiang\u02bban", last_name="Yan")

    (result,) = _route(predictor, [source])

    resolved = result.resolved_fields
    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == ("Xiang'an", "", "Yan")


@pytest.mark.parametrize("suffix", [None, "III", "Jr."])
def test_v3_uses_last_name_field_as_surname_first_initial_evidence(
    predictor: RoutingPredictorV3,
    suffix: str | None,
) -> None:
    (result,) = _route(
        predictor,
        [SourceAuthorFields(last_name="Masterov R. A.", suffix=suffix)],
    )

    resolved = result.resolved_fields
    assert (resolved.first_name, resolved.middle_names, resolved.last_name, resolved.suffix) == (
        "R.",
        "A.",
        "Masterov",
        suffix,
    )
    assert resolved.resolution_provenance is ResolutionProvenance.SOURCE
    assert resolved.resolution_action is ResolutionAction.ASSIGN
    assert resolved.resolution_reason is ResolutionReason.STRUCTURED_SURNAME_INITIAL_TAIL_ASSIGNMENT


def test_timo_config_registers_v3_without_replacing_v2() -> None:
    config = (Path(__file__).parents[1] / "sinonym" / "timo" / "config.yaml").read_text(encoding="utf-8")

    assert "sinonym_routing_v2:" in config
    assert "sinonym_routing_v3:" in config
    assert "instance: sinonym.timo.interface.RoutingInstanceV3" in config
    assert "prediction: sinonym.timo.interface.RoutedPaperPredictionV3" in config
    assert "predictor: sinonym.timo.interface.RoutingPredictorV3" in config
