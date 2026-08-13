"""Compact end-to-end tests for the TIMO interface."""

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
from sinonym.timo._resolution import (
    _pp_only_native_abstain_prefers_scalar,
    reviewed_initials_comma_reversal,
)
from sinonym.timo.interface import Instance, Predictor, PredictorConfig, SourceAuthorFields
from tests._korean_atomic_cases import ATOMIC_KOREAN_GIVEN_CASES, AtomicKoreanGivenCase


def _route(
    predictor: Predictor,
    authors: list[SourceAuthorFields],
    *,
    vys_other_names: list[str] | None = None,
):
    (paper,) = predictor.predict_batch(
        [Instance(pp_authors=authors, vys_other_names=vys_other_names or [])],
    )
    return paper.authors


def test_duplicate_names_remain_positionally_aligned(
    predictor: Predictor,
) -> None:
    first = SourceAuthorFields(first_name="UW", last_name="University")
    second = SourceAuthorFields(middle_names="UW", last_name="University")
    assert first.full_name() == second.full_name()

    results = _route(predictor, [first, second])

    assert [result.first_name for result in results] == ["UW", ""]
    assert [result.middle_names for result in results] == ["", "UW"]
    assert all(result.resolution_reason is ResolutionReason.NON_PERSON_SOURCE_PASSTHROUGH for result in results)


def test_reviewed_non_person_pattern_is_a_terminal_writer_decision(
    predictor: Predictor,
) -> None:
    source = SourceAuthorFields(first_name="STADT", last_name="NÜRNBERG")

    (result,) = _route(predictor, [source])

    resolved = result
    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == ("STADT", "", "NÜRNBERG")
    assert resolved.resolution_provenance is ResolutionProvenance.SOURCE
    assert resolved.resolution_action is ResolutionAction.SUPPRESS
    assert resolved.resolution_reason is ResolutionReason.REVIEWED_NON_PERSON_PATTERN


@pytest.mark.parametrize("last_name", ["北京大学", "國立臺灣大學"])
def test_last_only_han_institution_is_a_terminal_writer_suppression(
    predictor: Predictor,
    last_name: str,
) -> None:
    (resolved,) = _route(predictor, [SourceAuthorFields(last_name=last_name)])

    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == ("", "", last_name)
    assert resolved.resolution_provenance is ResolutionProvenance.SOURCE
    assert resolved.resolution_action is ResolutionAction.SUPPRESS
    assert resolved.resolution_reason is ResolutionReason.REVIEWED_NON_PERSON_PATTERN


def test_router_not_person_still_allows_a_real_person_scalar_parse(
    predictor: Predictor,
) -> None:
    (result,) = _route(predictor, [SourceAuthorFields(first_name="Babak", last_name="Esmaeili")])

    resolved = result
    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == ("Babak", "", "Esmaeili")
    assert resolved.resolution_action is ResolutionAction.ASSIGN
    assert resolved.resolution_reason is ResolutionReason.SCALAR_BASELINE


def test_structured_credential_only_cleanup_is_a_typed_source_passthrough(
    predictor: Predictor,
) -> None:
    source = SourceAuthorFields(last_name="MD, MRCP")

    (result,) = _route(predictor, [source])

    resolved = result
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
    predictor: Predictor,
) -> None:
    source = SourceAuthorFields(first_name="BEng", last_name="J.-P.")

    (result,) = _route(predictor, [source])

    resolved = result
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
def test_reviewed_exact_japanese_surface_preserves_timo_input_order(
    predictor: Predictor,
    first_name: str,
    last_name: str,
    expected_action: ResolutionAction,
    expected_reason: ResolutionReason,
) -> None:
    (result,) = _route(
        predictor,
        [SourceAuthorFields(first_name=first_name, last_name=last_name)],
    )

    resolved = result
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
def test_exact_case_credential_cleanup_reaches_terminal_scalar_output(
    predictor: Predictor,
    source: SourceAuthorFields,
    expected: tuple[str, str, str],
) -> None:
    (result,) = _route(predictor, [source])

    resolved = result
    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == expected
    assert resolved.resolution_action is ResolutionAction.ASSIGN
    assert resolved.resolution_reason is ResolutionReason.REVIEWED_SOURCE_PATTERN_ASSIGNMENT


def test_suffix_precedence_and_missingness(predictor: Predictor) -> None:
    results = _route(
        predictor,
        [
            SourceAuthorFields(first_name="Steve", last_name="Blando IV", suffix="Jr."),
            SourceAuthorFields(first_name="Steve", last_name="Blando IV", suffix=""),
            SourceAuthorFields(first_name="Michael", last_name="Johnson", suffix=None),
            SourceAuthorFields(first_name="Michael", last_name="Johnson", suffix=""),
        ],
    )

    assert [result.suffix for result in results] == ["Jr.", "IV", None, ""]


@pytest.mark.parametrize(
    "source_text",
    [
        pytest.param("A 𠮷", id="supplementary-han-u20bb7"),
        pytest.param("É 﨑", id="non-ascii-latin"),
    ],
)
def test_mixed_script_safety_covers_unicode_han_and_latin(
    predictor: Predictor,
    source_text: str,
) -> None:
    source = SourceAuthorFields(first_name=source_text)

    (result,) = _route(predictor, [source])

    resolved = result
    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == (source_text, "", "")
    assert resolved.resolution_provenance is ResolutionProvenance.SOURCE
    assert resolved.resolution_action is ResolutionAction.PRESERVE_INPUT
    assert resolved.resolution_reason is ResolutionReason.MIXED_SCRIPT_SAFETY_SUPPRESSION


def test_atomic_korean_token_repair_does_not_change_a_longer_hyphenated_name(
    predictor: Predictor,
) -> None:
    (result,) = _route(
        predictor,
        [SourceAuthorFields(first_name="Hana", last_name="Ha-Nam")],
    )

    assert (result.first_name, result.last_name) == ("Hana", "Ha-Nam")


@pytest.mark.parametrize(
    "case",
    ATOMIC_KOREAN_GIVEN_CASES,
    ids=lambda case: case.raw_name,
)
def test_timo_uses_the_shared_atomic_korean_given_tokens(
    predictor: Predictor,
    case: AtomicKoreanGivenCase,
) -> None:
    source = SourceAuthorFields(first_name=case.source_given, last_name=case.surname)
    (result,) = _route(predictor, [source])

    assert (result.first_name, result.last_name) == (case.source_given, case.surname)


@pytest.mark.parametrize(
    "source",
    [
        SourceAuthorFields(first_name="J", last_name="Aibar Manero"),
        SourceAuthorFields(first_name="L", last_name="Carreras Matas"),
        SourceAuthorFields(first_name="R", middle_names="F", last_name="Lai A Fat"),
        SourceAuthorFields(first_name="A", middle_names="Y F", last_name="Li Yim"),
        SourceAuthorFields(first_name="N.M.A.", last_name="Nik Long"),
    ],
)
def test_reviewed_compound_surnames_compare_canonical_initial_surfaces(
    predictor: Predictor,
    source: SourceAuthorFields,
) -> None:
    (result,) = _route(predictor, [source])

    resolved = result
    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == (
        source.first_name or "",
        source.middle_names or "",
        source.last_name or "",
    )
    assert resolved.resolution_provenance is ResolutionProvenance.SOURCE
    assert resolved.resolution_action is ResolutionAction.PRESERVE_INPUT
    assert resolved.resolution_reason is ResolutionReason.SCALAR_KNOWN_COMPOUND_SURNAME_PRESERVE_INPUT


def test_pp_only_abstain_is_a_terminal_input_order_decision(
    predictor: Predictor,
) -> None:
    (result,) = _route(
        predictor,
        [SourceAuthorFields(first_name="Wei", last_name="Wang")],
    )
    resolved = result

    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == ("Wei", "", "Wang")
    assert resolved.resolution_provenance is ResolutionProvenance.PP
    assert resolved.resolution_action is ResolutionAction.PRESERVE_INPUT
    assert resolved.resolution_reason is ResolutionReason.PP_ONLY_ABSTAIN_INPUT


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        pytest.param(SourceAuthorFields(first_name="王", last_name="伟"), ("Wei", "Wang"), id="wang-wei-chinese"),
        pytest.param(SourceAuthorFields(first_name="张", last_name="伟"), ("Wei", "Zhang"), id="zhang-wei-chinese"),
        pytest.param(SourceAuthorFields(first_name="金", last_name="正日"), ("正日", "金"), id="kim-jong-il-korean-hanja"),
        pytest.param(SourceAuthorFields(first_name="濱", last_name="定史"), ("定史", "濱"), id="hama-sadahumi-japanese"),
    ],
)
def test_pp_only_native_abstain_uses_non_chinese_scalar_evidence(
    predictor: Predictor,
    source: SourceAuthorFields,
    expected: tuple[str, str],
) -> None:
    """Chinese controls keep PP while Japanese/Korean-Hanja rows keep native script."""
    (resolved,) = _route(predictor, [source])

    assert (resolved.first_name, resolved.last_name) == expected
    if source.first_name in {"王", "张"}:
        assert resolved.resolution_provenance is ResolutionProvenance.PP
        assert resolved.resolution_reason is ResolutionReason.PP_ONLY_ABSTAIN_INPUT
    else:
        assert resolved.resolution_provenance is ResolutionProvenance.SCALAR
        assert resolved.resolution_reason is ResolutionReason.SCALAR_BASELINE


def test_pp_only_native_abstain_gate_scores_the_space_preserving_surface() -> None:
    scorer_inputs: list[str] = []

    assert _pp_only_native_abstain_prefers_scalar(
        "金 正日",
        lambda surface: scorer_inputs.append(surface) or 0.8,
    )
    assert scorer_inputs == ["金 正日"]


@pytest.mark.parametrize("surface", ["金John", "金 John", "金正日"])
def test_pp_only_native_abstain_gate_rejects_mixed_or_unspaced_surfaces(surface: str) -> None:
    def unexpected_score(_surface: str) -> float:
        pytest.fail("ineligible surfaces must not reach the Japanese classifier")

    assert not _pp_only_native_abstain_prefers_scalar(surface, unexpected_score)


@pytest.mark.parametrize(
    "source",
    [
        SourceAuthorFields(first_name="Huong Yong", last_name="Ting"),
        SourceAuthorFields(first_name="Huong", middle_names="Yong", last_name="Ting"),
        SourceAuthorFields(last_name="Huong Yong Ting"),
    ],
    ids=["compound-first", "split-first-middle", "packed-last"],
)
def test_identity_backed_huong_yong_ting_writes_atomic_fields(
    predictor: Predictor,
    source: SourceAuthorFields,
) -> None:
    (result,) = _route(predictor, [source])

    assert (result.first_name, result.middle_names, result.last_name, result.suffix) == (
        "Huong-Yong",
        "",
        "Ting",
        None,
    )
    assert result.resolution_provenance is ResolutionProvenance.SCALAR
    assert result.resolution_action is ResolutionAction.ASSIGN
    assert result.resolution_reason is ResolutionReason.IDENTITY_BACKED_EXACT_ASSIGNMENT


@pytest.mark.parametrize(
    ("source_given", "expected_given"),
    [
        ("Hoai", "Hoai"),
        ("Toan", "Toan"),
        ("Ho\u00e0i", "Hoai"),
        ("To\u00e0n", "Toan"),
    ],
)
def test_timo_preserves_reviewed_vietnamese_atomic_given_fields(
    predictor: Predictor,
    source_given: str,
    expected_given: str,
) -> None:
    source = SourceAuthorFields(first_name=source_given, last_name="Wang")
    (result,) = _route(predictor, [source])

    assert (result.first_name, result.middle_names, result.last_name) == (expected_given, "", "Wang")


@pytest.mark.parametrize(
    "hyphen",
    sorted(ROMAN_HYPHEN_LIKE, key=ord),
    ids=lambda character: f"U+{ord(character):04X}",
)
def test_pp_only_compound_surname_guard_matches_structural_roman_hyphens(
    predictor: Predictor,
    hyphen: str,
) -> None:
    source = SourceAuthorFields(first_name="Ka Ming", last_name=f"Au{hyphen}Yeung")

    (result,) = _route(predictor, [source])

    resolved = result
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
    predictor: Predictor,
    leading: str,
    trailing: str,
) -> None:
    source = SourceAuthorFields(first_name="Ka Ming", last_name=f"{leading}AU\u2011YEUNG{trailing}")

    (result,) = _route(predictor, [source])

    resolved = result
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
    predictor: Predictor,
    hyphen: str,
) -> None:
    source = SourceAuthorFields(first_name="Ka Ming", last_name=f"Au{hyphen}Yeung")

    (result,) = _route(predictor, [source])

    assert result.resolution_reason is not ResolutionReason.PP_ONLY_ABSTAIN_REVIEWED_COMPOUND_SURNAME


@pytest.mark.parametrize(
    "last_name",
    ["Au/Yeung", "Au \u2011 Yeung", "Au\u2011Yung", "Chan\u2011Yeung"],
)
def test_pp_only_compound_surname_guard_keeps_exact_reviewed_membership(
    predictor: Predictor,
    last_name: str,
) -> None:
    source = SourceAuthorFields(first_name="Ka Ming", last_name=last_name)

    (result,) = _route(predictor, [source])

    assert result.resolution_reason is not ResolutionReason.PP_ONLY_ABSTAIN_REVIEWED_COMPOUND_SURNAME


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
    predictor: Predictor,
    middle_names: str | None,
) -> None:
    source = SourceAuthorFields(first_name="A,", middle_names=middle_names, last_name="Smith")

    (result,) = _route(predictor, [source])

    resolved = result
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


def test_reorder_veto_does_not_use_vys_tail_as_paper_context(predictor: Predictor) -> None:
    (result,) = _route(
        predictor,
        [SourceAuthorFields(first_name="Satomi", last_name="Miwa")],
        vys_other_names=["Akira Suzuki"],
    )

    assert (result.first_name, result.last_name) == ("Miwa", "Satomi")
    assert result.resolution_reason is ResolutionReason.SCALAR_BASELINE


def test_possible_japanese_surname_repairs_wrong_batch_reversal_without_vys_context(
    predictor: Predictor,
) -> None:
    (result,) = _route(
        predictor,
        [SourceAuthorFields(first_name="Mai", last_name="Hata")],
    )

    assert (result.first_name, result.last_name) == ("Mai", "Hata")
    assert result.resolution_reason is ResolutionReason.JAPANESE_GIVEN_FIRST_REORDER_VETO_PRESERVE_INPUT


def test_failed_pp_vys_abstain_cannot_fall_through_to_scalar_reorder(
    predictor: Predictor,
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


def test_timo_runs_one_scalar_evidence_pass_per_focal_author(
    predictor: Predictor,
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
    predictor = Predictor(PredictorConfig(parallel="never"), ".")
    predictor._detector._ensure_initialized()  # noqa: SLF001
    service = predictor._detector._batch_analysis_service  # noqa: SLF001
    assert service is not None

    def programming_error(*_args, **_kwargs):
        message = "synthetic batch programming error"
        raise RuntimeError(message)

    monkeypatch.setattr(service, "analyze_related_name_batches", programming_error)

    with pytest.raises(RuntimeError, match="synthetic batch programming error"):
        _route(predictor, [SourceAuthorFields(first_name="Michael", last_name="Johnson")])


def test_predict_batch_forwards_execution_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    config = PredictorConfig(
        parallel="always",
        mp_max_workers=3,
        mp_chunk_size=7,
        mp_min_parallel_batches=2,
        mp_start_method="spawn",
    )
    predictor = Predictor(config, ".")
    captured: dict[str, object] = {}

    def analyze(requests, **kwargs):
        captured["requests"] = requests
        captured.update(kwargs)
        return []

    monkeypatch.setattr(predictor._detector, "_analyze_related_batch_requests", analyze)  # noqa: SLF001

    assert predictor.predict_batch([]) == []
    assert captured == {
        "requests": [],
        "parallel": "always",
        "min_parallel_batches": 2,
        "max_workers": 3,
        "chunk_size": 7,
        "mp_start_method": "spawn",
    }


def test_reordered_pp_result_is_an_invariant_failure(
    predictor: Predictor,
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
    predictor: Predictor,
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
    predictor: Predictor,
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
    predictor: Predictor,
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
    predictor: Predictor,
) -> None:
    source = SourceAuthorFields(first_name="Zhang", middle_names="Wei", last_name="Q.")

    (pp_only,) = _route(predictor, [source])
    (present_vys,) = _route(predictor, [source], vys_other_names=[])

    assert set(pp_only.dict()) == {
        "first_name",
        "middle_names",
        "last_name",
        "suffix",
        "resolution_provenance",
        "resolution_action",
        "resolution_reason",
    }
    assert (
        pp_only.first_name,
        pp_only.middle_names,
        pp_only.last_name,
    ) == ("Zhang", "Wei", "Q.")
    assert pp_only == present_vys


def test_timo_preserves_unicode_apostrophe_display(
    predictor: Predictor,
) -> None:
    source = SourceAuthorFields(first_name="Xiang\u02bban", last_name="Yan")

    (result,) = _route(predictor, [source])

    resolved = result
    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == ("Xiang'an", "", "Yan")


@pytest.mark.parametrize("suffix", [None, "III", "Jr."])
def test_timo_uses_last_name_field_as_surname_first_initial_evidence(
    predictor: Predictor,
    suffix: str | None,
) -> None:
    (result,) = _route(
        predictor,
        [SourceAuthorFields(last_name="Masterov R. A.", suffix=suffix)],
    )

    resolved = result
    assert (resolved.first_name, resolved.middle_names, resolved.last_name, resolved.suffix) == (
        "R.",
        "A.",
        "Masterov",
        suffix,
    )
    assert resolved.resolution_provenance is ResolutionProvenance.SOURCE
    assert resolved.resolution_action is ResolutionAction.ASSIGN
    assert resolved.resolution_reason is ResolutionReason.STRUCTURED_SURNAME_INITIAL_TAIL_ASSIGNMENT


def test_timo_config_registers_one_unversioned_variant() -> None:
    config = (Path(__file__).parents[1] / "sinonym" / "timo" / "config.yaml").read_text(encoding="utf-8")

    assert config.count("  sinonym:") == 1
    assert "sinonym_v" not in config
    assert "sinonym_routing" not in config
    assert "instance: sinonym.timo.interface.Instance" in config
    assert "prediction: sinonym.timo.interface.Prediction" in config
    assert "predictor: sinonym.timo.interface.Predictor" in config
    assert "predictor_config: sinonym.timo.interface.PredictorConfig" in config
    assert "integration_test: sinonym.timo.integration_test.TestIntegration" in config
    assert "cuda: false" in config
