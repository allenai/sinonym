"""Closed paper-context repairs from the PR29/PR30 manual diff audit."""

from __future__ import annotations

from sinonym.coretypes.routing_resolution import (
    ResolutionAction,
    ResolutionProvenance,
    ResolutionReason,
)
from sinonym.timo.interface import Instance, Predictor, ResolvedAuthorFields, SourceAuthorFields


def _route_paper(predictor: Predictor, sources: list[SourceAuthorFields]) -> list[ResolvedAuthorFields]:
    (paper,) = predictor.predict_batch([Instance(pp_authors=sources)])
    return paper.authors


def _assert_source_assignment(
    resolved: ResolvedAuthorFields,
    reason: ResolutionReason = ResolutionReason.REVIEWED_SOURCE_PATTERN_ASSIGNMENT,
) -> None:
    assert resolved.resolution_provenance is ResolutionProvenance.SOURCE
    assert resolved.resolution_action is ResolutionAction.ASSIGN
    assert resolved.resolution_reason is reason


def test_repeated_full_name_column_is_removed_under_complete_paper_consensus(predictor: Predictor) -> None:
    sources = [
        SourceAuthorFields(first_name="Sumio", middle_names="Okuyama", last_name="Sumio Okuyama"),
        SourceAuthorFields(first_name="Mariam", middle_names="Tsiklauri", last_name="Mariam Tsiklauri"),
    ]

    resolved = _route_paper(predictor, sources)

    assert [(row.first_name, row.middle_names, row.last_name) for row in resolved] == [
        ("Sumio", "", "Okuyama"),
        ("Mariam", "", "Tsiklauri"),
    ]
    for row in resolved:
        _assert_source_assignment(row)


def test_repeated_full_name_column_requires_complete_paper_consensus(predictor: Predictor) -> None:
    resolved = _route_paper(
        predictor,
        [
            SourceAuthorFields(first_name="Sumio", middle_names="Okuyama", last_name="Sumio Okuyama"),
            SourceAuthorFields(first_name="Jane", last_name="Smith"),
        ],
    )[0]

    assert resolved.resolution_reason is not ResolutionReason.REVIEWED_SOURCE_PATTERN_ASSIGNMENT


def test_uniform_titled_given_fields_are_flipped_and_titles_removed(predictor: Predictor) -> None:
    sources = [
        SourceAuthorFields(first_name="Lotti", last_name="Prof. Francesco"),
        SourceAuthorFields(first_name="Cipriani", last_name="Dr. Sarah"),
        SourceAuthorFields(first_name="Gasperetti", last_name="Dr. Beatrice"),
        SourceAuthorFields(first_name="Baraddi", last_name="Dr. Carolina"),
        SourceAuthorFields(first_name="Ciambrone", last_name="Dr. Viviana"),
        SourceAuthorFields(first_name="Maggi", last_name="Prof. Mario"),
    ]

    resolved = _route_paper(predictor, sources)

    assert [(row.first_name, row.middle_names, row.last_name) for row in resolved] == [
        ("Francesco", "", "Lotti"),
        ("Sarah", "", "Cipriani"),
        ("Beatrice", "", "Gasperetti"),
        ("Carolina", "", "Baraddi"),
        ("Viviana", "", "Ciambrone"),
        ("Mario", "", "Maggi"),
    ]
    for row in resolved:
        _assert_source_assignment(row)


def test_titled_given_inversion_requires_a_uniform_multi_author_paper(predictor: Predictor) -> None:
    resolved = _route_paper(
        predictor,
        [
            SourceAuthorFields(first_name="Cipriani", last_name="Dr. Sarah"),
            SourceAuthorFields(first_name="Jane", last_name="Smith"),
        ],
    )[0]

    assert resolved.resolution_reason is not ResolutionReason.REVIEWED_SOURCE_PATTERN_ASSIGNMENT


def test_uniform_dotted_surname_initial_sequence_is_rotated(predictor: Predictor) -> None:
    sources = [
        SourceAuthorFields(first_name="Etchike", last_name="D. A. B."),
        SourceAuthorFields(first_name="Ngassoum", last_name="M. B."),
        SourceAuthorFields(first_name="Mapongmetsem", last_name="P. M."),
    ]

    resolved = _route_paper(predictor, sources)

    assert [(row.first_name, row.middle_names, row.last_name) for row in resolved] == [
        ("D.", "A. B.", "Etchike"),
        ("M.", "B.", "Ngassoum"),
        ("P.", "M.", "Mapongmetsem"),
    ]
    for row in resolved:
        _assert_source_assignment(row, ResolutionReason.STRUCTURED_SURNAME_INITIAL_TAIL_ASSIGNMENT)


def test_uniform_surname_initial_sequence_requires_explicit_periods(predictor: Predictor) -> None:
    resolved = _route_paper(
        predictor,
        [
            SourceAuthorFields(first_name="Etchike", last_name="D A B"),
            SourceAuthorFields(first_name="Ngassoum", last_name="M B"),
        ],
    )

    assert all(row.resolution_reason is not ResolutionReason.STRUCTURED_SURNAME_INITIAL_TAIL_ASSIGNMENT for row in resolved)


def test_second_reviewed_surname_initial_roster_is_rotated(predictor: Predictor) -> None:
    sources = [
        SourceAuthorFields(first_name="Tsapaeva", last_name="N. L."),
        SourceAuthorFields(first_name="Tsapaev", last_name="V. G."),
    ]

    resolved = _route_paper(predictor, sources)

    assert [(row.first_name, row.middle_names, row.last_name) for row in resolved] == [
        ("N.", "L.", "Tsapaeva"),
        ("V.", "G.", "Tsapaev"),
    ]
    for row in resolved:
        _assert_source_assignment(row, ResolutionReason.STRUCTURED_SURNAME_INITIAL_TAIL_ASSIGNMENT)


def test_paper_wide_trailing_affiliation_marker_is_removed(predictor: Predictor) -> None:
    sources = [
        SourceAuthorFields(first_name="ing", last_name="Zhanga"),
        SourceAuthorFields(first_name="Yuan", last_name="Hua"),
        SourceAuthorFields(first_name="Lei", last_name="Songa"),
        SourceAuthorFields(first_name="Hongdian", last_name="Lua"),
        SourceAuthorFields(first_name="Jian", last_name="Wanga"),
        SourceAuthorFields(first_name="Qingqing", last_name="Liua"),
    ]

    resolved = _route_paper(predictor, sources)

    assert [(row.first_name, row.middle_names, row.last_name) for row in resolved] == [
        ("Ing", "", "Zhang"),
        ("Yuan", "", "Hu"),
        ("Lei", "", "Song"),
        ("Hongdian", "", "Lu"),
        ("Jian", "", "Wang"),
        ("Qingqing", "", "Liu"),
    ]
    for row in resolved:
        _assert_source_assignment(row)


def test_second_reviewed_trailing_affiliation_roster_is_repaired(predictor: Predictor) -> None:
    sources = [
        SourceAuthorFields(first_name="Su-Sui", last_name="Lina"),
        SourceAuthorFields(first_name="Wei-Shen", last_name="Taia"),
        SourceAuthorFields(first_name="Kwo-Ting", last_name="Fanga"),
    ]

    resolved = _route_paper(predictor, sources)

    assert [(row.first_name, row.middle_names, row.last_name) for row in resolved] == [
        ("Su-Sui", "", "Lin"),
        ("Wei-Shen", "", "Tai"),
        ("Kwo-Ting", "", "Fang"),
    ]
    for row in resolved:
        _assert_source_assignment(row)


def test_trailing_affiliation_marker_requires_an_exact_reviewed_roster(predictor: Predictor) -> None:
    resolved = _route_paper(
        predictor,
        [
            SourceAuthorFields(first_name="Ali", last_name="Kaya"),
            SourceAuthorFields(first_name="Yasemin", last_name="Kaya"),
            SourceAuthorFields(first_name="Mehmet", last_name="Kaya"),
        ],
    )

    assert [row.last_name for row in resolved] == ["Kaya", "Kaya", "Kaya"]
    assert all(row.resolution_reason is not ResolutionReason.REVIEWED_SOURCE_PATTERN_ASSIGNMENT for row in resolved)


def test_peer_supported_detached_diacritics_are_composed(predictor: Predictor) -> None:
    sources = [
        SourceAuthorFields(first_name="Jose", middle_names="Mar \u00b4", last_name="ia"),
        SourceAuthorFields(first_name="Mart", middle_names="\u00b4", last_name="inez"),
    ]

    resolved = _route_paper(predictor, sources)

    assert [(row.first_name, row.middle_names, row.last_name) for row in resolved] == [
        ("Jose", "", "María"),
        ("", "", "Martínez"),
    ]
    for row in resolved:
        _assert_source_assignment(row)


def test_detached_diacritic_repair_requires_a_second_valid_peer(predictor: Predictor) -> None:
    resolved = _route_paper(
        predictor,
        [SourceAuthorFields(first_name="Mart", middle_names="\u00b4", last_name="inez")],
    )[0]

    assert resolved.resolution_reason is not ResolutionReason.REVIEWED_SOURCE_PATTERN_ASSIGNMENT


def test_detached_diacritic_valid_peer_paper_rejects_an_invalid_focal(predictor: Predictor) -> None:
    sources = [
        SourceAuthorFields(first_name="Jose", middle_names="Mar \u00b4", last_name="ia"),
        SourceAuthorFields(first_name="Mart", middle_names="\u00b4", last_name="inez"),
        SourceAuthorFields(first_name="Ana", middle_names="Sa \u00b4", last_name="di"),
    ]

    resolved = _route_paper(predictor, sources)

    assert resolved[0].last_name == "María"
    assert resolved[1].last_name == "Martínez"
    assert resolved[2].resolution_reason is not ResolutionReason.REVIEWED_SOURCE_PATTERN_ASSIGNMENT


def test_south_indian_terminal_initial_triad_keeps_scalar_boundary(predictor: Predictor) -> None:
    sources = [
        SourceAuthorFields(first_name="Vijay", last_name="Chander S"),
        SourceAuthorFields(first_name="Ramakrishnan", last_name="K"),
        SourceAuthorFields(first_name="Muthu", last_name="D"),
    ]

    resolved = _route_paper(predictor, sources)

    assert (resolved[0].first_name, resolved[0].middle_names, resolved[0].last_name) == ("Vijay", "Chander", "S")
    assert resolved[0].resolution_reason is ResolutionReason.SCALAR_BASELINE
    assert [(row.first_name, row.middle_names, row.last_name) for row in resolved[1:]] == [
        ("Ramakrishnan", "", "K"),
        ("Muthu", "", "D"),
    ]


def test_south_indian_terminal_initial_exception_does_not_expand_to_four_authors(predictor: Predictor) -> None:
    sources = [
        SourceAuthorFields(first_name="Vijay", last_name="Chander S"),
        SourceAuthorFields(first_name="Ramakrishnan", last_name="K"),
        SourceAuthorFields(first_name="Muthu", last_name="D"),
        SourceAuthorFields(first_name="Other", last_name="P"),
    ]

    resolved = _route_paper(predictor, sources)[0]

    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == ("Vijay", "", "Chander S")
    assert resolved.resolution_reason is ResolutionReason.SCALAR_CLEAN_SOURCE_SURNAME_REPARTITION_ASSIGNMENT


def test_peer_supported_joined_uppercase_surname_prefix_is_assigned(predictor: Predictor) -> None:
    sources = [
        SourceAuthorFields(last_name="GUOWei"),
        SourceAuthorFields(last_name="XIONGZhong-wei"),
        SourceAuthorFields(last_name="LIZhi-tang"),
    ]

    resolved = _route_paper(predictor, sources)

    assert [(row.first_name, row.middle_names, row.last_name) for row in resolved] == [
        ("Wei", "", "Guo"),
        ("Zhong-wei", "", "Xiong"),
        ("Zhi-tang", "", "Li"),
    ]
    for row in resolved:
        _assert_source_assignment(row)


def test_joined_uppercase_rule_requires_recognized_peer_prefixes(predictor: Predictor) -> None:
    resolved = _route_paper(
        predictor,
        [SourceAuthorFields(last_name="QYLuo"), SourceAuthorFields(last_name="DRSpolaore")],
    )

    assert all(row.resolution_reason is not ResolutionReason.REVIEWED_SOURCE_PATTERN_ASSIGNMENT for row in resolved)


def test_joined_uppercase_rule_keeps_an_already_aligned_candidate_unchanged(predictor: Predictor) -> None:
    sources = [
        SourceAuthorFields(last_name="LIGuowu"),
        SourceAuthorFields(last_name="YANGGuangming"),
        SourceAuthorFields(last_name="MAZhengsheng"),
        SourceAuthorFields(last_name="SHINicheng"),
        SourceAuthorFields(last_name="XIONGMing"),
        SourceAuthorFields(last_name="FANHaifu"),
        SourceAuthorFields(last_name="SHENGGanfu"),
    ]

    resolved = _route_paper(predictor, sources)[0]

    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == ("Guo-Wu", "", "Li")
    assert resolved.resolution_reason is ResolutionReason.PP_SELECTED


def test_strong_chinese_context_bypasses_broad_japanese_veto_for_ma_kai(predictor: Predictor) -> None:
    sources = [
        SourceAuthorFields(first_name="Gu", last_name="Qianlei"),
        SourceAuthorFields(first_name="Zhang", last_name="Wanfu"),
        SourceAuthorFields(first_name="Chen", last_name="Luqi"),
        SourceAuthorFields(first_name="Ma", last_name="Kai"),
        SourceAuthorFields(first_name="Li", last_name="Chun"),
        SourceAuthorFields(first_name="Yang", last_name="Jian-gang"),
    ]

    resolved = _route_paper(predictor, sources)[3]

    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == ("Kai", "", "Ma")
    assert resolved.resolution_reason is ResolutionReason.PP_SELECTED
    assert resolved.resolution_provenance is ResolutionProvenance.PP
    assert resolved.resolution_action is ResolutionAction.ASSIGN
