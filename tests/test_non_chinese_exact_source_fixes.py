"""Full-corpus-reviewed exact source-tuple corrections."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias, cast

import pytest

from sinonym.coretypes import NameComponents
from sinonym.coretypes.routing_resolution import (
    ResolutionAction,
    ResolutionProvenance,
    ResolutionReason,
)
from sinonym.timo.routing_v3 import (
    ResolvedAuthorFields,
    RoutingInstanceV3,
    SourceAuthorFields,
    reviewed_exact_source_assignment,
    reviewed_exact_source_reversal,
    reviewed_fullwidth_katakana_alias_assignment,
    reviewed_katakana_middle_period_cyclic_reversal,
    reviewed_leading_jr_peer_assignment,
)

if TYPE_CHECKING:
    from sinonym.timo.interface import RoutingPredictorV3

_PATTERN_ORACLE = [
    json.loads(line)
    for line in (Path(__file__).parent / "data" / "reviewed_source_pattern_assignments.jsonl")
    .read_text(encoding="utf-8")
    .splitlines()
]
_JR_PATTERN_ORACLE = [row for row in _PATTERN_ORACLE if row["kind"] == "jr"]
_FULLWIDTH_PATTERN_ORACLE = [row for row in _PATTERN_ORACLE if row["kind"] == "fullwidth"]
_COMMA_CREDENTIAL_PATTERN_ORACLE = [row for row in _PATTERN_ORACLE if row["kind"] == "comma_credential"]
_CANONICAL_COMMA_CREDENTIAL_INITIALS = {
    "260895607:1": ("Jeffery", "R.", "Kimber", None),
    "38537137:0": ("John", "P. K.", "O'Dea", None),
    "561870:4": ("Mark", "F.", "O'Brien", None),
    "33720872:0": ("Mark", "F. H.", "Brougham", None),
    "73362146:3": ("Raja", "J.", "Selvaraj", None),
}
_SourceParts: TypeAlias = tuple[str | None, str | None, str | None, str | None]


@pytest.fixture(scope="module")
def predictor(routing_predictor_v3: RoutingPredictorV3) -> RoutingPredictorV3:
    """Alias the shared session predictor under this module's name."""
    return routing_predictor_v3


def _route(predictor: RoutingPredictorV3, source: SourceAuthorFields):
    (paper,) = predictor.predict_batch([RoutingInstanceV3(pp_authors=[source])])
    return paper.authors[0].resolved_fields


def _route_paper(predictor: RoutingPredictorV3, sources: list[SourceAuthorFields]):
    (paper,) = predictor.predict_batch([RoutingInstanceV3(pp_authors=sources)])
    return [author.resolved_fields for author in paper.authors]


def _source(parts: _SourceParts) -> SourceAuthorFields:
    first, middle, last, suffix = parts
    return SourceAuthorFields(first_name=first, middle_names=middle, last_name=last, suffix=suffix)


@pytest.mark.parametrize(
    ("source_parts", "expected"),
    [
        (("Abramova", "", "Na"), ("Na", "", "Abramova")),
        (("Arbabi", "", "Masoud"), ("Masoud", "", "Arbabi")),
        (("Bakulina", "", "Li"), ("Li", "", "Bakulina")),
        (("Barani", "", "Hossein"), ("Hossein", "", "Barani")),
        (("Batista", "", "Juanize Matias da Silva"), ("Juanize Matias da Silva", "", "Batista")),
        (("Boriskova", "", "Pi"), ("Pi", "", "Boriskova")),
        (("Cho", "", "Yk"), ("Yk", "", "Cho")),
        (("Clyman", "", "Mj"), ("Mj", "", "Clyman")),
        (("Dahl", "", "Mm"), ("Mm", "", "Dahl")),
        (("Firmbach", "", "F-P."), ("F.-P.", "", "Firmbach")),
        (("Gol'dman", "", "An"), ("An", "", "Gol'dman")),
        (("Grube", "", "Mr"), ("Mr", "", "Grube")),
        (("Ha", "", "Sh"), ("Sh", "", "Ha")),
        (("Hashimoto", "", "Keiichi"), ("Keiichi", "", "Hashimoto")),
        (("Ho", "", "Jm"), ("Jm", "", "Ho")),
        (("Hori", "", "Maiya"), ("Maiya", "", "Hori")),
        (("Im", "", "Jj"), ("Jj", "", "Im")),
        (("Iwasaki", "", "Tohru"), ("Tohru", "", "Iwasaki")),
        (("Khurs", "", "En"), ("En", "", "Khurs")),
        (("Kim", "", "Jina"), ("Jina", "", "Kim")),
        (("Kim", "", "Jy"), ("Jy", "", "Kim")),
        (("Kim", "", "Namseok"), ("Namseok", "", "Kim")),
        (("Kim", "", "WoanSub"), ("WoanSub", "", "Kim")),
        (("Kim", "", "Ys"), ("Ys", "", "Kim")),
        (("Korolev", "", "Vv"), ("Vv", "", "Korolev")),
        (("Kwon", "", "Yong"), ("Yong", "", "Kwon")),
        (("Lemomu", "", "KM"), ("KM", "", "Lemomu")),
        (("Mao", "", "Kai"), ("Kai", "", "Mao")),
        (("Nishanov", "", "D.A"), ("D.", "A.", "Nishanov")),
        (("Pappanikou", "", "Aj"), ("Aj", "", "Pappanikou")),
        (("Park", "", "Sh"), ("Sh", "", "Park")),
        (("Podol'nikova", "", "Np"), ("Np", "", "Podol'nikova")),
        (("Rorem", "", "Da"), ("Da", "", "Rorem")),
        (("Seo", "", "MyeongWhoon"), ("MyeongWhoon", "", "Seo")),
        (("Sodimu", "", "Isiaka"), ("Isiaka", "", "Sodimu")),
        (("Sugino", "", "Eiichi"), ("Eiichi", "", "Sugino")),
        (("Veselov", "", "Vf"), ("Vf", "", "Veselov")),
        (("Wada", "", "Shin-ichi"), ("Shin-ichi", "", "Wada")),
        (("\u4e2d\u5c71", "", "\u8fc5"), ("\u8fc5", "", "\u4e2d\u5c71")),
        (("\u6842", "", "\u7460\u4ee5"), ("\u7460\u4ee5", "", "\u6842")),
        (("برخورداری،", "", "وحید"), ("وحید", "", "برخورداری")),
        (("دعایی،", "", "فریما"), ("فریما", "", "دعایی")),
        (("昌谷", "", "忠海"), ("忠海", "", "昌谷")),
        (("赫勒", "", "M"), ("M.", "", "赫勒")),
        (("Augustin", "Mary", "Ann"), ("Mary", "Ann", "Augustin")),
        (("Choi", "Seung", "Wook"), ("Seung", "Wook", "Choi")),
        (("Do", "Thi Kim", "Lanh"), ("Lanh", "Thi Kim", "Do")),
        (("Karkabounas", "Spyridon", "Ch."), ("Spyridon", "Ch.", "Karkabounas")),
        (("Kim", "Sun", "Hyoung"), ("Sun", "Hyoung", "Kim")),
        (("Lee", "Joo", "Youn"), ("Joo", "Youn", "Lee")),
        (("Mai", "Dac", "Bien"), ("Bien", "Dac", "Mai")),
        (("Kim", "Tae", "In"), ("Tae", "In", "Kim")),
        (("Lee", "Joo", "Hee"), ("Joo", "Hee", "Lee")),
        (("Tormos", "Josep", "Maria"), ("Josep", "Maria", "Tormos")),
        pytest.param(
            ("モンゴメリー\uff0c", "エイチ\uff0e", "マンニング\uff0c"),
            ("エイチ.", "モンゴメリー", "マンニング"),
            id="montgomery-catalog-internal-roles",
        ),
        pytest.param(
            ("Fernando", "del.", "Pulgar"),
            ("Fernando", "", "del Pulgar"),
            id="fernando-del-pulgar-particle-period",
        ),
        pytest.param(
            ("Min", "-Fu", "Tsan"),
            ("Min-Fu", "", "Tsan"),
            id="min-fu-tsan-split-given-name",
        ),
    ],
)
def test_reviewed_exact_source_assignments_are_terminal(
    predictor: RoutingPredictorV3,
    source_parts: tuple[str, str, str],
    expected: tuple[str, str, str],
) -> None:
    first, middle, last = source_parts
    source = SourceAuthorFields(first_name=first, middle_names=middle or None, last_name=last)

    resolved = _route(predictor, source)

    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == expected
    assert resolved.resolution_provenance is ResolutionProvenance.SOURCE
    assert resolved.resolution_action is ResolutionAction.ASSIGN
    assert resolved.resolution_reason is ResolutionReason.REVIEWED_EXACT_SOURCE_ASSIGNMENT


@pytest.mark.parametrize(
    "reason",
    [
        ResolutionReason.NON_PERSON_SOURCE_PASSTHROUGH,
        ResolutionReason.REVIEWED_NON_PERSON_PATTERN,
        ResolutionReason.HANDLED_EVIDENCE_FAILURE,
    ],
)
def test_source_non_assignments_preserve_initial_spelling_byte_exact(reason: ResolutionReason) -> None:
    source = SourceAuthorFields(first_name="A", middle_names="D", last_name="UNIVERSITY", suffix=" raw ")

    resolved = ResolvedAuthorFields.from_source(source, reason=reason)

    assert (resolved.first_name, resolved.middle_names, resolved.last_name, resolved.suffix) == (
        "A",
        "D",
        "UNIVERSITY",
        " raw ",
    )


@pytest.mark.parametrize(
    "source",
    [
        SourceAuthorFields(first_name="Kai", last_name="Zenger"),
        SourceAuthorFields(first_name="Masaki", last_name="Morishige"),
        SourceAuthorFields(first_name="Miki", last_name="Toyota"),
        SourceAuthorFields(first_name="Shinsei", last_name="Ryu"),
    ],
)
def test_reviewed_exact_reversals_preserve_source_order(
    predictor: RoutingPredictorV3,
    source: SourceAuthorFields,
) -> None:
    resolved = _route(predictor, source)

    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == (
        source.first_name,
        "",
        source.last_name,
    )
    assert resolved.resolution_provenance is ResolutionProvenance.SOURCE
    assert resolved.resolution_action is ResolutionAction.PRESERVE_INPUT
    assert resolved.resolution_reason is ResolutionReason.REVIEWED_EXACT_SOURCE_REORDER_VETO_PRESERVE_INPUT


@pytest.mark.parametrize(
    "source",
    [
        SourceAuthorFields(first_name="Kim", middle_names="A", last_name="Jy"),
        SourceAuthorFields(first_name="Kim", last_name="Nam-Seok"),
        SourceAuthorFields(first_name="Kim", middle_names="A", last_name="Ys"),
        SourceAuthorFields(first_name="Park", middle_names="A", last_name="Sh"),
        SourceAuthorFields(first_name="Han", middle_names="W", last_name="Tun"),
        SourceAuthorFields(first_name="Kai", middle_names="A", last_name="Zenger"),
        SourceAuthorFields(first_name="Miki", middle_names="A", last_name="Toyota"),
        SourceAuthorFields(first_name="Shinsei", middle_names="A", last_name="Ryu"),
        SourceAuthorFields(first_name="\u4e2d\u5c71", last_name="\u7406"),
        SourceAuthorFields(first_name="\u6842", last_name="\u7460\u8863"),
    ],
)
def test_exact_source_rules_ignore_nearby_tuples(
    source: SourceAuthorFields,
) -> None:
    assert reviewed_exact_source_assignment(source) is None
    assert not reviewed_exact_source_reversal(
        source,
        NameComponents(given_name=source.last_name or "", surname=source.first_name or ""),
    )


@pytest.mark.parametrize(
    "source",
    [
        SourceAuthorFields(first_name="Han", middle_names="W.", last_name="Tun"),
        SourceAuthorFields(first_name="Kai", last_name="Zenger"),
        SourceAuthorFields(first_name="Masaki", last_name="Morishige"),
        SourceAuthorFields(first_name="Miki", last_name="Toyota"),
        SourceAuthorFields(first_name="Shinsei", last_name="Ryu"),
    ],
)
def test_exact_source_reversal_veto_requires_an_exact_reversal_candidate(
    source: SourceAuthorFields,
) -> None:
    assert reviewed_exact_source_reversal(
        source,
        NameComponents(
            given_name=source.last_name or "",
            middle_name=source.middle_names or "",
            surname=source.first_name or "",
        ),
    )
    assert not reviewed_exact_source_reversal(
        source,
        NameComponents(given_name=source.first_name or "", surname=source.last_name or ""),
    )
    assert not reviewed_exact_source_reversal(
        source,
        NameComponents(given_name=f"{source.last_name or ''}x", surname=source.first_name or ""),
    )


def test_reviewed_agudelo_sepulveda_preserves_the_complete_source_surname(
    predictor: RoutingPredictorV3,
) -> None:
    source = SourceAuthorFields(first_name="Natalia", last_name="Agudelo Sep\u00falveda")

    resolved = _route(predictor, source)

    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == (
        "Natalia",
        "",
        "Agudelo Sep\u00falveda",
    )
    assert resolved.resolution_provenance is ResolutionProvenance.SOURCE
    assert resolved.resolution_action is ResolutionAction.PRESERVE_INPUT
    assert resolved.resolution_reason is ResolutionReason.SCALAR_KNOWN_COMPOUND_SURNAME_PRESERVE_INPUT


def test_compound_surname_preservation_does_not_match_a_nearby_tuple(
    predictor: RoutingPredictorV3,
) -> None:
    resolved = _route(
        predictor,
        SourceAuthorFields(first_name="Natalie", last_name="Agudelo Sep\u00falveda"),
    )

    assert resolved.resolution_reason is not ResolutionReason.SCALAR_KNOWN_COMPOUND_SURNAME_PRESERVE_INPUT


@pytest.mark.parametrize(
    "source",
    [
        SourceAuthorFields(
            first_name="トーマス\uff0c",
            middle_names="ディー\uff0e",
            last_name="ウー\uff0c",
        ),
        SourceAuthorFields(
            first_name="デイビッド\uff0c",
            middle_names="リー\uff0e",
            last_name="スミッド\uff0c",
        ),
    ],
)
def test_katakana_middle_period_cyclic_rotation_preserves_source(
    predictor: RoutingPredictorV3,
    source: SourceAuthorFields,
) -> None:
    resolved = _route(predictor, source)

    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == (
        source.first_name,
        source.middle_names,
        source.last_name,
    )
    assert resolved.resolution_provenance is ResolutionProvenance.SOURCE
    assert resolved.resolution_action is ResolutionAction.PRESERVE_INPUT
    assert resolved.resolution_reason is ResolutionReason.INITIALS_COMMA_REORDER_VETO_PRESERVE_INPUT


@pytest.mark.parametrize(
    "source",
    [
        SourceAuthorFields(first_name="エルブルグ\uff0c", middle_names="ヨハネス", last_name="ヴァン\uff0c"),
        SourceAuthorFields(first_name="トーマス\uff0c", middle_names="ジュニア\uff0e", last_name="ウェルチ\uff0c"),
        SourceAuthorFields(first_name="トーマス\uff0c", middle_names="ジュニア・エー\uff0e", last_name="ウェルチ\uff0c"),
        SourceAuthorFields(first_name="Tom・\uff0c", middle_names="D・\uff0e", last_name="Wu・\uff0c"),
        SourceAuthorFields(first_name="Noa,", middle_names="M.", last_name="N,"),
    ],
)
def test_katakana_middle_period_veto_excludes_nearby_counterexamples(
    source: SourceAuthorFields,
) -> None:
    selected = NameComponents(
        given_name=source.middle_names or "",
        middle_name=(source.last_name or "").rstrip(",\uff0c\u3001"),
        surname=(source.first_name or "").rstrip(",\uff0c\u3001"),
    )

    assert not reviewed_katakana_middle_period_cyclic_reversal(source, selected)


def test_katakana_middle_period_veto_requires_exact_cyclic_candidate() -> None:
    source = SourceAuthorFields(
        first_name="トーマス\uff0c",
        middle_names="ディー\uff0e",
        last_name="ウー\uff0c",
    )

    assert not reviewed_katakana_middle_period_cyclic_reversal(
        source,
        NameComponents(given_name="トーマス", middle_name="ディー", surname="ウー"),
    )


@pytest.mark.parametrize(
    ("source_middle", "source_last", "peer"),
    [
        ("John B.", "Cobb", SourceAuthorFields(first_name="John", middle_names="B", last_name="Cobb")),
        (". R. T.", "Compton", SourceAuthorFields(first_name="R.", middle_names="T.", last_name="Compton")),
        ("J.M.", "Patrascu", SourceAuthorFields(first_name="J.", middle_names="M.", last_name="Patrascu")),
    ],
)
def test_reviewed_leading_jr_assignment_copies_the_exact_structured_peer(
    source_middle: str,
    source_last: str,
    peer: SourceAuthorFields,
) -> None:
    source = SourceAuthorFields(first_name="Jr.", middle_names=source_middle, last_name=source_last, suffix="")

    selected = reviewed_leading_jr_peer_assignment(source, [source, peer], 0)

    assert selected == NameComponents(
        given_name=peer.first_name or "",
        middle_name=peer.middle_names or "",
        surname=peer.last_name or "",
        suffix="Jr.",
    )


@pytest.mark.parametrize(
    ("first_name", "expected_suffix"),
    [
        pytest.param(" Jr ", "Jr", id="ascii-space"),
        pytest.param("\tJr.\r\n", "Jr.", id="ascii-controls"),
        pytest.param("\u2003jr.\u2003", "jr.", id="em-space"),
    ],
)
def test_reviewed_leading_jr_assignment_trims_only_the_decision_surface(
    first_name: str,
    expected_suffix: str,
) -> None:
    source = SourceAuthorFields(first_name=first_name, middle_names="John B.", last_name="Cobb")
    peer = SourceAuthorFields(first_name="John", middle_names="B", last_name="Cobb")

    selected = reviewed_leading_jr_peer_assignment(source, [source, peer], 0)

    assert selected == NameComponents(
        given_name="John",
        middle_name="B",
        surname="Cobb",
        suffix=expected_suffix,
    )
    assert source.first_name == first_name


def test_reviewed_leading_jr_assignment_is_terminal(predictor: RoutingPredictorV3) -> None:
    source = SourceAuthorFields(first_name="Jr.", middle_names="John B.", last_name="Cobb")
    peer = SourceAuthorFields(first_name="John", middle_names="B", last_name="Cobb")

    resolved = _route_paper(predictor, [source, peer])[0]

    assert (resolved.first_name, resolved.middle_names, resolved.last_name, resolved.suffix) == (
        "John",
        "B.",
        "Cobb",
        "Jr.",
    )
    assert resolved.resolution_provenance is ResolutionProvenance.SOURCE
    assert resolved.resolution_action is ResolutionAction.ASSIGN
    assert resolved.resolution_reason is ResolutionReason.REVIEWED_SOURCE_PATTERN_ASSIGNMENT


def test_reviewed_leading_jr_assignment_with_outer_whitespace_is_terminal(
    predictor: RoutingPredictorV3,
) -> None:
    source = SourceAuthorFields(first_name=" Jr. ", middle_names="John B.", last_name="Cobb")
    peer = SourceAuthorFields(first_name="John", middle_names="B", last_name="Cobb")

    resolved = _route_paper(predictor, [source, peer])[0]

    assert (resolved.first_name, resolved.middle_names, resolved.last_name, resolved.suffix) == (
        "John",
        "B.",
        "Cobb",
        "Jr.",
    )
    assert resolved.resolution_provenance is ResolutionProvenance.SOURCE
    assert resolved.resolution_action is ResolutionAction.ASSIGN
    assert resolved.resolution_reason is ResolutionReason.REVIEWED_SOURCE_PATTERN_ASSIGNMENT
    assert source.first_name == " Jr. "


def test_reviewed_leading_jr_assignment_nfkc_normalizes_the_organization_guard(
    predictor: RoutingPredictorV3,
) -> None:
    source = SourceAuthorFields(
        first_name=" Jr. ",
        middle_names="\uff35\uff4e\uff49\uff56\uff45\uff52\uff53\uff49\uff54\uff59",
        last_name="\uff22\uff4f\uff41\uff52\uff44",
    )
    peer = SourceAuthorFields(first_name="University", last_name="Board")

    assert reviewed_leading_jr_peer_assignment(source, [source, peer], 0) is None

    resolved = _route_paper(predictor, [source, peer])[0]
    assert (resolved.first_name, resolved.middle_names, resolved.last_name, resolved.suffix) == (
        source.first_name,
        source.middle_names,
        source.last_name,
        None,
    )
    assert resolved.resolution_provenance is ResolutionProvenance.SOURCE
    assert resolved.resolution_action is ResolutionAction.PRESERVE_INPUT
    assert resolved.resolution_reason is ResolutionReason.NON_PERSON_SOURCE_PASSTHROUGH


@pytest.mark.parametrize("case", _JR_PATTERN_ORACLE, ids=lambda row: row["stable_id"])
def test_all_reviewed_leading_jr_actions_match_the_frozen_oracle(case: dict[str, object]) -> None:
    source = _source(cast("_SourceParts", case["source"]))
    peer = _source(cast("_SourceParts", case["peer"]))

    selected = reviewed_leading_jr_peer_assignment(source, [source, peer], 0)

    assert selected is not None
    assert (selected.given_name, selected.middle_name, selected.surname, selected.suffix) == tuple(case["expected"])


@pytest.mark.parametrize(
    ("source_parts", "peer_parts"),
    [
        (("JR.", "John", "Cobb", None), [("John", None, "Cobb", None)]),
        (("Jr.", "John", "Cobb", "III"), [("John", None, "Cobb", None)]),
        (("Jr.", "University", "Board", None), [("University", None, "Board", None)]),
        (("Jr.", "John", "Cobb", None), []),
        (
            ("Jr.", "John", "Cobb", None),
            [
                ("John", None, "Cobb", None),
                ("John", None, "Cobb", None),
            ],
        ),
        (("Jr.", "John", "Cobb", None), [(None, "John", "Cobb", None)]),
        (
            ("jr", "Van", "Vaerenbergh", None),
            [(" Van ", " ", "Vaerenbergh", None)],
        ),
        (("Jr", "Ann A", "Smith", None), [("Anna", None, "Smith", None)]),
        ((" JR. ", "John", "Cobb", None), [("John", None, "Cobb", None)]),
        ((" \uff2a\uff52. ", "John", "Cobb", None), [("John", None, "Cobb", None)]),
        ((" Junior ", "John", "Cobb", None), [("John", None, "Cobb", None)]),
        ((" J r. ", "John", "Cobb", None), [("John", None, "Cobb", None)]),
        ((" Jr. ", "John", "Cobb", " "), [("John", None, "Cobb", None)]),
    ],
)
def test_reviewed_leading_jr_assignment_excludes_nearby_controls(
    source_parts: _SourceParts,
    peer_parts: list[_SourceParts],
) -> None:
    source = _source(source_parts)
    peers = [_source(parts) for parts in peer_parts]
    assert reviewed_leading_jr_peer_assignment(source, [source, *peers], 0) is None


def test_reviewed_fullwidth_alias_preserves_raw_segments_and_parentheses(
    predictor: RoutingPredictorV3,
) -> None:
    source = SourceAuthorFields(
        last_name="\u30b7\u30a2\u30f3\u30b0\uff0c\u30df\u30f3\uff08\u30a8\u30a4\u30df\u30fc\uff09",
        suffix="",
    )
    peer = SourceAuthorFields(last_name="\u30e0\u30b5\uff0c\u30cf\u30c3\u30b5\u30f3")

    selected = reviewed_fullwidth_katakana_alias_assignment(
        source,
        [source.full_name(), peer.full_name()],
        0,
    )
    assert selected == NameComponents(
        given_name="\u30df\u30f3\uff08\u30a8\u30a4\u30df\u30fc\uff09",
        surname="\u30b7\u30a2\u30f3\u30b0",
    )

    resolved = _route_paper(predictor, [source, peer])[0]
    assert (resolved.first_name, resolved.middle_names, resolved.last_name, resolved.suffix) == (
        "\u30df\u30f3\uff08\u30a8\u30a4\u30df\u30fc\uff09",
        "",
        "\u30b7\u30a2\u30f3\u30b0",
        "",
    )
    assert resolved.resolution_provenance is ResolutionProvenance.SOURCE
    assert resolved.resolution_action is ResolutionAction.ASSIGN
    assert resolved.resolution_reason is ResolutionReason.REVIEWED_SOURCE_PATTERN_ASSIGNMENT


@pytest.mark.parametrize("case", _FULLWIDTH_PATTERN_ORACLE, ids=lambda row: row["stable_id"])
def test_all_reviewed_fullwidth_alias_actions_match_the_frozen_oracle(case: dict[str, object]) -> None:
    source = _source(cast("_SourceParts", case["source"]))

    selected = reviewed_fullwidth_katakana_alias_assignment(
        source,
        [source.full_name(), "\u30e0\u30b5\uff0c\u30cf\u30c3\u30b5\u30f3"],
        0,
    )

    assert selected is not None
    assert (selected.given_name, selected.middle_name, selected.surname, selected.suffix) == tuple(case["expected"])


def test_reviewed_source_pattern_oracle_has_complete_activation_counts() -> None:
    assert len(_JR_PATTERN_ORACLE) == 51
    assert len(_FULLWIDTH_PATTERN_ORACLE) == 35
    assert len(_COMMA_CREDENTIAL_PATTERN_ORACLE) == 56
    assert sum(case["occurrences"] for case in _COMMA_CREDENTIAL_PATTERN_ORACLE) == 57


@pytest.mark.parametrize("case", _COMMA_CREDENTIAL_PATTERN_ORACLE, ids=lambda row: row["stable_id"])
def test_all_reviewed_closed_comma_credential_actions_match_the_frozen_oracle(
    predictor: RoutingPredictorV3,
    case: dict[str, object],
) -> None:
    source = _source(cast("_SourceParts", case["source"]))

    resolved = _route(predictor, source)
    expected_first, expected_middle, expected_last, expected_suffix = _CANONICAL_COMMA_CREDENTIAL_INITIALS.get(
        case["stable_id"],
        case["expected"],
    )

    assert (resolved.first_name, resolved.middle_names, resolved.last_name, resolved.suffix) == (
        expected_first,
        expected_middle or "",
        expected_last,
        expected_suffix,
    )
    assert resolved.resolution_provenance is ResolutionProvenance.SOURCE
    assert resolved.resolution_action is ResolutionAction.ASSIGN
    assert resolved.resolution_reason is ResolutionReason.REVIEWED_SOURCE_PATTERN_ASSIGNMENT


@pytest.mark.parametrize(
    "source",
    [
        SourceAuthorFields(first_name="John", last_name="Burn DM, MRCP"),
        SourceAuthorFields(first_name="Premanidhi", last_name="Panda MD (Medicine), MRCP, FRCP"),
        SourceAuthorFields(last_name="MDRD, DNB"),
        SourceAuthorFields(first_name="Dr.A.Thangavelu", last_name="MDS,DNB"),
        SourceAuthorFields(
            first_name="OSAMA",
            middle_names="MOHAMED ABU BAKR*;",
            last_name="BASHIR ABDELLATIF KHALED, MRCP",
        ),
    ],
)
def test_reviewed_closed_comma_credential_rule_excludes_full_corpus_controls(
    predictor: RoutingPredictorV3,
    source: SourceAuthorFields,
) -> None:
    resolved = _route(predictor, source)

    assert resolved.resolution_reason is not ResolutionReason.REVIEWED_SOURCE_PATTERN_ASSIGNMENT


@pytest.mark.parametrize(
    "source_parts",
    [
        ("X", None, "\u30b7\u30a2\u30f3\u30b0\uff0c\u30df\u30f3\uff08\u30a8\u30a4\u30df\u30fc\uff09", None),
        (None, "X", "\u30b7\u30a2\u30f3\u30b0\uff0c\u30df\u30f3\uff08\u30a8\u30a4\u30df\u30fc\uff09", None),
        (None, None, "\u30b7\u30a2\u30f3\u30b0\uff0c\u30df\u30f3\uff08\u30a8\u30a4\u30df\u30fc\uff09", "Jr."),
        (None, None, "\u30b7\u30a2\u30f3\u30b0,\u30df\u30f3\uff08\u30a8\u30a4\u30df\u30fc\uff09", None),
        (None, None, "\u30b7\u30a2\u30f3\u30b0\u3001\u30df\u30f3\uff08\u30a8\u30a4\u30df\u30fc\uff09", None),
        (None, None, "\u30b7\u30a2\u30f3\u30b0\uff0c\u30df\u30f3\uff0c\u30a8\u30a4\u30df\u30fc", None),
        (None, None, "\u30b7\u30fb\u30a2\u30f3\u30b0\uff0c\u30df\u30f3\uff08\u30a8\u30a4\u30df\u30fc\uff09", None),
        (None, None, "\u30b7\uff65\u30a2\u30f3\u30b0\uff0c\u30df\u30f3\uff08\u30a8\u30a4\u30df\u30fc\uff09", None),
        (None, None, "\u30b7\u30a2\u30f3\u30b0\uff0c\u30df\u30f3\uff0e\uff08\u30a8\u30a4\u30df\u30fc\uff09", None),
        (None, None, "\u30b7\u30a2\u30f3\u30b0\uff0c\u30df\u30f3", None),
        (None, None, "\u30b7\u30a2\u30f3\u30b0\uff0c\u30df\u30f3\uff08 \uff09", None),
        (None, None, "\u30b7\u30a2\u30f3\u30b0\uff0c\u30df\u30f3\uff08\u30a8\uff08\u30df\u30fc\uff09\uff09", None),
        (None, None, "\u30b7\u30a2\u30f3\u30b0\uff0c\u30df\u30f3\uff08\u30a8\u30a4\u30df\u30fc\uff09\u4e8c", None),
        (None, None, "\u30de\u30f3\u30cb\u30f3\u30b0\uff0c\u30b8\u30e5\u30cb\u30a2\uff08\u30a8\u30a4\u30df\u30fc\uff09", None),
    ],
)
def test_reviewed_fullwidth_alias_excludes_nearby_source_controls(source_parts: _SourceParts) -> None:
    source = _source(source_parts)
    peer_name = "\u30e0\u30b5\uff0c\u30cf\u30c3\u30b5\u30f3"
    assert reviewed_fullwidth_katakana_alias_assignment(source, [source.full_name(), peer_name], 0) is None


def test_reviewed_fullwidth_alias_requires_a_distinct_katakana_catalog_peer() -> None:
    source = SourceAuthorFields(last_name="\u30b7\u30a2\u30f3\u30b0\uff0c\u30df\u30f3\uff08\u30a8\u30a4\u30df\u30fc\uff09")

    assert reviewed_fullwidth_katakana_alias_assignment(source, [source.full_name()], 0) is None
    assert reviewed_fullwidth_katakana_alias_assignment(source, [source.full_name(), "Smith, John"], 0) is None
