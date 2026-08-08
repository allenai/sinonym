"""Regression tests for census-bounded residual error fixes."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from sinonym.coretypes.routing_resolution import ResolutionAction, ResolutionReason
from sinonym.timo.routing_v3 import (
    REVIEWED_EXACT_KOREAN_GIVEN_PREFIX_PACKS,
    RoutingInstanceV3,
    SourceAuthorFields,
    reviewed_exact_source_assignment,
)

if TYPE_CHECKING:
    from sinonym.timo.interface import RoutingPredictorV3


@pytest.fixture(scope="module")
def predictor(routing_predictor_v3: RoutingPredictorV3) -> RoutingPredictorV3:
    """Alias the shared session predictor under this module's name."""
    return routing_predictor_v3


def _route(
    predictor: RoutingPredictorV3,
    source: SourceAuthorFields,
):
    (paper,) = predictor.predict_batch([RoutingInstanceV3(pp_authors=[source])])
    return paper.authors[0].resolved_fields


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (SourceAuthorFields(first_name="FEBS", last_name="Prof. Dr. W. Schröder FACS"), ("W.", "", "Schröder", None)),
        (
            SourceAuthorFields(first_name="PhD", middle_names="FRACP", last_name="Prof. Murray Esler MBBS"),
            ("Murray", "", "Esler", None),
        ),
        (SourceAuthorFields(first_name="Apt", middle_names="Zullies", last_name="Ikawati"), ("Zullies", "", "Ikawati", None)),
        (
            SourceAuthorFields(first_name="Professur", middle_names="Prof. Dr. Wilfried", last_name="Krüger"),
            ("Wilfried", "", "Krüger", None),
        ),
        (SourceAuthorFields(first_name="ScD", last_name="Jennifer Massa"), ("Jennifer", "", "Massa", None)),
        (SourceAuthorFields(first_name="Angie", last_name="Sardina, MS, CTRS"), ("Angie", "", "Sardina", None)),
        (
            SourceAuthorFields(first_name="Se.Ak", middle_names=". Gede Adi", last_name="Yuniarta"),
            ("Gede", "Adi", "Yuniarta", None),
        ),
        (
            SourceAuthorFields(first_name="Thomas", middle_names="Dipl.-Ing. Fh", last_name="Schmidt-Behounek"),
            ("Thomas", "", "Schmidt-Behounek", None),
        ),
        (
            SourceAuthorFields(
                first_name="Suzanne",
                middle_names="M.",
                last_name="Thompson, MA, LRT/CTRS, FDRT, CHTP, BCTMB, LMBT",
            ),
            ("Suzanne", "M.", "Thompson", None),
        ),
        (
            SourceAuthorFields(first_name="Dominic", middle_names="J. Jr.", last_name="Varacalle"),
            ("Dominic", "J.", "Varacalle", "Jr."),
        ),
        (
            SourceAuthorFields(first_name="Array", middle_names="\u0410.", last_name="Галанина"),
            ("\u0410.", "", "Галанина", None),
        ),
        (SourceAuthorFields(first_name="Er.", middle_names="Surinder", last_name="Kumar"), ("Surinder", "", "Kumar", None)),
        (SourceAuthorFields(first_name="M.Pd", middle_names="Simson", last_name="Tarigan"), ("Simson", "", "Tarigan", None)),
        (SourceAuthorFields(first_name="MUDr.Zuzana", last_name="Blechová"), ("Zuzana", "", "Blechová", None)),
        (
            SourceAuthorFields(first_name="Assist", middle_names=".Lect. Muneera Mehdi", last_name="Muhsin"),
            ("Muneera", "Mehdi", "Muhsin", None),
        ),
    ],
)
def test_reviewed_source_cleanup_patterns_are_terminal(
    predictor: RoutingPredictorV3,
    source: SourceAuthorFields,
    expected: tuple[str, str, str, str | None],
) -> None:
    """Each censused source shape wins over PP/VYS and scalar candidates."""
    resolved = _route(predictor, source)

    assert (resolved.first_name, resolved.middle_names, resolved.last_name, resolved.suffix) == expected
    assert resolved.resolution_action is ResolutionAction.ASSIGN
    assert resolved.resolution_reason is ResolutionReason.REVIEWED_SOURCE_PATTERN_ASSIGNMENT


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            SourceAuthorFields(
                first_name="MSC",
                middle_names="CEng. Miee Mhkie Mieee",
                last_name="Winco K.C. Yung BEng",
            ),
            ("CEng", "Miee Mhkie Mieee Winco K. C.", "Yung"),
        ),
        (
            SourceAuthorFields(first_name="DNB", middle_names="DNB Tejas Patel", last_name="MBBS"),
            ("Dnb", "Dnb Tejas", "Patel"),
        ),
        (
            SourceAuthorFields(
                first_name="MS",
                middle_names="FRCS FACS FAMS FASc FNA",
                last_name="D.J. JUSSAWALLA",
            ),
            ("Frcs", "Facs Fams FASc Fna D. J.", "Jussawalla"),
        ),
        (
            SourceAuthorFields(first_name="Aurora", middle_names="M.T. Poon", last_name="FRACP"),
            ("Aurora", "M. T.", "Poon"),
        ),
        (
            SourceAuthorFields(
                first_name="FRCS",
                last_name="Univ.-Prof. Dr. med. Dr. h.c. N. Senninger FACS",
            ),
            ("Frcs", "Univ.-Prof Dr. Med. Dr. H. C. N.", "Senninger"),
        ),
    ],
)
def test_cleanup_does_not_expand_a_reviewed_surname_boundary(
    predictor: RoutingPredictorV3,
    source: SourceAuthorFields,
    expected: tuple[str, str, str],
) -> None:
    """Credential cleanup must not absorb initials or degrees into surname."""
    resolved = _route(predictor, source)

    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == expected
    assert resolved.resolution_reason is ResolutionReason.SCALAR_BASELINE


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            SourceAuthorFields(first_name="PhD", middle_names="MSN RN AOCNP", last_name="Carolyn S. Phillips"),
            ("Carolyn", "S.", "Phillips"),
        ),
        (
            SourceAuthorFields(first_name="MS", middle_names="FRCS FACS", last_name="Ronnie  T.  P.  Poon  MBBS"),
            ("Ronnie", "T. P.", "Poon"),
        ),
        (
            SourceAuthorFields(first_name="Johannes", middle_names="K Steinweg", last_name="MBBS"),
            ("Johannes", "K.", "Steinweg"),
        ),
        (
            SourceAuthorFields(first_name="iBSCHons", last_name="Ioanna Zimianiti MBBS"),
            ("Ioanna", "", "Zimianiti"),
        ),
    ],
)
def test_reviewed_credential_singletons_use_exact_clean_identity(
    predictor: RoutingPredictorV3,
    source: SourceAuthorFields,
    expected: tuple[str, str, str],
) -> None:
    """Full-corpus singletons remove credentials without changing identity."""
    resolved = _route(predictor, source)

    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == expected
    assert resolved.resolution_reason is ResolutionReason.REVIEWED_EXACT_SOURCE_ASSIGNMENT


def test_reviewed_shifted_alias_uses_exact_latin_identity(
    predictor: RoutingPredictorV3,
) -> None:
    """The singleton shifted alias emits the verified Chinese identity in Latin."""
    source = SourceAuthorFields(first_name="\u5289\u6bb7\u4f50", last_name="I-Ting Wang")
    resolved = _route(predictor, source)

    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == ("Yin-Zuo", "", "Liu")
    assert resolved.resolution_reason is ResolutionReason.REVIEWED_EXACT_SOURCE_ASSIGNMENT


def test_reviewed_packed_people_are_suppressed(
    predictor: RoutingPredictorV3,
) -> None:
    """Two people packed into one author slot cannot be safely materialized."""
    source = SourceAuthorFields(last_name="Anthony C. Laborte, Marissa C. Hitalia*")
    resolved = _route(predictor, source)

    assert resolved.resolution_action is ResolutionAction.SUPPRESS
    assert resolved.resolution_reason is ResolutionReason.REVIEWED_NON_PERSON_PATTERN


@pytest.mark.parametrize("source_last_name", ["Kai", "KAI"])
@pytest.mark.parametrize("vys_other_names", [None, ["Zhang Wei"]])
def test_mao_kai_exact_family_first_cohort_is_not_vetoed(
    predictor: RoutingPredictorV3,
    source_last_name: str,
    vys_other_names: list[str] | None,
) -> None:
    """The reviewed Chinese cohort keeps Mao as surname on both routed paths."""
    source = SourceAuthorFields(first_name="Mao", last_name=source_last_name)
    (paper,) = predictor.predict_batch(
        [RoutingInstanceV3(pp_authors=[source], vys_other_names=vys_other_names)],
    )
    resolved = paper.authors[0].resolved_fields

    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == ("Kai", "", "Mao")
    assert resolved.resolution_reason is ResolutionReason.REVIEWED_EXACT_SOURCE_ASSIGNMENT


@pytest.mark.parametrize(
    ("sources", "focal_index"),
    [
        (
            [
                SourceAuthorFields(first_name="Naeem", last_name="Ahmed"),
                SourceAuthorFields(first_name="Boyu", last_name="Hua"),
                SourceAuthorFields(first_name="Qiuming", last_name="Zhu"),
                SourceAuthorFields(first_name="Mao", last_name="Kai"),
            ],
            3,
        ),
        (
            [
                SourceAuthorFields(first_name="Haowen", last_name="Jin"),
                SourceAuthorFields(first_name="Weizhi", last_name="Zhong"),
                SourceAuthorFields(first_name="Liu", last_name="Xiang"),
                SourceAuthorFields(first_name="Qiuming", last_name="Zhu"),
                SourceAuthorFields(first_name="Zhipeng", last_name="Lin"),
                SourceAuthorFields(first_name="Mao", last_name="Kai"),
                SourceAuthorFields(first_name="Wang", last_name="Jie"),
            ],
            5,
        ),
    ],
)
def test_mao_kai_pp_abstention_contexts_still_use_reviewed_assignment(
    predictor: RoutingPredictorV3,
    sources: list[SourceAuthorFields],
    focal_index: int,
) -> None:
    """Terminal assignment covers the two corpus contexts that PP abstains on."""
    (paper,) = predictor.predict_batch([RoutingInstanceV3(pp_authors=sources)])
    resolved = paper.authors[focal_index].resolved_fields

    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == ("Kai", "", "Mao")
    assert resolved.resolution_reason is ResolutionReason.REVIEWED_EXACT_SOURCE_ASSIGNMENT


@pytest.mark.parametrize(
    "source",
    [
        SourceAuthorFields(first_name="Ирина", middle_names="Кузьминична", last_name="Русакович"),
        SourceAuthorFields(first_name="Лев", middle_names="Ильич", last_name="Могилевич"),
        SourceAuthorFields(first_name="Константин", middle_names="Мовчан", last_name="Николаевич"),
        SourceAuthorFields(first_name="Иванов", middle_names="Дмитрий", last_name="Николаевич"),
    ],
)
def test_cyrillic_morphology_alone_does_not_rotate_source_fields(
    predictor: RoutingPredictorV3,
    source: SourceAuthorFields,
) -> None:
    """Patronymic-like endings collide with valid given names and surnames."""
    resolved = _route(predictor, source)

    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == (
        source.first_name,
        source.middle_names,
        source.last_name,
    )
    assert resolved.resolution_reason is not ResolutionReason.REVIEWED_SOURCE_PATTERN_ASSIGNMENT


@pytest.mark.parametrize(
    "source",
    [
        SourceAuthorFields(
            first_name="\u0418\u0432\u0430\u043d\u043e\u0432\u0430",
            middle_names="\u0410\u043d\u043d\u0430",
            last_name="\u0421\u0435\u0440\u0433\u0435\u0435\u0432\u043d\u0430",
        ),
        SourceAuthorFields(
            first_name="\u0421\u0435\u0440\u0433\u0435\u0435\u0432\u0430",
            middle_names="\u041e\u043b\u044c\u0433\u0430",
            last_name="\u0418\u0432\u0430\u043d\u043e\u0432\u043d\u0430",
        ),
        SourceAuthorFields(
            first_name="\u041f\u0435\u0442\u0440\u043e\u0432\u0441\u043a\u0430\u044f",
            middle_names="\u0415\u043b\u0435\u043d\u0430",
            last_name="\u0410\u043b\u0435\u043a\u0441\u0430\u043d\u0434\u0440\u043e\u0432\u043d\u0430",
        ),
        SourceAuthorFields(
            first_name="\u041a\u0443\u0437\u043d\u0435\u0446\u043a\u0430\u044f",
            middle_names="\u041c\u0430\u0440\u0438\u044f",
            last_name="\u041f\u0435\u0442\u0440\u043e\u0432\u043d\u0430",
        ),
        SourceAuthorFields(
            first_name="\u041f\u0435\u0442\u0440\u043e\u0432\u0441\u043a\u0438\u0439",
            middle_names="\u0418\u0432\u0430\u043d",
            last_name="\u0421\u0435\u0440\u0433\u0435\u0435\u0432\u0438\u0447",
        ),
        SourceAuthorFields(
            first_name="\u041a\u0443\u0437\u043d\u0435\u0446\u043a\u0438\u0439",
            middle_names="\u041f\u0430\u0432\u0435\u043b",
            last_name="\u0418\u0432\u0430\u043d\u043e\u0432\u0438\u0447",
        ),
    ],
)
def test_gender_concordant_long_suffix_cyrillic_shape_rotates(
    predictor: RoutingPredictorV3,
    source: SourceAuthorFields,
) -> None:
    """The censused long-suffix class is surname-given-patronymic."""
    resolved = _route(predictor, source)

    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == (
        source.middle_names,
        source.last_name,
        source.first_name,
    )
    assert resolved.resolution_reason is ResolutionReason.REVIEWED_SOURCE_PATTERN_ASSIGNMENT


@pytest.mark.parametrize(
    "source",
    [
        SourceAuthorFields(
            first_name="\u0418\u0432\u0430\u043d\u043e\u0432",
            middle_names="\u0414\u043c\u0438\u0442\u0440\u0438\u0439",
            last_name="\u041d\u0438\u043a\u043e\u043b\u0430\u0435\u0432\u0438\u0447",
        ),
        SourceAuthorFields(
            first_name="\u0418\u0432\u0430\u043d\u043e\u0432\u0430",
            middle_names="\u0410\u043d\u043d\u0430",
            last_name="\u0421\u0435\u0440\u0433\u0435\u0435\u0432\u0438\u0447",
        ),
        SourceAuthorFields(
            first_name="\u041f\u0435\u0442\u0440\u043e\u0432\u0441\u043a\u0438\u0439",
            middle_names="\u0418\u0432\u0430\u043d\u043e\u0432\u0438\u0447",
            last_name="\u0421\u0435\u0440\u0433\u0435\u0435\u0432\u0438\u0447",
        ),
        SourceAuthorFields(
            first_name="\u041f\u0435\u0442\u0440\u043e\u0432\u0441\u043a\u0438\u0439",
            middle_names="\u0418\u0432\u0430\u043d \u041f\u0430\u0432\u0435\u043b",
            last_name="\u0421\u0435\u0440\u0433\u0435\u0435\u0432\u0438\u0447",
        ),
    ],
)
def test_cyrillic_rotation_declines_short_mismatched_or_ambiguous_shapes(
    predictor: RoutingPredictorV3,
    source: SourceAuthorFields,
) -> None:
    resolved = _route(predictor, source)

    assert resolved.resolution_reason is not ResolutionReason.REVIEWED_SOURCE_PATTERN_ASSIGNMENT


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            SourceAuthorFields(first_name="Nabiev", middle_names="Valery", last_name="Sharifyanovich"),
            ("Valery", "Sharifyanovich", "Nabiev"),
        ),
        (
            SourceAuthorFields(first_name="Turaxodjayeva", middle_names="Moxidil", last_name="Obidjonovna"),
            ("Moxidil", "Obidjonovna", "Turaxodjayeva"),
        ),
        (
            SourceAuthorFields(
                first_name="\u0410\u043b\u0435\u043a\u0441\u0435\u0435\u0432",
                middle_names="\u0413\u0435\u043d\u043d\u0430\u0434\u0438\u0439",
                last_name="\u0412\u0430\u043b\u0435\u043d\u0442\u0438\u043d\u043e\u0432\u0438\u0447",
            ),
            (
                "\u0413\u0435\u043d\u043d\u0430\u0434\u0438\u0439",
                "\u0412\u0430\u043b\u0435\u043d\u0442\u0438\u043d\u043e\u0432\u0438\u0447",
                "\u0410\u043b\u0435\u043a\u0441\u0435\u0435\u0432",
            ),
        ),
        (
            SourceAuthorFields(
                first_name="\u0411\u0435\u043b\u043e\u0432",
                middle_names="\u0412\u043b\u0430\u0434\u0438\u043c\u0438\u0440",
                last_name="\u041d\u0438\u043a\u043e\u043b\u0430\u0435\u0432\u0438\u0447",
            ),
            (
                "\u0412\u043b\u0430\u0434\u0438\u043c\u0438\u0440",
                "\u041d\u0438\u043a\u043e\u043b\u0430\u0435\u0432\u0438\u0447",
                "\u0411\u0435\u043b\u043e\u0432",
            ),
        ),
        (
            SourceAuthorFields(
                first_name="\u0412\u043e\u043b\u044b\u043d\u043e\u0432",
                middle_names="\u041c\u0438\u0445\u0430\u0438\u043b",
                last_name="\u0410\u043d\u0430\u0442\u043e\u043b\u044c\u0435\u0432\u0438\u0447",
            ),
            (
                "\u041c\u0438\u0445\u0430\u0438\u043b",
                "\u0410\u043d\u0430\u0442\u043e\u043b\u044c\u0435\u0432\u0438\u0447",
                "\u0412\u043e\u043b\u044b\u043d\u043e\u0432",
            ),
        ),
        (
            SourceAuthorFields(
                first_name="\u041a\u0438\u0440\u0441\u0430\u043d\u043e\u0432",
                middle_names="\u0410\u043d\u0434\u0440\u0435\u0439",
                last_name="\u0420\u043e\u043c\u0430\u043d\u043e\u0432\u0438\u0447",
            ),
            (
                "\u0410\u043d\u0434\u0440\u0435\u0439",
                "\u0420\u043e\u043c\u0430\u043d\u043e\u0432\u0438\u0447",
                "\u041a\u0438\u0440\u0441\u0430\u043d\u043e\u0432",
            ),
        ),
        (
            SourceAuthorFields(
                first_name="\u041a\u0438\u0441\u0442\u0435\u0440\u0441\u043a\u0438\u0439",
                middle_names="\u0410\u043b\u0435\u043a\u0441\u0430\u043d\u0434\u0440",
                last_name="\u041f\u0435\u0442\u0440\u043e\u0432\u0438\u0447",
            ),
            (
                "\u0410\u043b\u0435\u043a\u0441\u0430\u043d\u0434\u0440",
                "\u041f\u0435\u0442\u0440\u043e\u0432\u0438\u0447",
                "\u041a\u0438\u0441\u0442\u0435\u0440\u0441\u043a\u0438\u0439",
            ),
        ),
        (
            SourceAuthorFields(
                first_name="\u041c\u0430\u043a\u0430\u0440\u043e\u0432\u0430",
                middle_names="\u0415\u043a\u0430\u0442\u0435\u0440\u0438\u043d\u0430",
                last_name="\u0412\u043b\u0430\u0434\u0438\u043c\u0438\u0440\u043e\u0432\u043d\u0430",
            ),
            (
                "\u0415\u043a\u0430\u0442\u0435\u0440\u0438\u043d\u0430",
                "\u0412\u043b\u0430\u0434\u0438\u043c\u0438\u0440\u043e\u0432\u043d\u0430",
                "\u041c\u0430\u043a\u0430\u0440\u043e\u0432\u0430",
            ),
        ),
        (
            SourceAuthorFields(
                first_name="\u041d\u0435\u0432\u0435\u0440\u043e\u0432\u0430",
                middle_names="\u041e\u043b\u044c\u0433\u0430",
                last_name="\u0410\u043b\u0435\u043a\u0441\u0430\u043d\u0434\u0440\u043e\u0432\u043d\u0430",
            ),
            (
                "\u041e\u043b\u044c\u0433\u0430",
                "\u0410\u043b\u0435\u043a\u0441\u0430\u043d\u0434\u0440\u043e\u0432\u043d\u0430",
                "\u041d\u0435\u0432\u0435\u0440\u043e\u0432\u0430",
            ),
        ),
        (
            SourceAuthorFields(
                first_name="\u0421\u0432\u0438\u0434\u0443\u043d\u043e\u0432\u0438\u0447",
                middle_names="\u041d\u0438\u043a\u043e\u043b\u0430\u0439",
                last_name="\u0410\u043b\u0435\u043a\u0441\u0430\u043d\u0434\u0440\u043e\u0432\u0438\u0447",
            ),
            (
                "\u041d\u0438\u043a\u043e\u043b\u0430\u0439",
                "\u0410\u043b\u0435\u043a\u0441\u0430\u043d\u0434\u0440\u043e\u0432\u0438\u0447",
                "\u0421\u0432\u0438\u0434\u0443\u043d\u043e\u0432\u0438\u0447",
            ),
        ),
        (
            SourceAuthorFields(
                first_name="\u0422\u0440\u043e\u0444\u0438\u043c\u043e\u0432",
                middle_names="\u0410\u0440\u0442\u0435\u043c",
                last_name="\u0410\u043b\u0435\u043a\u0441\u0430\u043d\u0434\u0440\u043e\u0432\u0438\u0447",
            ),
            (
                "\u0410\u0440\u0442\u0435\u043c",
                "\u0410\u043b\u0435\u043a\u0441\u0430\u043d\u0434\u0440\u043e\u0432\u0438\u0447",
                "\u0422\u0440\u043e\u0444\u0438\u043c\u043e\u0432",
            ),
        ),
        (
            SourceAuthorFields(
                first_name="\u0422\u0443\u0445\u0442\u0430\u043c\u0443\u0440\u043e\u0434",
                middle_names="\u0417\u0438\u0451\u0434\u0443\u043b\u043b\u0430",
                last_name="\u0417\u0438\u043a\u0440\u0438\u043b\u043b\u0430",
            ),
            (
                "\u0417\u0438\u0451\u0434\u0443\u043b\u043b\u0430",
                "\u0417\u0438\u043a\u0440\u0438\u043b\u043b\u0430",
                "\u0422\u0443\u0445\u0442\u0430\u043c\u0443\u0440\u043e\u0434",
            ),
        ),
        (
            SourceAuthorFields(
                first_name="\u0428\u0435\u0432\u0447\u0435\u043d\u043a\u043e",
                middle_names="\u0415\u043b\u0435\u043d\u0430",
                last_name="\u0412\u0438\u043a\u0442\u043e\u0440\u043e\u0432\u043d\u0430",
            ),
            (
                "\u0415\u043b\u0435\u043d\u0430",
                "\u0412\u0438\u043a\u0442\u043e\u0440\u043e\u0432\u043d\u0430",
                "\u0428\u0435\u0432\u0447\u0435\u043d\u043a\u043e",
            ),
        ),
        (
            SourceAuthorFields(
                first_name="\u042f\u043c\u0430\u043b\u0434\u0438\u043d\u043e\u0432",
                middle_names="\u0422\u0438\u043c\u0443\u0440",
                last_name="\u0420\u0438\u0444\u0430\u0442\u043e\u0432\u0438\u0447",
            ),
            (
                "\u0422\u0438\u043c\u0443\u0440",
                "\u0420\u0438\u0444\u0430\u0442\u043e\u0432\u0438\u0447",
                "\u042f\u043c\u0430\u043b\u0434\u0438\u043d\u043e\u0432",
            ),
        ),
    ],
)
def test_reviewed_exact_surname_given_patronymic_tuples(
    predictor: RoutingPredictorV3,
    source: SourceAuthorFields,
    expected: tuple[str, str, str],
) -> None:
    """Every full-corpus occurrence of these exact tuples agreed on roles."""
    resolved = _route(predictor, source)

    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == expected
    assert resolved.resolution_reason is ResolutionReason.REVIEWED_EXACT_SOURCE_ASSIGNMENT


@pytest.mark.parametrize(
    "source",
    [
        SourceAuthorFields(first_name="Jidong", last_name="Sung"),
        SourceAuthorFields(first_name="Mung", last_name="Chiang"),
        SourceAuthorFields(first_name="Onchee", last_name="Yu"),
        SourceAuthorFields(first_name="Seah", middle_names="H.", last_name="Lim"),
        SourceAuthorFields(first_name="Seah", middle_names="H", last_name="Lim"),
    ],
)
def test_reviewed_korean_source_tuples_preserve_roles_and_canonicalize_initials(
    predictor: RoutingPredictorV3,
    source: SourceAuthorFields,
) -> None:
    """The complete exact cohorts support the supplied Korean roles and canonical output."""
    resolved = _route(predictor, source)

    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == (
        source.first_name,
        "H." if source.middle_names else "",
        source.last_name,
    )
    assert resolved.resolution_action is ResolutionAction.ASSIGN
    assert resolved.resolution_reason is ResolutionReason.REVIEWED_EXACT_SOURCE_ASSIGNMENT


def test_reviewed_seungbo_token_is_not_split_by_chinese_hyphenation(
    predictor: RoutingPredictorV3,
) -> None:
    """Every full-corpus Seungbo hit was Korean, independent of surname."""
    source = SourceAuthorFields(first_name="Seungbo", last_name="Choi")

    resolved = _route(predictor, source)

    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == ("Seungbo", "", "Choi")


def test_reviewed_spaced_korean_given_surface_is_restored_after_vys_selection(
    predictor: RoutingPredictorV3,
) -> None:
    source = SourceAuthorFields(first_name="So Young", last_name="Yun")
    (paper,) = predictor.predict_batch(
        [
            RoutingInstanceV3(
                pp_authors=[source],
                vys_other_names=[
                    "Jong An Lee",
                    "Yong Jun Kang",
                    "In Ho Choi",
                    "Se Young Lee",
                    "Chang-Woo Min",
                    "Seung Min Jung",
                ],
            ),
        ],
    )

    resolved = paper.authors[0].resolved_fields

    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == ("So Young", "", "Yun")
    assert resolved.resolution_reason is ResolutionReason.VYS_SELECTED


def test_reviewed_korean_source_assignment_inventory_matches_manual_ledger() -> None:
    ledger_path = Path(__file__).parent / "data" / "reviewed_korean_source_assignments.tsv"
    with ledger_path.open(encoding="utf-8", newline="") as ledger_file:
        rows = list(csv.DictReader(ledger_file, delimiter="\t"))

    assert len(rows) == 66
    assert sum(row["decision"] == "assign" for row in rows) == 59
    assert sum(row["decision"] == "exclude" for row in rows) == 6
    assert sum(row["decision"] == "abstain" for row in rows) == 1

    assigned_rows = [row for row in rows if row["decision"] == "assign"]
    ledger_inventory = {
        (
            row["source_first_name"].casefold(),
            row["source_middle_names"].casefold(),
            row["source_last_name"].casefold(),
        ): int(row["given_prefix_tokens"])
        for row in assigned_rows
    }
    assert ledger_inventory == REVIEWED_EXACT_KOREAN_GIVEN_PREFIX_PACKS

    for row in assigned_rows:
        resolved = reviewed_exact_source_assignment(
            SourceAuthorFields(
                first_name=row["source_first_name"],
                middle_names=row["source_middle_names"],
                last_name=row["source_last_name"],
            ),
        )
        assert resolved is not None
        assert (resolved.given_name, resolved.middle_name, resolved.surname) == (
            row["expected_given_name"],
            row["expected_middle_name"],
            row["expected_surname"],
        )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (SourceAuthorFields(first_name="Bong", middle_names="Soo", last_name="Cha"), ("Bong Soo", "", "Cha")),
        (SourceAuthorFields(first_name="Boo", middle_names="Young", last_name="Ko"), ("Boo Young", "", "Ko")),
        (SourceAuthorFields(first_name="Byoung", middle_names="Yoon", last_name="Kim"), ("Byoung Yoon", "", "Kim")),
        (SourceAuthorFields(first_name="Chang", middle_names="Hee", last_name="Lee"), ("Chang Hee", "", "Lee")),
        (SourceAuthorFields(first_name="Chang", middle_names="Mo", last_name="Yang"), ("Chang Mo", "", "Yang")),
        (SourceAuthorFields(first_name="Changwoo", last_name="Lee"), ("Changwoo", "", "Lee")),
        (SourceAuthorFields(first_name="Dong", middle_names="Soo", last_name="Han"), ("Dong Soo", "", "Han")),
        (SourceAuthorFields(first_name="Eun", middle_names="Kee", last_name="Jeong"), ("Eun Kee", "", "Jeong")),
        (SourceAuthorFields(first_name="Han", middle_names="Jin", last_name="Jung"), ("Han Jin", "", "Jung")),
        (SourceAuthorFields(first_name="Heung", middle_names="Soo", last_name="Lee"), ("Heung Soo", "", "Lee")),
        (SourceAuthorFields(first_name="Hoe", middle_names="Joon", last_name="Kim"), ("Hoe Joon", "", "Kim")),
        (SourceAuthorFields(first_name="Hyoung", middle_names="Sub", last_name="Kim"), ("Hyoung Sub", "", "Kim")),
        (SourceAuthorFields(first_name="Jae", middle_names="moon", last_name="Lee"), ("Jae moon", "", "Lee")),
        (SourceAuthorFields(first_name="Jae", middle_names="Won", last_name="Chung"), ("Jae Won", "", "Chung")),
        (SourceAuthorFields(first_name="Jeong", middle_names="Seon", last_name="Yeo"), ("Jeong Seon", "", "Yeo")),
        (SourceAuthorFields(first_name="Ji", middle_names="Hyun", last_name="Moon"), ("Ji Hyun", "", "Moon")),
        (SourceAuthorFields(first_name="Ji", middle_names="Soo", last_name="Lee"), ("Ji Soo", "", "Lee")),
        (SourceAuthorFields(first_name="Ji", middle_names="Woon", last_name="Ha"), ("Ji Woon", "", "Ha")),
        (SourceAuthorFields(first_name="Jin", middle_names="Cheul", last_name="Kim"), ("Jin Cheul", "", "Kim")),
        (SourceAuthorFields(first_name="Jong", middle_names="Hak", last_name="Kim"), ("Jong Hak", "", "Kim")),
        (SourceAuthorFields(first_name="Jong", middle_names="Ho", last_name="Kim"), ("Jong Ho", "", "Kim")),
        (SourceAuthorFields(first_name="Jong", middle_names="Hoon", last_name="Kang"), ("Jong Hoon", "", "Kang")),
        (SourceAuthorFields(first_name="Jong", middle_names="Soo", last_name="Woo"), ("Jong Soo", "", "Woo")),
        (SourceAuthorFields(first_name="Joon", middle_names="Young", last_name="Choi"), ("Joon Young", "", "Choi")),
        (SourceAuthorFields(first_name="Kang", middle_names="Ju", last_name="Kim"), ("Kang Ju", "", "Kim")),
        (SourceAuthorFields(first_name="Keum", middle_names="Seok", last_name="Bae"), ("Keum Seok", "", "Bae")),
        (SourceAuthorFields(first_name="Kyeong", middle_names="Ah", last_name="Kim"), ("Kyeong Ah", "", "Kim")),
        (SourceAuthorFields(first_name="Min", middle_names="Young", last_name="Lee"), ("Min Young", "", "Lee")),
        (SourceAuthorFields(first_name="Minwoo", last_name="Lee"), ("Minwoo", "", "Lee")),
        (SourceAuthorFields(first_name="Sang", middle_names="Hoon", last_name="Han"), ("Sang Hoon", "", "Han")),
        (SourceAuthorFields(first_name="Sang", middle_names="Hyub", last_name="Lee"), ("Sang Hyub", "", "Lee")),
        (SourceAuthorFields(first_name="Sang", middle_names="Min", last_name="Yoon"), ("Sang Min", "", "Yoon")),
        (SourceAuthorFields(first_name="Sang", middle_names="Yong", last_name="Shin"), ("Sang Yong", "", "Shin")),
        (SourceAuthorFields(first_name="Sang", middle_names="Yun", last_name="Han"), ("Sang Yun", "", "Han")),
        (SourceAuthorFields(first_name="Sanghun", last_name="Lee"), ("Sanghun", "", "Lee")),
        (SourceAuthorFields(first_name="Sangji", last_name="Lee"), ("Sangji", "", "Lee")),
        (SourceAuthorFields(first_name="Seok", middle_names="Yong", last_name="Kang"), ("Seok Yong", "", "Kang")),
        (SourceAuthorFields(first_name="Seung", middle_names="Jun", last_name="Lee"), ("Seung Jun", "", "Lee")),
        (SourceAuthorFields(first_name="Soo", middle_names="Ick", last_name="Cho"), ("Soo Ick", "", "Cho")),
        (SourceAuthorFields(first_name="Su", middle_names="Ja", last_name="Kim"), ("Su Ja", "", "Kim")),
        (SourceAuthorFields(first_name="Su", middle_names="Jin", last_name="Hwang"), ("Su Jin", "", "Hwang")),
        (SourceAuthorFields(first_name="Su", middle_names="Jung", last_name="Choi"), ("Su Jung", "", "Choi")),
        (SourceAuthorFields(first_name="Sumin", last_name="Lee"), ("Sumin", "", "Lee")),
        (SourceAuthorFields(first_name="Sung", middle_names="Hoon", last_name="Chung"), ("Sung Hoon", "", "Chung")),
        (SourceAuthorFields(first_name="Sung", middle_names="Ik", last_name="Lee"), ("Sung Ik", "", "Lee")),
        (SourceAuthorFields(first_name="Sunghak", last_name="Lee"), ("Sunghak", "", "Lee")),
        (SourceAuthorFields(first_name="Suji", last_name="Choi"), ("Suji", "", "Choi")),
        (SourceAuthorFields(first_name="Weon", middle_names="Ju", last_name="Lee"), ("Weon Ju", "", "Lee")),
        (
            SourceAuthorFields(first_name="Won", middle_names="Hyung A.", last_name="Ryu"),
            ("Won Hyung", "A.", "Ryu"),
        ),
        (SourceAuthorFields(first_name="Woo", middle_names="Sung", last_name="Jeon"), ("Woo Sung", "", "Jeon")),
        (SourceAuthorFields(first_name="Ye", middle_names="Hun", last_name="Choi"), ("Ye Hun", "", "Choi")),
        (SourceAuthorFields(first_name="Yi", middle_names="Ho", last_name="Lee"), ("Yi Ho", "", "Lee")),
        (SourceAuthorFields(first_name="Youme", last_name="Ko"), ("Youme", "", "Ko")),
        (SourceAuthorFields(first_name="Young", middle_names="Hee", last_name="Choi"), ("Young Hee", "", "Choi")),
        (SourceAuthorFields(first_name="Young", middle_names="In", last_name="Shin"), ("Young In", "", "Shin")),
        (SourceAuthorFields(first_name="Young", middle_names="Mo", last_name="Sung"), ("Young Mo", "", "Sung")),
        (SourceAuthorFields(first_name="Youn", middle_names="Sik", last_name="Kim"), ("Youn Sik", "", "Kim")),
        (SourceAuthorFields(first_name="Yoon", middle_names="Kyung", last_name="Choi"), ("Yoon Kyung", "", "Choi")),
        (SourceAuthorFields(first_name="Yunjin", last_name="Lee"), ("Yunjin", "", "Lee")),
    ],
)
def test_reviewed_korean_source_orthography_is_assigned_exactly(
    predictor: RoutingPredictorV3,
    source: SourceAuthorFields,
    expected: tuple[str, str, str],
) -> None:
    resolved = _route(predictor, source)

    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == expected
    assert resolved.resolution_action is ResolutionAction.ASSIGN
    assert resolved.resolution_reason is ResolutionReason.REVIEWED_EXACT_SOURCE_ASSIGNMENT


def test_attested_korean_hyphen_case_remains_unchanged(
    predictor: RoutingPredictorV3,
) -> None:
    resolved = _route(predictor, SourceAuthorFields(first_name="Yang-sook", last_name="Lee"))

    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == ("Yang-sook", "", "Lee")


@pytest.mark.parametrize(
    "source",
    [
        SourceAuthorFields(first_name="Young", middle_names="In", last_name="Lee"),
        SourceAuthorFields(first_name="Sung", middle_names="Ik", last_name="Kim"),
        SourceAuthorFields(first_name="Changwoo", last_name="Kim"),
        SourceAuthorFields(first_name="Young", middle_names="In", last_name="Shin", suffix="Jr."),
    ],
)
def test_reviewed_korean_source_orthography_assignment_is_exact_tuple_scoped(
    source: SourceAuthorFields,
) -> None:
    assert reviewed_exact_source_assignment(source) is None


@pytest.mark.parametrize(
    "source",
    [
        SourceAuthorFields(first_name="Seak", middle_names="Hee", last_name="Oh"),
        SourceAuthorFields(first_name="S.C.D.", last_name="Wright"),
        SourceAuthorFields(first_name="Jay", last_name="Apt"),
        SourceAuthorFields(first_name="Array", last_name="Array"),
        SourceAuthorFields(first_name="Array", last_name="Статья"),
        SourceAuthorFields(first_name="Chen", last_name="Tsung-Jr"),
        SourceAuthorFields(first_name="John", middle_names="FACS William", last_name="Smith"),
        SourceAuthorFields(first_name="John", middle_names="Jr. Jr.", last_name="Smith"),
        SourceAuthorFields(first_name="Apt", last_name="W"),
        SourceAuthorFields(first_name="Apt", last_name="Werner"),
        SourceAuthorFields(first_name="Er", middle_names="Qiang", last_name="Wang"),
        SourceAuthorFields(first_name="Er.", last_name="Shadab"),
        SourceAuthorFields(first_name="Er.", middle_names="Surinder", last_name="Kumar", suffix="PhD"),
        SourceAuthorFields(first_name="M.pd", middle_names="Simson", last_name="Tarigan"),
        SourceAuthorFields(first_name="M.Pd.", middle_names="Simson", last_name="Tarigan"),
        SourceAuthorFields(first_name="M.Pd", last_name="Ariyanto"),
        SourceAuthorFields(first_name="Simson", middle_names="M.Pd", last_name="Tarigan"),
        SourceAuthorFields(first_name="Mudrik", last_name="Alaydrus"),
        SourceAuthorFields(first_name="Mudr", middle_names="O.", last_name="Klaskova"),
        SourceAuthorFields(first_name="Mudr.Zuzana", last_name="Blechová"),
        SourceAuthorFields(first_name="MUDr.", last_name="Milan"),
        SourceAuthorFields(first_name="MUDr.", middle_names="Birgita", last_name="Slová"),
        SourceAuthorFields(first_name="Assist", middle_names="Lect. Muneera Mehdi", last_name="Muhsin"),
        SourceAuthorFields(first_name="Lect.", middle_names="Jasim Mohammed", last_name="Hassan"),
        SourceAuthorFields(first_name="Assist", middle_names=".Lect.", last_name="Muhsin"),
        SourceAuthorFields(first_name="Muneera", middle_names="Assist .Lect. Mehdi", last_name="Muhsin"),
    ],
)
def test_reviewed_source_cleanup_patterns_exclude_nearby_names(
    predictor: RoutingPredictorV3,
    source: SourceAuthorFields,
) -> None:
    """Case, position, punctuation, and usable-remainder gates stay narrow."""
    resolved = _route(predictor, source)

    assert resolved.resolution_reason is not ResolutionReason.REVIEWED_SOURCE_PATTERN_ASSIGNMENT


@pytest.mark.parametrize(
    "source",
    [
        SourceAuthorFields(first_name="Not Available", last_name="Not Available"),
        SourceAuthorFields(first_name="None", last_name="None"),
        SourceAuthorFields(first_name="Unknown", last_name="Author"),
        SourceAuthorFields(first_name="undefined", middle_names="No authorship", last_name="indicated"),
        SourceAuthorFields(last_name="January-February"),
        SourceAuthorFields(last_name="JANUARY-DECEMBER"),
        SourceAuthorFields(first_name="Wku", last_name="Libraries"),
        SourceAuthorFields(first_name="Petroleum", last_name="Geo-Services"),
        SourceAuthorFields(last_name="대한전자공학회"),
        SourceAuthorFields(first_name="Array", last_name="BioPharma"),
        SourceAuthorFields(first_name="Professur", last_name="Fördertechnik"),
        SourceAuthorFields(last_name="FACS"),
        SourceAuthorFields(first_name="ScD", last_name="MPH"),
    ],
)
def test_exact_metadata_patterns_are_terminal_suppressions(
    predictor: RoutingPredictorV3,
    source: SourceAuthorFields,
) -> None:
    """Reviewed whole-record metadata reaches the writer as suppression."""
    resolved = _route(predictor, source)

    assert resolved.resolution_action is ResolutionAction.SUPPRESS
    assert resolved.resolution_reason is ResolutionReason.REVIEWED_NON_PERSON_PATTERN


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (SourceAuthorFields(first_name="Yu.E.", last_name="Makarov"), ("Yu.E.", "", "Makarov")),
        (SourceAuthorFields(first_name="Yu.V.", last_name="Bulii"), ("Yu.V.", "", "Bulii")),
        (SourceAuthorFields(first_name="L", last_name="NoW."), ("L.", "", "Now.")),
    ],
)
def test_dotted_initial_casing_fix_is_narrow(
    predictor: RoutingPredictorV3,
    source: SourceAuthorFields,
    expected: tuple[str, str, str],
) -> None:
    resolved = _route(predictor, source)

    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == expected


@pytest.mark.parametrize(
    "source",
    [
        SourceAuthorFields(first_name="Syafira", middle_names="Elfithri", last_name="Universitas"),
        SourceAuthorFields(first_name="A.", middle_names="Hosp. Clínico Universitario Lozano Bles", last_name="Angusto"),
    ],
)
def test_organization_token_rules_do_not_suppress_reviewed_people(
    predictor: RoutingPredictorV3,
    source: SourceAuthorFields,
) -> None:
    resolved = _route(predictor, source)

    assert resolved.resolution_action is not ResolutionAction.SUPPRESS
    assert resolved.resolution_reason is not ResolutionReason.REVIEWED_NON_PERSON_PATTERN


@pytest.mark.parametrize(
    "source",
    [
        SourceAuthorFields(first_name="S", last_name="TucikeÅ¡iÄ‡"),
        SourceAuthorFields(first_name="Burak", last_name="KÄ±lanÃ§"),
        SourceAuthorFields(first_name="Mojibake", last_name="\u0421‡"),
    ],
)
def test_mojibake_tail_is_not_stripped_as_a_footnote(
    predictor: RoutingPredictorV3,
    source: SourceAuthorFields,
) -> None:
    resolved = _route(predictor, source)
    rendered = " ".join(
        value for value in (resolved.first_name, resolved.middle_names, resolved.last_name, resolved.suffix) if value
    )

    assert rendered.endswith(source.last_name or "")


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            SourceAuthorFields(
                first_name="MD",
                middle_names="PhD 영남대학교 의과대학 안과학교실",
                last_name="Junhyuk Son",
            ),
            ("Junhyuk", "", "Son"),
        ),
        (
            SourceAuthorFields(first_name="MD", middle_names="PhD 인제대학교 의과대학", last_name="Jung Lim Kim"),
            ("Jung Lim", "", "Kim"),
        ),
        (
            SourceAuthorFields(first_name="MD", middle_names="PhD 한림대학교 의과대학", last_name="Joo Yeon Lee"),
            ("Joo Yeon", "", "Lee"),
        ),
        (
            SourceAuthorFields(first_name="M.", middle_names="D", last_name="Services-REGINALD M. ATWATER"),
            ("Reginald", "M.", "Atwater"),
        ),
    ],
)
def test_reviewed_mixed_metadata_rows_salvage_the_person(
    predictor: RoutingPredictorV3,
    source: SourceAuthorFields,
    expected: tuple[str, str, str],
) -> None:
    resolved = _route(predictor, source)

    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == expected
    assert resolved.resolution_action is ResolutionAction.ASSIGN


@pytest.mark.parametrize(
    "source",
    [
        SourceAuthorFields(first_name="Benjamín", middle_names="Cristian", last_name="Corona-Comunidad"),
        SourceAuthorFields(first_name="Roque", middle_names="A.", last_name="Comunidad-Bonilla"),
        SourceAuthorFields(
            first_name="E.",
            middle_names="Slovenská poľnohospodárska univerzita v Nitre",
            last_name="Hazuchová",
        ),
        SourceAuthorFields(first_name="Miguel", middle_names="Angel Clínica Universitaria de Navarra", last_name="Monge"),
        SourceAuthorFields(first_name="Apt", last_name="W"),
        SourceAuthorFields(first_name="Apt", last_name="Werner"),
        SourceAuthorFields(last_name="ÃÍ§"),
        SourceAuthorFields(first_name="Irène", last_name="S***"),
    ],
)
def test_reviewed_cleanup_does_not_corrupt_person_collisions(
    predictor: RoutingPredictorV3,
    source: SourceAuthorFields,
) -> None:
    resolved = _route(predictor, source)
    rendered = " ".join(value for value in (resolved.first_name, resolved.middle_names, resolved.last_name) if value)

    assert resolved.resolution_action is not ResolutionAction.SUPPRESS
    assert source.last_name is None or rendered.endswith(source.last_name)
