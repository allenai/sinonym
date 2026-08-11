"""Regression tests for census-bounded residual error fixes."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import cast

import pytest

from sinonym.coretypes.routing_resolution import ResolutionAction, ResolutionReason
from sinonym.timo._resolution import (
    REVIEWED_EXACT_KOREAN_GIVEN_PREFIX_PACKS,
    reviewed_exact_source_assignment,
)
from sinonym.timo.interface import Instance, Predictor, SourceAuthorFields


def _load_reviewed_korean_rows() -> tuple[dict[str, str], ...]:
    """Load the manually adjudicated Korean source-assignment ledger."""
    ledger_path = Path(__file__).parent / "data" / "reviewed_korean_source_assignments.tsv"
    with ledger_path.open(encoding="utf-8", newline="") as ledger_file:
        return tuple(cast("dict[str, str]", row) for row in csv.DictReader(ledger_file, delimiter="\t"))


_REVIEWED_KOREAN_ROWS = _load_reviewed_korean_rows()
_REVIEWED_KOREAN_ASSIGNMENTS = tuple(row for row in _REVIEWED_KOREAN_ROWS if row["decision"] == "assign")


def _source_from_korean_row(row: dict[str, str]) -> SourceAuthorFields:
    """Materialize the source tuple recorded in a Korean ledger row."""
    return SourceAuthorFields(
        first_name=row["source_first_name"],
        middle_names=row["source_middle_names"],
        last_name=row["source_last_name"],
    )


def _route(
    predictor: Predictor,
    source: SourceAuthorFields,
):
    (paper,) = predictor.predict_batch([Instance(pp_authors=[source])])
    return paper.authors[0]


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
    predictor: Predictor,
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
    predictor: Predictor,
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
    predictor: Predictor,
    source: SourceAuthorFields,
    expected: tuple[str, str, str],
) -> None:
    """Full-corpus singletons remove credentials without changing identity."""
    resolved = _route(predictor, source)

    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == expected
    assert resolved.resolution_reason is ResolutionReason.REVIEWED_EXACT_SOURCE_ASSIGNMENT


def test_reviewed_shifted_alias_uses_exact_latin_identity(
    predictor: Predictor,
) -> None:
    """The singleton shifted alias emits the verified Chinese identity in Latin."""
    source = SourceAuthorFields(first_name="\u5289\u6bb7\u4f50", last_name="I-Ting Wang")
    resolved = _route(predictor, source)

    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == ("Yin-Zuo", "", "Liu")
    assert resolved.resolution_reason is ResolutionReason.REVIEWED_EXACT_SOURCE_ASSIGNMENT


def test_reviewed_packed_people_are_suppressed(
    predictor: Predictor,
) -> None:
    """Two people packed into one author slot cannot be safely materialized."""
    source = SourceAuthorFields(last_name="Anthony C. Laborte, Marissa C. Hitalia*")
    resolved = _route(predictor, source)

    assert resolved.resolution_action is ResolutionAction.SUPPRESS
    assert resolved.resolution_reason is ResolutionReason.REVIEWED_NON_PERSON_PATTERN


@pytest.mark.parametrize("source_last_name", ["Kai", "KAI"])
@pytest.mark.parametrize("vys_other_names", [None, ["Zhang Wei"]])
def test_mao_kai_exact_family_first_cohort_is_not_vetoed(
    predictor: Predictor,
    source_last_name: str,
    vys_other_names: list[str] | None,
) -> None:
    """The reviewed Chinese cohort keeps Mao as surname on both routed paths."""
    source = SourceAuthorFields(first_name="Mao", last_name=source_last_name)
    (paper,) = predictor.predict_batch(
        [Instance(pp_authors=[source], vys_other_names=vys_other_names or [])],
    )
    resolved = paper.authors[0]

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
    predictor: Predictor,
    sources: list[SourceAuthorFields],
    focal_index: int,
) -> None:
    """Terminal assignment covers the two corpus contexts that PP abstains on."""
    (paper,) = predictor.predict_batch([Instance(pp_authors=sources)])
    resolved = paper.authors[focal_index]

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
    predictor: Predictor,
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
    predictor: Predictor,
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
    predictor: Predictor,
    source: SourceAuthorFields,
) -> None:
    resolved = _route(predictor, source)

    assert resolved.resolution_reason is not ResolutionReason.REVIEWED_SOURCE_PATTERN_ASSIGNMENT


@pytest.mark.parametrize(
    "source",
    [
        SourceAuthorFields(
            first_name="Nabiev",
            middle_names="Valery",
            last_name="Sharifyanovich",
        ),
        SourceAuthorFields(
            first_name="Turaxodjayeva",
            middle_names="Moxidil",
            last_name="Obidjonovna",
        ),
        SourceAuthorFields(
            first_name="\u0410\u043b\u0435\u043a\u0441\u0435\u0435\u0432",
            middle_names="\u0413\u0435\u043d\u043d\u0430\u0434\u0438\u0439",
            last_name="\u0412\u0430\u043b\u0435\u043d\u0442\u0438\u043d\u043e\u0432\u0438\u0447",
        ),
        SourceAuthorFields(
            first_name="\u0411\u0435\u043b\u043e\u0432",
            middle_names="\u0412\u043b\u0430\u0434\u0438\u043c\u0438\u0440",
            last_name="\u041d\u0438\u043a\u043e\u043b\u0430\u0435\u0432\u0438\u0447",
        ),
        SourceAuthorFields(
            first_name="\u0412\u043e\u043b\u044b\u043d\u043e\u0432",
            middle_names="\u041c\u0438\u0445\u0430\u0438\u043b",
            last_name="\u0410\u043d\u0430\u0442\u043e\u043b\u044c\u0435\u0432\u0438\u0447",
        ),
        SourceAuthorFields(
            first_name="\u041a\u0438\u0440\u0441\u0430\u043d\u043e\u0432",
            middle_names="\u0410\u043d\u0434\u0440\u0435\u0439",
            last_name="\u0420\u043e\u043c\u0430\u043d\u043e\u0432\u0438\u0447",
        ),
        SourceAuthorFields(
            first_name="\u041a\u0438\u0441\u0442\u0435\u0440\u0441\u043a\u0438\u0439",
            middle_names="\u0410\u043b\u0435\u043a\u0441\u0430\u043d\u0434\u0440",
            last_name="\u041f\u0435\u0442\u0440\u043e\u0432\u0438\u0447",
        ),
        SourceAuthorFields(
            first_name="\u041c\u0430\u043a\u0430\u0440\u043e\u0432\u0430",
            middle_names="\u0415\u043a\u0430\u0442\u0435\u0440\u0438\u043d\u0430",
            last_name="\u0412\u043b\u0430\u0434\u0438\u043c\u0438\u0440\u043e\u0432\u043d\u0430",
        ),
        SourceAuthorFields(
            first_name="\u041d\u0435\u0432\u0435\u0440\u043e\u0432\u0430",
            middle_names="\u041e\u043b\u044c\u0433\u0430",
            last_name="\u0410\u043b\u0435\u043a\u0441\u0430\u043d\u0434\u0440\u043e\u0432\u043d\u0430",
        ),
        SourceAuthorFields(
            first_name="\u0421\u0432\u0438\u0434\u0443\u043d\u043e\u0432\u0438\u0447",
            middle_names="\u041d\u0438\u043a\u043e\u043b\u0430\u0439",
            last_name="\u0410\u043b\u0435\u043a\u0441\u0430\u043d\u0434\u0440\u043e\u0432\u0438\u0447",
        ),
        SourceAuthorFields(
            first_name="\u0422\u0440\u043e\u0444\u0438\u043c\u043e\u0432",
            middle_names="\u0410\u0440\u0442\u0435\u043c",
            last_name="\u0410\u043b\u0435\u043a\u0441\u0430\u043d\u0434\u0440\u043e\u0432\u0438\u0447",
        ),
        SourceAuthorFields(
            first_name="\u0422\u0443\u0445\u0442\u0430\u043c\u0443\u0440\u043e\u0434",
            middle_names="\u0417\u0438\u0451\u0434\u0443\u043b\u043b\u0430",
            last_name="\u0417\u0438\u043a\u0440\u0438\u043b\u043b\u0430",
        ),
        SourceAuthorFields(
            first_name="\u0428\u0435\u0432\u0447\u0435\u043d\u043a\u043e",
            middle_names="\u0415\u043b\u0435\u043d\u0430",
            last_name="\u0412\u0438\u043a\u0442\u043e\u0440\u043e\u0432\u043d\u0430",
        ),
        SourceAuthorFields(
            first_name="\u042f\u043c\u0430\u043b\u0434\u0438\u043d\u043e\u0432",
            middle_names="\u0422\u0438\u043c\u0443\u0440",
            last_name="\u0420\u0438\u0444\u0430\u0442\u043e\u0432\u0438\u0447",
        ),
    ],
)
def test_reviewed_exact_surname_given_patronymic_tuples(
    predictor: Predictor,
    source: SourceAuthorFields,
) -> None:
    """Every full-corpus occurrence of these exact tuples agreed on roles."""
    resolved = _route(predictor, source)

    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == (
        source.middle_names,
        source.last_name,
        source.first_name,
    )
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
    predictor: Predictor,
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
    predictor: Predictor,
) -> None:
    """Every full-corpus Seungbo hit was Korean, independent of surname."""
    source = SourceAuthorFields(first_name="Seungbo", last_name="Choi")

    resolved = _route(predictor, source)

    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == ("Seungbo", "", "Choi")


def test_reviewed_spaced_korean_given_surface_is_restored_after_vys_selection(
    predictor: Predictor,
) -> None:
    source = SourceAuthorFields(first_name="So Young", last_name="Yun")
    (paper,) = predictor.predict_batch(
        [
            Instance(
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

    resolved = paper.authors[0]

    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == ("So Young", "", "Yun")
    assert resolved.resolution_reason is ResolutionReason.VYS_SELECTED


def test_reviewed_korean_source_assignment_inventory_matches_manual_ledger() -> None:
    assert len(_REVIEWED_KOREAN_ROWS) == 66
    assert len(_REVIEWED_KOREAN_ASSIGNMENTS) == 59
    assert sum(row["decision"] == "exclude" for row in _REVIEWED_KOREAN_ROWS) == 6
    assert sum(row["decision"] == "abstain" for row in _REVIEWED_KOREAN_ROWS) == 1

    ledger_inventory = {
        (
            row["source_first_name"].casefold(),
            row["source_middle_names"].casefold(),
            row["source_last_name"].casefold(),
        ): int(row["given_prefix_tokens"])
        for row in _REVIEWED_KOREAN_ASSIGNMENTS
    }
    assert ledger_inventory == REVIEWED_EXACT_KOREAN_GIVEN_PREFIX_PACKS

    for row in _REVIEWED_KOREAN_ASSIGNMENTS:
        resolved = reviewed_exact_source_assignment(_source_from_korean_row(row))
        assert resolved is not None
        assert (resolved.given_name, resolved.middle_name, resolved.surname) == (
            row["expected_given_name"],
            row["expected_middle_name"],
            row["expected_surname"],
        )


@pytest.mark.parametrize(
    "row",
    _REVIEWED_KOREAN_ASSIGNMENTS,
    ids=lambda row: row["occurrence_ids"],
)
def test_reviewed_korean_source_orthography_is_assigned_exactly(
    predictor: Predictor,
    row: dict[str, str],
) -> None:
    source = _source_from_korean_row(row)
    resolved = _route(predictor, source)

    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == (
        row["expected_given_name"],
        row["expected_middle_name"],
        row["expected_surname"],
    )
    assert resolved.resolution_action is ResolutionAction.ASSIGN
    assert resolved.resolution_reason is ResolutionReason.REVIEWED_EXACT_SOURCE_ASSIGNMENT


def test_attested_korean_hyphen_case_remains_unchanged(
    predictor: Predictor,
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
    predictor: Predictor,
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
    predictor: Predictor,
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
    predictor: Predictor,
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
    predictor: Predictor,
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
    predictor: Predictor,
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
    predictor: Predictor,
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
    predictor: Predictor,
    source: SourceAuthorFields,
) -> None:
    resolved = _route(predictor, source)
    rendered = " ".join(value for value in (resolved.first_name, resolved.middle_names, resolved.last_name) if value)

    assert resolved.resolution_action is not ResolutionAction.SUPPRESS
    assert source.last_name is None or rendered.endswith(source.last_name)
