"""Focused regressions for manually adjudicated structured-source repairs."""

from __future__ import annotations

import pytest

from sinonym.coretypes.routing_resolution import (
    ResolutionAction,
    ResolutionProvenance,
    ResolutionReason,
)
from sinonym.timo._resolution import reviewed_middle_dot_packed_transliteration_assignment
from sinonym.timo.interface import Instance, Predictor, SourceAuthorFields


def _route(
    predictor: Predictor,
    source: SourceAuthorFields,
    *,
    use_vys: bool = False,
):
    """Resolve one structured author through PP-only or PP/VYS routing."""
    instance = Instance(
        pp_authors=[source],
        vys_other_names=["Alice Example"] if use_vys else [],
    )
    (paper,) = predictor.predict_batch([instance])
    return paper.authors[0]


@pytest.mark.parametrize("use_vys", [False, True], ids=["pp", "pp-vys"])
@pytest.mark.parametrize(
    ("packed", "expected"),
    [
        ("阿克塞尔·卡尔滕巴赫尔", ("阿克塞尔", "", "卡尔滕巴赫尔")),
        ("托马斯·斯克特尼科基", ("托马斯", "", "斯克特尼科基")),
        ("P·斯莱登", ("P.", "", "斯莱登")),
        ("保罗·乔治·本奈特", ("保罗", "乔治", "本奈特")),
        ("保罗·G·本奈特", ("保罗", "G.", "本奈特")),
    ],
)
def test_middle_dot_packed_transliteration_assignments_are_terminal(
    predictor: Predictor,
    packed: str,
    expected: tuple[str, str, str],
    use_vys: bool,
) -> None:
    """U+00B7 splits only the reviewed two- or three-part source shape."""
    resolved = _route(predictor, SourceAuthorFields(last_name=packed), use_vys=use_vys)

    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == expected
    assert resolved.suffix is None
    assert resolved.resolution_provenance is ResolutionProvenance.SOURCE
    assert resolved.resolution_action is ResolutionAction.ASSIGN
    assert resolved.resolution_reason is ResolutionReason.REVIEWED_SOURCE_PATTERN_ASSIGNMENT


@pytest.mark.parametrize(
    "source",
    [
        SourceAuthorFields(last_name="阿克塞尔・卡尔滕巴赫尔"),
        SourceAuthorFields(last_name="John·Smith"),
        SourceAuthorFields(last_name="甲"),
        SourceAuthorFields(last_name="甲··乙"),
        SourceAuthorFields(last_name="甲·乙·丙·丁"),
        SourceAuthorFields(last_name="北京大学·研究所"),
        SourceAuthorFields(last_name="서울대학교·연구원"),
        SourceAuthorFields(last_name="阿克塞尔·University"),
        SourceAuthorFields(last_name="张·李 & 王"),
        SourceAuthorFields(first_name="Alice", last_name="阿克塞尔·卡尔滕巴赫尔"),
    ],
)
def test_middle_dot_rule_declines_near_misses(
    predictor: Predictor,
    source: SourceAuthorFields,
) -> None:
    """Lookalikes, non-persons, bad arity, and populated peer fields decline."""
    assert reviewed_middle_dot_packed_transliteration_assignment(source) is None
    assert _route(predictor, source).resolution_reason is not ResolutionReason.REVIEWED_SOURCE_PATTERN_ASSIGNMENT


@pytest.mark.parametrize(
    "organization",
    [
        "香港大學",
        "香港學院",
        "香港實驗室",
        "香港編輯部",
        "香港科學院",
        "香港學部",
        "香港重點實驗室",
        "香港國家實驗室",
    ],
)
def test_middle_dot_rule_declines_traditional_organizations(
    predictor: Predictor,
    organization: str,
) -> None:
    """Traditional organization markers cannot become personal-name fields."""
    source = SourceAuthorFields(last_name=f"陳大文·{organization}")

    assert reviewed_middle_dot_packed_transliteration_assignment(source) is None
    resolved = _route(predictor, source)

    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == ("", "", source.last_name)
    assert resolved.resolution_provenance is ResolutionProvenance.SOURCE
    assert resolved.resolution_action is ResolutionAction.PRESERVE_INPUT
    assert resolved.resolution_reason is ResolutionReason.NON_PERSON_SOURCE_PASSTHROUGH


@pytest.mark.parametrize(
    ("audit_id", "source", "expected"),
    [
        (
            "253982312:1",
            SourceAuthorFields(first_name="L.", middle_names="S.", last_name="Lauria de Cidre"),
            ("L.", "S.", "Lauria de Cidre"),
        ),
        (
            "282588767:189",
            SourceAuthorFields(first_name="Y.", last_name="Pépin Dubois"),
            ("Y.", "", "Pépin Dubois"),
        ),
        (
            "12398174:1",
            SourceAuthorFields(first_name="R", last_name="Uribe Elías"),
            ("R.", "", "Uribe Elías"),
        ),
        (
            "256044778:597",
            SourceAuthorFields(first_name="M.", last_name="Ravonel Salzgeber"),
            ("M.", "", "Ravonel Salzgeber"),
        ),
        (
            "9089081:7",
            SourceAuthorFields(first_name="J", last_name="Robles Barba"),
            ("J.", "", "Robles Barba"),
        ),
        (
            "275754110:0",
            SourceAuthorFields(first_name="A", last_name="Fernandez Ajó"),
            ("A", "", "Fernandez Ajó"),
        ),
        (
            "256007940:1944",
            SourceAuthorFields(first_name="D.", last_name="Paredes Hernandez"),
            ("D.", "", "Paredes Hernandez"),
        ),
        (
            "80498088:6",
            SourceAuthorFields(first_name="V", last_name="Sánchez Margalet"),
            ("V.", "", "Sánchez Margalet"),
        ),
    ],
)
def test_reviewed_compound_surname_exact_assignments(
    predictor: Predictor,
    audit_id: str,
    source: SourceAuthorFields,
    expected: tuple[str, str, str],
) -> None:
    """Only the eight manually safe source tuples keep the compound surname."""
    resolved = _route(predictor, source)

    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == expected, audit_id
    assert resolved.suffix is None
    assert resolved.resolution_provenance is ResolutionProvenance.SOURCE
    assert resolved.resolution_action is ResolutionAction.ASSIGN
    assert resolved.resolution_reason is ResolutionReason.REVIEWED_EXACT_SOURCE_ASSIGNMENT


@pytest.mark.parametrize("use_vys", [False, True], ids=["pp", "pp-vys"])
@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (SourceAuthorFields(first_name="R", middle_names="K", last_name="Chew"), ("R.-K.", "", "Chew")),
        (SourceAuthorFields(first_name="J.", middle_names="I.", last_name="Yi"), ("J. I.", "", "Yi")),
        (SourceAuthorFields(first_name="增兴", last_name="游"), ("Zeng-Xing", "", "You")),
        (SourceAuthorFields(first_name="光", last_name="彩乃"), ("彩乃", "", "光")),
        (SourceAuthorFields(first_name="坂口", last_name="平"), ("平", "", "坂口")),
    ],
)
def test_reviewed_identity_and_language_exact_assignments(
    predictor: Predictor,
    source: SourceAuthorFields,
    expected: tuple[str, str, str],
    use_vys: bool,
) -> None:
    """Reviewed identity/language exceptions carry exact SOURCE provenance."""
    resolved = _route(predictor, source, use_vys=use_vys)

    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == expected
    assert resolved.suffix is None
    assert resolved.resolution_provenance is ResolutionProvenance.SOURCE
    assert resolved.resolution_action is ResolutionAction.ASSIGN
    assert resolved.resolution_reason is ResolutionReason.REVIEWED_EXACT_SOURCE_ASSIGNMENT


def test_unreviewed_de_compound_shape_is_unchanged(predictor: Predictor) -> None:
    """The exact repairs do not introduce a general X-de-Y surname rule."""
    source = SourceAuthorFields(first_name="Maria", middle_names="Helena de", last_name="Souza")
    resolved = _route(predictor, source)

    assert (resolved.first_name, resolved.middle_names, resolved.last_name) == ("Maria", "Helena", "de Souza")
    assert resolved.resolution_reason is ResolutionReason.SCALAR_BASELINE
