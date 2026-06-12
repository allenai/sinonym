"""Western surname-particle canonicalisation (opt-in flag) — comprehensive suite.

Sinonym rejects Western names by default. With
``ChineseNameDetector(enable_western_particles=True)`` a Western
"given + nobiliary/patronymic particle + surname" name is canonicalised: the
particle is folded into the surname, lowercased; other tokens keep input casing.

The bulk of the cases live in ``tests/data/western_particle_cases.csv`` (one row =
one name), so the corpus is data-driven and easy to extend. Columns:

    input    : the raw name fed to normalize_name
    expect   : 'fold' (canonicalises, success) or 'reject'
    output   : canonical string when expect=='fold' (ignored for reject)
    status   : 'ok' (logic must match) or 'xfail' (KNOWN LIMITATION — see note)
    note     : human description / why

`status==xfail` rows are documented failure modes (ambiguity + casing) where the
closed-list rule cannot disambiguate a particle from a real given name or an
integral capitalised surname element. They are marked xfail(strict=False) so the
suite stays green while tracking the gap; if the logic later improves they xpass.

Structural unit tests below cover: the DEFAULT detector still rejects every
fold-case (contract unchanged), parsed components, Chinese names are not hijacked,
and flag isolation between detector instances.
"""

import csv
from pathlib import Path

import pytest
from sinonym import ChineseNameDetector

_CSV = Path(__file__).parent / "data" / "western_particle_cases.csv"


def _load_cases():
    rows = []
    with _CSV.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if not r.get("input", "").strip():
                continue
            rows.append(r)
    return rows


_CASES = _load_cases()
_FOLD_CASES = [r for r in _CASES if r["expect"] == "fold"]

_PARTS_CSV = Path(__file__).parent / "data" / "western_particle_parts_cases.csv"


def _load_parts_cases():
    rows = []
    with _PARTS_CSV.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if not (r.get("first", "").strip() or r.get("last", "").strip()):
                continue
            rows.append(r)
    return rows


_PARTS_CASES = _load_parts_cases()


def _param(r):
    marks = []
    if r["status"].strip() == "xfail":
        marks = [pytest.mark.xfail(reason=r["note"], strict=False)]
    return pytest.param(r, id=r["input"], marks=marks)


@pytest.fixture(scope="module")
def western_detector():
    return ChineseNameDetector(enable_western_particles=True)


@pytest.fixture(scope="module")
def default_detector():
    return ChineseNameDetector()


def test_csv_loaded_and_sane():
    # Guard against an empty/garbled CSV silently passing the parametrised tests.
    assert len(_CASES) >= 60, f"expected a rich corpus, got {len(_CASES)} rows"
    for r in _CASES:
        assert r["expect"] in {"fold", "reject"}, r
        assert r["status"] in {"ok", "xfail"}, r
        if r["expect"] == "fold":
            assert r["output"].strip(), f"fold case needs an expected output: {r['input']}"


@pytest.mark.parametrize("r", [_param(r) for r in _CASES])
def test_western_particle_cases(western_detector, r):
    """Flag ON: each case canonicalises (fold) or is rejected, per the CSV."""
    res = western_detector.normalize_name(r["input"])
    if r["expect"] == "fold":
        assert res.success, f"{r['input']!r} should fold, rejected: {res.error_message}"
        assert res.result == r["output"], f"{r['input']!r} -> {res.result!r}, expected {r['output']!r}"
    else:
        assert res.success is False, f"{r['input']!r} should be rejected, got {res.result!r}"


@pytest.mark.parametrize("r", [pytest.param(r, id=r["input"]) for r in _FOLD_CASES])
def test_default_detector_still_rejects_every_fold_case(default_detector, r):
    """Flag OFF (default): the Chinese-only contract is unchanged — every name the
    Western router would fold is still rejected. (No xfail here: this must always
    hold regardless of the router's own limitations.)"""
    res = default_detector.normalize_name(r["input"])
    assert res.success is False, f"DEFAULT must reject {r['input']!r}, got {res.result!r}"


@pytest.mark.parametrize(
    "r",
    [pytest.param(r, id=f"{r['first']}|{r['middle']}|{r['last']}") for r in _PARTS_CASES],
)
def test_structured_parts_cases(western_detector, r):
    """STRUCTURED path: normalize_name_parts(first, middle, last). Folds confidently
    or fails closed (returns failure → caller keeps its split). The source split is a
    prior cross-checked against corpus stats; mis-split inputs fail rather than emit a
    wrong canonical."""
    res = western_detector.normalize_name_parts(r["first"], r["middle"], r["last"])
    if r["expect"] == "fold":
        assert res.success, f"({r['first']}|{r['middle']}|{r['last']}) should fold, got: {res.error_message}"
        assert res.result == r["output"], f"-> {res.result!r}, expected {r['output']!r}"
    else:
        assert res.success is False, f"({r['first']}|{r['middle']}|{r['last']}) should fail closed, got {res.result!r}"


def test_structured_default_off_rejects():
    """Default detector: normalize_name_parts must fail (flag off, contract intact)."""
    off = ChineseNameDetector()
    assert off.normalize_name_parts("Roeland", ["van"], "Hout").success is False


def test_structured_fixes_flat_false_negative(western_detector):
    """The flat path rejects `Della de Souza` (leading particle); the structured path
    fixes it because Della arrives in the first field."""
    assert western_detector.normalize_name("Della de Souza").success is False  # flat FN
    r = western_detector.normalize_name_parts("Della", "", "de Souza")          # structured fix
    assert r.success and r.result == "Della de Souza"


def test_use_prior_flag(western_detector):
    """use_prior=True uses the field split as a prior; use_prior=False ignores it and
    re-derives from the concatenated string (== normalize_name of the joined name)."""
    # Della de Souza: the prior makes it foldable; without the prior the flat router
    # hits the leading-particle false-negative and rejects.
    assert western_detector.normalize_name_parts("Della", "", "de Souza", use_prior=True).result == "Della de Souza"
    assert western_detector.normalize_name_parts("Della", "", "de Souza", use_prior=False).success is False
    # no-prior must match the flat normalize_name of the joined string, for fold and reject cases.
    for f, m, l in [("Roeland", "van", "Hout"), ("Della", "", "de Souza"), ("John", "Ben", "Carter")]:
        joined = " ".join([t for t in [f, *(m if isinstance(m, list) else m.split()), l] if t])
        np = western_detector.normalize_name_parts(f, m, l, use_prior=False)
        flat = western_detector.normalize_name(joined)
        assert np.success == flat.success and np.result == flat.result


def test_structured_middle_as_list_or_string(western_detector):
    """middle accepts both a string and a list of tokens."""
    assert western_detector.normalize_name_parts("Roeland", "van", "Hout").result == "Roeland van Hout"
    assert western_detector.normalize_name_parts("Roeland", ["van"], "Hout").result == "Roeland van Hout"


def test_parsed_components_carry_the_particle(western_detector):
    res = western_detector.normalize_name("Roeland van Hout")
    assert res.success and res.parsed is not None
    assert res.parsed.given_name == "Roeland"
    assert res.parsed.surname == "van Hout"
    assert res.parsed.surname_tokens == ["van", "Hout"]
    assert res.parsed.given_tokens == ["Roeland"]


def test_split_correct_for_germanic_givens(western_detector):
    """When every pre-particle token is a given name (Dutch/German), the parsed
    given/surname boundary is correct."""
    r = western_detector.normalize_name("Maximilian Pierer von Esch")
    assert r.parsed.given_name == "Maximilian Pierer"
    assert r.parsed.surname == "von Esch"


@pytest.mark.parametrize(
    "raw,given,surname",
    [
        # surname dict pulls pre-particle surnames (Ferreira/Garcia/Silva) into the
        # surname; given-name tokens (Luiz/Maria/Paula/Pierer) stay given.
        ("Osmar Luiz Ferreira de Carvalho", "Osmar Luiz", "Ferreira de Carvalho"),
        ("Maria Garcia de Souza", "Maria", "Garcia de Souza"),
        ("Pedro Silva dos Santos", "Pedro", "Silva dos Santos"),
        ("Ana Paula da Silva", "Ana Paula", "da Silva"),
        ("Jose Maria de Souza", "Jose Maria", "de Souza"),
    ],
)
def test_split_lusophone_multisurname(western_detector, raw, given, surname):
    """Pre-particle surnames are absorbed into the surname (Lusophone/Hispanic
    two-surname names) via the corpus-derived surname dictionary. The formatted
    STRING is order-preserving and unchanged; only the parsed split is refined."""
    r = western_detector.normalize_name(raw)
    assert r.result == raw                       # string unchanged
    assert r.parsed.given_name == given
    assert r.parsed.surname == surname


def test_multi_given_then_particle(western_detector):
    """Two/three given names before the particle: everything before the first
    particle is the given part."""
    assert western_detector.normalize_name("Maria Luisa von Trapp").result == "Maria Luisa von Trapp"
    assert (
        western_detector.normalize_name("John Michael Robert van Dijk").result
        == "John Michael Robert van Dijk"
    )


def test_chinese_names_not_hijacked_when_enabled():
    on = ChineseNameDetector(enable_western_particles=True)
    assert on.normalize_name("Wei Zhang").result == "Wei Zhang"
    # surname-first Chinese still reorders correctly (Western router skipped on
    # all-Chinese input)
    assert on.normalize_name("Zhang Wei").result == "Wei Zhang"


def test_flag_isolation_between_instances():
    on = ChineseNameDetector(enable_western_particles=True)
    off = ChineseNameDetector()
    assert on.normalize_name("Vincent van Gogh").success is True
    assert off.normalize_name("Vincent van Gogh").success is False
