"""Score bare-ASCII Vietnamese family-first routing against frozen blind judge labels.

`tests/data/vietnamese_name_order_gold.json` holds 690 labels for names whose head is one of the
admitted Vietnamese surnames: which token is the family name. The judges saw the raw string only, so
the labels score any routing rule, guard or lexicon change without buying a new judge round.

The rule decides 23,462 names / 101,119 corpus mentions, and a further 9,144 names / 53,067 mentions
are held back by the trailing-surname guard, so both the routed and the declined side are represented.

Two accuracy views, because the sampling differs and each answers a different question:

  * `PPS_FLOOR` covers the 395 items drawn with probability proportional to corpus mentions. Under PPS
    the sample's unweighted rate estimates the MENTION-WEIGHTED corpus rate, which is the number the
    change was justified on: 97.5% (95% CI 95.4-98.6%).
  * `SHAPE_FLOORS` cover every item by name count. The stratified rounds deliberately oversampled the
    hard shapes, so these are lower by construction and must not be read as corpus accuracy.

`risky_2tok` — two bare-ASCII surnames, which the guard declines — scores 36.7% by NAME while being
right on 74% of mentions: the shape's high-mention rows are inverted diaspora bylines and its long tail
is domestic order. Its floor records that trade rather than hiding it. If a future change makes this
class name-accurate at the cost of the PPS floor, that is a regression, not an improvement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from sinonym import ChineseNameDetector

GOLD_PATH = Path(__file__).parent / "data" / "vietnamese_name_order_gold.json"

# Mention-weighted floor over the PPS subset — the headline claim of the 26-head relaxation.
PPS_FLOOR = 0.95

# Per-shape floors over every item, by name count. Enriched sampling, so lower than corpus accuracy.
SHAPE_FLOORS = {
    "multi_tok": 0.93,
    "two_tok": 0.70,
}

# Per-stratum floors. `risky_2tok` is the guard's own class: declined on purpose, and right on
# mentions rather than on names.
STRATUM_FLOORS = {
    "pps_routed": 0.95,
    "multi_tok": 0.84,
    "plain_2tok": 0.83,
    "risky_2tok": 0.33,
}


@pytest.fixture(scope="module")
def gold() -> dict:
    return json.loads(GOLD_PATH.read_text())


def test_gold_fixture_is_wellformed(gold: dict) -> None:
    items = gold["items"]

    assert len(items) == gold["provenance"]["items"]
    assert len({item["name"] for item in items}) == len(items)
    assert set(SHAPE_FLOORS) == {item["shape"] for item in items}
    assert set(STRATUM_FLOORS) == {item["stratum"] for item in items}
    for item in items:
        assert item["family"] in item["name"].split(), item["name"]
        assert item["instrument"] in {"open_ended", "two_candidate"}
        assert item["mentions"] >= 0


def _routed_surname(detector: ChineseNameDetector, name: str) -> str | None:
    person = detector.normalize_person_name(name)
    return None if person is None else person.normalized.surname


def _hits(detector: ChineseNameDetector, items: list[dict]) -> tuple[int, list[str]]:
    correct = 0
    wrong: list[str] = []
    for item in items:
        got = _routed_surname(detector, item["name"])
        if got is not None and got.casefold() == item["family"].casefold():
            correct += 1
        else:
            wrong.append(f"{item['name']}: gold {item['family']!r}, routed {got!r} [{item['stratum']}]")
    return correct, wrong


def test_mention_weighted_accuracy_on_the_pps_sample(detector: ChineseNameDetector, gold: dict) -> None:
    """The claim the relaxation shipped on, measured without projecting a per-name rate."""
    items = [item for item in gold["items"] if item["stratum"] == "pps_routed"]
    correct, wrong = _hits(detector, items)

    assert correct / len(items) >= PPS_FLOOR, "\n".join(
        [f"PPS accuracy {correct}/{len(items)} = {correct / len(items):.3f} < floor {PPS_FLOOR}", "", *wrong[:20]],
    )


def test_accuracy_holds_per_token_shape(detector: ChineseNameDetector, gold: dict) -> None:
    by_shape: dict[str, list[dict]] = {}
    for item in gold["items"]:
        by_shape.setdefault(item["shape"], []).append(item)

    failures = []
    details: list[str] = []
    for shape, items in by_shape.items():
        correct, wrong = _hits(detector, items)
        if correct / len(items) < SHAPE_FLOORS[shape]:
            failures.append(f"{shape}: {correct}/{len(items)} = {correct / len(items):.3f} < {SHAPE_FLOORS[shape]}")
            details.extend(wrong[:10])

    assert not failures, "\n".join([*failures, "", *details])


def test_accuracy_holds_per_stratum(detector: ChineseNameDetector, gold: dict) -> None:
    by_stratum: dict[str, list[dict]] = {}
    for item in gold["items"]:
        by_stratum.setdefault(item["stratum"], []).append(item)

    failures = []
    for stratum, items in by_stratum.items():
        correct, _ = _hits(detector, items)
        if correct / len(items) < STRATUM_FLOORS[stratum]:
            failures.append(f"{stratum}: {correct}/{len(items)} = {correct / len(items):.3f} < {STRATUM_FLOORS[stratum]}")

    assert not failures, "\n".join(failures)


def test_no_multi_token_surname_is_emitted_for_a_routed_name(
    detector: ChineseNameDetector,
    gold: dict,
) -> None:
    """A fabricated two-token surname is the defect the relaxation exists to remove.

    "Vu Van Quang" shipped surname "Van Quang", which matches no real person and blocks with nobody.
    Wherever the gold family name is a single token and the rule routes, the emitted surname must be
    one token too — a name the guard declines may still carry the old multi-token reading, so only
    routed names are asserted.
    """
    offenders = []
    for item in gold["items"]:
        got = _routed_surname(detector, item["name"])
        if got is None:
            continue
        if got.casefold() == item["name"].split()[0].casefold() and " " in got:
            offenders.append(f"{item['name']}: routed surname {got!r}")

    assert not offenders, "\n".join(offenders)


# Representative names the rule gets right, one per mechanism, so a regression names itself instead of
# only moving a floor.
KNOWN_GOOD = (
    ("Vu Van Quang", "Vu"),  # 3 tokens, admitted head: was surname "Van Quang"
    ("Bui Huu Tai", "Bui"),  # middle particle `Huu` marks domestic order
    ("Huynh Quang Huy", "Huynh"),
    ("Vo Thi My Hanh", "Vo"),  # `Thi` only ever follows the family name
    ("Vu Thi", "Vu"),  # truncated record: family + particle, given name lost
    ("Nguyen Van Hieu", "Nguyen"),  # the three pre-existing heads keep working
    ("Nguyen Hy", "Nguyen"),  # `Hy` is not globally a trailing family name
    ("Đinh Thị Hiền Lê", "Đinh"),  # a trailing top-four surname is not globally decisive
    ("Bùi Văn Lễ", "Bùi"),  # `Lễ` folds to top-four `Le` but is the given name here
    ("Bùi Hoàng Thảo Trân", "Bùi"),  # `Trân` must not accent-fold into reviewed `Trần`
    ("Truong Khang Nguyen", "Nguyen"),  # trailing surname: inverted byline, correctly declined
    ("Thuong Le-Tien", "Le-Tien"),  # trailing surname behind a hyphen, correctly declined
    ("Ngo Si-Huy", "Ngo"),  # hyphenated GIVEN name: still routes
    ("Kim Overvad", "Overvad"),  # excluded head, Danish name untouched
    ("Ho Jin Kim", "Kim"),  # excluded head, Korean name untouched
)

# Representative unresolved rows, with the surname currently emitted.
KNOWN_RESIDUE = (
    ("Vu Dinh", "Vu", "Dinh"),  # two bare surnames: declined, right on mentions
    ("Hoang Ha", "Hoang", "Ha"),  # same shape
    ("Hoang Van Luong", "Hoang", "Van Luong"),  # guard declines, old two-token surname survives
    ("Vu Lam", "Vu", "Lam"),
    ("Hoang Vu-Thien", "Hoang", "Vu-Thien"),  # hyphen guard fires where judges split
)


def test_named_successes_stay_correct(detector: ChineseNameDetector) -> None:
    wrong = []
    for surface, surname in KNOWN_GOOD:
        got = _routed_surname(detector, surface)
        if got != surname:
            wrong.append(f"{surface}: expected {surname!r}, got {got!r}")

    assert not wrong, "\n".join(wrong)


def test_named_residue_is_unchanged(detector: ChineseNameDetector) -> None:
    """Fails when a known defect is fixed as well as when a new one appears — update the list."""
    moved = []
    for surface, gold_surname, current in KNOWN_RESIDUE:
        got = _routed_surname(detector, surface)
        if got != current:
            verdict = "FIXED" if got == gold_surname else "changed"
            moved.append(f"{surface}: was {current!r}, now {got!r} (gold {gold_surname!r}) — {verdict}")

    assert not moved, "\n".join([*moved, "", "Update KNOWN_RESIDUE: a pinned defect moved."])
