"""Score the spaced-CJK family-first router against frozen blind judge labels.

`tests/data/spaced_cjk_name_order_gold.json` holds 520 human-convention labels for
two-token Han/kana author names: which whitespace token is the family name. The judges saw
the raw string only, never any sinonym output, so the labels score any routing rule or
lexicon asset without buying a new judge round — a rule change or an asset bump reshuffles
which sub-class a name falls into, and this file says what the right answer was all along.

Only the non-Chinese stratum is scored. A name whose `normalize_name` succeeds is a Chinese
parse, and production (Scholar #42004 `applyParse`) takes the routed trio for it, so the
family-first router never decides its surname; the split is recomputed here rather than
frozen into the fixture, because which names pass Chinese detection is version-dependent.

Floors are per sub-class because the classes are not equally decidable. The routed classes
must stay perfect. The abstain classes carry the residue the rule knowingly declines, and
their floors record that residue instead of hiding it.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from sinonym import ChineseNameDetector

GOLD_PATH = Path(__file__).parent / "data" / "spaced_cjk_name_order_gold.json"

# sub-class -> minimum share of scored gold items whose surname the router must get right.
ACCURACY_FLOORS = {
    # Routed classes: one-sided or two-sided dictionary evidence, all reordered family-first.
    "FLIP(shipped)": 1.0,
    "A: t0 sur, t1 unknown": 1.0,
    "B: t1 giv, t0 unknown": 1.0,
    # Abstain classes. Declining is right for reverse-plausible pairs (60/60 here, 150/150 in
    # a later round) and for most of the no-evidence pairs (41/50 and 34/49 here, 74.7% and
    # 64.1% of class occ in a later round).
    "REV(t0 giv, t1 sur)": 1.0,
    "D: neither known": 0.80,
    "D: neither known/kata": 0.65,
    # Both-sided classes now route, so these are routed floors rather than residue floors.
    # 10/11 and 1/1 here; the wider evidence is 91.6% and 89.3% of class occ (297 and 159
    # PPS-sampled names) and 84.0% / 93.9% by name over 486 and 428 names.
    "C: both given-plausible": 0.85,
    "AMBIG(both surnames)": 1.0,
    # Names carrying a character the gate rejects (々 U+3005 is Script=Han but outside
    # sinonym's Han ranges), so the rule never runs. These 13 items are family-first and so
    # all wrong today, but they are NOT representative: a uniform draw of 59 more from the
    # class is only 20.3% family-first, because most 々 names are given-first bylines
    # ("ささぶね 佐々木", "勉 野々山") where refusing to route is the right answer. Widening
    # the gate would break ~1,433 occ to fix ~366, so the floor stays 0.0 for this slice and
    # the class is deliberately left alone.
    "BLOCKED(gate rejects a char)": 0.0,
    "attention_check": 1.0,
}
OVERALL_FLOOR = 0.85
OCC_WEIGHTED_FLOOR = 0.95


@pytest.fixture(scope="module")
def gold() -> dict:
    return json.loads(GOLD_PATH.read_text())


def test_gold_fixture_is_wellformed(gold: dict) -> None:
    items = gold["items"]

    assert len(items) == gold["provenance"]["items"]
    assert len({item["name"] for item in items}) == len(items)
    assert set(ACCURACY_FLOORS) == {item["class"] for item in items}
    for item in items:
        assert item["family"] in {"first", "second", "unsure", "not_a_person"}
        assert len(item["name"].split(" ")) == 2, item["name"]


def _judged_surname(item: dict) -> str:
    first, last = item["name"].split(" ")
    return first if item["family"] == "first" else last


def _score(detector: ChineseNameDetector, gold: dict) -> tuple[Counter, Counter, list[str]]:
    scored: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    wrong: list[str] = []
    for item in gold["items"]:
        if item["family"] not in {"first", "second"}:
            continue  # non-person and unsure items carry no surname to score
        if detector.normalize_name(item["name"]).success:
            continue  # Chinese parse: the routed trio decides, not this router
        person = detector.normalize_person_name(item["name"])
        got = None if person is None else person.normalized.surname
        scored[item["class"]] += 1
        if got == _judged_surname(item):
            correct[item["class"]] += 1
        else:
            wrong.append(f"{item['name']}: judged {_judged_surname(item)!r}, routed {got!r} [{item['class']}]")
    return scored, correct, wrong


def test_router_matches_the_blind_labels(detector: ChineseNameDetector, gold: dict) -> None:
    scored, correct, wrong = _score(detector, gold)

    failures = [
        f"{cls}: {correct[cls]}/{scored[cls]} < floor {ACCURACY_FLOORS[cls]:.2f}"
        for cls in scored
        if correct[cls] / scored[cls] < ACCURACY_FLOORS[cls]
    ]
    overall = sum(correct.values()) / sum(scored.values())
    if overall < OVERALL_FLOOR:
        failures.append(f"overall: {overall:.3f} < floor {OVERALL_FLOOR}")

    assert not failures, "\n".join([*failures, "", *wrong[:20]])


def test_occ_weighted_accuracy_holds(detector: ChineseNameDetector, gold: dict) -> None:
    """Per-class gold accuracy weighted by the class's corpus occ share.

    Gold sampling is stratified, not occ-proportional, so a raw item count understates the
    routed classes; weighting by `class_population` is what makes the number comparable to
    the corpus. This guards the headline claim of the one-sided-evidence rule: 12.3% of the
    stratum's occ was surname-correct before spaced kanji was routed at all, 76.2% with
    two-sided evidence required, 97.6% once one unopposed side is enough.
    """
    population = gold["class_population"]
    scored, correct, _ = _score(detector, gold)
    classes = [cls for cls in scored if cls in population]

    weighted = sum(population[cls]["occ"] * correct[cls] / scored[cls] for cls in classes)
    total_occ = sum(population[cls]["occ"] for cls in classes)

    assert weighted / total_occ >= OCC_WEIGHTED_FLOOR, (
        f"occ-weighted accuracy {weighted / total_occ:.3f} < floor {OCC_WEIGHTED_FLOOR}"
    )
