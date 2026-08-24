"""Score the Hangul 3-syllable router against frozen blind judge labels.

`tests/data/hangul_name_order_gold.json` holds 135 labels for all-Hangul author strings: where the
family-name boundary falls, one syllable or two, or that the string is not a person at all. The judges
saw the raw string only, so the labels score any boundary rule or future Hangul asset without buying a
new judge round.

Nothing else in the suite pins this rule, and it decides 231,786 names / 6,258,735 corpus mentions --
every all-Hangul name fails Chinese detection, so production consumes `canonical_name.normalized` and
the rule's 1+2 (or 2+1) split is what reaches a cluster block key.

Floors are per class because the classes are not equally decidable. `A_compound` must be perfect: the
judges never once placed the boundary after the first syllable there. `C_translit` carries one known
residue -- 샤오젠 (Xiao Jian), where a one-syllable Chinese surname is transcribed as two Hangul
syllables -- which no shipped asset can currently detect, so its floor records the residue instead of
hiding it. `ambiguous` and `unsure` items carry no scoreable boundary and are skipped.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from sinonym import ChineseNameDetector

GOLD_PATH = Path(__file__).parent / "data" / "hangul_name_order_gold.json"

ACCURACY_FLOORS = {
    "A_compound": 1.0,
    "baseline": 1.0,
    "B_nonname": 1.0,
    "B4_top_mention": 1.0,
    # 11 of 12 scoreable items are one-syllable heads the rule gets right; 샤오젠 is the residue.
    "C_translit": 0.9,
    "control": 1.0,
}
SCOREABLE = {"one", "two"}
BOUNDARY = {"one": 1, "two": 2}


@pytest.fixture(scope="module")
def gold() -> dict:
    return json.loads(GOLD_PATH.read_text(encoding="utf-8"))


def test_gold_fixture_is_wellformed(gold: dict) -> None:
    items = gold["items"]

    assert len(items) == gold["provenance"]["items"]
    assert len({item["name"] for item in items}) == len(items)
    assert set(ACCURACY_FLOORS) == {item["class"] for item in items}
    for item in items:
        assert item["family"] in {"one", "two", "not_a_person", "ambiguous", "unsure"}
        assert len(item["name"]) == 3, item["name"]
        assert all("가" <= character <= "힣" for character in item["name"]), item["name"]
    for item in items:
        if "expected" in item:
            assert item["family"] == item["expected"], item["name"]


def _score(detector: ChineseNameDetector, gold: dict) -> tuple[Counter, Counter, list[str]]:
    scored: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    wrong: list[str] = []
    for item in gold["items"]:
        if item["family"] not in SCOREABLE:
            continue
        if detector.normalize_name(item["name"]).success:
            continue  # a Chinese parse: the routed trio would decide, not this rule
        person = detector.normalize_person_name(item["name"])
        got = None if person is None else person.normalized.surname
        want = item["name"][: BOUNDARY[item["family"]]]
        scored[item["class"]] += 1
        if got == want:
            correct[item["class"]] += 1
        else:
            wrong.append(f"{item['name']}: judged surname {want!r}, routed {got!r} [{item['class']}]")
    return scored, correct, wrong


def test_boundary_matches_the_blind_labels(detector: ChineseNameDetector, gold: dict) -> None:
    scored, correct, wrong = _score(detector, gold)

    failures = [
        f"{cls}: {correct[cls]}/{scored[cls]} < floor {ACCURACY_FLOORS[cls]:.2f}"
        for cls in scored
        if correct[cls] / scored[cls] < ACCURACY_FLOORS[cls]
    ]

    assert not failures, "\n".join([*failures, "", *wrong[:20]])


def test_ambiguous_items_still_produce_a_two_way_split(detector: ChineseNameDetector, gold: dict) -> None:
    """A compound head the judges could not resolve must still be segmented, either way.

    황보석 reads as 황 + 보석 or 황보 + 석 and the labels record the tie. Whichever the rule picks, it
    must not fall through to the whole string as one surname — that would drop the segmentation for a
    name that certainly has one.
    """
    offenders = []
    for item in gold["items"]:
        if item["family"] != "ambiguous":
            continue
        person = detector.normalize_person_name(item["name"])
        if person is None or not person.normalized.given_name or not person.normalized.surname:
            offenders.append(f"{item['name']}: {person and person.normalized}")

    assert not offenders, "\n".join(offenders)


def test_compound_surnames_are_never_split_after_one_syllable(
    detector: ChineseNameDetector,
    gold: dict,
) -> None:
    """The defect this fixture was built for, stated directly rather than via a floor."""
    offenders = [
        item["name"]
        for item in gold["items"]
        if item["class"] == "A_compound"
        and (person := detector.normalize_person_name(item["name"])) is not None
        and person.normalized.surname == item["name"][:1]
    ]

    assert not offenders, f"compound surname split after one syllable: {offenders}"


# One name per mechanism, so a regression names itself rather than only moving a floor.
KNOWN_GOOD = (
    ("남궁원", "남궁"),  # compound surname, the defect this fixture was built for
    ("황보관", "황보"),
    ("독고석", "독고"),
    ("김민수", "김"),  # ordinary single-syllable surname, unchanged
    ("이상훈", "이"),
    ("강원실", "강"),  # compound-looking head (강원 is a province) but 강 is the surname
    ("남기웅", "남"),  # 남 alone, not the compound 남궁
)

# Judged wrong today and pinned so the defect cannot move silently. 샤오젠 is Xiao Jian: the Chinese
# surname Xiao occupies two Hangul syllables, so the boundary falls after syllable 2, and no shipped
# asset can detect that — sinonym/data has no Hangul lexicon at all.
KNOWN_RESIDUE = (("샤오젠", "샤오", "샤"),)


def test_named_successes_stay_correct(detector: ChineseNameDetector) -> None:
    wrong = []
    for surface, surname in KNOWN_GOOD:
        person = detector.normalize_person_name(surface)
        got = None if person is None else person.normalized.surname
        if got != surname:
            wrong.append(f"{surface}: expected {surname!r}, got {got!r}")

    assert not wrong, "\n".join(wrong)


def test_named_residue_is_unchanged(detector: ChineseNameDetector) -> None:
    """Fails when the known defect is fixed as well as when it worsens — update the list."""
    moved = []
    for surface, gold_surname, current in KNOWN_RESIDUE:
        person = detector.normalize_person_name(surface)
        got = None if person is None else person.normalized.surname
        if got != current:
            verdict = "FIXED" if got == gold_surname else "changed"
            moved.append(f"{surface}: was {current!r}, now {got!r} (gold {gold_surname!r}) — {verdict}")

    assert not moved, "\n".join([*moved, "", "Update KNOWN_RESIDUE: a pinned defect moved."])
