import json
from collections import Counter
from pathlib import Path

from sinonym.pipeline.name_order_routing import route_pp_abstain_rows, route_pp_vys_abstain_rows

DATA_DIR = Path("sinonym/data/name_order_routing")
PP_VYS_FIXTURE_ROWS = 1000
PP_VYS_DECISIVE_ROWS = 806
PP_VYS_EITHER_ROWS = 163
PP_ABSTAIN_FIXTURE_ROWS = 750
PP_ABSTAIN_DECISIVE_ROWS = 608


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _confusion(rows: list[dict[str, object]], labels: set[str], pred_key: str) -> Counter[tuple[str, str]]:
    return Counter((str(row["decision"]), str(row[pred_key])) for row in rows if row["decision"] in labels)


def _labeled_string(row: dict[str, object]) -> str:
    return str(row["pp_result"] if row["decision"] == "pp" else row["vys_result"])


def _emitted_string(row: dict[str, object]) -> str | None:
    """Return the string the pipeline emits for this routed row, or None if indeterminate.

    Router `abstain` emits the preprocessed input-order parse; its string is the
    PP/VYS result on whichever side `input_order_candidate` says preserves input
    order. With candidate `unknown` (or terminal `not_person`) no emitted string
    can be derived from the fixture, so the row can never score as correct.
    """
    route = row["router_prediction"]
    if route == "abstain":
        route = row["input_order_candidate"]
    if route == "pp":
        return str(row["pp_result"])
    if route == "vys":
        return str(row["vys_result"])
    return None


def test_pp_vys_abstain_label_fixture_reproduces_validation_metrics():
    """Score the pp-vys router at the emitted-string (output) level.

    A decisive (label `pp`/`vys`) row counts correct iff the string the pipeline
    would emit under the router's route equals the labeled route's string
    (`pp_result` for label `pp`, `vys_result` for label `vys`).

    Rows labeled `either` are the ones whose PP and VYS parses emit the
    identical string under the current parser: the route choice is cosmetic, so
    the row scores correct for any person route (pp, vys, or abstain resolving
    to either side). The invariant `pp_result == vys_result` is asserted per
    row — if a parser change makes an `either` row's strings diverge again, the
    fixture is stale and the row must go back for relabeling (see the fixture
    README), so this test fails loudly rather than trusting the label.
    """
    rows = _load_jsonl(DATA_DIR / "pp_vys_abstain_labels.jsonl")
    routed = route_pp_vys_abstain_rows(rows)

    either = [row for row in routed if row["decision"] == "either"]
    stale_either = [
        str(row["item_id"])
        for row in either
        if not (row["pp_success"] and row["vys_success"] and row["pp_result"] == row["vys_result"])
    ]
    assert stale_either == [], f"either-labeled rows whose PP/VYS strings diverged (relabel them): {stale_either}"
    either_scored = Counter(
        "match" if _emitted_string(row) == str(row["pp_result"]) else "mismatch" for row in either
    )

    decisive = [row for row in routed if row["decision"] in {"pp", "vys"}]
    confusion = Counter(
        (str(row["decision"]), "match" if _emitted_string(row) == _labeled_string(row) else "mismatch") for row in decisive
    )
    reason_counts = Counter(row["router_reason"] for row in decisive)

    assert len(rows) == PP_VYS_FIXTURE_ROWS
    assert len(decisive) == PP_VYS_DECISIVE_ROWS
    assert len(either) == PP_VYS_EITHER_ROWS
    assert either_scored == {"match": PP_VYS_EITHER_ROWS}
    assert confusion == {
        ("pp", "match"): 386,
        ("pp", "mismatch"): 61,
        ("vys", "match"): 343,
        ("vys", "mismatch"): 16,
    }
    assert {reason: count for reason, count in reason_counts.items() if reason.startswith("name_prior_")} == {
        "name_prior_cantonese_given_first": 6,
        "name_prior_korean_given_first_three_token": 30,
        "name_prior_repeated_tail_given_surname_first": 4,
    }


def test_pp_abstain_label_fixture_reproduces_current_validation_metrics():
    rows = _load_jsonl(DATA_DIR / "pp_abstain_labels.jsonl")
    routed = route_pp_abstain_rows(rows)

    decisive = [row for row in routed if row["decision"] in {"pp", "abstain"}]
    confusion = _confusion(decisive, {"pp", "abstain"}, "router_prediction")
    reason_counts = Counter(row["router_reason"] for row in decisive)

    assert len(rows) == PP_ABSTAIN_FIXTURE_ROWS
    assert len(decisive) == PP_ABSTAIN_DECISIVE_ROWS
    assert confusion == {
        ("abstain", "abstain"): 438,
        ("abstain", "pp"): 6,
        ("pp", "abstain"): 15,
        ("pp", "pp"): 149,
    }
    assert reason_counts == {
        "clean_bilingual_given_first": 34,
        "default_abstain": 162,
        "spaced_cjk_zero_batch_surname_first": 7,
        "weak_zero_batch": 206,
        "zero_batch_mixed_long": 78,
        "surname_first_two_token": 121,
    }
