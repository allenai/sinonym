import json
from collections import Counter
from pathlib import Path

from sinonym.pipeline.name_order_routing import route_pp_abstain_rows, route_pp_vys_abstain_rows

DATA_DIR = Path("sinonym/data/name_order_routing")
PP_VYS_FIXTURE_ROWS = 1000
PP_VYS_DECISIVE_ROWS = 969
PP_ABSTAIN_FIXTURE_ROWS = 750
PP_ABSTAIN_DECISIVE_ROWS = 550


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _confusion(rows: list[dict[str, object]], labels: set[str], pred_key: str) -> Counter[tuple[str, str]]:
    return Counter((str(row["decision"]), str(row[pred_key])) for row in rows if row["decision"] in labels)


def test_pp_vys_abstain_label_fixture_reproduces_validation_metrics():
    rows = _load_jsonl(DATA_DIR / "pp_vys_abstain_labels.jsonl")
    routed = route_pp_vys_abstain_rows(rows)
    for row in routed:
        row["effective_router_prediction"] = (
            row["input_order_candidate"] if row["router_prediction"] == "abstain" else row["router_prediction"]
        )

    decisive = [row for row in routed if row["decision"] in {"pp", "vys"}]
    confusion = _confusion(decisive, {"pp", "vys"}, "effective_router_prediction")
    reason_counts = Counter(row["router_reason"] for row in decisive)

    assert len(rows) == PP_VYS_FIXTURE_ROWS
    assert len(decisive) == PP_VYS_DECISIVE_ROWS
    assert confusion == {
        ("pp", "pp"): 409,
        ("pp", "vys"): 65,
        ("vys", "pp"): 74,
        ("vys", "vys"): 421,
    }
    assert {reason: count for reason, count in reason_counts.items() if reason.startswith("name_prior_")} == {
        "name_prior_cantonese_given_first": 6,
        "name_prior_korean_given_first_three_token": 33,
        "name_prior_ouyang_surname_first": 1,
        "name_prior_repeated_tail_given_surname_first": 5,
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
        ("abstain", "abstain"): 433,
        ("abstain", "pp"): 5,
        ("pp", "abstain"): 7,
        ("pp", "pp"): 105,
    }
    assert reason_counts == {
        "default_abstain": 149,
        "spaced_cjk_zero_batch_surname_first": 5,
        "weak_zero_batch": 211,
        "zero_batch_mixed_long": 75,
        "surname_first_two_token": 110,
    }
