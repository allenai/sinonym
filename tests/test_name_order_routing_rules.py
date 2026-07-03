import csv
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sinonym.coretypes import ParsedName, ParseResult
from sinonym.pipeline.name_order_routing import (
    PP_ABSTAIN_REQUIRED_COLUMNS,
    PP_ABSTAIN_TWO_TOKEN_RESULT_COUNT,
    PP_VYS_ABSTAIN_REQUIRED_COLUMNS,
    build_pp_abstain_rows,
    build_pp_vys_abstain_rows,
    input_order_parsed,
    route_pp_abstain_rows,
    route_pp_vys_abstain_batches,
    route_pp_vys_abstain_rows,
)

JSONL_SCALAR_COUNT = 3
JSONL_SCALAR_SCORE = 0.25


def _base_pp_vys_row():
    return {
        "name": "Xi Jiang",
        "pp_success": True,
        "pp_result": "Xi Jiang",
        "pp_selected_format": "surname_first",
        "pp_selected_surname_position": "first",
        "pp_selected_surname_frequency": 10.0,
        "pp_batch_dominant_format": "surname_first",
        "pp_batch_threshold_met": True,
        "pp_batch_vote_margin": 0.5,
        "pp_batch_total_count": 3,
        "pp_batch_confidence": 0.70,
        "pp_selected_surname_frequency_ratio": 1.0,
        "vys_success": True,
        "vys_result": "Xi Jiang",
        "vys_selected_format": "surname_first",
        "vys_selected_surname_position": "first",
        "vys_selected_surname_frequency": 10.0,
        "vys_batch_dominant_format": "surname_first",
        "vys_batch_threshold_met": False,
        "vys_batch_confidence": 0.0,
        "vys_batch_vote_margin": 0.0,
    }


def _apply_pp_vys_old_fixture(row, desired_old):
    if desired_old == "pp":
        return
    if desired_old == "vys":
        row.update(
            {
                "pp_result": "Jiang Xi",
                "vys_result": "Xi Jiang",
                "pp_selected_surname_position": "first",
                "pp_selected_surname_frequency": 1.0,
                "vys_selected_surname_position": "last",
                "vys_selected_surname_frequency": 10.0,
                "vys_batch_vote_margin": 0.80,
            },
        )
        return
    if desired_old == "not_person":
        row.update({"pp_success": False, "pp_result": ""})
        return
    row["pp_success"] = desired_old


def _apply_pp_vys_pp_fixture(row, reason):
    if reason == "endpoint_frequency_strongly_favors_pp":
        row["pp_selected_surname_frequency_ratio"] = 20.0
        row["pp_batch_threshold_met"] = False
        return
    if reason == "strong_pp_paper_context":
        row.update(
            {
                "pp_selected_surname_frequency_ratio": 1.0,
                "pp_batch_dominant_format": row["pp_selected_format"],
                "pp_batch_threshold_met": True,
                "pp_batch_total_count": 3,
                "pp_batch_confidence": 0.70,
                "vys_batch_threshold_met": False,
            },
        )
        return
    row["pp_selected_surname_frequency_ratio"] = reason


def _apply_pp_vys_vys_fixture(row, reason):
    if reason == "endpoint_frequency_strongly_favors_vys":
        row["pp_selected_surname_frequency_ratio"] = 0.05
        row["pp_batch_threshold_met"] = False
        return
    if reason == "strong_vys_batch_context":
        row.update(
            {
                "pp_selected_surname_frequency_ratio": 1.0,
                "pp_batch_threshold_met": False,
                "vys_batch_dominant_format": row["vys_selected_format"],
                "vys_batch_threshold_met": True,
                "vys_batch_confidence": 0.85,
            },
        )
        return
    row["vys_batch_threshold_met"] = reason


def _apply_pp_vys_new_fixture(row, desired_new, desired_reason):
    if desired_new == "pp":
        _apply_pp_vys_pp_fixture(row, desired_reason or "strong_pp_paper_context")
        return
    if desired_new == "vys":
        _apply_pp_vys_vys_fixture(row, desired_reason or "strong_vys_batch_context")
        return
    if desired_new == "abstain":
        row.update(
            {
                "pp_selected_surname_frequency_ratio": 1.0,
                "pp_batch_threshold_met": False,
                "vys_batch_threshold_met": False,
            },
        )
        return
    if desired_new == "not_person":
        row.update({"pp_success": False, "pp_result": ""})
        return
    row["pp_selected_surname_frequency_ratio"] = desired_new


def _align_pp_vys_context_overrides(row, desired_new, desired_reason, override_keys):
    pp_strong = desired_new == "pp" and (desired_reason or "strong_pp_paper_context") == "strong_pp_paper_context"
    if pp_strong and "pp_batch_dominant_format" not in override_keys:
        row["pp_batch_dominant_format"] = row["pp_selected_format"]

    vys_strong = desired_new == "vys" and (desired_reason or "strong_vys_batch_context") == "strong_vys_batch_context"
    if vys_strong and "vys_batch_dominant_format" not in override_keys:
        row["vys_batch_dominant_format"] = row["vys_selected_format"]


def _pp_vys_row(**overrides):
    desired_old = overrides.pop("old_prediction", "pp")
    desired_new = overrides.pop("new_prediction", "pp")
    desired_reason = overrides.pop("new_reason", None)
    override_keys = set(overrides)
    row = _base_pp_vys_row()

    _apply_pp_vys_old_fixture(row, desired_old)
    _apply_pp_vys_new_fixture(row, desired_new, desired_reason)
    row.update(overrides)
    _align_pp_vys_context_overrides(row, desired_new, desired_reason, override_keys)
    return row


def _pp_abstain_row(**overrides):
    row = {
        "pp_result_token_count": 2,
        "selected_format": "surname_first",
        "batch_total_count": 3,
        "selected_surname_frequency": 10_000,
        "has_cjk": False,
        "has_latin": True,
        "cjk_has_space": False,
        "raw_tokens": 2,
    }
    row.update(overrides)
    return row


def _routing_result(text: str, *, success: bool = True):
    return SimpleNamespace(success=success, result=text)


def _routing_pattern(**overrides):
    pattern = {
        "dominant_format": "surname_first",
        "confidence": 1.0,
        "decision_confidence": 1.0,
        "threshold_met": False,
        "total_count": 1,
        "vote_margin": 0.0,
    }
    pattern.update(overrides)
    return SimpleNamespace(**pattern)


def _routing_evidence(**overrides):
    evidence = {
        "raw_name": "Wang X",
        "selected_format": "surname_first",
        "selected_surname_position": "first",
        "selected_surname_frequency": 1.0,
        "selected_over_alternate_surname_frequency_ratio": 1.0,
    }
    evidence.update(overrides)
    return SimpleNamespace(**evidence)


def _routing_batch(names, results, evidence, pattern):
    return SimpleNamespace(names=names, results=results, name_order_evidence=evidence, format_pattern=pattern)


def test_name_order_routing_script_is_in_sdist_include():
    pyproject_text = Path("pyproject.toml").read_text(encoding="utf-8")

    assert '"scripts/name_order_routing_rules.py"' in pyproject_text


def test_name_order_routing_script_preserves_empty_csv_schema(tmp_path, monkeypatch):
    input_path = tmp_path / "empty.csv"
    output_path = tmp_path / "routed.csv"
    fieldnames = list(_pp_abstain_row())
    input_path.write_text(",".join(fieldnames) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/name_order_routing_rules.py",
            "pp-abstain",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
    )
    runpy.run_path("scripts/name_order_routing_rules.py", run_name="__main__")

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == [*fieldnames, "router_prediction", "router_reason"]
        assert list(reader) == []


def test_name_order_routing_script_preserves_empty_pp_vys_csv_schema(tmp_path, monkeypatch):
    input_path = tmp_path / "empty.csv"
    output_path = tmp_path / "routed.csv"
    fieldnames = list(_pp_vys_row())
    input_path.write_text(",".join(fieldnames) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/name_order_routing_rules.py",
            "pp-vys-abstain",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
    )
    runpy.run_path("scripts/name_order_routing_rules.py", run_name="__main__")

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == [
            *fieldnames,
            "old_prediction",
            "old_reason",
            "new_prediction",
            "new_reason",
            "input_order_candidate",
            "router_prediction",
            "router_reason",
        ]
        assert list(reader) == []


def test_name_order_routing_script_writes_numpy_scalars_to_jsonl(tmp_path):
    output_path = tmp_path / "rows.jsonl"
    script_module = runpy.run_path("scripts/name_order_routing_rules.py")

    script_module["_write_rows"](
        [
            {
                "count": np.int64(JSONL_SCALAR_COUNT),
                "flag": np.bool_(True),
                "score": np.float32(JSONL_SCALAR_SCORE),
            },
        ],
        output_path,
        ["count", "flag", "score"],
    )

    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert row["count"] == JSONL_SCALAR_COUNT
    assert row["flag"] is True
    assert row["score"] == pytest.approx(JSONL_SCALAR_SCORE)


def test_name_order_routing_script_rejects_empty_csv_with_missing_schema(tmp_path, monkeypatch):
    input_path = tmp_path / "empty.csv"
    output_path = tmp_path / "routed.csv"
    input_path.write_text("name\n", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/name_order_routing_rules.py",
            "pp-abstain",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
    )

    with pytest.raises(ValueError, match="missing required columns"):
        runpy.run_path("scripts/name_order_routing_rules.py", run_name="__main__")


def test_name_order_routing_script_rejects_headerless_empty_csv(tmp_path, monkeypatch):
    input_path = tmp_path / "empty.csv"
    output_path = tmp_path / "routed.csv"
    input_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/name_order_routing_rules.py",
            "pp-abstain",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
    )

    with pytest.raises(ValueError, match="missing required columns"):
        runpy.run_path("scripts/name_order_routing_rules.py", run_name="__main__")


def test_pp_abstain_builder_converts_batch_evidence_to_router_rows(detector):
    batch = detector.analyze_name_batch(["Wang An", "Yan Li", "Wu Gang", "Li Bao"])

    with pytest.raises(ValueError, match="missing required columns"):
        route_pp_abstain_rows([vars(batch.name_order_evidence[0])])

    rows = build_pp_abstain_rows(batch, detector)
    routed = route_pp_abstain_rows(rows)

    assert len(rows) == len(batch.names)
    assert all(set(PP_ABSTAIN_REQUIRED_COLUMNS) <= row.keys() for row in rows)
    assert all("router_prediction" in row for row in routed)


def test_pp_abstain_builder_counts_result_text_tokens_for_hyphenated_given_names(detector):
    batch = detector.analyze_name_batch(["Zhang Chang-Qing"])

    rows = build_pp_abstain_rows(batch, detector)

    assert batch.results[0].result == "Chang-Qing Zhang"
    assert rows[0]["pp_result_token_count"] == PP_ABSTAIN_TWO_TOKEN_RESULT_COUNT


def test_pp_abstain_builder_derives_cjk_context_fields(detector):
    batch = detector.analyze_name_batch(["\u9ec4 \u5609\u5e73"])

    row = build_pp_abstain_rows(batch, detector)[0]

    assert row["has_cjk"] is True
    assert row["has_latin"] is False
    assert row["cjk_has_space"] is True


def test_pp_vys_builder_converts_aligned_batch_results_to_router_rows(detector):
    names = ["Wang An", "Yan Li", "Wu Gang", "Li Bao"]
    pp_batch = detector.analyze_name_batch(names)
    vys_batch = detector.analyze_name_batch(names)

    rows = build_pp_vys_abstain_rows(pp_batch, vys_batch)
    routed = route_pp_vys_abstain_rows(rows)
    routed_direct = route_pp_vys_abstain_batches(pp_batch, vys_batch)

    assert len(rows) == len(names)
    assert all(set(PP_VYS_ABSTAIN_REQUIRED_COLUMNS) <= row.keys() for row in rows)
    assert all("old_reason" not in row for row in rows)
    assert all("router_prediction" in row for row in routed)
    assert all("old_reason" in row for row in routed)
    assert routed_direct == routed


def test_pp_vys_builder_rejects_unaligned_batch_results(detector):
    pp_batch = detector.analyze_name_batch(["Wang An", "Yan Li"])
    vys_batch = detector.analyze_name_batch(["Wang An", "Wu Gang"])

    with pytest.raises(ValueError, match="aligned names"):
        build_pp_vys_abstain_rows(pp_batch, vys_batch)


def test_pp_vys_builder_derives_internal_old_new_policy_fields():
    pp_batch = _routing_batch(
        ["Xi Jiang"],
        [_routing_result("Jiang Xi")],
        [
            _routing_evidence(
                raw_name="Xi Jiang",
                selected_surname_position="first",
                selected_surname_frequency=10.0,
                selected_over_alternate_surname_frequency_ratio=0.05,
            ),
        ],
        _routing_pattern(total_count=1),
    )
    vys_batch = _routing_batch(
        ["Xi Jiang"],
        [_routing_result("Xi Jiang")],
        [
            _routing_evidence(
                raw_name="Xi Jiang",
                selected_format="given_first",
                selected_surname_position="last",
                selected_surname_frequency=100.0,
            ),
        ],
        _routing_pattern(dominant_format="given_first", threshold_met=True, total_count=10, vote_margin=0.8),
    )

    rows = build_pp_vys_abstain_rows(pp_batch, vys_batch)
    routed = route_pp_vys_abstain_rows(rows)

    assert "old_prediction" not in rows[0]
    assert "new_prediction" not in rows[0]
    assert routed[0]["old_prediction"] == "vys"
    assert routed[0]["old_reason"] == "vys_backoff_rule"
    assert routed[0]["new_prediction"] == "vys"
    assert routed[0]["new_reason"] == "endpoint_frequency_strongly_favors_vys"


def test_pp_vys_abstain_rule_keeps_existing_vys_agreements():
    rows = [
        _pp_vys_row(old_prediction="vys", new_prediction="vys"),
        _pp_vys_row(old_prediction="pp", new_prediction="vys"),
        _pp_vys_row(old_prediction="vys", new_prediction="abstain"),
    ]

    routed = route_pp_vys_abstain_rows(rows)

    assert [row["router_prediction"] for row in routed] == ["vys", "vys", "vys"]
    assert [row["router_reason"] for row in routed] == [
        "old_new_vys",
        "old_pp_new_vys",
        "old_vys_new_abstain",
    ]


def test_pp_vys_abstain_rule_allows_strong_pp_override_before_abstain_guard():
    routed = route_pp_vys_abstain_rows(
        [
            _pp_vys_row(
                old_prediction="vys",
                new_prediction="pp",
                new_reason="endpoint_frequency_strongly_favors_pp",
                pp_batch_total_count=3,
            ),
            _pp_vys_row(
                old_prediction="vys",
                new_prediction="pp",
                new_reason="strong_pp_paper_context",
                pp_selected_surname_frequency_ratio=12,
            ),
        ],
    )

    assert [row["router_prediction"] for row in routed] == ["pp", "pp"]
    assert [row["router_reason"] for row in routed] == [
        "endpoint_pp_with_real_paper_votes",
        "strong_pp_allowed_ratio",
    ]


def test_pp_vys_abstain_rule_routes_reliable_input_order_to_abstain():
    routed = route_pp_vys_abstain_rows(
        [
            _pp_vys_row(
                old_prediction="pp",
                new_prediction="pp",
                new_reason="endpoint_frequency_strongly_favors_pp",
                pp_selected_format="given_first",
                vys_selected_format="surname_first",
            ),
            _pp_vys_row(
                old_prediction="vys",
                new_prediction="vys",
                new_reason="endpoint_frequency_strongly_favors_vys",
                pp_selected_format="surname_first",
                vys_selected_format="given_first",
                pp_batch_total_count=2,
            ),
        ],
    )

    assert [row["input_order_candidate"] for row in routed] == ["pp", "vys"]
    assert [row["router_prediction"] for row in routed] == ["abstain", "abstain"]
    assert [row["router_reason"] for row in routed] == [
        "reliable_input_order_abstain",
        "reliable_input_order_abstain",
    ]


def test_pp_abstain_rule_accepts_balanced_high_confidence_pp_slices():
    routed = route_pp_abstain_rows(
        [
            _pp_abstain_row(),
            _pp_abstain_row(
                selected_format="given_first",
                has_cjk=True,
                has_latin=True,
                raw_tokens=2,
                selected_surname_frequency=2500,
            ),
        ],
    )

    assert [row["router_prediction"] for row in routed] == ["pp", "pp"]
    assert [row["router_reason"] for row in routed] == [
        "surname_first_two_token",
        "clean_bilingual_given_first",
    ]


def test_pp_abstain_rule_accepts_numpy_bool_scalars():
    routed = route_pp_abstain_rows(
        [
            _pp_abstain_row(
                has_cjk=np.bool_(True),
                has_latin=np.bool_(True),
                cjk_has_space=np.bool_(False),
                selected_format="given_first",
                raw_tokens=2,
                selected_surname_frequency=2500,
            ),
        ],
    )

    assert routed[0]["router_prediction"] == "pp"
    assert routed[0]["router_reason"] == "clean_bilingual_given_first"


def test_pp_abstain_rule_defaults_to_input_order_for_weak_or_ambiguous_zero_batch():
    routed = route_pp_abstain_rows(
        [
            _pp_abstain_row(selected_format="given_first", batch_total_count=0, selected_surname_frequency=499),
            _pp_abstain_row(batch_total_count=0, has_cjk=True, has_latin=True, raw_tokens=3),
        ],
    )

    assert [row["router_prediction"] for row in routed] == ["abstain", "abstain"]
    assert [row["router_reason"] for row in routed] == [
        "weak_zero_batch",
        "zero_batch_mixed_long",
    ]


def test_pp_abstain_rule_accepts_surname_first_despite_weak_zero_batch():
    routed = route_pp_abstain_rows([_pp_abstain_row(batch_total_count=0, selected_surname_frequency=499)])

    assert routed[0]["router_prediction"] == "pp"
    assert routed[0]["router_reason"] == "surname_first_two_token"


def test_pp_abstain_rule_applies_spaced_cjk_guard():
    routed = route_pp_abstain_rows(
        [
            _pp_abstain_row(
                batch_total_count=0,
                selected_surname_frequency=10_000,
                has_cjk=True,
                has_latin=False,
                cjk_has_space=True,
            ),
        ],
    )

    assert routed[0]["router_prediction"] == "abstain"
    assert routed[0]["router_reason"] == "spaced_cjk_zero_batch_surname_first"


def test_pp_abstain_rule_rejects_malformed_feature_values():
    with pytest.raises(ValueError, match="selected_surname_frequency"):
        route_pp_abstain_rows([_pp_abstain_row(batch_total_count=0, selected_surname_frequency="not-a-number")])

    with pytest.raises(ValueError, match="has_cjk"):
        route_pp_abstain_rows([_pp_abstain_row(has_cjk="definitely")])

    with pytest.raises(ValueError, match="selected_format"):
        route_pp_abstain_rows([_pp_abstain_row(selected_format="sideways")])


def test_pp_vys_abstain_rule_rejects_unknown_enums():
    with pytest.raises(ValueError, match="vys_batch_dominant_format"):
        route_pp_vys_abstain_rows(
            [
                _pp_vys_row(
                    new_prediction="vys",
                    new_reason="strong_vys_batch_context",
                    vys_batch_dominant_format="sideways",
                ),
            ],
        )

    with pytest.raises(ValueError, match="pp_selected_format"):
        route_pp_vys_abstain_rows([_pp_vys_row(pp_selected_format="mixed")])

    with pytest.raises(ValueError, match="pp_selected_surname_frequency_ratio"):
        route_pp_vys_abstain_rows([_pp_vys_row(pp_selected_surname_frequency_ratio="not-a-number")])


def test_pp_vys_abstain_rule_keeps_not_person_terminal():
    routed = route_pp_vys_abstain_rows(
        [
            _pp_vys_row(
                new_prediction="not_person",
                pp_selected_format="mixed",
                vys_selected_format="mixed",
                pp_batch_total_count="not-a-number",
                pp_batch_confidence="not-a-number",
                pp_selected_surname_frequency_ratio="not-a-number",
            ),
        ],
    )

    assert routed[0]["input_order_candidate"] == "unknown"
    assert routed[0]["router_prediction"] == "not_person"
    assert routed[0]["router_reason"] == "not_person"


def test_pp_vys_abstain_rule_derives_policy_fields_from_evidence_not_input_audit_columns():
    row = _pp_vys_row(old_prediction="vys", new_prediction="vys")
    row["old_prediction"] = "pp"
    row["old_reason"] = "stale"
    row["new_prediction"] = "pp"
    row["new_reason"] = "strong_pp_paper_context"

    routed = route_pp_vys_abstain_rows([row])

    assert routed[0]["old_prediction"] == "vys"
    assert routed[0]["old_reason"] == "vys_backoff_rule"
    assert routed[0]["new_prediction"] == "vys"
    assert routed[0]["new_reason"] == "strong_vys_batch_context"


def test_pp_vys_abstain_rule_allows_empty_ratio_sentinel():
    routed = route_pp_vys_abstain_rows(
        [
            _pp_vys_row(
                old_prediction="vys",
                new_prediction="vys",
                new_reason="strong_vys_batch_context",
                pp_selected_format="surname_first",
                vys_selected_format="given_first",
                pp_batch_total_count=2,
                pp_selected_surname_frequency_ratio="",
            ),
        ],
    )

    assert routed[0]["router_prediction"] == "vys"
    assert routed[0]["router_reason"] == "old_new_vys"


def test_pp_vys_builder_preserves_missing_ratio_evidence():
    pp_batch = _routing_batch(
        ["Wang X"],
        [_routing_result("X Wang")],
        [
            _routing_evidence(
                selected_over_alternate_surname_frequency_ratio=None,
            ),
        ],
        _routing_pattern(),
    )
    vys_batch = _routing_batch(
        ["Wang X"],
        [_routing_result("Wang X")],
        [
            _routing_evidence(
                selected_format="given_first",
                selected_surname_position="last",
                selected_surname_frequency=1.0,
            ),
        ],
        _routing_pattern(dominant_format="given_first"),
    )

    rows = build_pp_vys_abstain_rows(pp_batch, vys_batch)
    routed = route_pp_vys_abstain_rows(rows)

    assert rows[0]["pp_selected_surname_frequency_ratio"] is None
    assert "new_prediction" not in rows[0]
    assert routed[0]["new_prediction"] == "abstain"
    assert routed[0]["new_reason"] == "weak_or_conflicting_evidence"
    assert routed[0]["router_prediction"] == "pp"
    assert routed[0]["router_reason"] == "default_pp"


def test_pp_vys_builder_accepts_mixed_format_for_not_person(detector):
    names = ["John Smith"]
    pp_batch = detector.analyze_name_batch(names)
    vys_batch = detector.analyze_name_batch(names)

    rows = build_pp_vys_abstain_rows(pp_batch, vys_batch)
    routed = route_pp_vys_abstain_rows(rows)

    assert rows[0]["pp_selected_format"] == "mixed"
    assert rows[0]["vys_selected_format"] == "mixed"
    assert "old_prediction" not in rows[0]
    assert "new_prediction" not in rows[0]
    assert routed[0]["old_prediction"] == "not_person"
    assert routed[0]["new_prediction"] == "not_person"
    assert routed[0]["router_prediction"] == "not_person"


def test_pp_vys_abstain_rule_applies_effective_plus_pp_overrides():
    routed = route_pp_vys_abstain_rows(
        [
            _pp_vys_row(
                old_prediction="vys",
                new_prediction="pp",
                new_reason="endpoint_frequency_strongly_favors_pp",
                pp_batch_total_count=2,
                pp_batch_confidence=0.85,
            ),
            _pp_vys_row(
                old_prediction="pp",
                new_prediction="vys",
                new_reason="strong_vys_batch_context",
                pp_batch_total_count=6,
            ),
        ],
    )

    assert [row["router_prediction"] for row in routed] == ["pp", "pp"]
    assert [row["router_reason"] for row in routed] == [
        "endpoint_pp_high_conf_two_vote",
        "larger_pp_paper_overrides_vys_batch",
    ]


def test_pp_vys_abstain_rule_applies_promoted_name_priors():
    routed = route_pp_vys_abstain_rows(
        [
            _pp_vys_row(
                name="Chong Pei Pei",
                old_prediction="vys",
                new_prediction="vys",
                new_reason="strong_vys_batch_context",
                pp_selected_format="surname_first",
                vys_selected_format="given_first",
            ),
            _pp_vys_row(
                name="Woo Jin Chung",
                pp_selected_format="surname_first",
                vys_selected_format="given_first",
            ),
            _pp_vys_row(
                name="Ching Yee Yong",
                pp_selected_format="surname_first",
                vys_selected_format="given_first",
            ),
        ],
    )

    assert [row["router_prediction"] for row in routed] == ["pp", "vys", "vys"]
    assert [row["router_reason"] for row in routed] == [
        "name_prior_repeated_tail_given_surname_first",
        "name_prior_korean_given_first_three_token",
        "name_prior_cantonese_given_first",
    ]


def test_pp_vys_abstain_repeated_tail_name_prior_beats_broader_given_first_priors():
    routed = route_pp_vys_abstain_rows(
        [
            _pp_vys_row(
                name="Su Yun Yun",
                pp_selected_format="surname_first",
                vys_selected_format="given_first",
            ),
            _pp_vys_row(
                name="Han Yun Yun",
                pp_selected_format="surname_first",
                vys_selected_format="given_first",
            ),
        ],
    )

    assert [row["router_prediction"] for row in routed] == ["pp", "pp"]
    assert [row["router_reason"] for row in routed] == [
        "name_prior_repeated_tail_given_surname_first",
        "name_prior_repeated_tail_given_surname_first",
    ]


def test_pp_vys_abstain_rule_does_not_apply_rejected_broad_name_priors():
    routed = route_pp_vys_abstain_rows(
        [
            _pp_vys_row(
                name="Sima Hay",
                old_prediction="vys",
                new_prediction="vys",
                pp_selected_format="surname_first",
                vys_selected_format="given_first",
                pp_batch_total_count=6,
            ),
            _pp_vys_row(
                name="Li Wei",
                old_prediction="vys",
                new_prediction="vys",
                pp_selected_format="surname_first",
                vys_selected_format="given_first",
                pp_batch_total_count=6,
            ),
        ],
    )

    assert [row["router_prediction"] for row in routed] == ["vys", "vys"]
    assert [row["router_reason"] for row in routed] == ["old_new_vys", "old_new_vys"]


def _parse_result(surname, given_tokens, order, middle_tokens=()):
    parsed_original = ParsedName(
        surname=surname,
        given_name="-".join(given_tokens),
        surname_tokens=[surname],
        given_tokens=list(given_tokens),
        middle_name=" ".join(middle_tokens),
        middle_tokens=list(middle_tokens),
        order=order,
    )
    return ParseResult.success_with_name(
        formatted_name=f"{'-'.join(given_tokens)} {surname}",
        parsed=parsed_original,
        parsed_original_order=parsed_original,
    )


def test_input_order_parsed_flips_surname_first_reading():
    result = _parse_result("Wang", ["Wei"], ["surname", "given"])
    as_typed = input_order_parsed(result)
    assert (as_typed.given_name, as_typed.surname) == ("Wang", "Wei")
    assert as_typed.order == ["given", "middle", "surname"]


def test_input_order_parsed_keeps_surname_last_reading():
    result = _parse_result("Zhang", ["Wei"], ["given", "surname"], middle_tokens=["Y", "Z"])
    assert input_order_parsed(result) is result.parsed


def test_input_order_parsed_hyphenates_multi_token_given():
    result = _parse_result("Huang", ["Yu", "Qiang"], ["surname", "given"])
    as_typed = input_order_parsed(result)
    assert (as_typed.given_name, as_typed.surname) == ("Huang-Yu", "Qiang")
    assert as_typed.given_tokens == ["Huang", "Yu"]


def test_input_order_parsed_rejects_failed_and_single_token_parses():
    failed = ParseResult(success=False, result="", error_message="no parse")
    assert input_order_parsed(failed) is None
    single = _parse_result("Wang", [], ["surname", "given"])
    assert input_order_parsed(single) is None
