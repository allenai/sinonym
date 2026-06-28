import csv
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from sinonym.pipeline.name_order_routing import (
    PP_ABSTAIN_REQUIRED_COLUMNS,
    PP_VYS_ABSTAIN_REQUIRED_COLUMNS,
    PPVysRoutingDecision,
    build_pp_abstain_rows,
    build_pp_vys_abstain_rows,
    route_pp_abstain_rows,
    route_pp_vys_abstain_rows,
)


def _pp_vys_row(**overrides):
    row = {
        "name": "Xi Jiang",
        "old_prediction": "pp",
        "new_prediction": "pp",
        "new_reason": "weak_or_conflicting_evidence",
        "pp_selected_format": "surname_first",
        "vys_selected_format": "surname_first",
        "pp_batch_total_count": 0,
        "pp_batch_confidence": 0,
        "pp_selected_surname_frequency_ratio": 0,
    }
    row.update(overrides)
    return row


def _pp_abstain_row(**overrides):
    row = {
        "pp_result_token_count": 2,
        "selected_format": "surname_first",
        "batch_total_count": 3,
        "selected_surname_frequency": 10_000,
        "selected_over_alternate_ratio": 100,
        "has_cjk": False,
        "has_latin": True,
        "cjk_has_space": False,
        "compact_cjk": "",
        "jp_probability": 0,
        "raw_tokens": 2,
    }
    row.update(overrides)
    return row


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


def test_pp_abstain_builder_converts_batch_evidence_to_router_rows(detector):
    batch = detector.analyze_name_batch(["Wang An", "Yan Li", "Wu Gang", "Li Bao"])

    with pytest.raises(ValueError, match="missing required columns"):
        route_pp_abstain_rows([vars(batch.name_order_evidence[0])])

    rows = build_pp_abstain_rows(batch, detector)
    routed = route_pp_abstain_rows(rows)

    assert len(rows) == len(batch.names)
    assert all(set(PP_ABSTAIN_REQUIRED_COLUMNS) <= row.keys() for row in rows)
    assert all("router_prediction" in row for row in routed)


def test_pp_abstain_builder_derives_cjk_context_fields(detector):
    batch = detector.analyze_name_batch(["\u9ec4 \u5609\u5e73"])

    row = build_pp_abstain_rows(batch, detector)[0]

    assert row["has_cjk"] is True
    assert row["has_latin"] is False
    assert row["cjk_has_space"] is True
    assert row["compact_cjk"] == "\u9ec4\u5609\u5e73"
    assert isinstance(row["jp_probability"], float)


def test_pp_vys_builder_converts_aligned_batch_results_to_router_rows(detector):
    names = ["Wang An", "Yan Li", "Wu Gang", "Li Bao"]
    pp_batch = detector.analyze_name_batch(names)
    vys_batch = detector.analyze_name_batch(names)
    decisions = [
        PPVysRoutingDecision(
            old_prediction="pp",
            new_prediction="pp",
            new_reason="weak_or_conflicting_evidence",
        )
        for _name in names
    ]

    rows = build_pp_vys_abstain_rows(pp_batch, vys_batch, decisions)
    routed = route_pp_vys_abstain_rows(rows)

    assert len(rows) == len(names)
    assert all(set(PP_VYS_ABSTAIN_REQUIRED_COLUMNS) <= row.keys() for row in rows)
    assert all("router_prediction" in row for row in routed)


def test_pp_vys_builder_rejects_unaligned_batch_results(detector):
    pp_batch = detector.analyze_name_batch(["Wang An", "Yan Li"])
    vys_batch = detector.analyze_name_batch(["Wang An", "Wu Gang"])
    decisions = [
        PPVysRoutingDecision("pp", "pp", "weak_or_conflicting_evidence"),
        PPVysRoutingDecision("pp", "pp", "weak_or_conflicting_evidence"),
    ]

    with pytest.raises(ValueError, match="aligned names"):
        build_pp_vys_abstain_rows(pp_batch, vys_batch, decisions)


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
            _pp_abstain_row(batch_total_count=0, selected_surname_frequency=499),
            _pp_abstain_row(batch_total_count=0, has_cjk=True, has_latin=True, raw_tokens=3),
            _pp_abstain_row(
                batch_total_count=0,
                has_cjk=False,
                has_latin=True,
                selected_over_alternate_ratio=10,
            ),
        ],
    )

    assert [row["router_prediction"] for row in routed] == ["abstain", "abstain", "abstain"]
    assert [row["router_reason"] for row in routed] == [
        "weak_zero_batch",
        "zero_batch_mixed_long",
        "zero_batch_latin_ambiguous_endpoint",
    ]


def test_pp_abstain_rule_applies_spaced_cjk_and_jp_likelihood_guards():
    routed = route_pp_abstain_rows(
        [
            _pp_abstain_row(
                batch_total_count=0,
                selected_surname_frequency=10_000,
                has_cjk=True,
                has_latin=False,
                cjk_has_space=True,
            ),
            _pp_abstain_row(
                compact_cjk="原田泰夫",
                jp_probability=0.60,
            ),
        ],
    )

    assert [row["router_prediction"] for row in routed] == ["abstain", "abstain"]
    assert [row["router_reason"] for row in routed] == [
        "spaced_cjk_zero_batch_surname_first",
        "jp_likelihood_060",
    ]


def test_pp_abstain_rule_rejects_malformed_feature_values():
    with pytest.raises(ValueError, match="selected_surname_frequency"):
        route_pp_abstain_rows([_pp_abstain_row(batch_total_count=0, selected_surname_frequency="not-a-number")])

    with pytest.raises(ValueError, match="has_cjk"):
        route_pp_abstain_rows([_pp_abstain_row(has_cjk="definitely")])

    with pytest.raises(ValueError, match="selected_format"):
        route_pp_abstain_rows([_pp_abstain_row(selected_format="sideways")])


def test_pp_vys_abstain_rule_rejects_unknown_enums():
    with pytest.raises(ValueError, match="new_prediction"):
        route_pp_vys_abstain_rows([_pp_vys_row(new_prediction="maybe")])

    with pytest.raises(ValueError, match="pp_selected_format"):
        route_pp_vys_abstain_rows([_pp_vys_row(pp_selected_format="mixed")])

    with pytest.raises(ValueError, match="pp_selected_surname_frequency_ratio"):
        route_pp_vys_abstain_rows([_pp_vys_row(pp_selected_surname_frequency_ratio="not-a-number")])


def test_pp_vys_abstain_rule_rejects_contradictory_new_prediction_reason():
    with pytest.raises(ValueError, match="new_reason"):
        route_pp_vys_abstain_rows(
            [
                _pp_vys_row(
                    new_prediction="pp",
                    new_reason="endpoint_frequency_strongly_favors_vys",
                    pp_selected_format="surname_first",
                    vys_selected_format="given_first",
                ),
            ],
        )


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


def test_pp_vys_abstain_rule_applies_effective_plus_vys_override():
    routed = route_pp_vys_abstain_rows(
        [
            _pp_vys_row(
                old_prediction="pp",
                new_prediction="abstain",
                pp_selected_format="surname_first",
                vys_selected_format="given_first",
                pp_batch_total_count=2,
                pp_selected_surname_frequency_ratio=0.5,
            ),
        ],
    )

    assert routed[0]["router_prediction"] == "vys"
    assert routed[0]["router_reason"] == "pp_abstain_two_vote_small_ratio_vys"


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
            _pp_vys_row(
                old_prediction="vys",
                new_prediction="vys",
                new_reason="strong_vys_batch_context",
                pp_batch_total_count=7,
                pp_selected_surname_frequency_ratio=3,
            ),
            _pp_vys_row(
                old_prediction="vys",
                new_prediction="vys",
                new_reason="strong_vys_batch_context",
                pp_batch_total_count=3,
                pp_selected_surname_frequency_ratio=8,
            ),
            _pp_vys_row(
                old_prediction="vys",
                new_prediction="pp",
                new_reason="endpoint_frequency_strongly_favors_pp",
                pp_batch_total_count=1,
                pp_batch_confidence=0.5,
                pp_selected_surname_frequency_ratio=101,
            ),
        ],
    )

    assert [row["router_prediction"] for row in routed] == ["pp", "pp", "pp", "pp", "pp"]
    assert [row["router_reason"] for row in routed] == [
        "endpoint_pp_high_conf_two_vote",
        "larger_pp_paper_overrides_vys_batch",
        "strong_vys_large_pp_count_ratio_looks_pp",
        "strong_vys_three_vote_mid_ratio_pp",
        "endpoint_pp_low_count_low_conf_very_high_ratio",
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
                name="Ouyang Yu",
                old_prediction="vys",
                new_prediction="vys",
                new_reason="weak_or_conflicting_evidence",
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

    assert [row["router_prediction"] for row in routed] == ["pp", "pp", "vys", "vys"]
    assert [row["router_reason"] for row in routed] == [
        "name_prior_repeated_tail_given_surname_first",
        "name_prior_ouyang_surname_first",
        "name_prior_korean_given_first_three_token",
        "name_prior_cantonese_given_first",
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
