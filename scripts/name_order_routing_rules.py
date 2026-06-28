# ruff: noqa: INP001
"""Apply external name-order routing rules for PP/VYS/abstain experiments.

This script intentionally lives outside the `sinonym` package. The rules combine
two or more `sinonym.analyze_name_batch()` runs plus external context metadata;
they are not part of the single-name detector API.

Route semantics:
- `pp`: emit the PP parse.
- `vys`: emit the VYS parse.
- `abstain`: emit the preprocessed input-order parse.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

Route = Literal["pp", "vys", "abstain"]
Row = Mapping[str, object]
MutableRow = dict[str, object]
FORMAT_VALUES = ("surname_first", "given_first", "mixed")
ORDER_FORMAT_VALUES = ("surname_first", "given_first")
PREDICTION_VALUES = ("pp", "vys", "abstain", "not_person")
PP_VYS_REASON_VALUES = (
    "endpoint_frequency_strongly_favors_pp",
    "endpoint_frequency_strongly_favors_vys",
    "strong_pp_paper_context",
    "strong_vys_batch_context",
    "weak_or_conflicting_evidence",
)

PP_VYS_ABSTAIN_REQUIRED_COLUMNS = (
    "old_prediction",
    "new_prediction",
    "new_reason",
    "pp_selected_format",
    "vys_selected_format",
    "pp_batch_total_count",
    "pp_batch_confidence",
    "pp_selected_surname_frequency_ratio",
)

PP_ABSTAIN_REQUIRED_COLUMNS = (
    "pp_result_token_count",
    "selected_format",
    "batch_total_count",
    "selected_surname_frequency",
    "selected_over_alternate_ratio",
    "has_cjk",
    "has_latin",
    "cjk_has_space",
    "compact_cjk",
    "jp_probability",
    "raw_tokens",
)

PP_ABSTAIN_WEAK_SURNAME_FREQUENCY_MAX = 500.0
PP_ABSTAIN_LATIN_AMBIGUOUS_RATIO_MAX = 10.0
PP_ABSTAIN_CLEAN_BILINGUAL_SURNAME_FREQUENCY_MIN = 2500.0
PP_ABSTAIN_JP_PROBABILITY_MIN = 0.60
PP_ABSTAIN_TWO_TOKEN_RESULT_COUNT = 2
PP_ABSTAIN_MIXED_LONG_RAW_TOKEN_MIN = 3
PP_ABSTAIN_CLEAN_BILINGUAL_RAW_TOKEN_COUNT = 2
PP_VYS_REAL_PAPER_VOTES_MIN = 3
PP_VYS_LOW_BATCH_MAX = 2
PP_VYS_VERY_LOW_BATCH_MAX = 1
PP_VYS_LOW_RATIO_MAX = 5.0
PP_VYS_ALLOWED_RATIO_LOW_MAX = 1.0
PP_VYS_ALLOWED_RATIO_HIGH_MIN = 5.0
PP_VYS_ALLOWED_RATIO_HIGH_MAX = 20.0
PP_VYS_TINY_RATIO_MAX = 0.1
PP_VYS_MEDIUM_RATIO_MAX = 5.0
PP_VYS_HIGH_RATIO_MAX = 100.0
PP_VYS_HIGH_CONFIDENCE_MIN = 0.85
PP_VYS_LARGER_PAPER_VOTES_MIN = 6
PP_VYS_TOTAL_BUCKET_ZERO_ONE_MAX = 1
PP_VYS_TOTAL_BUCKET_TWO_MAX = 2
PP_VYS_TOTAL_BUCKET_THREE_MAX = 3
PP_VYS_TOTAL_BUCKET_FIVE_MAX = 5
PP_VYS_TOTAL_BUCKET_TEN_MAX = 10
PP_VYS_CONF_BUCKET_LOW_MAX = 0.50
PP_VYS_CONF_BUCKET_MID_MAX = 0.70


@dataclass(frozen=True)
class PPVysFeatures:
    """Parsed feature values for one PP/VYS/abstain routing row."""

    ratio: float
    pp_batch_total_count: float
    pp_batch_confidence: float
    old_prediction: str
    new_prediction: str
    new_reason: str
    input_order_candidate: str
    pp_total_bucket: str
    pp_conf_bucket: str
    pp_ratio_bucket: str
    old_new_vys: bool
    old_pp_new_vys: bool
    old_vys_new_abstain: bool
    old_vys_new_pp: bool


def route_pp_vys_abstain_rows(rows: Iterable[Row]) -> list[MutableRow]:
    """Route PP/VYS disagreements, defaulting to PP except known VYS/abstain slices.

    Required input columns are listed in `PP_VYS_ABSTAIN_REQUIRED_COLUMNS`.
    The returned rows add:
    - `input_order_candidate`
    - `router_prediction`
    - `router_reason`
    """
    return [_route_pp_vys_abstain_row(row) for row in rows]


def _route_pp_vys_abstain_row(row: Row) -> MutableRow:
    _require_columns(row, PP_VYS_ABSTAIN_REQUIRED_COLUMNS)
    features = _pp_vys_features(row)
    route, reason = _base_pp_vys_route(features)
    override = _effective_plus_override(features)
    if override is not None:
        route, reason = override

    routed = dict(row)
    routed["input_order_candidate"] = features.input_order_candidate
    routed["router_prediction"] = route
    routed["router_reason"] = reason
    return routed


def _pp_vys_features(row: Row) -> PPVysFeatures:
    ratio = _optional_float(row, "pp_selected_surname_frequency_ratio", default=math.nan)
    pp_batch_total_count = _required_float(row, "pp_batch_total_count")
    pp_batch_confidence = _required_float(row, "pp_batch_confidence")
    old_prediction = _required_string(row, "old_prediction", allowed=("pp", "vys"))
    new_prediction = _required_string(row, "new_prediction", allowed=PREDICTION_VALUES)
    new_reason = _required_string(row, "new_reason", allowed=PP_VYS_REASON_VALUES)
    old_new_vys = old_prediction == "vys" and new_prediction == "vys"
    old_pp_new_vys = old_prediction == "pp" and new_prediction == "vys"
    old_vys_new_abstain = old_prediction == "vys" and new_prediction in {"abstain", "not_person"}
    old_vys_new_pp = old_prediction == "vys" and new_prediction == "pp"

    return PPVysFeatures(
        ratio=ratio,
        pp_batch_total_count=pp_batch_total_count,
        pp_batch_confidence=pp_batch_confidence,
        old_prediction=old_prediction,
        new_prediction=new_prediction,
        new_reason=new_reason,
        input_order_candidate=_input_order_candidate(row),
        pp_total_bucket=_pp_total_bucket(pp_batch_total_count),
        pp_conf_bucket=_pp_conf_bucket(pp_batch_confidence),
        pp_ratio_bucket=_pp_ratio_bucket(ratio),
        old_new_vys=old_new_vys,
        old_pp_new_vys=old_pp_new_vys,
        old_vys_new_abstain=old_vys_new_abstain,
        old_vys_new_pp=old_vys_new_pp,
    )


def _base_pp_vys_route(features: PPVysFeatures) -> tuple[Route, str]:
    route: Route = "pp"
    reason = "default_pp"

    base_overrides: tuple[tuple[bool, Route, str], ...] = (
        (features.old_new_vys, "vys", "old_new_vys"),
        (features.old_pp_new_vys, "vys", "old_pp_new_vys"),
        (features.old_vys_new_abstain, "vys", "old_vys_new_abstain"),
        (features.old_vys_new_pp, "vys", "old_vys_new_pp_default_vys"),
        (_endpoint_pp_with_real_paper_votes(features), "pp", "endpoint_pp_with_real_paper_votes"),
        (_strong_pp_allowed_ratio(features), "pp", "strong_pp_allowed_ratio"),
        (_reliable_input_order_abstain(features), "abstain", "reliable_input_order_abstain"),
    )
    for condition, candidate_route, candidate_reason in base_overrides:
        if condition:
            route = candidate_route
            reason = candidate_reason
    return route, reason


def _effective_plus_override(features: PPVysFeatures) -> tuple[Route, str] | None:
    selected: tuple[Route, str] | None = None
    overrides: tuple[tuple[bool, Route, str], ...] = (
        (_pp_abstain_small_ratio_vys(features), "vys", "pp_abstain_two_vote_small_ratio_vys"),
        (_endpoint_pp_high_conf_two_vote(features), "pp", "endpoint_pp_high_conf_two_vote"),
        (_pp_context_over_vys_batch_for_larger_papers(features), "pp", "larger_pp_paper_overrides_vys_batch"),
        (_strong_vys_large_pp_looks_pp(features), "pp", "strong_vys_large_pp_count_ratio_looks_pp"),
        (_strong_vys_three_vote_mid_ratio_pp(features), "pp", "strong_vys_three_vote_mid_ratio_pp"),
        (
            _endpoint_pp_low_count_low_conf_very_high_ratio(features),
            "pp",
            "endpoint_pp_low_count_low_conf_very_high_ratio",
        ),
    )
    for condition, route, reason in overrides:
        if condition:
            selected = (route, reason)
    return selected


def _endpoint_pp_with_real_paper_votes(features: PPVysFeatures) -> bool:
    return (
        features.old_vys_new_pp
        and features.new_reason == "endpoint_frequency_strongly_favors_pp"
        and features.pp_batch_total_count >= PP_VYS_REAL_PAPER_VOTES_MIN
    )


def _strong_pp_allowed_ratio(features: PPVysFeatures) -> bool:
    return (
        features.old_vys_new_pp
        and features.new_reason == "strong_pp_paper_context"
        and (
            features.ratio <= PP_VYS_ALLOWED_RATIO_LOW_MAX
            or PP_VYS_ALLOWED_RATIO_HIGH_MIN < features.ratio <= PP_VYS_ALLOWED_RATIO_HIGH_MAX
        )
    )


def _reliable_input_order_abstain(features: PPVysFeatures) -> bool:
    input_pp_old_new_pp = (
        features.input_order_candidate == "pp" and features.old_prediction == "pp" and features.new_prediction == "pp"
    )
    endpoint_vys_small_pp_context = (
        features.input_order_candidate == "vys"
        and features.new_reason == "endpoint_frequency_strongly_favors_vys"
        and features.pp_batch_total_count <= PP_VYS_LOW_BATCH_MAX
    )
    weak_vys_small_pp_context = (
        features.input_order_candidate == "vys"
        and features.new_reason == "weak_or_conflicting_evidence"
        and features.pp_batch_total_count <= PP_VYS_VERY_LOW_BATCH_MAX
        and features.ratio <= PP_VYS_LOW_RATIO_MAX
    )
    strong_vys_small_pp_context = (
        features.input_order_candidate == "vys"
        and features.new_reason == "strong_vys_batch_context"
        and features.pp_batch_total_count <= PP_VYS_LOW_BATCH_MAX
        and features.ratio <= PP_VYS_LOW_RATIO_MAX
    )
    return input_pp_old_new_pp or endpoint_vys_small_pp_context or weak_vys_small_pp_context or strong_vys_small_pp_context


def _pp_abstain_small_ratio_vys(features: PPVysFeatures) -> bool:
    return (
        features.old_prediction == "pp"
        and features.new_prediction in {"abstain", "not_person"}
        and features.pp_batch_total_count == PP_VYS_LOW_BATCH_MAX
        and PP_VYS_TINY_RATIO_MAX < features.ratio <= PP_VYS_ALLOWED_RATIO_LOW_MAX
    )


def _endpoint_pp_high_conf_two_vote(features: PPVysFeatures) -> bool:
    return (
        features.old_vys_new_pp
        and features.new_reason == "endpoint_frequency_strongly_favors_pp"
        and features.pp_batch_total_count == PP_VYS_LOW_BATCH_MAX
        and features.pp_batch_confidence >= PP_VYS_HIGH_CONFIDENCE_MIN
    )


def _pp_context_over_vys_batch_for_larger_papers(features: PPVysFeatures) -> bool:
    return (
        features.old_pp_new_vys
        and features.new_reason == "strong_vys_batch_context"
        and features.pp_batch_total_count >= PP_VYS_LARGER_PAPER_VOTES_MIN
    )


def _strong_vys_large_pp_looks_pp(features: PPVysFeatures) -> bool:
    return (
        features.old_new_vys
        and features.new_reason == "strong_vys_batch_context"
        and features.pp_total_bucket == "6-10"
        and features.pp_ratio_bucket == "1-5"
    )


def _strong_vys_three_vote_mid_ratio_pp(features: PPVysFeatures) -> bool:
    return (
        features.old_new_vys
        and features.new_reason == "strong_vys_batch_context"
        and features.pp_batch_total_count == PP_VYS_REAL_PAPER_VOTES_MIN
        and PP_VYS_ALLOWED_RATIO_HIGH_MIN < features.ratio <= PP_VYS_ALLOWED_RATIO_HIGH_MAX
    )


def _endpoint_pp_low_count_low_conf_very_high_ratio(features: PPVysFeatures) -> bool:
    return (
        features.old_vys_new_pp
        and features.new_reason == "endpoint_frequency_strongly_favors_pp"
        and features.pp_total_bucket == "0-1"
        and features.pp_conf_bucket == "<=0.50"
        and features.pp_ratio_bucket == "100+"
    )


def route_pp_abstain_rows(rows: Iterable[Row]) -> list[MutableRow]:
    """Route no-VYS rows to PP or input-order abstain.

    Required input columns are listed in `PP_ABSTAIN_REQUIRED_COLUMNS`.
    The returned rows add:
    - `router_prediction`
    - `router_reason`
    """
    output: list[MutableRow] = []
    for row in rows:
        _require_columns(row, PP_ABSTAIN_REQUIRED_COLUMNS)
        routed = dict(row)
        predicates = _pp_abstain_predicates(row)

        route: Route = "abstain"
        reason = "default_abstain"
        if predicates["accept_surname_first"]:
            route = "pp"
            reason = "surname_first_two_token"
        if predicates["clean_bilingual_given_first"]:
            route = "pp"
            reason = "clean_bilingual_given_first"
        if route == "abstain":
            reason = _first_true_reason(
                predicates,
                (
                    "spaced_cjk_zero_batch_surname_first",
                    "jp_likelihood_060",
                    "weak_zero_batch",
                    "zero_batch_mixed_long",
                    "zero_batch_latin_ambiguous_endpoint",
                ),
                default="default_abstain",
            )

        routed["router_prediction"] = route
        routed["router_reason"] = reason
        output.append(routed)
    return output


def pp_vys_abstain_router(dataframe: Any) -> Any:
    """Pandas wrapper for `route_pp_vys_abstain_rows`.

    This keeps notebook/pipeline usage close to the original scratch rule while
    letting the tested implementation stay dependency-light.
    """
    _require_frame_columns(dataframe, PP_VYS_ABSTAIN_REQUIRED_COLUMNS)
    rows = route_pp_vys_abstain_rows(dataframe.to_dict("records"))
    return _with_routing_columns(dataframe, rows, ("input_order_candidate", "router_prediction", "router_reason"))


def pp_abstain_router(dataframe: Any) -> Any:
    """Pandas wrapper for `route_pp_abstain_rows`."""
    _require_frame_columns(dataframe, PP_ABSTAIN_REQUIRED_COLUMNS)
    rows = route_pp_abstain_rows(dataframe.to_dict("records"))
    return _with_routing_columns(dataframe, rows, ("router_prediction", "router_reason"))


def _input_order_candidate(row: Row) -> str:
    pp_preserves_input_order = (
        _required_string(
            row,
            "pp_selected_format",
            allowed=ORDER_FORMAT_VALUES,
        )
        == "given_first"
    )
    vys_preserves_input_order = (
        _required_string(
            row,
            "vys_selected_format",
            allowed=ORDER_FORMAT_VALUES,
        )
        == "given_first"
    )
    if pp_preserves_input_order and not vys_preserves_input_order:
        return "pp"
    if vys_preserves_input_order and not pp_preserves_input_order:
        return "vys"
    return "unknown"


def _pp_total_bucket(value: float) -> str:
    buckets = (
        (PP_VYS_TOTAL_BUCKET_ZERO_ONE_MAX, "0-1"),
        (PP_VYS_TOTAL_BUCKET_TWO_MAX, "2"),
        (PP_VYS_TOTAL_BUCKET_THREE_MAX, "3"),
        (PP_VYS_TOTAL_BUCKET_FIVE_MAX, "4-5"),
        (PP_VYS_TOTAL_BUCKET_TEN_MAX, "6-10"),
    )
    for upper_bound, label in buckets:
        if value <= upper_bound:
            return label
    return "11+"


def _pp_conf_bucket(value: float) -> str:
    buckets = (
        (PP_VYS_CONF_BUCKET_LOW_MAX, "<=0.50"),
        (PP_VYS_CONF_BUCKET_MID_MAX, "0.50-0.70"),
        (PP_VYS_HIGH_CONFIDENCE_MIN, "0.70-0.85"),
    )
    for upper_bound, label in buckets:
        if value <= upper_bound:
            return label
    return "0.85-1.00"


def _pp_ratio_bucket(value: float) -> str:
    if math.isnan(value):
        return "missing"
    buckets = (
        (PP_VYS_TINY_RATIO_MAX, "<=0.1"),
        (PP_VYS_ALLOWED_RATIO_LOW_MAX, "0.1-1"),
        (PP_VYS_MEDIUM_RATIO_MAX, "1-5"),
        (PP_VYS_ALLOWED_RATIO_HIGH_MAX, "5-20"),
        (PP_VYS_HIGH_RATIO_MAX, "20-100"),
    )
    for upper_bound, label in buckets:
        if value <= upper_bound:
            return label
    return "100+"


def _pp_abstain_predicates(row: Row) -> dict[str, bool]:
    pp_result_token_count = _required_float(row, "pp_result_token_count")
    batch_total_count = _required_float(row, "batch_total_count")
    selected_surname_frequency = _required_float(row, "selected_surname_frequency")
    selected_over_alternate_ratio = _required_float(row, "selected_over_alternate_ratio")
    raw_tokens = _required_float(row, "raw_tokens")
    jp_probability = _required_float(row, "jp_probability")
    has_cjk = _required_bool(row, "has_cjk")
    has_latin = _required_bool(row, "has_latin")
    cjk_has_space = _required_bool(row, "cjk_has_space")
    compact_cjk = _optional_string(row, "compact_cjk")
    selected_format = _required_string(row, "selected_format", allowed=FORMAT_VALUES)

    surname_first_two_token = pp_result_token_count == PP_ABSTAIN_TWO_TOKEN_RESULT_COUNT and selected_format == "surname_first"
    weak_zero_batch = batch_total_count == 0 and selected_surname_frequency < PP_ABSTAIN_WEAK_SURNAME_FREQUENCY_MAX
    zero_batch_mixed_long = batch_total_count == 0 and has_cjk and has_latin and raw_tokens >= PP_ABSTAIN_MIXED_LONG_RAW_TOKEN_MIN
    zero_batch_latin_ambiguous_endpoint = (
        batch_total_count == 0
        and not has_cjk
        and has_latin
        and 0 < selected_over_alternate_ratio <= PP_ABSTAIN_LATIN_AMBIGUOUS_RATIO_MAX
    )
    spaced_cjk_zero_batch_surname_first = (
        surname_first_two_token and batch_total_count == 0 and has_cjk and not has_latin and cjk_has_space
    )
    jp_likelihood_060 = surname_first_two_token and compact_cjk != "" and jp_probability >= PP_ABSTAIN_JP_PROBABILITY_MIN
    accept_surname_first = (
        surname_first_two_token
        and not weak_zero_batch
        and not zero_batch_mixed_long
        and not zero_batch_latin_ambiguous_endpoint
        and not spaced_cjk_zero_batch_surname_first
        and not jp_likelihood_060
    )
    clean_bilingual_given_first = (
        pp_result_token_count == PP_ABSTAIN_TWO_TOKEN_RESULT_COUNT
        and selected_format == "given_first"
        and has_cjk
        and has_latin
        and raw_tokens == PP_ABSTAIN_CLEAN_BILINGUAL_RAW_TOKEN_COUNT
        and selected_surname_frequency >= PP_ABSTAIN_CLEAN_BILINGUAL_SURNAME_FREQUENCY_MIN
    )
    return {
        "surname_first_two_token": surname_first_two_token,
        "weak_zero_batch": weak_zero_batch,
        "zero_batch_mixed_long": zero_batch_mixed_long,
        "zero_batch_latin_ambiguous_endpoint": zero_batch_latin_ambiguous_endpoint,
        "spaced_cjk_zero_batch_surname_first": spaced_cjk_zero_batch_surname_first,
        "jp_likelihood_060": jp_likelihood_060,
        "accept_surname_first": accept_surname_first,
        "clean_bilingual_given_first": clean_bilingual_given_first,
    }


def _first_true_reason(predicates: Mapping[str, bool], names: tuple[str, ...], *, default: str) -> str:
    for name in names:
        if predicates[name]:
            return name
    return default


def _require_columns(row: Row, required_columns: tuple[str, ...]) -> None:
    missing = [column for column in required_columns if column not in row]
    if missing:
        message = f"missing required columns: {', '.join(missing)}"
        raise ValueError(message)


def _require_frame_columns(dataframe: Any, required_columns: tuple[str, ...]) -> None:
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        message = f"missing required columns: {', '.join(missing)}"
        raise ValueError(message)


def _with_routing_columns(dataframe: Any, rows: list[MutableRow], columns: tuple[str, ...]) -> Any:
    output = dataframe.copy()
    for column in columns:
        output[column] = [row[column] for row in rows]
    return output


def _is_nan(value: object) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _required_string(row: Row, column: str, *, allowed: tuple[str, ...] | None = None) -> str:
    value = row[column]
    if value is None or _is_nan(value):
        message = f"{column} is required"
        raise ValueError(message)
    text = str(value)
    if text == "":
        message = f"{column} is required"
        raise ValueError(message)
    if allowed is not None and text not in allowed:
        message = f"{column} must be one of {', '.join(allowed)}"
        raise ValueError(message)
    return text


def _optional_string(row: Row, column: str) -> str:
    value = row[column]
    if value is None:
        return ""
    if _is_nan(value):
        return ""
    return str(value)


def _required_float(row: Row, column: str) -> float:
    value = row[column]
    if value is None or _is_nan(value):
        message = f"{column} must be a finite number"
        raise ValueError(message)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        message = f"{column} must be a finite number"
        raise ValueError(message) from None
    if not math.isfinite(parsed):
        message = f"{column} must be a finite number"
        raise ValueError(message)
    return parsed


def _optional_float(row: Row, column: str, *, default: float) -> float:
    value = row[column]
    if value is None or value == "" or _is_nan(value):
        return default
    return _required_float(row, column)


def _required_bool(row: Row, column: str) -> bool:
    value = row[column]
    if isinstance(value, bool):
        return value
    if value is None or _is_nan(value):
        message = f"{column} must be a boolean"
        raise ValueError(message)
    if isinstance(value, int | float) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"1", "true", "t", "yes", "y"}:
            return True
        if normalized in {"0", "false", "f", "no", "n"}:
            return False
    message = f"{column} must be a boolean"
    raise ValueError(message)


def _read_rows(path: Path) -> list[MutableRow]:
    suffix = path.suffix.casefold()
    if suffix == ".parquet":
        return _read_parquet_rows(path)
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8-sig") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(rows: list[MutableRow], path: Path) -> None:
    suffix = path.suffix.casefold()
    if suffix == ".parquet":
        _write_parquet_rows(rows, path)
    elif suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        fieldnames = _fieldnames(rows)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def _read_parquet_rows(path: Path) -> list[MutableRow]:
    try:
        import pandas as pd  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        message = "Parquet input requires pandas and a parquet engine in this environment."
        raise RuntimeError(message) from exc
    return pd.read_parquet(path).to_dict("records")


def _write_parquet_rows(rows: list[MutableRow], path: Path) -> None:
    try:
        import pandas as pd  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        message = "Parquet output requires pandas and a parquet engine in this environment."
        raise RuntimeError(message) from exc
    pd.DataFrame(rows).to_parquet(path, index=False)


def _fieldnames(rows: list[MutableRow]) -> list[str]:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "regime",
        choices=("pp-vys-abstain", "pp-abstain"),
        help="Routing regime to apply.",
    )
    parser.add_argument("--input", type=Path, required=True, help="Input CSV, JSONL, or Parquet file.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV, JSONL, or Parquet file.")
    args = parser.parse_args()

    rows = _read_rows(args.input)
    routed = route_pp_vys_abstain_rows(rows) if args.regime == "pp-vys-abstain" else route_pp_abstain_rows(rows)
    _write_rows(routed, args.output)


if __name__ == "__main__":
    main()
