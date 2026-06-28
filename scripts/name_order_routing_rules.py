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
from pathlib import Path
from typing import Any, Literal

Route = Literal["pp", "vys", "abstain"]
Row = Mapping[str, object]
MutableRow = dict[str, object]

PP_VYS_ABSTAIN_REQUIRED_COLUMNS = (
    "old_prediction",
    "new_prediction",
    "new_reason",
    "pp_selected_format",
    "vys_selected_format",
    "pp_batch_total_count",
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


def route_pp_vys_abstain_rows(rows: Iterable[Row]) -> list[MutableRow]:
    """Route PP/VYS disagreements, defaulting to PP except known VYS/abstain slices.

    Required input columns are listed in `PP_VYS_ABSTAIN_REQUIRED_COLUMNS`.
    The returned rows add:
    - `input_order_candidate`
    - `router_prediction`
    - `router_reason`
    """
    output: list[MutableRow] = []
    for row in rows:
        _require_columns(row, PP_VYS_ABSTAIN_REQUIRED_COLUMNS)
        routed = dict(row)
        ratio = _float(row["pp_selected_surname_frequency_ratio"])
        pp_batch_total_count = _float(row["pp_batch_total_count"])
        old_prediction = _string(row["old_prediction"])
        new_prediction = _string(row["new_prediction"])
        new_reason = _string(row["new_reason"])

        input_order_candidate = _input_order_candidate(row)
        route: Route = "pp"
        reason = "default_pp"

        old_new_vys = old_prediction == "vys" and new_prediction == "vys"
        old_pp_new_vys = old_prediction == "pp" and new_prediction == "vys"
        old_vys_new_abstain = old_prediction == "vys" and new_prediction in {"abstain", "not_person"}
        old_vys_new_pp = old_prediction == "vys" and new_prediction == "pp"

        if old_new_vys:
            route = "vys"
            reason = "old_new_vys"
        if old_pp_new_vys:
            route = "vys"
            reason = "old_pp_new_vys"
        if old_vys_new_abstain:
            route = "vys"
            reason = "old_vys_new_abstain"
        if old_vys_new_pp:
            route = "vys"
            reason = "old_vys_new_pp_default_vys"

        endpoint_pp_with_real_paper_votes = (
            old_vys_new_pp
            and new_reason == "endpoint_frequency_strongly_favors_pp"
            and pp_batch_total_count >= PP_VYS_REAL_PAPER_VOTES_MIN
        )
        if endpoint_pp_with_real_paper_votes:
            route = "pp"
            reason = "endpoint_pp_with_real_paper_votes"

        strong_pp_allowed_ratio = (
            old_vys_new_pp
            and new_reason == "strong_pp_paper_context"
            and (ratio <= PP_VYS_ALLOWED_RATIO_LOW_MAX or PP_VYS_ALLOWED_RATIO_HIGH_MIN < ratio <= PP_VYS_ALLOWED_RATIO_HIGH_MAX)
        )
        if strong_pp_allowed_ratio:
            route = "pp"
            reason = "strong_pp_allowed_ratio"

        reliable_input_order_abstain = (
            (input_order_candidate == "pp" and old_prediction == "pp" and new_prediction == "pp")
            or (
                input_order_candidate == "vys"
                and new_reason == "endpoint_frequency_strongly_favors_vys"
                and pp_batch_total_count <= PP_VYS_LOW_BATCH_MAX
            )
            or (
                input_order_candidate == "vys"
                and new_reason == "weak_or_conflicting_evidence"
                and pp_batch_total_count <= PP_VYS_VERY_LOW_BATCH_MAX
                and ratio <= PP_VYS_LOW_RATIO_MAX
            )
            or (
                input_order_candidate == "vys"
                and new_reason == "strong_vys_batch_context"
                and pp_batch_total_count <= PP_VYS_LOW_BATCH_MAX
                and ratio <= PP_VYS_LOW_RATIO_MAX
            )
        )
        if reliable_input_order_abstain:
            route = "abstain"
            reason = "reliable_input_order_abstain"

        routed["input_order_candidate"] = input_order_candidate
        routed["router_prediction"] = route
        routed["router_reason"] = reason
        output.append(routed)
    return output


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
    pp_preserves_input_order = _string(row["pp_selected_format"]) == "given_first"
    vys_preserves_input_order = _string(row["vys_selected_format"]) == "given_first"
    if pp_preserves_input_order and not vys_preserves_input_order:
        return "pp"
    if vys_preserves_input_order and not pp_preserves_input_order:
        return "vys"
    return "unknown"


def _pp_abstain_predicates(row: Row) -> dict[str, bool]:
    pp_result_token_count = _float(row["pp_result_token_count"])
    batch_total_count = _float(row["batch_total_count"])
    selected_surname_frequency = _float(row["selected_surname_frequency"])
    selected_over_alternate_ratio = _float(row["selected_over_alternate_ratio"])
    raw_tokens = _float(row["raw_tokens"])
    jp_probability = _float(row["jp_probability"])
    has_cjk = _bool(row["has_cjk"])
    has_latin = _bool(row["has_latin"])
    cjk_has_space = _bool(row["cjk_has_space"])
    compact_cjk = _string(row["compact_cjk"])

    surname_first_two_token = (
        pp_result_token_count == PP_ABSTAIN_TWO_TOKEN_RESULT_COUNT and _string(row["selected_format"]) == "surname_first"
    )
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
        and _string(row["selected_format"]) == "given_first"
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


def _string(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def _float(value: object) -> float:
    if value is None:
        return 0.0
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(parsed):
        return 0.0
    return parsed


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "t", "yes", "y"}
    return bool(value)


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
