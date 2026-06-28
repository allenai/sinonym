# ruff: noqa: INP001
"""CLI wrapper for external PP/VYS/abstain name-order routing rules.

The routing policy lives in `sinonym.pipeline.name_order_routing` so packaged
pipeline code can import it directly. This script only handles CSV, JSONL, and
Parquet file IO.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from sinonym.pipeline.name_order_routing import (
    MutableRow,
    route_pp_abstain_rows,
    route_pp_vys_abstain_rows,
)

PP_VYS_ABSTAIN_ROUTING_COLUMNS = ("input_order_candidate", "router_prediction", "router_reason")
PP_ABSTAIN_ROUTING_COLUMNS = ("router_prediction", "router_reason")


@dataclass(frozen=True)
class RowTable:
    """Rows plus source field order for empty-output schema preservation."""

    rows: list[MutableRow]
    fieldnames: list[str]


def _read_rows(path: Path) -> RowTable:
    suffix = path.suffix.casefold()
    if suffix == ".parquet":
        return _read_parquet_rows(path)
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8-sig") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        return RowTable(rows=rows, fieldnames=_fieldnames(rows))
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return RowTable(rows=list(reader), fieldnames=list(reader.fieldnames or []))


def _write_rows(rows: list[MutableRow], path: Path, fieldnames: list[str]) -> None:
    suffix = path.suffix.casefold()
    if suffix == ".parquet":
        _write_parquet_rows(rows, path, fieldnames)
    elif suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def _read_parquet_rows(path: Path) -> RowTable:
    try:
        import pandas as pd  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        message = "Parquet input requires pandas and a parquet engine in this environment."
        raise RuntimeError(message) from exc
    dataframe = pd.read_parquet(path)
    return RowTable(rows=dataframe.to_dict("records"), fieldnames=list(dataframe.columns))


def _write_parquet_rows(rows: list[MutableRow], path: Path, fieldnames: list[str]) -> None:
    try:
        import pandas as pd  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        message = "Parquet output requires pandas and a parquet engine in this environment."
        raise RuntimeError(message) from exc
    pd.DataFrame(rows, columns=fieldnames).to_parquet(path, index=False)


def _fieldnames(rows: list[MutableRow]) -> list[str]:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def _output_fieldnames(input_fieldnames: list[str], rows: list[MutableRow], routing_columns: tuple[str, ...]) -> list[str]:
    fieldnames = _fieldnames(rows)
    if not fieldnames:
        fieldnames = list(input_fieldnames)

    for column in routing_columns:
        if column not in fieldnames:
            fieldnames.append(column)
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

    table = _read_rows(args.input)
    if args.regime == "pp-vys-abstain":
        routed = route_pp_vys_abstain_rows(table.rows)
        routing_columns = PP_VYS_ABSTAIN_ROUTING_COLUMNS
    else:
        routed = route_pp_abstain_rows(table.rows)
        routing_columns = PP_ABSTAIN_ROUTING_COLUMNS
    _write_rows(routed, args.output, _output_fieldnames(table.fieldnames, routed, routing_columns))


if __name__ == "__main__":
    main()
