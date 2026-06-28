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
from pathlib import Path

from sinonym.pipeline.name_order_routing import MutableRow, route_pp_abstain_rows, route_pp_vys_abstain_rows


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
