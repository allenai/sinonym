#!/usr/bin/env python
"""Merge rebuilt ``token_role_prior.csv.gz`` and ``two_token_order_prior.csv.gz`` into the shipped tables.

A rebuild from a newer snapshot decides every entry by that day's counts. Entries near a gate (a given-name share near
90% or 10%, a pair majority near 80%, a token near 200 mentions or a pair near 100) cross it in either direction on
small count changes, and each crossing turns a two-word name around (``Petralia / Salvatore``) or stops turning it.
On the 2026-11-12 rebuild the plain new tables fixed the family name of 183 ORCID people whose source names they
touch and broke 70; most of the breaks came from entries the rebuild dropped.

This step keeps every shipped entry whose role or majority the candidate does not reverse, with its shipped row, and
adds every candidate-only entry (the rebuild's own gates); ``--margin`` asks a candidate-only or reversing entry to
clear the gate by that much more:

  - token: ``given_share >= 0.90 + margin`` (given) or ``<= 0.10 - margin`` (surname), at the builder's mentions gate;
  - pair: majority share ``>= 0.80 + margin``, at the builder's mentions gate;
  - a shipped entry the candidate lists with the opposite role or majority takes the candidate's row when the
    candidate clears the same margin; otherwise the shipped row stays.

Chosen on the ORCID people whose source names contain a changed token or pair (registered family name, 0.7.0 tables
as the baseline): the plain rebuild fixed 183 and broke 70; keeping shipped entries with margin 0 fixed 177 and broke
25; margin 0.01 67 / 5, 0.02 56 / 2. Blind labels sided with margin 0: on names where keeping the dropped entries
changes the rebuild's output, 18 better / 0 worse; on names where a 0.02 margin would hold back an addition, the
addition was better 63 times and worse 3.

Usage: stabilize_order_priors.py --previous-dir sinonym/data --candidate-dir rebuilt/ --output-dir sinonym/data
           [--margin 0.0]
"""

from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path

MARGIN = 0.0
TOKEN_MIN_MENTIONS = 200
GIVEN_MIN_SHARE = 0.9
SURNAME_MAX_GIVEN = 0.1
PAIR_MIN_MENTIONS = 100
PAIR_MIN_SHARE = 0.8

TOKEN_TABLE = "token_role_prior.csv.gz"  # noqa: S105
PAIR_TABLE = "two_token_order_prior.csv.gz"


def read_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_table(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def clear_token(row: dict[str, str], margin: float) -> bool:
    share = float(row["given_share"])
    if int(float(row["mentions"])) < TOKEN_MIN_MENTIONS:
        return False
    return share >= GIVEN_MIN_SHARE + margin or share <= SURNAME_MAX_GIVEN - margin


def pair_majority(row: dict[str, str]) -> str:
    return "ab" if float(row["n_ab"]) >= float(row["n_ba"]) else "ba"


def clear_pair(row: dict[str, str], margin: float) -> bool:
    n_ab, n_ba = float(row["n_ab"]), float(row["n_ba"])
    total = n_ab + n_ba
    return total >= PAIR_MIN_MENTIONS and max(n_ab, n_ba) / total >= PAIR_MIN_SHARE + margin


def merge_tokens(previous: list[dict[str, str]], candidate: list[dict[str, str]], margin: float) -> list[dict[str, str]]:
    new = {row["token"]: row for row in candidate}
    merged = {}
    for row in previous:
        update = new.get(row["token"])
        reversed_role = update is not None and update["role"] != row["role"] and clear_token(update, margin)
        merged[row["token"]] = update if reversed_role else row
    for token, row in new.items():
        if token not in merged and clear_token(row, margin):
            merged[token] = row
    return [merged[token] for token in sorted(merged)]


def merge_pairs(previous: list[dict[str, str]], candidate: list[dict[str, str]], margin: float) -> list[dict[str, str]]:
    new = {(row["a"], row["b"]): row for row in candidate}
    merged = {}
    for row in previous:
        key = (row["a"], row["b"])
        update = new.get(key)
        reversed_majority = update is not None and pair_majority(update) != pair_majority(row) and clear_pair(update, margin)
        merged[key] = update if reversed_majority else row
    for key, row in new.items():
        if key not in merged and clear_pair(row, margin):
            merged[key] = row
    return [merged[key] for key in sorted(merged)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--previous-dir", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--margin", type=float, default=MARGIN)
    args = parser.parse_args()
    fields, previous = read_table(args.previous_dir / TOKEN_TABLE)
    _, candidate = read_table(args.candidate_dir / TOKEN_TABLE)
    tokens = merge_tokens(previous, candidate, args.margin)
    pair_fields, previous_pairs = read_table(args.previous_dir / PAIR_TABLE)
    _, candidate_pairs = read_table(args.candidate_dir / PAIR_TABLE)
    pairs = merge_pairs(previous_pairs, candidate_pairs, args.margin)
    write_table(args.output_dir / TOKEN_TABLE, fields, tokens)
    write_table(args.output_dir / PAIR_TABLE, pair_fields, pairs)
    print(f"tokens {len(previous)} -> {len(tokens)}; pairs {len(previous_pairs)} -> {len(pairs)}")


if __name__ == "__main__":
    main()
