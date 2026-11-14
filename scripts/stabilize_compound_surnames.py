#!/usr/bin/env python
"""Decide which surfaces of a rebuilt ``compound_surnames.csv.gz`` change status against the shipped table.

A rebuild from a newer snapshot lists or drops every surface by the evidence of the day. For a two-word pair that
ORCID registrants write about as often whole (``Rodrigues Gomes``) as split (``Gomes``), the majority flips between
snapshots, and the flips break almost as many people's family names as they fix. This step keeps the shipped status
unless the ORCID registry evidence is clear.

Evidence (``--stats``, Parquet with columns ``surf``, ``j``, ``t``, one row per surface): for every ORCID person whose
source-supplied mentions carry the surface as the last 2-3 words of ``middle + last`` (folded like the table's
``surface``: lower case, accents stripped, hyphens as spaces, single spaces), look up their registered family name in
the ORCID public data file, folded the same way. ``j`` (join) counts the people whose family name is the surface or
ends with it (``de Sousa Almeida`` for ``sousa almeida``); ``t`` (tail) counts those whose family name is the surface
without its first word. Each person counts once per surface.

  - clear compound: ``j + t >= MIN_SUPPORT`` and ``j >= MIN_SHARE * (j + t)``: listed.
  - clear split: ``j + t >= MIN_SUPPORT`` and ``t >= MIN_SHARE * (j + t)``: not listed.
  - ambiguous (enough evidence, neither side clear): ``--ambiguous keep`` (the default) keeps the shipped status;
    ``--ambiguous abstain`` drops the surface, so sinonym follows the source's own field split (``Ana / Paula
    Rodrigues / Gomes`` keeps ``Gomes``, ``Ana Paula / - / Rodrigues Gomes`` keeps ``Rodrigues Gomes``).
  - weak evidence (``j + t < MIN_SUPPORT``) or none: the shipped status (the candidate's without ``--previous``).

The defaults were chosen on held-out ORCID people (evidence from half A, scored on half B, against the 0.7.0 table):
abstaining lost at every margin and support tried (0.67 and 20: 2,702 family names fixed, 3,642 broken), because
sources split pairs that their authors registered whole about as often as the reverse; keeping the shipped status
with a 0.67 margin and 20 registrants changed no held-out person for the worse (3 fixed, 0 broken), while a 0.6 margin
re-admitted the churn (support 5: 300 fixed, 254 broken).

Only surfaces of the shipped or the candidate table are considered. A surface that stays listed keeps its shipped row,
so its counts do not churn either; a newly listed one takes the candidate's row and goes after them, so the file diff
shows only the changes. A shipped surface with weak evidence stays even if the current builders would no longer list
it, so after a change to the builders' gates run without ``--previous``.

Usage: stabilize_compound_surnames.py --candidate new.csv.gz --stats registry_stats.parquet
           [--previous sinonym/data/compound_surnames.csv.gz] [--min-share 0.67] [--min-support 20]
           [--ambiguous keep|abstain] --output sinonym/data/compound_surnames.csv.gz
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
from dataclasses import dataclass
from pathlib import Path

MIN_SHARE = 0.67
MIN_SUPPORT = 20
MAJORITY = 0.5


@dataclass(frozen=True)
class Policy:
    min_share: float = MIN_SHARE
    min_support: int = MIN_SUPPORT
    ambiguous: str = "keep"


def read_table(path: str) -> tuple[list[str], dict[str, list[str]]]:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        msg = f"{path}: empty table"
        raise ValueError(msg)
    table: dict[str, list[str]] = {}
    for row in rows[1:]:
        if not row:
            continue
        if row[0] in table:
            msg = f"{path}: duplicate surface {row[0]!r}"
            raise ValueError(msg)
        table[row[0]] = row
    return rows[0], table


def decide(join: int, tail: int, *, previous: bool | None, candidate: bool, policy: Policy) -> bool:
    """Whether the surface is listed. ``previous`` is None when there is no shipped table."""
    fallback = candidate if previous is None else previous
    support = join + tail
    if support < policy.min_support:
        return fallback
    if join >= policy.min_share * support:
        return True
    if tail >= policy.min_share * support:
        return False
    return False if policy.ambiguous == "abstain" else fallback


def stabilize(
    candidate: dict[str, list[str]],
    previous: dict[str, list[str]] | None,
    stats: dict[str, tuple[int, int]],
    policy: Policy,
) -> tuple[list[list[str]], dict[str, int]]:
    """The listed rows (shipped order, then new ones by tier and surface) and the counts of status changes."""
    base = previous if previous is not None else candidate
    listed = {}
    changes = {"listed": 0, "dropped": 0}
    for surface in dict.fromkeys([*base, *candidate]):
        before = None if previous is None else surface in previous
        listed[surface] = decide(*stats.get(surface, (0, 0)), previous=before, candidate=surface in candidate, policy=policy)
        if before is not None and listed[surface] != before:
            changes["listed" if listed[surface] else "dropped"] += 1
    rows = [row for surface, row in base.items() if listed[surface]]
    added = (row for surface, row in candidate.items() if surface not in base and listed[surface])
    return rows + sorted(added, key=lambda row: (row[1], row[0])), changes


def main() -> None:
    import duckdb  # noqa: PLC0415  (only the build needs it; tests import decide)

    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--stats", required=True, help="Parquet with columns surf, j, t (see the module docstring)")
    ap.add_argument("--previous")
    ap.add_argument("--min-share", type=float, default=MIN_SHARE)
    ap.add_argument("--min-support", type=int, default=MIN_SUPPORT)
    ap.add_argument("--ambiguous", choices=["keep", "abstain"], default="keep")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    if not MAJORITY < a.min_share <= 1:
        ap.error("--min-share must be in (0.5, 1]")
    policy = Policy(a.min_share, a.min_support, a.ambiguous)

    header, cand = read_table(a.candidate)
    prev_header, prev = read_table(a.previous) if a.previous else (header, None)
    if prev_header != header:
        ap.error(f"{a.previous}: header {prev_header}, expected {header}")
    rows = duckdb.execute("SELECT surf, j, t FROM read_parquet(?) WHERE surf IS NOT NULL", [a.stats]).fetchall()
    stats = {surf: (int(j or 0), int(t or 0)) for surf, j, t in rows}
    if len(stats) != len(rows):
        ap.error(f"{a.stats}: {len(rows) - len(stats):,} duplicate surfaces")
    matched = len(stats.keys() & (set(cand) | set(prev or ())))
    if not matched:
        ap.error(f"{a.stats}: no surface matches the tables; check the folding")

    listed, changes = stabilize(cand, prev, stats, policy)
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(header)
    writer.writerows(listed)
    output = Path(a.output)
    with output.open("wb") as raw, gzip.GzipFile(filename="compound_surnames.csv", mode="wb", fileobj=raw, mtime=0) as fh:
        fh.write(out.getvalue().encode("utf-8"))
    print(
        f"MIN_SHARE={a.min_share} MIN_SUPPORT={a.min_support} AMBIGUOUS={a.ambiguous}: {matched:,} table surfaces with "
        f"registry evidence; {len(listed):,} listed"
        + ("" if prev is None else f"; against the shipped table {changes['listed']:,} listed, {changes['dropped']:,} dropped")
        + f" -> {output}",
    )


if __name__ == "__main__":
    main()
