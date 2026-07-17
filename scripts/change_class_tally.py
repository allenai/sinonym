"""Tally name-normalization change classes over a canonical parquet (reference tool).

Runs a set of SQL slice-detectors over per-name normalization output and prints,
for each change class, the distinct-name count, the occurrence count, and the
share of NON-CHINESE occurrences. DuckDB streams the file, so it is memory-safe on
a multi-GB / 100M-row canonical (a pandas/pyarrow read_table would OOM).

This is the detector behind the "change-class sizes" table in the PR review. The
counts are exact for each detector's *signature*, but a signature is a heuristic:
it can miss variant shapes (under-capture) and, for the convention/routing classes
(order swap, compound-surname split), it lumps correct cases in with wrong ones —
so those rows are a CEILING, not an error count. It is not ground truth.

INPUT DATA — one row per distinct production (first, middle, last) field-split of a
corpus author name. The same joined name string can appear on several rows (one per
split it occurs under), so row count > distinct name count. `nm` is space-joined
from the production split and is the ONLY thing sinonym sees; the db_* fields are the
untouched upstream reference, and the norm_* fields are sinonym's output — i.e.
production vs sinonym side by side, after a sinonym 0.4.0 run. Required columns:
  nm             the joined input name string (given+middle+last, space-joined)
  occ            author-rows carrying that exact production split (weighting)
  db_first/db_middle/db_last   the production (upstream) field split, for comparison
  chinese        True if sinonym routed it through the Chinese path (excluded here)
  rejected       True if sinonym returned no person (non-person / abstain)
  canonical_text sinonym's rendered full name
  norm_given/norm_middle/norm_surname/norm_suffix   sinonym's component split
Optional (present in the corpus dump, unused here): script, error_message, norm_order.

Counts below are per-row (per production split): the "distinct" column is distinct
splits, not distinct name strings, and occ is author-row occurrences.

Usage:
  python scripts/change_class_tally.py <canonical_parquet>
"""
import sys
from pathlib import Path

import duckdb

# Bare organization nouns that, alone in db_last, mark a pure institution.
ORGS = (
    "('university','institute','laboratory','center','centre','society','department',"
    "'hospital','association','committee','consortium','corporation','company','inc','ltd','team')"
)

# class name -> SQL predicate over a non-Chinese, per-name row.
CHANGE_CLASSES = {
    "leading-initial dup on particle": (
        "not rejected and norm_surname like norm_given || ' %' and norm_given like '%.'"
    ),
    "compound-surname split (ceiling)": (
        "not rejected and db_middle = '' and db_last like '% %' and norm_middle <> '' "
        "and lower(norm_surname) = lower(split_part(db_last, ' ', -1))"
    ),
    "given<->surname swap (EA order, ceiling)": (
        "not rejected and db_middle = '' and lower(norm_surname) = lower(db_first) "
        "and lower(norm_given) = lower(db_last)"
    ),
    "dangling '-and' reject": (
        "rejected and regexp_matches(lower(nm), '(^|[ -])and([ -]|$)')"
    ),
    "senior/junior -> suffix (surname emptied)": (
        "not rejected and norm_suffix in ('Sr.','Jr.') and norm_surname = ''"
    ),
    "org-word kept as person": f"not rejected and lower(db_last) in {ORGS}",
    "nobiliary particle Title-cased": (
        "not rejected and regexp_matches(canonical_text, ' (Ten|Ur|Dem|Ud) ') "
        "and regexp_matches(lower(nm), ' (ten|ur|dem|ud) ')"
    ),
}


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: python scripts/change_class_tally.py <canonical_parquet>")
    parquet = Path(sys.argv[1])
    if not parquet.exists():
        sys.exit(f"no such file: {parquet}")
    src = f"read_parquet('{parquet.as_posix()}')"

    con = duckdb.connect()
    con.execute("PRAGMA threads=6")
    total, dnm, tocc = con.execute(f"select count(*), count(distinct nm), sum(occ) from {src}").fetchone()
    cd, cocc = con.execute(f"select count(*), sum(occ) from {src} where chinese=true").fetchone()
    den = con.execute(f"select sum(occ) from {src} where chinese=false").fetchone()[0]
    print(f"file: {parquet}")
    print(f"rows (production splits): {total:,}   distinct name strings: {dnm:,}   total occ: {tocc:,}")
    print(f"chinese: splits={cd:,} occ={cocc:,}")
    print(f"non-chinese denominator occ = {den:,}\n")
    print(f"{'change class':44}{'splits':>12}{'occ':>14}{'%nonCh':>9}")
    for name, where in CHANGE_CLASSES.items():
        d, o = con.execute(
            f"select count(*), coalesce(sum(occ), 0) from {src} where chinese=false and {where}",
        ).fetchone()
        print(f"{name:44}{d:>12,}{o:>14,}{100 * o / den:>8.4f}%")


if __name__ == "__main__":
    main()
