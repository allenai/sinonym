#!/usr/bin/env python
"""Build ``sinonym/data/token_role_prior.csv.gz``: Latin name tokens the corpus uses almost only as a given name or
almost only as a surname (see services/token_order_prior.py).

Input: partition_prior_tokens.parquet (per folded token, mentions as a given name vs as the surname word, from the
pre-normalization paper_authors dump; built by author-names/pp_vys_orcid_conc/scripts/partition_prior.py).
A token ships when it has >= MIN_MENTIONS mentions and a given-name share <= SURNAME_MAX_GIVEN (a surname word)
or >= GIVEN_MIN_SHARE (a given-name word). Kill test before shipping: orderflip_analysis.py (690k index,
production key, both halves).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
from pathlib import Path

import duckdb

MIN_MENTIONS = 200
SURNAME_MAX_GIVEN = 0.1
GIVEN_MIN_SHARE = 0.9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    rows = duckdb.connect().execute(
        f"""SELECT t, CASE WHEN given_share >= {GIVEN_MIN_SHARE} THEN 'given' ELSE 'surname' END AS token_role,
                   CAST(as_first + as_last AS BIGINT) mentions, round(given_share, 3) given_share
            FROM '{a.tokens}'
            WHERE as_first + as_last >= {MIN_MENTIONS} AND length(t) >= 2 AND regexp_matches(t, '^[a-z]+$')
              AND (given_share <= {SURNAME_MAX_GIVEN} OR given_share >= {GIVEN_MIN_SHARE})
            ORDER BY t""",  # noqa: S608
    ).fetchall()
    out = Path(a.output)
    with gzip.open(out, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["token", "role", "mentions", "given_share"])
        w.writerows(rows)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
    given = sum(1 for r in rows if r[1] == "given")
    print(f"{len(rows):,} tokens ({given:,} given, {len(rows) - given:,} surname) -> {out} (sha256 {digest})")


if __name__ == "__main__":
    main()
