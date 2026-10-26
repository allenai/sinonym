#!/usr/bin/env python
"""Build ``sinonym/data/two_token_order_prior.csv.gz`` from a pair-orientation table (see services/order_prior.py).

Input: pair_order_prior.parquet (author-names/pp_vys_orcid_conc/scripts/order_prior.py): per unordered token pair
``(a, b)``, mentions and distinct papers written ``first=a last=b`` (``n_ab``, ``papers_ab``) and the reverse, from
pre-normalization source names. A pair ships when it has >= MIN_MENTIONS mentions, a >= MIN_SHARE majority by
mentions *and* by distinct papers (so one prolific author cannot set it), and the Chinese parser reads the pair as a
name (the rule only ever fires on the Chinese path). Columns match the loader: a, b, n_ab, n_ba, authors_ab, authors_ba
(authors = distinct ORCID people in the October counts).
Usage: build_order_prior_asset.py --pairs pair_order_prior.parquet --output sinonym/data/two_token_order_prior.csv.gz
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
from pathlib import Path

import duckdb

from sinonym.detector import ChineseNameDetector

MIN_MENTIONS = 100
MIN_SHARE = 0.8


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    rows = duckdb.connect().execute(
        f"""SELECT a, b, n_ab, n_ba, authors_ab, authors_ba FROM '{a.pairs}'
            WHERE n_ab + n_ba >= {MIN_MENTIONS}
              AND GREATEST(n_ab, n_ba) / (n_ab + n_ba) >= {MIN_SHARE}
              AND (CASE WHEN n_ab >= n_ba THEN papers_ab ELSE papers_ba END) / GREATEST(papers_ab + papers_ba, 1) >= {MIN_SHARE}
              AND regexp_matches(a, '^[a-z]+$') AND regexp_matches(b, '^[a-z]+$')
            ORDER BY a, b""",  # noqa: S608
    ).fetchall()
    detector = ChineseNameDetector()
    kept = []
    for row in rows:
        pair_a, pair_b = row[0], row[1]
        if detector.is_chinese_name(f"{pair_a} {pair_b}").success or detector.is_chinese_name(f"{pair_b} {pair_a}").success:
            kept.append(row)
    out = Path(a.output)
    with gzip.open(out, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["a", "b", "n_ab", "n_ba", "authors_ab", "authors_ba"])
        for pair_a, pair_b, n_ab, n_ba, au_ab, au_ba in kept:
            w.writerow([pair_a, pair_b, float(n_ab), float(n_ba), float(au_ab or 0), float(au_ba or 0)])
    digest = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
    print(f"{len(rows):,} pairs pass the thresholds, {len(kept):,} read as Chinese names -> {out} (sha256 {digest})")


if __name__ == "__main__":
    main()
