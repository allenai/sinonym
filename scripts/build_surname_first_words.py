#!/usr/bin/env python
"""Build ``sinonym/data/surname_first_words.csv.gz``: words the corpus writes surname-first before dotted initials
(``Dmitrieva / - / N.M.``, ``Назарова / - / А.Р.``) and otherwise uses mostly as a surname.

Input: the source-name counts export (one row per distinct source ``first_name / middle_names / last_name`` triple with
``mention_count``; see author-names ``NAME_COUNTS.md``). A word qualifies when
  - it appears as the whole ``first_name`` of a triple whose ``middle_names`` is empty and whose ``last_name`` is 2-3
    initials, each with its own dot (Latin or Cyrillic), and
  - across all single-word fields of the corpus it is a ``last_name`` at least MIN_AS_LAST times and more often than a
    ``first_name``.
The second condition keeps out names written ``Sowmya / - / K. R.``, where the word is the given name and the initials
are the family name. Measured on every ORCID person (dataset v4): 3,072 people reunited / 47 split; against the ORCID
registry the resulting surname is right 4,863 times and wrong 409.

Usage: build_surname_first_words.py --counts '<name_counts parquet glob>' --output sinonym/data/surname_first_words.csv.gz
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
from pathlib import Path

import duckdb

MIN_AS_LAST = 3
FOLD = "lower(strip_accents(regexp_replace(trim(coalesce({x},'')), '[^\\p{{L}}\\p{{N}}]', '', 'g')))"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    con = duckdb.connect()
    con.execute(f"""CREATE TABLE role AS SELECT w, SUM(as_first) as_first, SUM(as_last) as_last FROM (
        SELECT {FOLD.format(x='first_name')} w, mention_count as_first, 0 as_last FROM read_parquet('{a.counts}') WHERE NOT contains(trim(first_name), ' ')
        UNION ALL SELECT {FOLD.format(x='last_name')} w, 0, mention_count FROM read_parquet('{a.counts}') WHERE NOT contains(trim(last_name), ' ')) GROUP BY 1""")
    rows = con.execute(f"""
        WITH shape AS (
          SELECT {FOLD.format(x='first_name')} w, SUM(mention_count) n_shape FROM read_parquet('{a.counts}')
          WHERE regexp_full_match(trim(first_name), '\\p{{Lu}}[\\p{{Ll}}''\\-]{{2,}}') AND coalesce(trim(middle_names), '') = ''
            AND regexp_full_match(trim(last_name), '\\p{{Lu}}\\.(?:\\s*\\p{{Lu}}\\.){{1,2}}')
          GROUP BY 1)
        SELECT s.w, s.n_shape, r.as_last, r.as_first FROM shape s JOIN role r USING (w)
        WHERE r.as_last >= {MIN_AS_LAST} AND r.as_last > r.as_first ORDER BY s.n_shape DESC, s.w""").fetchall()
    out = Path(a.output)
    with gzip.open(out, "wt", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["word", "surname_first_mentions", "as_last", "as_first"])
        wr.writerows((w, int(n), int(l), int(f)) for w, n, l, f in rows)
    print(f"{len(rows):,} words -> {out} (sha256 {hashlib.sha256(out.read_bytes()).hexdigest()[:16]})")


if __name__ == "__main__":
    main()
