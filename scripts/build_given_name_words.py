#!/usr/bin/env python
"""Build ``sinonym/data/given_name_words.csv.gz``: words that ORCID registry people use almost only as a given name.

Input: the ORCID public data file as Parquet with ``orcid``, ``given_names`` and ``family_name``. Each word (lower-case,
accents stripped, split on spaces, hyphens, dots and commas, three or more letters) is counted once per person as a given
name and once per person as a family name. A word is listed when at least 200 people use it and at least 95% of them use
it as a given name (``jonathan``, ``gareth``, ``myriam``; not ``lee`` or ``kumar``, which are also surnames), and fewer
than 0.5% of those given-name people open a multi-word family name with it (not ``andres``, ``blanca``, ``david``,
``jose``: Iberian and South Asian double surnames start with them). Words that miss those bars but that at least 80% use as
a given name and fewer than 1.5% open a multi-word family name with (``philip``, ``james``, ``paul``) are listed too: the
runtime reads them only before a common Chinese surname. The output holds only words and counts, no identifiers.

Usage: build_given_name_words.py --registry '<registry parquet glob>' --output sinonym/data/given_name_words.csv.gz
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
from pathlib import Path

import duckdb

MIN_PEOPLE = 200
# the general bars (95% given, under 0.5% family head) live in sinonym/utils/given_name_words.py
# looser bars for a word before a common Chinese surname (C. / L. Philip / Chen), where no double surname opens with it
MIN_GIVEN_SHARE_CHINESE = 0.80
MAX_FAMILY_HEAD_SHARE_CHINESE = 0.015


def build(registry: str) -> list[tuple]:
    return duckdb.sql(f"""
        WITH r AS (SELECT orcid, lower(strip_accents(coalesce(given_names, ''))) g,
                          lower(strip_accents(coalesce(family_name, ''))) f FROM read_parquet('{registry}')),
        gw AS (SELECT DISTINCT orcid, unnest(regexp_split_to_array(g, '[\\s\\-.,]+')) w FROM r),
        fw AS (SELECT DISTINCT orcid, unnest(regexp_split_to_array(f, '[\\s\\-.,]+')) w FROM r),
        hw AS (SELECT DISTINCT orcid, split_part(f, ' ', 1) w FROM r WHERE contains(f, ' ')),
        g AS (SELECT w, count(*) n FROM gw GROUP BY 1),
        f AS (SELECT w, count(*) n FROM fw GROUP BY 1),
        h AS (SELECT w, count(*) n FROM hw GROUP BY 1)
        SELECT g.w, g.n, coalesce(f.n, 0), coalesce(h.n, 0) FROM g LEFT JOIN f USING (w) LEFT JOIN h USING (w)
        WHERE length(g.w) >= 3 AND regexp_full_match(g.w, '[a-z]+')
          AND g.n + coalesce(f.n, 0) >= {MIN_PEOPLE} AND g.n >= {MIN_GIVEN_SHARE_CHINESE} * (g.n + coalesce(f.n, 0))
          AND coalesce(h.n, 0) < {MAX_FAMILY_HEAD_SHARE_CHINESE} * g.n
        ORDER BY 1""").fetchall()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    rows = build(a.registry)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["word", "given_people", "family_people", "family_head_people"])
    w.writerows(rows)
    Path(a.output).write_bytes(gzip.compress(buf.getvalue().encode("utf-8"), mtime=0))
    print(len(rows), "words")


if __name__ == "__main__":
    main()
