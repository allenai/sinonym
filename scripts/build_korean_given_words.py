#!/usr/bin/env python
"""Build ``sinonym/data/korean_given_words.csv.gz``: two-syllable Korean given names and where their syllables meet.

Input: the ORCID public data file as Parquet with ``orcid``, ``given_names`` and ``countries``.

A given name written as one word (``Jinho``) or as two parts (``Jin-Ho``, ``Jin Ho``) is listed under its joined
spelling when
  - at least MIN_KR_PEOPLE registry people in South Korea write it, and they are at least MIN_KR_SHARE of the people
    with a country who write it (``Juan``, ``Lina`` and ``Hana`` are not listed);
  - at least MIN_SPLIT_PEOPLE of those Korean people write it in two parts, and at least MIN_CUT_SHARE of them cut it
    at the same place (that cut is the listed boundary; a word Korean people cut in two ways is not listed).

Usage: build_korean_given_words.py --registry '<registry parquet glob>' --output sinonym/data/korean_given_words.csv.gz
       [--half a]   (evidence from ORCID iDs ending in an even digit only, for held-out evaluation)
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
from pathlib import Path

import duckdb

MIN_KR_PEOPLE = 5
MIN_KR_SHARE = 0.2
MIN_SPLIT_PEOPLE = 3
MIN_CUT_SHARE = 0.9


def build(registry: str, half: str | None) -> list[tuple]:
    con = duckdb.connect()
    half_filter = "AND upper(substr(orcid, -1, 1)) IN ('0','2','4','6','8')" if half == "a" else ""
    con.execute(f"""CREATE TABLE ev AS SELECT
        replace(g, ' ', '') word, CASE WHEN contains(g, ' ') THEN length(split_part(g, ' ', 1)) END cut,
        regexp_matches(countries, '(^|[^A-Z])KR([^A-Z]|$)') kr,
        regexp_matches(countries, '(^|[^A-Z])(CN|TW|HK|MO|SG)([^A-Z]|$)') cn
      FROM (SELECT orcid, countries,
              regexp_replace(lower(strip_accents(trim(coalesce(given_names, '')))), '[\\s\\-\u2010\u2011\u2013]+', ' ', 'g') g
            FROM read_parquet('{registry}') WHERE coalesce(countries, '') <> '' {half_filter})
      WHERE (regexp_full_match(g, '[a-z]{{3,10}}') OR regexp_full_match(g, '[a-z]{{1,6}} [a-z]{{1,6}}'))
        AND length(replace(g, ' ', '')) BETWEEN 3 AND 10""")
    return con.execute(f"""
      WITH w AS (SELECT word, count(*) people, count(*) FILTER (WHERE kr) kr_people, count(*) FILTER (WHERE cn) cn_people,
                   count(*) FILTER (WHERE kr AND cut IS NOT NULL) kr_split FROM ev GROUP BY word),
           c AS (SELECT word, cut, count(*) n, row_number() OVER (PARTITION BY word ORDER BY count(*) DESC, cut) rk
                 FROM ev WHERE kr AND cut IS NOT NULL GROUP BY word, cut)
      SELECT w.word, c.cut, w.kr_people, w.cn_people, w.people, round(w.kr_people / w.people, 3), w.kr_split,
        round(c.n / w.kr_split, 3)
      FROM w JOIN c ON c.word = w.word AND c.rk = 1
      WHERE w.kr_people >= {MIN_KR_PEOPLE} AND w.kr_people >= {MIN_KR_SHARE} * w.people
        AND w.kr_split >= {MIN_SPLIT_PEOPLE} AND c.n >= {MIN_CUT_SHARE} * w.kr_split
      ORDER BY w.kr_people DESC, w.word""").fetchall()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--half", choices=["a"])
    a = ap.parse_args()
    rows = build(a.registry, a.half)
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(
        ["word", "cut", "kr_people", "chinese_region_people", "registry_people", "kr_share", "kr_split_people", "cut_share"],
    )
    writer.writerows(rows)
    Path(a.output).write_bytes(gzip.compress(out.getvalue().encode("utf-8"), mtime=0))
    print(len(rows), "words")


if __name__ == "__main__":
    main()
