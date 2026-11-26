#!/usr/bin/env python
"""Build ``sinonym/data/allcaps_given_words.csv.gz``: three-letter words that are usually a person's given name.

Input: the ORCID public data file as Parquet with ``orcid`` and ``given_names``.

A source writes initials as one capitalized word (``ESL Pedersen`` for Eva Sophie Lunde), and sometimes a given name in
capitals too (``ANA Dias``). A three-letter word is listed when registry people use it far more as a given name than as
initials:
  - at least MIN_NAME_PEOPLE people write it as the first word of their given names, not in capitals (``Ana Maria``);
  - those people are at least MIN_NAME_SHARE of them plus the people whose three given names start with its letters
    (``Eva Sophie Lunde`` for ``esl``, ``Melissa A. E.`` for ``mae``).

Usage: build_allcaps_given_words.py --registry '<registry parquet glob>' --output sinonym/data/allcaps_given_words.csv.gz
       [--half a]   (evidence from ORCID iDs ending in an even digit only, for held-out evaluation)
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
from pathlib import Path

import duckdb

MIN_NAME_PEOPLE = 10
MIN_NAME_SHARE = 0.8


def build(registry: str, half: str | None) -> list[tuple]:
    con = duckdb.connect()
    half_filter = "WHERE upper(substr(orcid, -1, 1)) IN ('0','2','4','6','8')" if half == "a" else ""
    con.execute(f"""CREATE TABLE ev AS SELECT w[1] first_raw, lower(w[1]) first_word, len(w) words,
        lower(list_aggregate(list_transform(w, lambda x: x[1]), 'string_agg', '')) letters
      FROM (SELECT list_filter(regexp_split_to_array(trim(strip_accents(coalesce(given_names, ''))),
                                                     '[\\s.\\-\u2010\u2011]+'), lambda x: x <> '') w
            FROM read_parquet('{registry}') {half_filter})
      WHERE len(w) > 0""")
    return con.execute(f"""
      WITH n AS (SELECT first_word word, count(*) name_people FROM ev
                 WHERE regexp_full_match(first_raw, '[A-Za-z]{{3}}') AND first_raw <> upper(first_raw) GROUP BY 1),
           i AS (SELECT letters word, count(*) initials_people FROM ev
                 WHERE words = 3 AND regexp_full_match(letters, '[a-z]{{3}}') GROUP BY 1)
      SELECT n.word, name_people, coalesce(initials_people, 0) initials_people,
        round(name_people / (name_people + coalesce(initials_people, 0)), 3) name_share
      FROM n LEFT JOIN i USING (word)
      WHERE name_people >= {MIN_NAME_PEOPLE}
        AND name_people >= {MIN_NAME_SHARE} * (name_people + coalesce(initials_people, 0))
      ORDER BY n.word""").fetchall()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--half", choices=["a"])
    a = ap.parse_args()
    rows = build(a.registry, a.half)
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["word", "name_people", "initials_people", "name_share"])
    writer.writerows(rows)
    Path(a.output).write_bytes(gzip.compress(out.getvalue().encode("utf-8"), mtime=0))
    print(len(rows), "words")


if __name__ == "__main__":
    main()
