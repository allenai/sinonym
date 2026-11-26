#!/usr/bin/env python
"""Build ``sinonym/data/glued_initial_prefixes.csv.gz``: the given words after which a glued capital is a middle initial
(``DamianG.`` -> ``Damian G.``; see services/source_repairs.py).

Input: source-name counts (Parquet with ``first_name``, ``last_name``, ``mention_count``; source names as published,
never sinonym output). A word ships when it is a capitalized word of 3+ letters used as a first name at least
MIN_FIRST_MENTIONS times and at least as often as a surname. For each word, ``block_dotted`` / ``block_undotted`` list
the letters L for which word + lower(L) is itself a first name with at least BLOCK_DOTTED_SHARE / BLOCK_UNDOTTED_SHARE
of the word's own first-name count (``Martin`` blocks ``E``: ``MartinE.`` may be Martine).
"""

from __future__ import annotations

import argparse

import duckdb

MIN_FIRST_MENTIONS = 100
BLOCK_DOTTED_SHARE = 0.05
BLOCK_UNDOTTED_SHARE = 0.01


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name-counts", required=True, help="Parquet file or glob of source-name counts")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    con = duckdb.connect()
    con.execute(f"""
        CREATE TABLE firsts AS SELECT first_name AS f, sum(mention_count)::BIGINT AS m FROM read_parquet('{a.name_counts}')
            WHERE first_name IS NOT NULL GROUP BY 1;
        CREATE TABLE lasts AS SELECT last_name AS f, sum(mention_count)::BIGINT AS m FROM read_parquet('{a.name_counts}')
            WHERE last_name IS NOT NULL GROUP BY 1;
        CREATE TABLE words AS SELECT f.f AS prefix, f.m FROM firsts f LEFT JOIN lasts l USING (f)
            WHERE regexp_full_match(f.f, '\\p{{Lu}}\\p{{Ll}}{{2,}}') AND f.m >= {MIN_FIRST_MENTIONS} AND f.m >= coalesce(l.m, 0);
        CREATE TABLE blocked AS SELECT w.prefix, chr(c::INTEGER) AS letter, x.m AS lm, w.m
            FROM words w CROSS JOIN range(65, 91) t(c) JOIN firsts x ON x.f = w.prefix || lower(chr(c::INTEGER));
    """)  # noqa: S608
    con.execute(f"""
        COPY (SELECT w.prefix,
                     coalesce(string_agg(b.letter, '' ORDER BY b.letter)
                              FILTER (WHERE b.lm >= {BLOCK_DOTTED_SHARE} * w.m), '') AS block_dotted,
                     coalesce(string_agg(b.letter, '' ORDER BY b.letter)
                              FILTER (WHERE b.lm >= {BLOCK_UNDOTTED_SHARE} * w.m), '') AS block_undotted
              FROM words w LEFT JOIN blocked b USING (prefix) GROUP BY w.prefix ORDER BY w.prefix)
        TO '{a.output}' (HEADER, COMPRESSION gzip)
    """)  # noqa: S608
    print(con.execute("SELECT count(*) FROM words").fetchone()[0], "words ->", a.output)


if __name__ == "__main__":
    main()
