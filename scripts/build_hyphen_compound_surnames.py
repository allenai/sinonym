#!/usr/bin/env python
"""Add the ``hyphen`` tier to ``sinonym/data/compound_surnames.csv.gz`` (see data/COMPOUND_SURNAMES.md).

A source that writes ``Mantovani-Sarti`` in ``last_name`` declares the two words one surname. The surface ships when the
corpus writes it hyphenated in ``last_name`` for given names with at least MIN_INITIALS distinct initials (one person,
whose sources share an initial, cannot admit their own surname) and, counting the hyphenated mentions as whole, writes it
whole at least MIN_WHOLE_SHARE of the time (``Kumar-Singh`` is hyphenated by a few people but ``X / Kumar / Singh`` is
written 59k times: Kumar is a given name there). Behind the
same non-statistical gates as the other tiers (suffix tails, East Asian and Chinese compound heads, Vietnamese pairs,
one-letter words, head-word given-name gate with particles and construct prefixes exempt). Surfaces made only of East
Asian syllables are left out: surname-first sources put hyphenated given names in ``last_name`` (``Zhong-Hua``,
``Hyun-Woo``). Surfaces already in the asset keep their tier.

  uv run python scripts/build_hyphen_compound_surnames.py --name-counts '<name_counts dir>/*' \\
      --tokens partition_prior_tokens.parquet --partition partition_prior.parquet --asset sinonym/data/compound_surnames.csv.gz
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

import duckdb
from build_compound_surnames import (
    CHINESE_COMPOUND_HEADS,
    CONSTRUCT_PREFIXES,
    CORPUS_MIN_WHOLE_SHARE,
    EAST_ASIAN_HEADS,
    MAX_GIVEN_SHARE,
    PARTICLES,
    SUFFIX_TAILS,
    VIETNAMESE_MIDDLE,
    VIETNAMESE_TAILS,
)

from sinonym import ChineseNameDetector
from sinonym.chinese_names_data import (
    JAPANESE_SURNAMES,
    KOREAN_GIVEN_PATTERNS,
    KOREAN_ONLY_SURNAMES,
    NAME_ORDER_ROUTING_KOREAN_SURNAMES,
    OVERLAPPING_KOREAN_SURNAMES,
    VALID_CHINESE_ONSETS,
    VALID_CHINESE_RIMES,
    VIETNAMESE_SURNAMES,
    WADE_GILES_SYLLABLE_RULES,
)
from sinonym.services.person_name_normalization import EAST_ASIAN_GIVEN_SYLLABLES

MIN_INITIALS = 2
MIN_WHOLE_SHARE = CORPUS_MIN_WHOLE_SHARE
TIER = "hyphen"


def sql_list(words: tuple[str, ...]) -> str:
    return ",".join(f"'{w}'" for w in words)


def east_asian_syllable(word: str) -> bool:
    if word in EAST_ASIAN_GIVEN_SYLLABLES or word in KOREAN_GIVEN_PATTERNS or word in WADE_GILES_SYLLABLE_RULES:
        return True
    return any(word[:i] in VALID_CHINESE_ONSETS and word[i:] in VALID_CHINESE_RIMES for i in range(len(word)))


def east_asian_surnames() -> set[str]:
    path = Path(__file__).parent.parent / "sinonym/data/east_asian_roman_lexicons.json.gz"
    lexicons = json.loads(gzip.decompress(path.read_bytes()))
    return (
        set(ChineseNameDetector()._data.surnames)  # noqa: SLF001
        | set(lexicons["korean_surnames"]) | set(lexicons["japanese_surnames"]) | set(lexicons["vietnamese_surnames"])
        | KOREAN_ONLY_SURNAMES | OVERLAPPING_KOREAN_SURNAMES | NAME_ORDER_ROUTING_KOREAN_SURNAMES
        | JAPANESE_SURNAMES | VIETNAMESE_SURNAMES
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name-counts", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--partition", required=True)
    ap.add_argument("--min-whole-share", type=float, default=MIN_WHOLE_SHARE)
    ap.add_argument("--asset", required=True)
    a = ap.parse_args()
    asset = Path(a.asset)
    text = gzip.decompress(asset.read_bytes()).decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    fields = reader.fieldnames
    kept = [row for row in reader if row["tier"] != TIER]
    known = {row["surface"] for row in kept}
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute("CREATE TABLE known (surface VARCHAR)")
    con.executemany("INSERT INTO known VALUES (?)", [(s,) for s in known])
    # a surname-first source puts the family name in first_name and the hyphenated given name in last_name (Wang / Cai-hon)
    con.execute("CREATE TABLE ea (s VARCHAR)")
    con.executemany("INSERT INTO ea VALUES (?)", [(s,) for s in east_asian_surnames() if len(s) > 1])  # not bare initials (A, E)
    hyphens = "[-\u2010\u2011\u2012\u2013\u2212]"
    particles = sql_list((*PARTICLES, *CONSTRUCT_PREFIXES))
    vietnamese = (
        f"(len(w) = 2 AND w[1] IN ({sql_list(VIETNAMESE_MIDDLE)}) AND w[2] IN ({sql_list(VIETNAMESE_TAILS)}))"
    )
    rows = con.execute(f"""
        WITH r AS (
          SELECT substr(lower(strip_accents(trim(first_name))), 1, 1) ini,
                 trim(regexp_replace(regexp_replace(lower(strip_accents(trim(last_name))), '{hyphens}', ' ', 'g'),
                                     '\\s+', ' ', 'g')) surface,
                 mention_count n
          FROM read_parquet('{a.name_counts}')
          WHERE first_name IS NOT NULL AND regexp_matches(trim(first_name), '^[A-Za-zÀ-ÿ]')
            AND lower(strip_accents(trim(first_name))) NOT IN (SELECT s FROM ea)
            AND regexp_matches(last_name, '{hyphens}')
            AND regexp_matches(trim(last_name), '^[A-Za-zÀ-ÿ''\\-\u2010\u2011\u2012\u2013\u2212 ]+$')
        ),
        s AS (
          SELECT surface, string_split(surface, ' ') w, SUM(n) mentions, COUNT(DISTINCT ini) initials
          FROM r GROUP BY 1 HAVING COUNT(DISTINCT ini) >= {MIN_INITIALS}
        )
        SELECT s.surface, s.mentions, s.initials,
               (s.mentions + COALESCE(p.n_whole, 0)) / (s.mentions + COALESCE(p.n_total, 0)) whole_share
        FROM s
        JOIN '{a.tokens}' th ON th.t = s.w[1]
        LEFT JOIN '{a.partition}' p ON p.tail = s.surface
        WHERE len(s.w) BETWEEN 2 AND 3
          AND (s.mentions + COALESCE(p.n_whole, 0)) / (s.mentions + COALESCE(p.n_total, 0)) >= {a.min_whole_share}
          AND s.surface NOT IN (SELECT surface FROM known)
          AND s.w[-1] NOT IN ({sql_list(SUFFIX_TAILS)}) AND s.w[-1] NOT IN ({sql_list(VIETNAMESE_TAILS)})
          AND s.w[1] NOT IN ({sql_list(EAST_ASIAN_HEADS)}) AND s.w[1] NOT IN ({sql_list(CHINESE_COMPOUND_HEADS)})
          AND NOT {vietnamese}
          AND NOT EXISTS (SELECT 1 FROM unnest(s.w) AS u(x) WHERE length(u.x) < 2)
          AND (s.w[1] IN ({particles}) OR th.given_share <= {MAX_GIVEN_SHARE})
          AND NOT EXISTS (SELECT 1 FROM unnest(s.w[1:-2]) AS u(x) JOIN '{a.tokens}' t ON t.t = u.x
                          WHERE u.x NOT IN ({particles}) AND t.given_share > {MAX_GIVEN_SHARE})
        ORDER BY s.initials DESC, s.mentions DESC""").fetchall()  # noqa: S608
    rows = [r for r in rows if not all(east_asian_syllable(word) for word in r[0].split())]
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=fields)
    w.writeheader()
    w.writerows(kept)
    for surface, mentions, initials, share in rows:
        w.writerow({"surface": surface, "tier": TIER, "mentions": int(mentions), "whole_share": round(share, 2),
                    "given_names": int(initials)})
    with asset.open("wb") as raw, gzip.GzipFile(filename=asset.name.removesuffix(".gz"), mode="wb", fileobj=raw, mtime=0) as fh:
        fh.write(out.getvalue().encode("utf-8"))
    digest = hashlib.sha256(asset.read_bytes()).hexdigest()[:16]
    print(f"{len(kept):,} kept + {len(rows):,} {TIER} surfaces -> {asset} (sha256 {digest})")
    print("top by initials:", [r[0] for r in rows[:15]])


if __name__ == "__main__":
    main()
