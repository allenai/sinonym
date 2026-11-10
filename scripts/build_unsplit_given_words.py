#!/usr/bin/env python
"""Build ``sinonym/data/unsplit_given_words.csv.gz``: words the Chinese given-name splitter would cut into two pinyin
syllables (``Annie`` -> ``An-Nie``, ``Dana`` -> ``Da-Na``, ``Jihye`` -> ``Jih-Ye``) that are not Chinese given names.

Inputs (the counts export is built from the Semantic Scholar corpus and is not public):
  --counts    the source-name counts export (one row per distinct source ``first_name / middle_names / last_name``
              triple with ``mention_count``, built from the names as published, never from sinonym's output);
  --registry  the ORCID public data file as Parquet with ``orcid``, ``given_names`` and ``countries``.

Evidence is about the given word itself. A word the splitter would split is listed when
  - at least MIN_REGISTRY_PEOPLE ORCID people with a country write it in their given name (whole or split), fewer than
    MAX_CHINESE_COUNTRY_SHARE of them are in China, Taiwan, Hong Kong, Macau or Singapore, and fewer than
    MAX_CHINESE_CONTEXT_SHARE of its corpus mentions are in a name with a common mainland Chinese surname
    (``NAME_ORDER_ROUTING_COMMON_CHINESE_SURNAMES`` without ``lee``); or
  - the registry has too few such people, but the word has at least MIN_MENTIONS corpus mentions, fewer than
    MAX_UNKNOWN_CONTEXT_SHARE of them beside a common Chinese surname, and the sources write the split itself
    (``an-nie``, ``An Nie``, ``AnNie``) less than MAX_SPLIT_SHARE as often as the word.
``Xiaoming`` (79% of registry people in a Chinese country) and ``Bangti`` (one registry person, 99.6% of its mentions
beside a Chinese surname) are not listed. Reduplicated words (``Nini``, ``Sasa``) are never listed: on ORCID people the
split is as often right as wrong. A word the table does not list keeps the splitter's behaviour.

Usage: build_unsplit_given_words.py --counts '<name counts parquet glob>' --registry '<registry parquet glob>'
       --output sinonym/data/unsplit_given_words.csv.gz
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
from pathlib import Path

import duckdb

from sinonym.chinese_names_data import NAME_ORDER_ROUTING_COMMON_CHINESE_SURNAMES
from sinonym.detector import ChineseNameDetector
from sinonym.utils import string_manipulation, unsplit_given_words
from sinonym.utils.string_manipulation import StringManipulationUtils

MIN_REGISTRY_PEOPLE = 5
MAX_CHINESE_COUNTRY_SHARE = 0.4
MAX_CHINESE_CONTEXT_SHARE = 0.5
MIN_MENTIONS = 100
MAX_UNKNOWN_CONTEXT_SHARE = 0.2
MAX_SPLIT_SHARE = 0.02
MIN_CANDIDATE_MENTIONS = 5
CHINESE_COUNTRIES = "(^|[^A-Z])(CN|TW|HK|MO|SG)([^A-Z]|$)"
WORDS = "string_split(concat_ws(' ', first_name, middle_names, last_name), ' ')"
TOKENS = (
    f"list_transform(string_split(lower(strip_accents(array_to_string({WORDS}, ' '))), ' '), "
    "x -> regexp_replace(x, '[^a-z]', '', 'g'))"
)


def splits(words: list[str]) -> dict[str, tuple[str, str]]:
    """The two-part split the splitter makes for each word, ignoring the table this script builds."""
    unsplit_given_words._words = frozenset  # noqa: SLF001
    string_manipulation.is_unsplit_given_word = lambda _token: False
    detector = ChineseNameDetector()
    detector._ensure_initialized()  # noqa: SLF001
    out = {}
    for word in words:
        parts = StringManipulationUtils.split_concatenated_name(
            word.capitalize(), None, detector._data, detector._normalizer, detector._config,  # noqa: SLF001
        )
        if parts and len(parts) == 2 and parts[0].lower() != parts[1].lower():  # noqa: PLR2004
            out[word] = (parts[0].lower(), parts[1].lower())
    return out


def registry_words(con: duckdb.DuckDBPyConnection, registry: str) -> None:
    """Per folded given word: registry people with a country who write it whole or split, and how many are Chinese."""
    con.execute(f"""CREATE TABLE reg AS SELECT orcid, regexp_matches(coalesce(countries, ''), '{CHINESE_COUNTRIES}') cn,
        list_filter(string_split(regexp_replace(lower(strip_accents(coalesce(given_names, ''))), '[^a-z\\- ]', ' ', 'g'), ' '),
                    x -> x <> '') t
        FROM read_parquet('{registry}') WHERE coalesce(countries, '') <> ''""")
    con.execute("""CREATE TABLE regw AS SELECT w, count(DISTINCT orcid) people, count(DISTINCT orcid) FILTER (WHERE cn) cn_people
        FROM (
          SELECT orcid, cn, replace(w, '-', '') w FROM (SELECT orcid, cn, unnest(t) w FROM reg)
          WHERE regexp_full_match(w, '[a-z]+(-[a-z]+)?')
          UNION ALL
          SELECT orcid, cn, t[i] || t[i + 1] FROM reg, LATERAL (SELECT unnest(range(1, len(t))) i)
          WHERE NOT contains(t[i], '-') AND NOT contains(t[i + 1], '-')
        ) WHERE length(w) >= 3 GROUP BY 1""")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE src AS SELECT row_number() OVER () rid, first_name, middle_names, last_name, mention_count n "
        f"FROM read_parquet('{a.counts}')",
    )
    con.execute(f"CREATE TABLE rt AS SELECT rid, n, unnest(toks) w FROM (SELECT rid, n, {TOKENS} toks FROM src)")
    con.execute("CREATE TABLE cs(s VARCHAR)")
    con.executemany("INSERT INTO cs VALUES (?)", [(s,) for s in sorted(NAME_ORDER_ROUTING_COMMON_CHINESE_SURNAMES - {"lee"})])
    con.execute("CREATE TABLE rowcs AS SELECT DISTINCT rid FROM rt JOIN cs ON cs.s = rt.w")
    con.execute(f"""CREATE TABLE ctx AS SELECT w, sum(n) mentions, coalesce(sum(n) FILTER (WHERE chinese), 0) chinese
        FROM (SELECT rt.w, rt.n, rowcs.rid IS NOT NULL chinese FROM rt LEFT JOIN rowcs USING (rid))
        WHERE regexp_full_match(w, '[a-z]{{3,}}') GROUP BY 1 HAVING sum(n) >= {MIN_CANDIDATE_MENTIONS}""")
    registry_words(con, a.registry)
    con.execute(f"""CREATE TABLE cand AS SELECT ctx.*, coalesce(regw.people, 0) reg_people, coalesce(regw.cn_people, 0) reg_cn
        FROM ctx LEFT JOIN regw USING (w)
        WHERE ctx.chinese < {MAX_CHINESE_CONTEXT_SHARE} * ctx.mentions
          AND (coalesce(regw.people, 0) >= {MIN_REGISTRY_PEOPLE} OR ctx.mentions >= {MIN_MENTIONS})""")
    candidates = splits([w for (w,) in con.execute("SELECT w FROM cand ORDER BY w").fetchall()])
    con.execute("CREATE TABLE sp(w VARCHAR, form VARCHAR)")
    con.executemany("INSERT INTO sp VALUES (?, ?)", [(w, f"{x}-{y}") for w, (x, y) in candidates.items()])
    con.execute(f"""CREATE TABLE forms AS SELECT form, sum(c) c FROM (
        SELECT lower(strip_accents(t)) form, n c FROM (SELECT unnest({WORDS}) t, n FROM src)
        WHERE regexp_full_match(t, '\\p{{L}}+-\\p{{L}}+')
        UNION ALL
        SELECT lower(strip_accents(regexp_replace(t, '^(\\p{{Lu}}\\p{{Ll}}+)(\\p{{Lu}}\\p{{Ll}}+)$', '\\1-\\2'))), n
        FROM (SELECT unnest({WORDS}) t, n FROM src) WHERE regexp_full_match(t, '\\p{{Lu}}\\p{{Ll}}+\\p{{Lu}}\\p{{Ll}}+')
        UNION ALL
        SELECT lower(strip_accents(l[i] || '-' || l[i + 1])), n FROM (
          SELECT string_split(trim(concat_ws(' ', first_name, middle_names)), ' ') l, n FROM src
          WHERE middle_names IS NOT NULL OR contains(trim(first_name), ' ')),
          LATERAL (SELECT unnest(range(1, len(l))) i)
        WHERE regexp_full_match(l[i], '\\p{{L}}+') AND regexp_full_match(l[i + 1], '\\p{{L}}+')
      ) WHERE form IN (SELECT form FROM sp) GROUP BY 1""")
    rows = con.execute(f"""
        SELECT cand.w, cand.mentions, sp.form, coalesce(forms.c, 0), round(cand.chinese / cand.mentions, 3),
               cand.reg_people, round(cand.reg_cn / nullif(cand.reg_people, 0), 3)
        FROM cand JOIN sp USING (w) LEFT JOIN forms USING (form)
        WHERE (cand.reg_people >= {MIN_REGISTRY_PEOPLE} AND cand.reg_cn < {MAX_CHINESE_COUNTRY_SHARE} * cand.reg_people)
           OR (cand.reg_people < {MIN_REGISTRY_PEOPLE} AND cand.mentions >= {MIN_MENTIONS}
               AND cand.chinese < {MAX_UNKNOWN_CONTEXT_SHARE} * cand.mentions
               AND coalesce(forms.c, 0) < {MAX_SPLIT_SHARE} * (cand.mentions + coalesce(forms.c, 0)))
        ORDER BY cand.mentions DESC, cand.w""").fetchall()
    out = Path(a.output)
    with gzip.open(out, "wt", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["word", "mentions", "split", "split_mentions", "chinese_context_share", "registry_people",
                     "registry_chinese_country_share"])
        wr.writerows(rows)
    print(f"{len(rows):,} words -> {out} (sha256 {hashlib.sha256(out.read_bytes()).hexdigest()[:16]})")


if __name__ == "__main__":
    main()
