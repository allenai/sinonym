#!/usr/bin/env python
"""Build ``sinonym/data/unsplit_given_words.csv.gz``: words the Chinese given-name splitter would cut into two pinyin
syllables (``Annie`` -> ``An-Nie``, ``Hoang`` -> ``Ho-Ang``, ``Jihye`` -> ``Jih-Ye``) that the corpus uses as one word.

Input: the source-name counts export (one row per distinct source ``first_name / middle_names / last_name`` triple with
``mention_count``, built from the names as published, never from sinonym's output). Every name is folded to lowercase
letters without accents and split on spaces. A word qualifies when
  - it occurs at least MIN_MENTIONS times as a whole word in any field,
  - fewer than MAX_CHINESE_CONTEXT_SHARE of those mentions are in a name that also contains a common mainland Chinese
    surname (``NAME_ORDER_ROUTING_COMMON_CHINESE_SURNAMES`` without ``lee``, which Korean and English names share), and
  - the split the splitter makes is written by the sources themselves (``an-nie`` as one hyphenated word, ``An Nie`` as
    two adjacent given words, ``AnNie``) less than MAX_SPLIT_SHARE as often as the word plus its split, both over all
    names and over the names that contain a common Chinese surname.
A word Chinese authors write (``Xiaoming``: 85% of mentions beside a common Chinese surname, ``Xiao-Ming`` 20% of the
time) fails these conditions and stays splittable, and so does a word that is mostly non-Chinese but that Chinese
authors write split (``Yudi``: 16% beside a Chinese surname, written ``Yu-Di`` 9% of the time there); a word the table
does not list keeps the splitter's behaviour.

Usage: build_unsplit_given_words.py --counts '<name_counts parquet glob>' --output sinonym/data/unsplit_given_words.csv.gz
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
from sinonym.utils import unsplit_given_words
from sinonym.utils.string_manipulation import StringManipulationUtils

MIN_MENTIONS = 100
MAX_CHINESE_CONTEXT_SHARE = 0.2
MAX_SPLIT_SHARE = 0.02
WORDS = "string_split(concat_ws(' ', first_name, middle_names, last_name), ' ')"
TOKENS = (
    f"list_transform(string_split(lower(strip_accents(array_to_string({WORDS}, ' '))), ' '), "
    "x -> regexp_replace(x, '[^a-z]', '', 'g'))"
)


def splits(words: list[str]) -> dict[str, tuple[str, str]]:
    """The two-part split the splitter makes for each word, ignoring the table this script builds."""
    unsplit_given_words._words = frozenset  # noqa: SLF001
    detector = ChineseNameDetector()
    detector._ensure_initialized()  # noqa: SLF001
    out = {}
    for word in words:
        parts = StringManipulationUtils.split_concatenated_name(
            word.capitalize(), None, detector._data, detector._normalizer, detector._config,  # noqa: SLF001
        )
        if parts and len(parts) == 2:  # noqa: PLR2004
            out[word] = (parts[0].lower(), parts[1].lower())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", required=True)
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
        WHERE regexp_full_match(w, '[a-z]{{3,}}') GROUP BY 1
        HAVING sum(n) >= {MIN_MENTIONS} AND coalesce(sum(n) FILTER (WHERE chinese), 0) < {MAX_CHINESE_CONTEXT_SHARE} * sum(n)""")
    candidates = splits([w for (w,) in con.execute("SELECT w FROM ctx ORDER BY w").fetchall()])
    con.execute("CREATE TABLE sp(w VARCHAR, form VARCHAR)")
    con.executemany("INSERT INTO sp VALUES (?, ?)", [(w, f"{x}-{y}") for w, (x, y) in candidates.items()])
    con.execute(f"""CREATE TABLE forms AS
      SELECT form, sum(c) c, coalesce(sum(c) FILTER (WHERE rid IN (SELECT rid FROM rowcs)), 0) c_cn FROM (
        SELECT rid, lower(strip_accents(t)) form, n c FROM (SELECT rid, unnest({WORDS}) t, n FROM src)
        WHERE regexp_full_match(t, '\\p{{L}}+-\\p{{L}}+')
        UNION ALL
        SELECT rid, lower(strip_accents(regexp_replace(t, '^(\\p{{Lu}}\\p{{Ll}}+)(\\p{{Lu}}\\p{{Ll}}+)$', '\\1-\\2'))), n
        FROM (SELECT rid, unnest({WORDS}) t, n FROM src) WHERE regexp_full_match(t, '\\p{{Lu}}\\p{{Ll}}+\\p{{Lu}}\\p{{Ll}}+')
        UNION ALL
        SELECT rid, lower(strip_accents(l[i] || '-' || l[i + 1])), n FROM (
          SELECT rid, string_split(trim(concat_ws(' ', first_name, middle_names)), ' ') l, n FROM src
          WHERE middle_names IS NOT NULL OR contains(trim(first_name), ' ')),
          LATERAL (SELECT unnest(range(1, len(l))) i)
        WHERE regexp_full_match(l[i], '\\p{{L}}+') AND regexp_full_match(l[i + 1], '\\p{{L}}+')
      ) WHERE form IN (SELECT form FROM sp) GROUP BY 1""")
    rows = con.execute(f"""
        SELECT ctx.w, ctx.mentions, sp.form, coalesce(forms.c, 0), round(ctx.chinese / ctx.mentions, 3)
        FROM ctx JOIN sp USING (w) LEFT JOIN forms USING (form)
        WHERE coalesce(forms.c, 0) < {MAX_SPLIT_SHARE} * (ctx.mentions + coalesce(forms.c, 0))
          AND (coalesce(forms.c_cn, 0) = 0 OR forms.c_cn < {MAX_SPLIT_SHARE} * (ctx.chinese + forms.c_cn))
        ORDER BY ctx.mentions DESC, ctx.w""").fetchall()
    out = Path(a.output)
    with gzip.open(out, "wt", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["word", "mentions", "split", "split_mentions", "chinese_context_share"])
        wr.writerows((w, int(m), form, int(c), share) for w, m, form, c, share in rows)
    print(f"{len(rows):,} words -> {out} (sha256 {hashlib.sha256(out.read_bytes()).hexdigest()[:16]})")


if __name__ == "__main__":
    main()
