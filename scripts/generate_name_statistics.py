#!/usr/bin/env python3
# ruff: noqa: PLR2004, SLF001, PLC0415
"""Generate corpus-derived given-name statistics assets for sinonym/data.

Produces two CSVs consumed by ``DataInitializationService``:

* ``given_position.csv``  - per-syllable initial/final counts inside two-syllable
  given names. Feeds the positional feature for 3-token parses.
* ``given_bigrams.csv``   - ordered (initial, final) pair counts for two-syllable
  given names. Feeds the ordered-vs-reversed bigram feature for 3-token parses.

(The surname-position usage table, surname_usage_acl.csv, intentionally stays
ACL-mined: replacing it with corpus-scale marginal counts was measured to be
incompatible with the current 1+1 mirror expectations - see the data README.)

Sources
-------
1. wainshine/Chinese-Names-Corpus 120W (Apache-2.0), ~1.14M Han full names.
   The raw 12MB corpus is NOT committed; download it once with:

       curl -L -o Chinese_Names_Corpus_120W.txt \
         "https://github.com/wainshine/Chinese-Names-Corpus/raw/refs/heads/master/Chinese_Names_Corpus/Chinese_Names_Corpus%EF%BC%88120W%EF%BC%89.txt"

   Names are split deterministically against the familyname_orcid.csv inventory
   (2-char compound surname first, then 1-char) and romanized per character with
   pypinyin plus conventional surname-reading fixes.
2. Optional in-house byline supplement (``--parquet``): two-token romanized
   author names with exactly one hyphenated token, mined from an author-mention
   parquet (columns ``name``, ``script``). The hyphen convention (the hyphenated
   token is the given name) yields a surname/given split without running the
   parser, so the counts are independent of sinonym's own decisions. Only
   curated compound-surname variants ("Au-Yeung", "Sze-To", ...) are dropped;
   fused given names that collide with the surname inventory ("Hao-Ran") are
   kept. Byline counts are rescaled to the 120W pair total (so both sources
   carry comparable probability mass) and POOLED with the corpus counts: this
   both covers spellings pypinyin cannot produce ("kung", "pao") and keeps
   real byline conventions alive for syllables that are rare in the Han corpus
   but common in romanized bylines ("sang" as a given initial).

Thresholds (documented in sinonym/data/README.md):
* bigrams: pooled pairs need count >= 5; byline-only pairs need >= 3 distinct names.
* given position: byline-only syllables need >= 10 distinct names across positions.

Usage:
    uv run --with pyarrow python scripts/generate_name_statistics.py \
        --corpus "scratch/corpus_scoping/Chinese_Names_Corpus_120W.txt" \
        --parquet "scratch/baseline-sinonym-exp/inputs/batches_pp.parquet"
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pypinyin

from sinonym import ChineseNameDetector
from sinonym.chinese_names_data import COMPOUND_VARIANTS
from sinonym.resources import open_csv_reader
from sinonym.services.parsing import CURATED_COMPOUND_TARGETS
from sinonym.utils.string_manipulation import StringManipulationUtils

DATA_DIR = Path(__file__).resolve().parent.parent / "sinonym" / "data"

BIGRAM_MIN_COUNT = 5
BYLINE_BIGRAM_MIN_DISTINCT = 3
BYLINE_POSITION_MIN_DISTINCT = 10

# Conventional surname readings where pypinyin's default differs
# (mirrors PYPINYIN_FREQUENCY_ALIASES, applied at romanization time).
SURNAME_READING_FIX = {
    "曾": "zeng",
    "单": "shan",
    "解": "xie",
    "仇": "qiu",
    "区": "ou",
    "查": "zha",
    "繆": "miao",
}

HYPHEN_NAME_PATTERN = re.compile(
    r"^([A-Za-z]+) ([A-Za-z]+-[A-Za-z]+)$|^([A-Za-z]+-[A-Za-z]+) ([A-Za-z]+)$",
)


@dataclass
class PositionStats:
    """Positional and pair counts from one source corpus."""

    initial: Counter[str] = field(default_factory=Counter)
    final: Counter[str] = field(default_factory=Counter)
    bigrams: Counter[tuple[str, str]] = field(default_factory=Counter)


def mine_corpus(corpus_path: Path, norm_light) -> PositionStats:
    """Mine the 120W Han-name corpus with deterministic surname splits."""
    surname_inventory = {row["surname"] for row in open_csv_reader("familyname_orcid.csv")}

    pinyin_cache: dict[str, str] = {}

    def romanize(ch: str) -> str:
        syllable = pinyin_cache.get(ch)
        if syllable is None:
            syllable = norm_light(pypinyin.lazy_pinyin(ch, style=pypinyin.Style.NORMAL)[0])
            pinyin_cache[ch] = syllable
        return syllable

    stats = PositionStats()
    n_total = n_used = 0
    lines = corpus_path.read_text(encoding="utf-8-sig").splitlines()[3:]  # skip corpus header
    for line in lines:
        name = line.strip()
        if not (2 <= len(name) <= 4) or not name.isalpha():
            continue
        n_total += 1
        if name[:2] in surname_inventory and len(name) >= 3:
            given = name[2:]
        elif name[0] in surname_inventory:
            given = name[1:]
        else:
            continue
        n_used += 1
        if len(given) == 2:
            g1, g2 = romanize(given[0]), romanize(given[1])
            stats.initial[g1] += 1
            stats.final[g2] += 1
            stats.bigrams[(g1, g2)] += 1

    print(
        f"120W corpus: {n_total:,} Han names, {n_used:,} split "
        f"({n_total - n_used:,} skipped, no inventory surname)",
    )
    print(
        f"  2-syllable given pairs={sum(stats.bigrams.values()):,}  "
        f"distinct bigrams={len(stats.bigrams):,}",
    )
    return stats


def _resolve_byline_name(name, is_surname, is_curated_compound, syllables) -> tuple[str, str] | None:
    """Return the (initial, final) given parts of a resolvable byline name."""
    match = HYPHEN_NAME_PATTERN.match(name)
    if match is None:
        return None
    if match.group(1) is not None:
        surname_token, given_token = match.group(1), match.group(2)  # "Li Wen-Gang"
    else:
        given_token, surname_token = match.group(3), match.group(4)  # "Wen-Gang Li"
    part1, part2 = given_token.lower().split("-")
    if not is_surname(surname_token) or part1 not in syllables or part2 not in syllables:
        return None
    # Drop only curated compound-surname variants (Au-Yeung Ka, ...); fused
    # given names that collide with the surname inventory (Hao-Ran) stay.
    if is_curated_compound(given_token.lower()):
        return None
    return part1, part2


def mine_byline(parquet_path: Path, detector: ChineseNameDetector) -> PositionStats:
    """Mine hyphen-convention surname/given splits from an author-mention parquet.

    Distinct-name weighted: every distinct byline string counts once, avoiding
    prolific-author bias. Uses only inventory membership and the hyphen
    orthographic convention - never the parser's own split decisions.
    """
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    data = detector._data
    normalizer = detector._normalizer
    syllables = data.plausible_components

    def is_surname(token: str) -> bool:
        return data.is_surname(token, StringManipulationUtils.remove_spaces(normalizer.norm(token)))

    def is_curated_compound(token_lower: str) -> bool:
        if token_lower in COMPOUND_VARIANTS:
            return True
        return data.compound_hyphen_map.get(token_lower) in CURATED_COMPOUND_TARGETS

    distinct_names: set[str] = set()
    scanner = ds.dataset(parquet_path).scanner(
        columns=["name"],
        filter=pc.field("script") == "latin",
        batch_size=1_000_000,
    )
    for batch in scanner.to_batches():
        column = batch.column(0)
        mask = pc.match_substring_regex(column, HYPHEN_NAME_PATTERN.pattern)
        distinct_names.update(column.filter(mask).to_pylist())
    print(f"byline parquet: {len(distinct_names):,} distinct two-token names with one hyphenated token")

    stats = PositionStats()
    kept = 0
    for name in distinct_names:
        parts = _resolve_byline_name(name, is_surname, is_curated_compound, syllables)
        if parts is None:
            continue
        kept += 1
        g1, g2 = normalizer.norm_light(parts[0]), normalizer.norm_light(parts[1])
        stats.initial[g1] += 1
        stats.final[g2] += 1
        stats.bigrams[(g1, g2)] += 1

    print(f"  kept {kept:,} distinct resolved names, {len(stats.bigrams):,} distinct given bigrams")
    return stats


def _scale(count: int, factor: float) -> int:
    return max(1, round(count * factor))


def emit_given_position(corpus: PositionStats, byline: PositionStats | None) -> None:
    """Emit given_position.csv: pooled initial/final counts inside 2-syllable given names."""
    rows: list[tuple[str, int, int, str]] = []
    factor = sum(corpus.bigrams.values()) / sum(byline.bigrams.values()) if byline is not None else 0.0
    corpus_vocabulary = set(corpus.initial) | set(corpus.final)
    byline_vocabulary = (set(byline.initial) | set(byline.final)) if byline is not None else set()
    for syllable in sorted(corpus_vocabulary | byline_vocabulary):
        in_corpus = syllable in corpus_vocabulary
        i_byline = byline.initial.get(syllable, 0) if byline is not None else 0
        f_byline = byline.final.get(syllable, 0) if byline is not None else 0
        if not in_corpus and i_byline + f_byline < BYLINE_POSITION_MIN_DISTINCT:
            continue
        i_count = corpus.initial.get(syllable, 0) + (_scale(i_byline, factor) if i_byline else 0)
        f_count = corpus.final.get(syllable, 0) + (_scale(f_byline, factor) if f_byline else 0)
        source = ("corpus+byline" if i_byline + f_byline else "corpus") if in_corpus else "byline"
        rows.append((syllable, i_count, f_count, source))

    out_path = DATA_DIR / "given_position.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["syllable", "initial_count", "final_count", "source"])
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")


def emit_given_bigrams(corpus: PositionStats, byline: PositionStats | None) -> None:
    """Emit given_bigrams.csv: pooled ordered (initial, final) given-pair counts."""
    rows: list[tuple[str, str, int, str]] = []
    factor = sum(corpus.bigrams.values()) / sum(byline.bigrams.values()) if byline is not None else 0.0
    byline_pairs = set(byline.bigrams) if byline is not None else set()
    for pair in sorted(set(corpus.bigrams) | byline_pairs):
        corpus_count = corpus.bigrams.get(pair, 0)
        byline_count = byline.bigrams.get(pair, 0) if byline is not None else 0
        if corpus_count == 0 and byline_count < BYLINE_BIGRAM_MIN_DISTINCT:
            continue
        total = corpus_count + (_scale(byline_count, factor) if byline_count else 0)
        if total < BIGRAM_MIN_COUNT:
            continue
        source = ("corpus+byline" if byline_count else "corpus") if corpus_count else "byline"
        rows.append((pair[0], pair[1], total, source))

    out_path = DATA_DIR / "given_bigrams.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["initial", "final", "count", "source"])
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")


def print_verification_targets(corpus: PositionStats) -> None:
    """Print the pairs whose measured values gate the parser features."""
    print("\nverification targets (bigrams):")
    for pair in (("jian", "feng"), ("feng", "jian"), ("wen", "xing"), ("xing", "wen")):
        print(f"  {pair[0]}+{pair[1]}: {corpus.bigrams.get(pair, 0):,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--corpus",
        required=True,
        help="path to Chinese_Names_Corpus_120W.txt (see module docstring for the download command)",
    )
    parser.add_argument(
        "--parquet",
        default=None,
        help="optional author-mention parquet (columns: name, script) for the byline supplement",
    )
    args = parser.parse_args()

    detector = ChineseNameDetector()
    corpus = mine_corpus(Path(args.corpus), detector._normalizer.norm_light)
    byline = mine_byline(Path(args.parquet), detector) if args.parquet else None
    if byline is None:
        print("no --parquet given: emitting corpus-only tables (committed assets include the byline supplement)")

    emit_given_position(corpus, byline)
    emit_given_bigrams(corpus, byline)
    print_verification_targets(corpus)


if __name__ == "__main__":
    main()
