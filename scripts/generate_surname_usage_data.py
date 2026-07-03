#!/usr/bin/env python3
# ruff: noqa: PLR2004, SLF001
"""Generate surname/given position usage counts from ACL 2025 authors.

Mines per-syllable surname-position and given-position counts from
``sinonym/data/acl_2025_authors.txt`` and writes them to
``sinonym/data/surname_usage_acl.csv``. The counts feed the surname-position
usage log-odds feature used when scoring 1+1 parses (see NameParsingService).

Author entries are assumed to be in Western order (given names first, surname
last), which holds for the large majority of entries. Syllable keys are passed
through the same ``norm_light`` normalization the scorer uses at lookup time,
so diacritic romanizations ("Köksal") land on the key the scorer queries
("koksal") instead of a dead accented key.

Usage:
    uv run python scripts/generate_surname_usage_data.py
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from sinonym import ChineseNameDetector
from sinonym.resources import read_text

OUTPUT_NAME = "surname_usage_acl.csv"


def mine_position_counts(detector: ChineseNameDetector) -> tuple[Counter[str], Counter[str]]:
    """Count surname-position and given-position occurrences per normalized syllable."""
    norm_light = detector._normalizer.norm_light
    surname_counts: Counter[str] = Counter()
    given_counts: Counter[str] = Counter()

    for line in read_text("acl_2025_authors.txt").splitlines():
        tokens = [t.lower().strip(".,'") for t in line.split()]
        if not 2 <= len(tokens) <= 3:
            continue
        if not all(t.isalpha() or "-" in t for t in tokens):
            continue
        if "-" not in tokens[-1]:
            surname_counts[norm_light(tokens[-1])] += 1
        for token in tokens[:-1]:
            for part in token.split("-"):
                if part.isalpha():
                    given_counts[norm_light(part)] += 1

    return surname_counts, given_counts


def main() -> None:
    detector = ChineseNameDetector()
    surname_counts, given_counts = mine_position_counts(detector)
    print(f"Mined {sum(surname_counts.values())} surname and {sum(given_counts.values())} given occurrences")

    out_path = Path(__file__).resolve().parent.parent / "sinonym" / "data" / OUTPUT_NAME
    vocabulary = sorted(surname_counts.keys() | given_counts.keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["syllable", "surname_count", "given_count"])
        for syllable in vocabulary:
            writer.writerow([syllable, surname_counts.get(syllable, 0), given_counts.get(syllable, 0)])

    print(f"Wrote {len(vocabulary)} syllables to {out_path}")
    for label, counts in (("surname", surname_counts), ("given", given_counts)):
        top = ", ".join(f"{s}={c}" for s, c in counts.most_common(10))
        print(f"top {label}: {top}")


if __name__ == "__main__":
    main()
