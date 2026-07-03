#!/usr/bin/env python3
"""Generate position-conditional given-syllable counts from ACL 2025 authors.

Mines unambiguous two-syllable given names from ``sinonym/data/acl_2025_authors.txt``
and writes per-syllable initial/final position counts to
``sinonym/data/given_position_acl.csv``. The counts feed the positional given-name
feature used when scoring 3-token spaced parses (see NameParsingService).

A pair (g1, g2) is mined from a two-token "Given Surname" author entry when the
last token is a known Chinese surname, the given token is not itself a surname,
and the given token is either:
  * hyphenated into exactly two alphabetic parts ("Hao-Ran"), or
  * fused with exactly one split into two valid syllables ("Haoran" -> hao+ran).

Syllables are kept AS WRITTEN (lowercased, no Mandarin romanization remapping)
so that Cantonese/Wade-Giles forms like "ka"/"fai"/"ting" retain their own
positional statistics.

Usage:
    uv run python scripts/generate_given_position_data.py
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from sinonym import ChineseNameDetector
from sinonym.resources import read_text
from sinonym.utils.string_manipulation import StringManipulationUtils

OUTPUT_NAME = "given_position_acl.csv"


def _fused_split(token: str, syllables: frozenset[str]) -> tuple[str, str] | None:
    """Return the unique two-syllable split of a fused token, if exactly one exists."""
    splits = [(token[:i], token[i:]) for i in range(1, len(token)) if token[:i] in syllables and token[i:] in syllables]
    if len(splits) == 1:
        return splits[0]
    return None


def mine_pairs(detector: ChineseNameDetector) -> list[tuple[str, str]]:
    """Extract unambiguous (initial, final) given-syllable pairs from ACL authors."""
    data = detector._data
    normalizer = detector._normalizer
    syllables = data.plausible_components

    def is_surname(token: str) -> bool:
        return data.is_surname(
            token,
            StringManipulationUtils.remove_spaces(normalizer.norm(token)),
        )

    pairs: list[tuple[str, str]] = []
    for line in read_text("acl_2025_authors.txt").splitlines():
        tokens = line.strip().split()
        if len(tokens) != 2:
            continue
        given_tok, surname_tok = tokens
        if not is_surname(surname_tok) or is_surname(given_tok):
            continue

        given_lower = given_tok.lower()
        if "-" in given_lower:
            parts = given_lower.split("-")
            if len(parts) == 2 and all(p.isalpha() and p.isascii() for p in parts):
                pairs.append((parts[0], parts[1]))
        elif given_lower.isalpha() and given_lower.isascii():
            split = _fused_split(given_lower, syllables)
            if split is not None:
                pairs.append(split)

    return pairs


def main() -> None:
    detector = ChineseNameDetector()
    pairs = mine_pairs(detector)
    print(f"Mined {len(pairs)} (initial, final) given-syllable pairs")

    initial_counts: Counter[str] = Counter(g1 for g1, _ in pairs)
    final_counts: Counter[str] = Counter(g2 for _, g2 in pairs)

    out_path = Path(__file__).resolve().parent.parent / "sinonym" / "data" / OUTPUT_NAME
    vocabulary = sorted(set(initial_counts) | set(final_counts))
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["syllable", "initial_count", "final_count"])
        for syllable in vocabulary:
            writer.writerow([syllable, initial_counts.get(syllable, 0), final_counts.get(syllable, 0)])

    print(f"Wrote {len(vocabulary)} syllables to {out_path}")
    for label, counts in (("initial", initial_counts), ("final", final_counts)):
        top = ", ".join(f"{s}={c}" for s, c in counts.most_common(10))
        print(f"top {label}: {top}")


if __name__ == "__main__":
    main()
