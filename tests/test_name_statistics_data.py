# ruff: noqa: SLF001
"""Invariants for the corpus-derived positional asset.

given_position.csv is generated offline by scripts/generate_name_statistics.py
from the wainshine 120W Han-name corpus plus the in-house byline hyphen-mining
supplement; the scorer looks syllables up through norm_light. These tests guard
the generator/lookup contract and the measured properties the 3-token feature
operating point was calibrated against.
"""

from sinonym.resources import open_csv_reader


def test_given_position_keys_match_lookup_normalization(detector):
    """Every positional key must be a norm_light fixed point (scorer-reachable)."""
    data = detector._data
    keys = set(data.given_initial_log_probabilities) | set(data.given_final_log_probabilities)
    assert len(keys) > 300
    unstable = [key for key in keys if detector._normalizer.norm_light(key) != key]
    assert unstable == []


def test_given_position_includes_byline_supplement():
    """Non-pinyin romanizations must be covered via the byline supplement rows."""
    rows = {row["syllable"]: row for row in open_csv_reader("given_position.csv")}
    assert any(row["source"] == "byline" for row in rows.values())
    # kung cannot be produced by pypinyin; it comes from the byline supplement
    assert rows["kung"]["source"] == "byline"


def test_given_position_scale(detector):
    """Positional counts must stay at corpus scale (deadband calibrated to it)."""
    total = sum(int(row["initial_count"]) for row in open_csv_reader("given_position.csv"))
    assert total == 1_927_487
