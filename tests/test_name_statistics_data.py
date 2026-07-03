# ruff: noqa: PLR2004, SLF001
"""Invariants for the corpus-derived positional and bigram assets.

given_position.csv and given_bigrams.csv are generated offline by
scripts/generate_name_statistics.py from the wainshine 120W Han-name corpus
plus the in-house byline hyphen-mining supplement; the scorer looks syllables
up through norm_light. These tests guard the generator/lookup contract and the
measured properties the 3-token feature operating points were calibrated
against.
"""

import math

from sinonym.resources import open_csv_reader
from sinonym.services.parsing import DEFAULT_WEIGHTS, GIVEN_BIGRAM_MAGNITUDE_CAP


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
    assert total > 900_000


def test_given_bigram_keys_match_lookup_normalization(detector):
    """Every bigram key must be a norm_light fixed point (scorer-reachable)."""
    counts = detector._data.given_bigram_counts
    assert len(counts) > 15_000
    norm_light = detector._normalizer.norm_light
    unstable = [pair for pair in counts if norm_light(pair[0]) != pair[0] or norm_light(pair[1]) != pair[1]]
    assert unstable == []


def test_given_bigram_counts_are_order_sensitive(detector):
    """The bigram table must keep decisive ordered-pair asymmetry (jian+feng)."""
    data = detector._data
    forward = data.get_given_bigram_count("jian", "feng")
    backward = data.get_given_bigram_count("feng", "jian")
    assert forward >= 500
    assert math.log((forward + 2) / (backward + 2)) > 3.0


def test_given_bigram_feature_swing_stays_below_batch_vote_margin():
    """Cap times weight must keep the max swing under the 0.261 vote margin."""
    assert 2 * GIVEN_BIGRAM_MAGNITUDE_CAP * DEFAULT_WEIGHTS[10] < 0.261
