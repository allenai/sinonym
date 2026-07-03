# ruff: noqa: PLR2004, SLF001
"""Invariants for the surname-position usage data (surname_usage_acl.csv).

The counts are mined offline by scripts/generate_surname_usage_data.py; the
scorer looks syllables up through norm_light. These tests guard the contract
between the two: every mined key must be reachable by the lookup normalization.
"""

from sinonym.resources import open_csv_reader


def test_surname_usage_keys_match_lookup_normalization(detector):
    """Every key in the loaded log-odds map must be a norm_light fixed point.

    A key that norm_light rewrites (e.g. a raw diacritic form like 'köksal')
    can never be queried by the scorer and silently drops its corpus counts.
    """
    usage = detector._data.surname_usage_logodds
    assert len(usage) > 500
    unstable = [key for key in usage if detector._normalizer.norm_light(key) != key]
    assert unstable == []


def test_surname_usage_counts_include_diacritic_romanizations(detector):
    """Diacritic corpus occurrences must fold onto the key the scorer queries."""
    rows = {row["syllable"]: row for row in open_csv_reader("surname_usage_acl.csv")}
    koksal = rows[detector._normalizer.norm_light("Köksal")]
    assert int(koksal["surname_count"]) >= 1
