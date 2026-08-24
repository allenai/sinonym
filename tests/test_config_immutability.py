"""Regression tests for immutable, process-safe detector configuration."""

from __future__ import annotations

import operator
import pickle

import pytest

from sinonym.coretypes import ChineseNameConfig
from sinonym.name_punctuation import NAME_JOINER_DELETE_TRANSLATION


def test_default_translation_tables_are_read_only_mappings():
    """Mapping and dict-level mutation attempts cannot contaminate defaults."""
    first = ChineseNameConfig.create_default()
    second = ChineseNameConfig.create_default()
    before = dict(NAME_JOINER_DELETE_TRANSLATION)
    table = first.hyphens_apostrophes_tr

    with pytest.raises(TypeError):
        operator.setitem(table, ord("!"), None)
    with pytest.raises(TypeError):
        dict.__setitem__(table, ord("!"), None)
    with pytest.raises(TypeError):
        operator.setitem(table._values, ord("!"), None)  # noqa: SLF001 - verify the backing store is read-only

    assert all(not hasattr(table, method) for method in ("clear", "pop", "popitem", "setdefault", "update"))
    assert dict(table) == before
    assert dict(second.hyphens_apostrophes_tr) == before
    assert before == NAME_JOINER_DELETE_TRANSLATION
    assert table is not second.hyphens_apostrophes_tr


def test_config_translation_tables_survive_pickle_and_replace():
    """Multiprocessing and immutable updates retain the frozen table contract."""
    config = ChineseNameConfig.create_default()
    round_trip = pickle.loads(pickle.dumps(config))  # noqa: S301 - trusted in-memory config round trip
    updated = config.with_log_probabilities(-12.0, -13.0)

    assert round_trip == config
    assert type(round_trip.hyphens_apostrophes_tr) is type(config.hyphens_apostrophes_tr)
    assert type(round_trip.roman_punctuation_fold_tr) is type(config.roman_punctuation_fold_tr)
    assert updated.hyphens_apostrophes_tr is config.hyphens_apostrophes_tr
    assert updated.roman_punctuation_fold_tr is config.roman_punctuation_fold_tr
    sample = "Cui-e O'Yang"
    assert sample.translate(round_trip.hyphens_apostrophes_tr) == sample.translate(NAME_JOINER_DELETE_TRANSLATION)

    with pytest.raises(TypeError):
        round_trip.roman_punctuation_fold_tr[ord("!")] = "'"
