"""Order-prior table refreshes: shipped entries stay unless clearly reversed; new entries pass the builder gates, plus
an optional margin."""

import gzip
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from sinonym.timo.interface import Instance, Predictor, PredictorConfig, SourceAuthorFields

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "stabilize_order_priors.py"
_SPEC = importlib.util.spec_from_file_location("stabilize_order_priors", SCRIPT)
stabilize = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = stabilize
_SPEC.loader.exec_module(stabilize)


def token(name, role, mentions, share):
    return {"token": name, "role": role, "mentions": str(mentions), "given_share": str(share)}


def pair(a, b, n_ab, n_ba):
    return {"a": a, "b": b, "n_ab": str(n_ab), "n_ba": str(n_ba), "authors_ab": "0", "authors_ba": "0"}


def test_shipped_token_missing_from_the_rebuild_stays():
    merged = stabilize.merge_tokens([token("aldo", "given", 900, 0.902)], [], 0.02)
    assert [row["token"] for row in merged] == ["aldo"]


@pytest.mark.parametrize(
    ("row", "added"),
    [
        (token("brenno", "given", 250, 0.95), True),
        (token("cudrow", "surname", 250, 0.05), True),
        # just over a gate: share or mentions
        (token("dallin", "given", 250, 0.905), False),
        (token("evrard", "surname", 250, 0.095), False),
        (token("fenwil", "surname", 199, 0.0), False),
    ],
)
def test_new_token_needs_a_margin(row, added):
    merged = stabilize.merge_tokens([], [row], 0.02)
    assert bool(merged) is added


def test_default_margin_adds_every_new_entry_at_the_builder_gates():
    rows = [token("dallin", "given", 250, 0.905), token("fenwil", "surname", 199, 0.0)]
    merged = stabilize.merge_tokens([], rows, stabilize.MARGIN)
    assert [row["token"] for row in merged] == ["dallin"]


@pytest.mark.parametrize(("share", "flips"), [(0.05, True), (0.095, False)])
def test_reversed_token_changes_only_on_clear_evidence(share, flips):
    merged = stabilize.merge_tokens([token("gorran", "given", 900, 0.93)], [token("gorran", "surname", 900, share)], 0.02)
    assert merged[0]["role"] == ("surname" if flips else "given")


def test_pairs_follow_the_same_policy():
    shipped = [pair("hanu", "lirro", 500, 50)]
    candidate = [pair("hanu", "lirro", 90, 600), pair("mesk", "olvan", 160, 40), pair("pell", "quess", 1000, 10)]
    merged = {(row["a"], row["b"]): row for row in stabilize.merge_pairs(shipped, candidate, 0.02)}
    assert merged[("hanu", "lirro")]["n_ba"] == "600"
    assert ("mesk", "olvan") not in merged
    assert ("pell", "quess") in merged


def test_command_line_writes_both_tables(tmp_path):
    for directory, tokens, pairs in (
        ("previous", [token("aldo", "given", 900, 0.902)], [pair("hanu", "lirro", 500, 50)]),
        ("candidate", [token("brenno", "given", 250, 0.95)], []),
    ):
        (tmp_path / directory).mkdir()
        stabilize.write_table(tmp_path / directory / stabilize.TOKEN_TABLE, list(tokens[0]), tokens)
        stabilize.write_table(
            tmp_path / directory / stabilize.PAIR_TABLE,
            list(pairs[0] if pairs else pair("x", "y", 0, 0)),
            pairs,
        )
    (tmp_path / "out").mkdir()
    subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(SCRIPT),
            "--previous-dir",
            str(tmp_path / "previous"),
            "--candidate-dir",
            str(tmp_path / "candidate"),
            "--output-dir",
            str(tmp_path / "out"),
        ],
        check=True,
        capture_output=True,
    )
    with gzip.open(tmp_path / "out" / stabilize.TOKEN_TABLE, "rt") as handle:
        assert handle.read().splitlines()[1:] == ["aldo,given,900,0.902", "brenno,given,250,0.95"]


@pytest.fixture(scope="module")
def predictor() -> Predictor:
    return Predictor(PredictorConfig(parallel="never"), ".")


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (("Petralia", None, "Salvatore"), ("Salvatore", "Petralia")),
        (("Boudaoud", None, "Souhila"), ("Souhila", "Boudaoud")),
        (("Franić", None, "Josip"), ("Josip", "Franić")),
    ],
)
def test_shipped_order_entries_still_turn_reversed_names(predictor, source, expected):
    fields = SourceAuthorFields(first_name=source[0], middle_names=source[1], last_name=source[2])
    author = predictor.predict_batch([Instance(pp_authors=[fields])])[0].authors[0]
    assert (author.first_name, author.last_name) == expected
