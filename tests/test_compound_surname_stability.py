"""Compound-surname table refreshes: a pair changes status only on clear registry evidence, and an unlisted pair follows
the source's own field split."""

import csv
import gzip
import importlib.util
import io
import subprocess
import sys
from pathlib import Path

import pytest

from sinonym.timo.interface import Instance, Predictor, PredictorConfig, SourceAuthorFields

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "stabilize_compound_surnames.py"
_SPEC = importlib.util.spec_from_file_location("stabilize_compound_surnames", SCRIPT)
stabilize = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = stabilize
_SPEC.loader.exec_module(stabilize)


@pytest.mark.parametrize(
    "case",
    [
        # clear evidence decides whatever the shipped table said
        (30, 5, False, True, "keep", True),
        (10, 25, True, False, "keep", False),
        (10, 25, True, False, "abstain", False),
        # registrants about evenly split: the shipped status stays (keep) or the pair is dropped (abstain)
        (22, 22, True, False, "keep", True),
        (22, 22, False, True, "keep", False),
        (22, 22, True, False, "abstain", False),
        # boundaries: 0.67 of exactly MIN_SUPPORT registrants is clear; exactly 2/3 is not
        (14, 6, False, True, "keep", True),
        (6, 14, True, False, "keep", False),
        (20, 10, False, True, "keep", False),
        (13, 6, False, True, "keep", False),
        # too few registrants: the shipped status, not the rebuild's
        (8, 1, False, True, "keep", False),
        (1, 8, True, False, "abstain", True),
        # no shipped table: the rebuild decides below the support bar
        (8, 1, None, True, "keep", True),
    ],
)
def test_decide(case):
    join, tail, previous, candidate, ambiguous, expected = case
    policy = stabilize.Policy(ambiguous=ambiguous)
    assert stabilize.decide(join, tail, previous=previous, candidate=candidate, policy=policy) is expected


HEADER = ["surface", "tier", "mentions", "whole_share", "given_names", "people", "people_whole", "people_split_today"]


def _row(surface, tier, mentions="5"):
    return [surface, tier, mentions, "0.5", "2", "", "", ""]


def _write(path, rows):
    out = io.StringIO()
    csv.writer(out).writerows([HEADER, *rows])
    path.write_bytes(gzip.compress(out.getvalue().encode("utf-8"), mtime=0))


def test_main_keeps_shipped_rows_and_order(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    _write(
        tmp_path / "prev.csv.gz",
        [_row("da silva", "listed"), _row("silva silva", "hyphen"), _row("rodrigues gomes", "registry")],
    )
    _write(
        tmp_path / "cand.csv.gz",
        [
            _row("da silva", "listed", "9"),
            _row("rodrigues oliveira", "registry"),
            _row("pereira neves", "corpus"),
            _row("de jesus oliveira", "corpus"),
            _row("costa e silva", "corpus"),
        ],
    )
    duckdb.sql(
        """SELECT * FROM (VALUES ('da silva', 80, 20), ('silva silva', 8, 26), ('rodrigues gomes', 22, 22),
           ('rodrigues oliveira', 14, 11), ('pereira neves', 3, 7), ('de jesus oliveira', 15, 2),
           ('costa e silva', 30, 2)) v(surf, j, t)""",
    ).write_parquet(str(tmp_path / "stats.parquet"))
    args = [
        sys.executable,
        str(SCRIPT),
        "--candidate",
        str(tmp_path / "cand.csv.gz"),
        "--previous",
        str(tmp_path / "prev.csv.gz"),
        "--stats",
        str(tmp_path / "stats.parquet"),
        "--output",
        str(tmp_path / "out.csv.gz"),
    ]
    subprocess.run(args, check=True, capture_output=True)  # noqa: S603
    text = gzip.decompress((tmp_path / "out.csv.gz").read_bytes()).decode("utf-8")
    rows = list(csv.reader(io.StringIO(text)))
    # da silva keeps its shipped row; silva silva is clearly split; the ambiguous rodrigues gomes stays listed and the
    # ambiguous rodrigues oliveira stays out; pereira neves and de jesus oliveira have too few registrants to change;
    # costa e silva is clearly whole and is added after the shipped rows
    assert rows == [HEADER, _row("da silva", "listed"), _row("rodrigues gomes", "registry"), _row("costa e silva", "corpus")]
    assert text.endswith("\r\n")
    first = (tmp_path / "out.csv.gz").read_bytes()
    subprocess.run(args, check=True, capture_output=True)  # noqa: S603
    assert (tmp_path / "out.csv.gz").read_bytes() == first


@pytest.fixture(scope="module")
def predictor() -> Predictor:
    return Predictor(PredictorConfig(parallel="never"), ".")


def resolve(predictor: Predictor, first, middle, last):
    source = SourceAuthorFields(first_name=first or None, middle_names=middle or None, last_name=last)
    author = predictor.predict_batch([Instance(pp_authors=[source])])[0].authors[0]
    return author.first_name, author.middle_names, author.last_name


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        # registrants write the pair whole and split about equally and it is not listed: the source's split stands
        (("Ana", "Paula Rodrigues", "Oliveira"), ("Ana", "Paula Rodrigues", "Oliveira")),
        (("Ana Paula", None, "Rodrigues Oliveira"), ("Ana Paula", "", "Rodrigues Oliveira")),
        # an ambiguous pair the table already listed stays listed rather than flipping with the refresh
        (("Ana", "Paula Rodrigues", "Gomes"), ("Ana", "Paula", "Rodrigues Gomes")),
        (("Ana Paula", None, "Rodrigues Gomes"), ("Ana Paula", "", "Rodrigues Gomes")),
        # clear compounds are still joined
        (("Ana", "Maria da", "Silva"), ("Ana", "Maria", "da Silva")),
        (("Juan", "Carlos Gonzalez", "Lopez"), ("Juan", "Carlos", "Gonzalez Lopez")),
    ],
)
def test_compound_surname_follows_table(predictor, source, expected):
    assert resolve(predictor, *source) == expected
