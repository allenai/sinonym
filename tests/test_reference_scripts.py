"""Regression tests for retained reference and historical scripts."""

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sinonym import ChineseNameDetector

REPO_ROOT = Path(__file__).parents[1]


def test_historical_generators_pass_spaced_compound_context() -> None:
    detector = ChineseNameDetector()
    normalized = detector._normalizer.apply("Wei Zhang")  # noqa: SLF001

    corpus_script = runpy.run_path(REPO_ROOT / "scripts/generate_chinese_name_corpus_data.py")
    candidates = corpus_script["generate_parse_candidates"](
        list(normalized.roman_tokens),
        detector._parsing_service,  # noqa: SLF001
        normalized.norm_map,
        normalized.compound_metadata,
    )
    assert candidates

    acl_script = runpy.run_path(REPO_ROOT / "scripts/generate_acl_data.py")
    examples = acl_script["convert_to_training_format"](
        [{"original": "Wei Zhang", "tokens": ["Wei", "Zhang"]}],
        detector,
    )
    assert len(examples) == 1
    assert examples[0]["parses"]


def test_historical_corpus_generator_surfaces_parser_failures() -> None:
    class FailingParser:
        def _generate_all_parses_with_format(self, *_args):
            message = "parser failed"
            raise RuntimeError(message)

    corpus_script = runpy.run_path(REPO_ROOT / "scripts/generate_chinese_name_corpus_data.py")

    with pytest.raises(RuntimeError, match="parser failed"):
        corpus_script["generate_parse_candidates"](
            ["Wei", "Zhang"],
            FailingParser(),
            {},
            {},
        )


@pytest.mark.parametrize(
    "case",
    [
        ((1, 1, 1), (1, 1), 0, ("total occ: 1", "chinese: splits=1 occ=1")),
        ((1, 1, 1), (0, 0), 1, ("total occ: 1", "chinese: splits=0 occ=0")),
        ((0, 0, 0), (0, 0), 0, ("total occ: 0", "chinese: splits=0 occ=0")),
    ],
    ids=["all-chinese", "no-chinese", "empty"],
)
def test_change_class_tally_coalesces_empty_sums(
    monkeypatch,
    tmp_path: Path,
    capsys,
    case: tuple[tuple[int, int, int], tuple[int, int], int, tuple[str, str]],
) -> None:
    totals, chinese, non_chinese_occ, expected_lines = case

    class FakeConnection:
        def __init__(self) -> None:
            self.query = ""
            self.queries: list[str] = []

        def execute(self, query: str) -> "FakeConnection":
            self.query = query
            self.queries.append(query)
            return self

        def fetchone(self) -> tuple[int, ...]:
            if "count(*), count(distinct nm)" in self.query:
                return totals
            if "count(*)" in self.query and "chinese=true" in self.query:
                return chinese
            if self.query.startswith("select coalesce(sum(occ), 0)"):
                return (non_chinese_occ,)
            return (0, 0)

    connection = FakeConnection()
    monkeypatch.setitem(sys.modules, "duckdb", SimpleNamespace(connect=lambda: connection))
    parquet = tmp_path / "all_chinese.parquet"
    parquet.touch()
    monkeypatch.setattr(sys, "argv", ["change_class_tally.py", str(parquet)])

    runpy.run_path(REPO_ROOT / "scripts/change_class_tally.py", run_name="__main__")

    output = capsys.readouterr().out
    assert f"non-chinese denominator occ = {non_chinese_occ}" in output
    assert all(expected_line in output for expected_line in expected_lines)
    assert output.count("0.0000%") == 7
    assert all("coalesce(sum(occ), 0)" in query for query in connection.queries if "sum(occ)" in query)
