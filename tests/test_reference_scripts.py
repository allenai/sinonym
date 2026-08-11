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


def test_change_class_tally_handles_an_empty_non_chinese_population(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    class FakeConnection:
        query = ""

        def execute(self, query: str) -> "FakeConnection":
            self.query = query
            return self

        def fetchone(self) -> tuple[int | None, ...]:
            if "count(*), count(distinct nm), sum(occ)" in self.query:
                return (1, 1, 1)
            if "count(*), sum(occ)" in self.query and "chinese=true" in self.query:
                return (1, 1)
            if "select sum(occ)" in self.query:
                return (None,)
            return (0, 0)

    monkeypatch.setitem(sys.modules, "duckdb", SimpleNamespace(connect=FakeConnection))
    parquet = tmp_path / "all_chinese.parquet"
    parquet.touch()
    monkeypatch.setattr(sys, "argv", ["change_class_tally.py", str(parquet)])

    runpy.run_path(REPO_ROOT / "scripts/change_class_tally.py", run_name="__main__")

    output = capsys.readouterr().out
    assert "non-chinese denominator occ = 0" in output
    assert output.count("0.0000%") == 7
