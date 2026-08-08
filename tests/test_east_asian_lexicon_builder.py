"""Tests for the reviewed East Asian lexicon build inputs."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from scripts import build_east_asian_name_lexicons as builder


def _write_reviewed_csv(path: Path, row: str) -> None:
    path.write_text(
        "surname_key,evidence_id,example_surface,evidence_url,given_first_exact_surface\n" + row,
        encoding="utf-8",
    )


def test_encode_has_a_portable_deterministic_gzip_header() -> None:
    payload: dict[str, object] = {"schema_version": builder.ASSET_SCHEMA_VERSION, "values": ["a", "b"]}

    first = builder.encode(payload)
    second = builder.encode(payload)

    assert first == second
    assert first[:10] == bytes.fromhex("1f8b08000000000002ff")
    assert json.loads(gzip.decompress(first)) == payload


def test_reviewed_possible_surname_exact_surface_is_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "reviewed.csv"
    _write_reviewed_csv(
        source,
        "hiroya,record:1,Kou Hiroya,https://example.test,Kou Hiroya\nmiwa,record:2,Takaya Miwa,https://example.test,\n",
    )
    monkeypatch.setattr(builder, "REVIEWED_POSSIBLE_SURNAMES_PATH", source)

    surnames, exact_surfaces, metadata = builder.reviewed_possible_surnames()

    assert surnames == ["hiroya", "miwa"]
    assert exact_surfaces == ["kou hiroya"]
    assert metadata["sha256"]


def test_reviewed_possible_surname_evidence_must_end_in_that_surname(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "reviewed.csv"
    _write_reviewed_csv(source, "miwa,record:1,Takaya Sato,https://example.test,\n")
    monkeypatch.setattr(builder, "REVIEWED_POSSIBLE_SURNAMES_PATH", source)

    with pytest.raises(ValueError, match="evidence surfaces"):
        builder.reviewed_possible_surnames()


def test_shipped_reviewed_surnames_keep_general_evidence_separate_from_exact_repairs() -> None:
    surnames, exact_surfaces, _metadata = builder.reviewed_possible_surnames()

    assert len(surnames) == 29
    assert {"azuma", "haba", "kadono", "shiota", "yoshinaka"} <= set(surnames)
    assert exact_surfaces == [
        "haruki kadono",
        "kou hiroya",
        "masaki takamoto",
        "masaki tomonaga",
        "shoji kagami",
        "takaya miwa",
    ]


def test_country_spelling_variants_are_possible_but_not_directional_surnames() -> None:
    asset = Path("sinonym/data/east_asian_roman_lexicons.json.gz")
    payload = json.loads(gzip.decompress(asset.read_bytes()))
    directional = set(payload["japanese_surnames"])
    possible = set(payload["japanese_possible_surnames"])

    assert {
        "fujihara",
        "hirano",
        "miyasaki",
        "nakashima",
        "sugahara",
        "takaki",
        "takata",
        "taketa",
        "ueta",
        "yamasaki",
    } <= possible - directional
