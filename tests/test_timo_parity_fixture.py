import hashlib
import json
from pathlib import Path

from sinonym.coretypes.routing_resolution import ResolutionReason
from sinonym.timo.interface import (
    Instance,
    Prediction,
    Predictor,
    PredictorConfig,
    ResolvedAuthorFields,
    SourceAuthorFields,
)

FIXTURE_DIR = Path(__file__).parent / "data" / "timo_parity"
TERMINAL_FIELD_NAMES = {
    "first_name",
    "middle_names",
    "last_name",
    "suffix",
    "resolution_provenance",
    "resolution_action",
    "resolution_reason",
}
TERMINAL_OCCURRENCE_SHA256 = "347bd45913039f01edcc107afda5d1d66575eeac5fcaffab099055a7b9c8f733"


def _records() -> list[dict]:
    return [json.loads(line) for line in (FIXTURE_DIR / "requests.jsonl").read_text(encoding="utf-8").splitlines()]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _terminal_occurrences(records: list[dict]) -> list[dict]:
    return [
        {"paper_id": record["paper_id"], "position": position, "fields": fields}
        for record in records
        for position, fields in enumerate(record["expected"]["authors"])
    ]


def _terminal_dict(prediction: Prediction) -> dict:
    """Compare only the fixture's terminal fields; ``chinese_detected`` is unpinned."""
    result = prediction.dict()
    for author in result["authors"]:
        author.pop("chinese_detected", None)
    return result


def test_timo_parity_fixture_is_content_addressed_and_well_formed() -> None:
    manifest = json.loads((FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8"))
    fixture_bytes = (FIXTURE_DIR / manifest["fixture_file"]).read_bytes()
    records = _records()

    assert manifest["schema_version"] == 2
    assert hashlib.sha256(fixture_bytes).hexdigest() == manifest["fixture_sha256"]
    assert len(records) == manifest["case_count"] == 19
    assert [record["case_id"] for record in records] == sorted({record["case_id"] for record in records})
    assert all(hashlib.sha256(_canonical_bytes(record["request"])).hexdigest() == record["request_sha256"] for record in records)
    assert (
        hashlib.sha256(_canonical_bytes([record["request_sha256"] for record in records])).hexdigest()
        == manifest["request_set_sha256"]
    )

    assert all(set(record["expected"]) == {"authors"} for record in records)
    assert all("vys_other_names" in record["request"] for record in records)
    assert all(len(record["request"]["pp_authors"]) == len(record["expected"]["authors"]) for record in records)
    assert all(set(fields) == TERMINAL_FIELD_NAMES for record in records for fields in record["expected"]["authors"])

    occurrences = _terminal_occurrences(records)
    occurrence_keys = [(occurrence["paper_id"], occurrence["position"]) for occurrence in occurrences]
    assert len(occurrence_keys) == len(set(occurrence_keys)) == manifest["occurrence_count"] == 37
    assert manifest["terminal_occurrence_sha256"] == TERMINAL_OCCURRENCE_SHA256
    assert hashlib.sha256(_canonical_bytes(occurrences)).hexdigest() == TERMINAL_OCCURRENCE_SHA256


def test_timo_parity_fixture_round_trips_the_wire_contract() -> None:
    for record in _records():
        request = Instance(**record["request"])
        expected = Prediction(**record["expected"])

        assert request.dict() == record["request"]
        assert _terminal_dict(expected) == record["expected"]


def test_timo_matches_the_occurrence_keyed_field_parity_oracle() -> None:
    records = [record for record in _records() if record.get("evaluation_scope", "end_to_end") == "end_to_end"]
    instances = [Instance(**record["request"]) for record in records]
    predictor = Predictor(config=PredictorConfig(parallel="never"), artifacts_dir=".")

    predictions = predictor.predict_batch(instances)

    assert len(predictions) == len(records)
    for record, prediction in zip(records, predictions, strict=True):
        assert _terminal_dict(prediction) == record["expected"], record["case_id"]


def test_source_passthrough_materializer_preserves_the_two_juan_boundaries() -> None:
    records = [record for record in _records() if record.get("evaluation_scope") == "source_passthrough_materializer"]

    assert len(records) == 2
    for record in records:
        expected = record["expected"]["authors"][0]
        source = SourceAuthorFields(**record["request"]["pp_authors"][0])
        resolved = ResolvedAuthorFields.from_source(
            source,
            reason=ResolutionReason(expected["resolution_reason"]),
        )
        assert _terminal_dict(Prediction(authors=[resolved])) == record["expected"]
