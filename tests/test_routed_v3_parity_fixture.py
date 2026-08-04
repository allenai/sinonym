import hashlib
import json
from pathlib import Path

from sinonym.coretypes.routing_resolution import ResolutionReason
from sinonym.timo.interface import PredictorConfig, RoutingInstance, RoutingPredictorV2, RoutingPredictorV3
from sinonym.timo.routing_v3 import ResolvedAuthorFields, RoutingInstanceV3, SourceAuthorFields

FIXTURE_DIR = Path(__file__).parent / "data" / "routed_v3_parity"


def _records() -> list[dict]:
    return [json.loads(line) for line in (FIXTURE_DIR / "requests.jsonl").read_text(encoding="utf-8").splitlines()]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def test_routed_v3_parity_fixture_is_content_addressed_and_well_formed() -> None:
    manifest = json.loads((FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8"))
    fixture_bytes = (FIXTURE_DIR / manifest["fixture_file"]).read_bytes()
    records = _records()
    assert manifest["schema_version"] == 1
    assert hashlib.sha256(fixture_bytes).hexdigest() == manifest["fixture_sha256"]
    assert len(records) == manifest["case_count"] == 19
    assert [record["case_id"] for record in records] == sorted({record["case_id"] for record in records})
    assert all(hashlib.sha256(_canonical_bytes(record["request"])).hexdigest() == record["request_sha256"] for record in records)
    assert (
        hashlib.sha256(_canonical_bytes([record["request_sha256"] for record in records])).hexdigest()
        == manifest["request_set_sha256"]
    )
    occurrence_keys = [(expected["paper_id"], expected["position"]) for record in records for expected in record["expected"]]
    assert len(occurrence_keys) == len(set(occurrence_keys)) == manifest["occurrence_count"] == 37


def test_routed_v3_parity_fixture_round_trips_the_wire_contract() -> None:
    for record in _records():
        request = RoutingInstanceV3(**record["request"])
        assert request.dict() == record["request"]
        assert len(request.pp_authors) == len(record["expected"])
        for expected in record["expected"]:
            resolved = ResolvedAuthorFields(**expected["resolved_author_fields"])
            assert resolved.dict() == expected["resolved_author_fields"]


def test_routing_v3_matches_the_occurrence_keyed_field_parity_oracle() -> None:
    records = [record for record in _records() if record.get("evaluation_scope", "end_to_end") == "end_to_end"]
    instances = [RoutingInstanceV3(**record["request"]) for record in records]
    predictor = RoutingPredictorV3(config=PredictorConfig(parallel="never"), artifacts_dir=".")

    predictions = predictor.predict_batch(instances)

    assert len(predictions) == len(records)
    for record, prediction in zip(records, predictions, strict=True):
        assert len(prediction.authors) == len(record["expected"])
        for expected, author in zip(record["expected"], prediction.authors, strict=True):
            assert author.resolved_fields.dict() == expected["resolved_author_fields"], (
                record["case_id"],
                expected["paper_id"],
                expected["position"],
            )


def test_source_passthrough_materializer_preserves_the_two_juan_boundaries() -> None:
    records = [record for record in _records() if record.get("evaluation_scope") == "source_passthrough_materializer"]

    assert len(records) == 2
    for record in records:
        expected = record["expected"][0]["resolved_author_fields"]
        source = SourceAuthorFields(**record["request"]["pp_authors"][0])
        resolved = ResolvedAuthorFields.from_source(
            source,
            reason=ResolutionReason(expected["resolution_reason"]),
        )
        assert resolved.dict() == expected


def test_production_fixture_legacy_evidence_still_matches_v2() -> None:
    records = [record for record in _records() if record["case_kind"] == "production_replay"]
    v3_instances = [RoutingInstanceV3(**record["request"]) for record in records]
    v2_instances = [
        RoutingInstance(pp_names=instance.pp_names, vys_pool_names=instance.vys_pool_names) for instance in v3_instances
    ]
    predictor = RoutingPredictorV2(config=PredictorConfig(parallel="never"), artifacts_dir=".")

    predictions = predictor.predict_batch(v2_instances)

    for record, prediction in zip(records, predictions, strict=True):
        for expected, author in zip(record["expected"], prediction.authors, strict=True):
            legacy = expected["legacy_evidence"]
            if author.success and author.surname:
                apply_path = "chinese"
            elif author.canonical_name is not None and author.canonical_name.normalized.surname:
                apply_path = "person"
            else:
                apply_path = "passthrough"
            assert {
                "apply_path": apply_path,
                "router_prediction": author.router_prediction.value,
                "router_reason": author.router_reason,
            } == legacy
