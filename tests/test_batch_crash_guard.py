"""analyze_name_batch must never let one pathological name crash the whole batch.

It is the entry point every served/batch caller flows through; a per-name exception
there fails the entire batch. These tests inject crashes at the two guarded sites
(canonical attachment, batch phase + per-name fallback) — synthetic by necessity,
since the guard exists for crash classes that don't exist yet — and pin that the
known production crash classes stay batch-safe with real inputs.
"""

import pytest

from sinonym import ChineseNameDetector

CHINESE_POISON = "Wang Fang"
POISON = "Poison Name"
BATCH = ["Zhang Wei", CHINESE_POISON, POISON, "Li Ming"]


@pytest.fixture(scope="module")
def detector():
    return ChineseNameDetector()


def test_canonical_attachment_crash_keeps_base_result(detector, monkeypatch):
    # Poison a Chinese batch participant: its canonical attachment happens only in the
    # final per-name pass, isolating that guard layer. (Poisoning a non-participant
    # instead crashes inside the batch phases and exercises the whole-batch degrade,
    # which the next test covers directly.)
    original = ChineseNameDetector._attach_canonical_name

    def crashing(self, name, result):
        if name == CHINESE_POISON:
            raise RuntimeError("synthetic canonical crash")
        return original(self, name, result)

    monkeypatch.setattr(ChineseNameDetector, "_attach_canonical_name", crashing)
    batch = detector.analyze_name_batch(BATCH)

    assert len(batch.results) == len(BATCH)
    by_name = dict(zip(batch.names, batch.results, strict=True))
    # A Chinese name whose canonical attachment crashes keeps its successful parse.
    assert by_name[CHINESE_POISON].success
    assert by_name[CHINESE_POISON].canonical_name is None
    assert by_name["Zhang Wei"].success
    assert by_name["Zhang Wei"].canonical_name is not None


def test_batch_phase_crash_degrades_to_guarded_per_name(detector, monkeypatch):
    def crashing_batch(*args, **kwargs):
        raise RuntimeError("synthetic batch-phase crash")

    monkeypatch.setattr(
        detector._batch_analysis_service, "analyze_name_batch", crashing_batch
    )
    original = ChineseNameDetector.normalize_name

    def crashing_normalize(self, raw_name):
        if raw_name == POISON:
            raise RuntimeError("synthetic per-name crash")
        return original(self, raw_name)

    monkeypatch.setattr(ChineseNameDetector, "normalize_name", crashing_normalize)
    batch = detector.analyze_name_batch(BATCH)

    assert len(batch.results) == len(BATCH)
    by_name = dict(zip(batch.names, batch.results, strict=True))
    assert by_name[POISON].success is False
    assert by_name["Zhang Wei"].success
    assert by_name[CHINESE_POISON].success
    assert by_name["Li Ming"].success


def test_production_crash_corpus_classes_stay_batch_safe(detector):
    # Real inputs from the prod bad_names log (EA-route crashes, fixed in 0.4.1) plus
    # the combining-marks crash (fixed in 0.3.1): none may raise inside a batch, the
    # EA-class names abstain (update if classification ever improves), and the Chinese
    # cohort still parses.
    ea_crashers = ["Yi -Hung Choh", "Shin -ichi Kudô", "O -T Carter"]
    combining_marks = "彬人 樽\U000E0100井"
    batch = detector.analyze_name_batch(["Zhang Wei", *ea_crashers, combining_marks, "Wang Fang"])

    assert len(batch.results) == len(ea_crashers) + 3
    by_name = dict(zip(batch.names, batch.results, strict=True))
    for name in ea_crashers:
        assert by_name[name].success is False
    assert by_name["Zhang Wei"].success
    assert by_name["Wang Fang"].success
