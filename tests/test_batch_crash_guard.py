"""analyze_name_batch must never let one pathological name crash the whole batch.

It is the entry point every served/batch caller flows through; a per-name exception
there fails the entire batch. These tests inject crashes at the two guarded sites
(canonical attachment, batch phase + per-name fallback) — synthetic by necessity,
since the guard exists for crash classes that don't exist yet — and pin that the
known production crash classes stay batch-safe with real inputs.
"""

from dataclasses import replace

import pytest

from sinonym import ChineseNameDetector

CHINESE_POISON = "Wang Fang"
POISON = "Poison Name"
BATCH = ["Zhang Wei", CHINESE_POISON, POISON, "Li Ming"]


@pytest.fixture(scope="module")
def detector():
    return ChineseNameDetector()


def test_canonical_attachment_crash_keeps_base_result(detector, monkeypatch):
    original = ChineseNameDetector._attach_canonical_name  # noqa: SLF001

    def crashing(self, name, result):
        if name == CHINESE_POISON:
            message = "synthetic canonical crash"
            raise RuntimeError(message)
        return original(self, name, result)

    monkeypatch.setattr(ChineseNameDetector, "_attach_canonical_name", crashing)
    batch = detector.analyze_name_batch(BATCH)

    assert len(batch.results) == len(BATCH)
    by_name = dict(zip(batch.names, batch.results, strict=True))
    assert by_name[CHINESE_POISON].success
    assert by_name[CHINESE_POISON].canonical_name is None
    assert by_name["Zhang Wei"].success
    assert by_name["Zhang Wei"].canonical_name is not None


def test_batch_phase_crash_degrades_to_guarded_per_name(detector, monkeypatch):
    def crashing_batch(*args, **kwargs):
        message = "synthetic batch-phase crash"
        raise RuntimeError(message)

    monkeypatch.setattr(
        detector._batch_analysis_service,  # noqa: SLF001
        "analyze_name_batch",
        crashing_batch,
    )
    original = ChineseNameDetector.normalize_name

    def crashing_normalize(self, raw_name):
        if raw_name == POISON:
            message = "synthetic per-name crash"
            raise RuntimeError(message)
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
    ea_crashers = ["Yi -Hung Choh", "Shin -ichi Kudô", "O -T Carter"]
    combining_marks = "彬人 樽\U000e0100井"
    batch = detector.analyze_name_batch(["Zhang Wei", *ea_crashers, combining_marks, "Wang Fang"])

    assert len(batch.results) == len(ea_crashers) + 3
    by_name = dict(zip(batch.names, batch.results, strict=True))
    for name in ea_crashers:
        assert by_name[name].success is False
    assert by_name["Zhang Wei"].success
    assert by_name["Wang Fang"].success


def test_strict_and_forgiving_batches_share_analysis_but_not_sidecars(detector):
    """Both policies use one analyzer; only the compatibility path adds canonicals."""
    names = ["Dr. Steve Marsh PhD", "Li Wei"]

    forgiving = detector.analyze_name_batch(names)
    (strict,) = detector.analyze_name_batches_strict([names], parallel="never")

    assert strict.names == forgiving.names
    assert strict.format_pattern == forgiving.format_pattern
    assert strict.individual_analyses == forgiving.individual_analyses
    assert strict.improvements == forgiving.improvements
    assert strict.name_order_evidence == forgiving.name_order_evidence
    assert strict.results == [replace(result, canonical_name=None) for result in forgiving.results]
    assert any(result.canonical_name is not None for result in forgiving.results)
    assert all(result.canonical_name is None for result in strict.results)


def test_strict_batches_propagate_analysis_failures(detector, monkeypatch):
    """The strict public twin must not degrade a programming failure to fallback rows."""
    def crashing_batch(*args, **kwargs):
        message = "synthetic strict batch crash"
        raise RuntimeError(message)

    monkeypatch.setattr(
        detector._batch_analysis_service,  # noqa: SLF001
        "analyze_name_batch",
        crashing_batch,
    )

    with pytest.raises(RuntimeError, match="synthetic strict batch crash"):
        detector.analyze_name_batches_strict([BATCH], parallel="never")
