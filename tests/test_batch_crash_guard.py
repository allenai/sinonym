"""Fail-fast batch analysis and known rejected-input behavior."""

import pytest

from sinonym import ChineseNameDetector

CHINESE_POISON = "Wang Fang"
POISON = "Poison Name"
BATCH = ["Zhang Wei", CHINESE_POISON, POISON, "Li Ming"]


@pytest.fixture(scope="module")
def detector():
    return ChineseNameDetector()


def test_canonical_attachment_crash_propagates(detector, monkeypatch):
    original = ChineseNameDetector._attach_canonical_name  # noqa: SLF001

    def crashing(self, name, result):
        if name == CHINESE_POISON:
            message = "synthetic canonical crash"
            raise RuntimeError(message)
        return original(self, name, result)

    monkeypatch.setattr(ChineseNameDetector, "_attach_canonical_name", crashing)
    with pytest.raises(RuntimeError, match="synthetic canonical crash"):
        detector.analyze_name_batch(BATCH)


def test_batch_phase_crash_propagates(detector, monkeypatch):
    def crashing_batch(*args, **kwargs):
        message = "synthetic batch-phase crash"
        raise RuntimeError(message)

    monkeypatch.setattr(
        detector._batch_analysis_service,  # noqa: SLF001
        "analyze_name_batch",
        crashing_batch,
    )
    with pytest.raises(RuntimeError, match="synthetic batch-phase crash"):
        detector.analyze_name_batch(BATCH)


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


def test_singular_and_plural_batch_analysis_match_with_canonical_sidecars(detector):
    """The two public shapes share one analysis and sidecar contract."""
    names = ["Dr. Steve Marsh PhD", "Li Wei"]

    singular = detector.analyze_name_batch(names)
    (plural,) = detector.analyze_name_batches([names], parallel="never")

    assert plural == singular
    assert any(result.canonical_name is not None for result in plural.results)


def test_plural_batches_propagate_analysis_failures(detector, monkeypatch):
    """The plural API must not degrade a programming failure to fallback rows."""
    def crashing_batch(*args, **kwargs):
        message = "synthetic batch crash"
        raise RuntimeError(message)

    monkeypatch.setattr(
        detector._batch_analysis_service,  # noqa: SLF001
        "analyze_name_batch",
        crashing_batch,
    )

    with pytest.raises(RuntimeError, match="synthetic batch crash"):
        detector.analyze_name_batches([BATCH], parallel="never")


def test_missing_batch_service_is_an_invariant_failure(detector, monkeypatch):
    """Initialized detectors must not invent fallback batch results."""
    monkeypatch.setattr(detector, "_batch_analysis_service", None)

    with pytest.raises(RuntimeError, match="batch analysis service is not initialized"):
        detector.analyze_name_batch(BATCH)


def test_strict_public_twin_is_removed(detector):
    """There is one public failure policy for plural batch analysis."""
    assert not hasattr(detector, "analyze_name_batches_strict")
