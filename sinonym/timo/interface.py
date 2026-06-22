from typing import List, Optional

from pydantic import BaseModel, BaseSettings, Field

from sinonym.detector import ChineseNameDetector


class Instance(BaseModel):
    name: str = Field(description="Name string to detect/normalize as Chinese")


class FormatPattern(BaseModel):
    """Batch-level order detection (surname-first vs given-first)."""

    dominant_format: str = Field(description="surname_first | given_first | mixed")
    confidence: float = Field(description="dominant_count / total_count")
    surname_first_count: int
    given_first_count: int
    total_count: int
    threshold_met: bool = Field(description="confidence >= format_threshold")


class Prediction(BaseModel):
    success: bool = Field(description="Whether the name was recognized as Chinese")
    result: str = Field(
        description="Normalized name in 'Given-Name Surname' format, or empty on failure"
    )
    error_message: Optional[str] = Field(default=None, description="Reason for failure")
    given_name: Optional[str] = Field(default=None)
    surname: Optional[str] = Field(default=None)
    middle_name: Optional[str] = Field(default=None)
    original_compound_surname: Optional[str] = Field(default=None)
    confidence: Optional[float] = Field(
        default=None, description="per-name confidence (softmax over candidate scores)"
    )
    format_pattern: Optional[FormatPattern] = Field(
        default=None, description="shared batch order pattern (same on every row)"
    )


class Candidate(BaseModel):
    surname_tokens: List[str]
    given_tokens: List[str]
    score: float
    format: str = Field(description="surname_first | given_first | mixed")
    original_compound_format: Optional[str] = None


class IndividualAnalysis(BaseModel):
    """Per-name analysis, pre batch-override."""

    raw_name: str
    candidates: List[Candidate]
    best_candidate: Optional[Candidate] = None
    confidence: float = Field(description="softmax over candidate scores for best candidate")


class BatchPrediction(BaseModel):
    """Full result of analyze_name_batch."""

    names: List[str]
    results: List[Prediction]
    format_pattern: FormatPattern
    individual_analyses: List[IndividualAnalysis]
    improvements: List[int] = Field(description="indices of names changed by batch context")


class BatchSummary(BaseModel):
    """Trimmed analyze_name_batch result: drops candidates, keeps per-name confidence only."""

    names: List[str]
    results: List[Prediction]
    format_pattern: FormatPattern
    confidences: List[float] = Field(
        description="per-name confidence from individual_analyses, aligned with results"
    )


class PredictorConfig(BaseSettings):
    pass


class Predictor:
    _config: PredictorConfig
    _artifacts_dir: str

    def __init__(self, config: PredictorConfig, artifacts_dir: str):
        self._config = config
        self._artifacts_dir = artifacts_dir
        self._detector = ChineseNameDetector()

    # ---- converters -------------------------------------------------------

    def _to_prediction(self, parse_result) -> Prediction:
        return Prediction(
            success=parse_result.success,
            result=parse_result.result if isinstance(parse_result.result, str) else "",
            error_message=parse_result.error_message,
            given_name=parse_result.parsed.given_name if parse_result.parsed else None,
            surname=parse_result.parsed.surname if parse_result.parsed else None,
            middle_name=(
                parse_result.parsed.middle_name
                if parse_result.parsed and parse_result.parsed.middle_name
                else None
            ),
            original_compound_surname=parse_result.original_compound_surname,
        )

    def _to_format_pattern(self, pattern) -> FormatPattern:
        return FormatPattern(
            dominant_format=pattern.dominant_format.value,
            confidence=pattern.confidence,
            surname_first_count=pattern.surname_first_count,
            given_first_count=pattern.given_first_count,
            total_count=pattern.total_count,
            threshold_met=pattern.threshold_met,
        )

    def _to_candidate(self, candidate) -> Candidate:
        return Candidate(
            surname_tokens=list(candidate.surname_tokens),
            given_tokens=list(candidate.given_tokens),
            score=candidate.score,
            format=candidate.format.value,
            original_compound_format=candidate.original_compound_format,
        )

    def _to_individual_analysis(self, analysis) -> IndividualAnalysis:
        return IndividualAnalysis(
            raw_name=analysis.raw_name,
            candidates=[self._to_candidate(c) for c in analysis.candidates],
            best_candidate=(
                self._to_candidate(analysis.best_candidate)
                if analysis.best_candidate is not None
                else None
            ),
            confidence=analysis.confidence,
        )

    def _to_batch_prediction(self, batch_result) -> BatchPrediction:
        return BatchPrediction(
            names=list(batch_result.names),
            results=[self._to_prediction(r) for r in batch_result.results],
            format_pattern=self._to_format_pattern(batch_result.format_pattern),
            individual_analyses=[
                self._to_individual_analysis(a) for a in batch_result.individual_analyses
            ],
            improvements=list(batch_result.improvements),
        )

    def _to_batch_summary(self, batch_result) -> BatchSummary:
        return BatchSummary(
            names=list(batch_result.names),
            results=[self._to_prediction(r) for r in batch_result.results],
            format_pattern=self._to_format_pattern(batch_result.format_pattern),
            confidences=[a.confidence for a in batch_result.individual_analyses],
        )

    # ---- timo entrypoint --------------------------------------------------

    def predict_batch(self, instances: List[Instance]) -> List[Prediction]:
        """timo HTTP entrypoint. One Prediction per instance.

        Always runs batch analysis (cross-name format detection on). Each Prediction
        carries the normalized result plus per-name `confidence` and the shared batch
        `format_pattern` (replicated per row — timo has no batch-level response slot).
        Clients read whichever fields they need.
        """
        if not instances:
            return []

        names = [i.name for i in instances]
        batch_result = self._detector.analyze_name_batch(names)
        pattern = self._to_format_pattern(batch_result.format_pattern)

        predictions = []
        for parse_result, analysis in zip(
            batch_result.results, batch_result.individual_analyses
        ):
            prediction = self._to_prediction(parse_result)
            prediction.confidence = analysis.confidence
            prediction.format_pattern = pattern
            predictions.append(prediction)
        return predictions

    # ---- exposed detector functions --------------------------------------

    @staticmethod
    def _batch_kwargs(format_threshold, minimum_batch_size=...) -> dict:
        """Forward only caller-set tuning params; let sinonym own the defaults."""
        kw = {}
        if format_threshold is not None:
            kw["format_threshold"] = format_threshold
        if minimum_batch_size is not ... and minimum_batch_size is not None:
            kw["minimum_batch_size"] = minimum_batch_size
        return kw

    def analyze_name_batch(
        self,
        names: List[str],
        format_threshold: Optional[float] = None,
        minimum_batch_size: Optional[int] = None,
    ) -> BatchPrediction:
        batch_result = self._detector.analyze_name_batch(
            names, **self._batch_kwargs(format_threshold, minimum_batch_size)
        )
        return self._to_batch_prediction(batch_result)

    def process_name_batch(
        self,
        names: List[str],
        format_threshold: Optional[float] = None,
        minimum_batch_size: Optional[int] = None,
    ) -> List[Prediction]:
        results = self._detector.process_name_batch(
            names, **self._batch_kwargs(format_threshold, minimum_batch_size)
        )
        return [self._to_prediction(r) for r in results]

    def detect_batch_format(
        self,
        names: List[str],
        format_threshold: Optional[float] = None,
    ) -> FormatPattern:
        pattern = self._detector.detect_batch_format(
            names, **self._batch_kwargs(format_threshold)
        )
        return self._to_format_pattern(pattern)

    def process_name_batch_multiprocess(
        self,
        names: List[str],
        max_workers: Optional[int] = None,
        chunk_size: int = 64,
    ) -> List[Prediction]:
        results = self._detector.process_name_batch_multiprocess(
            names, max_workers=max_workers, chunk_size=chunk_size
        )
        return [self._to_prediction(r) for r in results]

    def score_name_batch(
        self,
        names: List[str],
        format_threshold: Optional[float] = None,
        minimum_batch_size: Optional[int] = None,
    ) -> BatchSummary:
        """analyze_name_batch trimmed to names, results, format_pattern, per-name confidence."""
        batch_result = self._detector.analyze_name_batch(
            names, **self._batch_kwargs(format_threshold, minimum_batch_size)
        )
        return self._to_batch_summary(batch_result)
