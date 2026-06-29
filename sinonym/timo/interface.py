from enum import Enum

from pydantic import BaseModel, BaseSettings, Field, root_validator

from sinonym.detector import ChineseNameDetector


class TimoModel(BaseModel):
    """Base Pydantic model for TIMO request and response contracts."""

    def dict(self, *args, **kwargs):
        """Return plain Python serialization values for enum fields."""
        return _serialize_enum_values(super().dict(*args, **kwargs))


class NameFormatValue(str, Enum):
    """Serialized batch name-order values."""

    SURNAME_FIRST = "surname_first"
    GIVEN_FIRST = "given_first"
    MIXED = "mixed"


class ScriptRepresentationValue(str, Enum):
    """Serialized script provenance values exposed in batch evidence."""

    LATIN_ONLY = "latin_only"
    HAN_ONLY = "han_only"
    BILINGUAL_ALIGNED = "bilingual_aligned"
    MIXED_SCRIPT = "mixed_script"
    REJECTED_INPUT = "rejected_input"
    UNKNOWN = "unknown"


class SurnamePositionValue(str, Enum):
    """Serialized selected surname positions in source tokens."""

    FIRST = "first"
    LAST = "last"
    INTERNAL = "internal"
    UNKNOWN = "unknown"


class Instance(TimoModel):
    name: str = Field(description="Name string to detect/normalize as Chinese")


class FormatPattern(TimoModel):
    """Batch-level order detection (surname-first vs given-first)."""

    dominant_format: NameFormatValue
    confidence: float = Field(description="dominant_count / total_count")
    decision_confidence: float = Field(description="score used to decide threshold_met")
    surname_first_count: int
    given_first_count: int
    total_count: int
    voting_count: int = Field(description="count of names contributing a confident direction vote")
    vote_margin_count: int = Field(description="absolute difference between surname-first and given-first votes")
    vote_margin: float = Field(description="vote_margin_count / total_count")
    threshold_met: bool = Field(description="decision_confidence >= format_threshold plus gating checks")

    @root_validator(pre=True)
    def _fill_derived_fields(cls, values):  # noqa: N805
        """Accept legacy six-field payloads and derive the new vote metrics."""
        if not isinstance(values, dict):
            return values

        output = dict(values)
        confidence = float(output.get("confidence", 0.0) or 0.0)
        surname_first_count = int(output.get("surname_first_count", 0) or 0)
        given_first_count = int(output.get("given_first_count", 0) or 0)
        total_count = int(output.get("total_count", 0) or 0)
        vote_margin_count = abs(surname_first_count - given_first_count)

        output.setdefault("decision_confidence", confidence)
        output.setdefault("voting_count", surname_first_count + given_first_count)
        output.setdefault("vote_margin_count", vote_margin_count)
        output.setdefault("vote_margin", vote_margin_count / total_count if total_count > 0 else 0.0)
        return output


def _serialize_enum_values(value):
    """Recursively convert Enum objects to their values for `.dict()` output."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, list):
        return [_serialize_enum_values(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_serialize_enum_values(item) for item in value)
    if isinstance(value, dict):
        return {key: _serialize_enum_values(item) for key, item in value.items()}
    return value


class Prediction(TimoModel):
    success: bool = Field(description="Whether the name was recognized as Chinese")
    error_message: str | None = Field(default=None, description="Reason for failure")
    given_name: str | None = Field(default=None)
    surname: str | None = Field(default=None)
    middle_name: str | None = Field(default=None)
    confidence: float | None = Field(default=None, description="per-name confidence (softmax over candidate scores)")
    format_pattern: FormatPattern | None = Field(default=None, description="shared batch order pattern (same on every row)")


class Candidate(TimoModel):
    surname_tokens: list[str]
    given_tokens: list[str]
    score: float
    format: NameFormatValue
    original_compound_format: str | None = None


class IndividualAnalysis(TimoModel):
    """Per-name analysis, pre batch-override."""

    raw_name: str
    candidates: list[Candidate]
    best_candidate: Candidate | None = None
    confidence: float = Field(description="softmax over candidate scores for best candidate")


class NameOrderEvidence(TimoModel):
    """Per-name evidence aligned with batch names and results."""

    raw_name: str
    raw_tokens: list[str]
    raw_token_count: int
    script_representation: ScriptRepresentationValue
    batch_participant: bool
    batch_applied: bool
    batch_changed_format: bool
    individual_format: NameFormatValue
    selected_format: NameFormatValue
    selected_surname_position: SurnamePositionValue
    first_token_surname_frequency: float | None = None
    last_token_surname_frequency: float | None = None
    selected_surname_frequency: float | None = None
    alternate_endpoint_surname_frequency: float | None = None
    selected_over_alternate_surname_frequency_ratio: float | None = None
    has_all_caps_token: bool
    all_caps_tokens: list[str]


class BatchPrediction(TimoModel):
    """Full result of analyze_name_batch."""

    names: list[str]
    results: list[Prediction]
    format_pattern: FormatPattern
    individual_analyses: list[IndividualAnalysis]
    improvements: list[int] = Field(description="indices of names changed by batch context")
    name_order_evidence: list[NameOrderEvidence]


class BatchSummary(TimoModel):
    """Trimmed analyze_name_batch result: drops candidates, keeps per-name confidence only."""

    names: list[str]
    results: list[Prediction]
    format_pattern: FormatPattern
    confidences: list[float] = Field(description="per-name confidence from individual_analyses, aligned with results")


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

    def _to_prediction(
        self,
        parse_result,
        *,
        confidence: float | None = None,
        format_pattern: FormatPattern | None = None,
    ) -> Prediction:
        return Prediction(
            success=parse_result.success,
            error_message=parse_result.error_message,
            given_name=parse_result.parsed.given_name if parse_result.parsed else None,
            surname=parse_result.parsed.surname if parse_result.parsed else None,
            middle_name=(parse_result.parsed.middle_name if parse_result.parsed and parse_result.parsed.middle_name else None),
            confidence=confidence,
            format_pattern=format_pattern,
        )

    def _to_format_pattern(self, pattern) -> FormatPattern:
        return FormatPattern(
            dominant_format=pattern.dominant_format.value,
            confidence=pattern.confidence,
            decision_confidence=pattern.decision_confidence,
            surname_first_count=pattern.surname_first_count,
            given_first_count=pattern.given_first_count,
            total_count=pattern.total_count,
            voting_count=pattern.voting_count,
            vote_margin_count=pattern.vote_margin_count,
            vote_margin=pattern.vote_margin,
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
            best_candidate=(self._to_candidate(analysis.best_candidate) if analysis.best_candidate is not None else None),
            confidence=analysis.confidence,
        )

    def _to_name_order_evidence(self, evidence) -> NameOrderEvidence:
        return NameOrderEvidence(
            raw_name=evidence.raw_name,
            raw_tokens=list(evidence.raw_tokens),
            raw_token_count=evidence.raw_token_count,
            script_representation=evidence.script_representation or ScriptRepresentationValue.UNKNOWN.value,
            batch_participant=evidence.batch_participant,
            batch_applied=evidence.batch_applied,
            batch_changed_format=evidence.batch_changed_format,
            individual_format=evidence.individual_format.value,
            selected_format=evidence.selected_format.value,
            selected_surname_position=evidence.selected_surname_position,
            first_token_surname_frequency=evidence.first_token_surname_frequency,
            last_token_surname_frequency=evidence.last_token_surname_frequency,
            selected_surname_frequency=evidence.selected_surname_frequency,
            alternate_endpoint_surname_frequency=evidence.alternate_endpoint_surname_frequency,
            selected_over_alternate_surname_frequency_ratio=(evidence.selected_over_alternate_surname_frequency_ratio),
            has_all_caps_token=evidence.has_all_caps_token,
            all_caps_tokens=list(evidence.all_caps_tokens),
        )

    def _to_batch_prediction(self, batch_result) -> BatchPrediction:
        return BatchPrediction(
            names=list(batch_result.names),
            results=[self._to_prediction(r) for r in batch_result.results],
            format_pattern=self._to_format_pattern(batch_result.format_pattern),
            individual_analyses=[self._to_individual_analysis(a) for a in batch_result.individual_analyses],
            improvements=list(batch_result.improvements),
            name_order_evidence=[self._to_name_order_evidence(e) for e in batch_result.name_order_evidence],
        )

    def _to_batch_summary(self, batch_result) -> BatchSummary:
        return BatchSummary(
            names=list(batch_result.names),
            results=[self._to_prediction(r) for r in batch_result.results],
            format_pattern=self._to_format_pattern(batch_result.format_pattern),
            confidences=[a.confidence for a in batch_result.individual_analyses],
        )

    # ---- timo entrypoint --------------------------------------------------

    def predict_batch(self, instances: list[Instance]) -> list[Prediction]:
        """timo HTTP entrypoint. Analyzes the whole batch jointly.

        Names are processed together (cross-batch order detection), returning one
        Prediction per name (index-aligned) with surname/given_name/middle_name,
        per-name `confidence`, and the shared `format_pattern` (same on every row).
        """
        if not instances:
            return []

        names = [i.name for i in instances]
        batch_result = self._detector.analyze_name_batch(names)
        pattern = self._to_format_pattern(batch_result.format_pattern)

        predictions = []
        for parse_result, analysis in zip(batch_result.results, batch_result.individual_analyses, strict=False):
            prediction = self._to_prediction(
                parse_result,
                confidence=analysis.confidence,
                format_pattern=pattern,
            )
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
        names: list[str],
        format_threshold: float | None = None,
        minimum_batch_size: int | None = None,
    ) -> BatchPrediction:
        batch_result = self._detector.analyze_name_batch(names, **self._batch_kwargs(format_threshold, minimum_batch_size))
        return self._to_batch_prediction(batch_result)

    def process_name_batch(
        self,
        names: list[str],
        format_threshold: float | None = None,
        minimum_batch_size: int | None = None,
    ) -> list[Prediction]:
        results = self._detector.process_name_batch(names, **self._batch_kwargs(format_threshold, minimum_batch_size))
        return [self._to_prediction(r) for r in results]

    def detect_batch_format(
        self,
        names: list[str],
        format_threshold: float | None = None,
    ) -> FormatPattern:
        pattern = self._detector.detect_batch_format(names, **self._batch_kwargs(format_threshold))
        return self._to_format_pattern(pattern)

    def process_name_batch_multiprocess(
        self,
        names: list[str],
        max_workers: int | None = None,
        chunk_size: int = 64,
    ) -> list[Prediction]:
        results = self._detector.process_name_batch_multiprocess(names, max_workers=max_workers, chunk_size=chunk_size)
        return [self._to_prediction(r) for r in results]

    def score_name_batch(
        self,
        names: list[str],
        format_threshold: float | None = None,
        minimum_batch_size: int | None = None,
    ) -> BatchSummary:
        """analyze_name_batch trimmed to names, results, format_pattern, per-name confidence."""
        batch_result = self._detector.analyze_name_batch(names, **self._batch_kwargs(format_threshold, minimum_batch_size))
        return self._to_batch_summary(batch_result)
