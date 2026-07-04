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
    has_all_caps_token: bool
    all_caps_tokens: list[str]
    first_token_surname_frequency: float | None = None
    last_token_surname_frequency: float | None = None
    selected_surname_frequency: float | None = None
    alternate_endpoint_surname_frequency: float | None = None
    selected_over_alternate_surname_frequency_ratio: float | None = None


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


class RoutingDecisionValue(str, Enum):
    """Serialized pp-vys-abstain router decision."""

    PP = "pp"
    VYS = "vys"
    ABSTAIN = "abstain"
    NOT_PERSON = "not_person"


class InputOrderCandidateValue(str, Enum):
    """Which batch (pp/vys) preserves the input/given-first order; unknown if neither uniquely does."""

    PP = "pp"
    VYS = "vys"
    UNKNOWN = "unknown"


class RoutedPrediction(TimoModel):
    """PP-vs-VYS routed result for one author, plus everything needed to inspect/re-run the router.

    `given_name`/`surname`/`middle_name`/`success` are the FINAL routed answer: the PP parse
    when `router_prediction=='pp'`, the VYS parse when `'vys'`, the input-order side
    (`input_order_candidate`) when `'abstain'`, and empty when `'not_person'`.
    """

    success: bool = Field(description="Whether the routed answer is a recognized Chinese person")
    given_name: str | None = Field(default=None)
    surname: str | None = Field(default=None)
    middle_name: str | None = Field(default=None)
    router_prediction: RoutingDecisionValue = Field(description="pp / vys / abstain / not_person")
    router_reason: str = Field(description="rule that produced router_prediction")
    input_order_candidate: InputOrderCandidateValue | None = Field(
        default=None,
        description="pp/vys/unknown; abstain resolves to this side. None in PP-only mode.",
    )
    pp: Prediction = Field(description="the paper-batch (PP) parse for this author")
    vys: Prediction | None = Field(default=None, description="the VYS parse; None in PP-only mode (no venue pool)")


class PPRoutedPrediction(TimoModel):
    """PP-only (pp-abstain) routed result for one author.

    `given_name`/`surname`/`middle_name`/`success` are the FINAL routed answer: the PP parse when
    `router_prediction=='pp'`, the input-order parse when `'abstain'`, empty when `'not_person'`.
    """

    success: bool = Field(description="Whether the routed answer is a recognized Chinese person")
    given_name: str | None = Field(default=None)
    surname: str | None = Field(default=None)
    middle_name: str | None = Field(default=None)
    router_prediction: RoutingDecisionValue = Field(description="pp / abstain (pp-abstain router never emits vys or not_person)")
    router_reason: str = Field(description="rule that produced router_prediction")
    pp: Prediction = Field(description="the paper-batch (PP) parse for this author")


class RoutingInstance(TimoModel):
    """One paper's authors for the routing-served variant (`RoutingPredictor.predict_batch`).

    Self-contained per instance (the timo runner feeds one Instance at a time), so each carries
    its own venue pool. One `RoutingInstance` maps to exactly one `RoutedPaperPrediction`, whose
    `authors` list holds `len(pp_names)` routed authors in `pp_names` order.
    """

    pp_names: list[str] = Field(description="the paper's author names (PP batch); output aligns to this order")
    vys_pool_names: list[str] | None = Field(
        default=None,
        description=(
            "venue-source-year author pool with the paper's authors as the FIRST len(pp_names) "
            "entries (same strings/order as pp_names), then the other venue authors. None/empty => "
            "PP-only routing fallback."
        ),
    )


class RoutedPaperPrediction(TimoModel):
    """One paper's routed result — the timo `Prediction` for the `sinonym_routing_v1` variant.

    Keeps the timo contract 1:1 (one Prediction per `RoutingInstance`) while carrying the paper's
    per-author routed results in `authors` (aligned to `pp_names`). A paper with no authors yields
    `authors=[]`, so instances never silently vanish from the output stream.
    """

    authors: list[RoutedPrediction] = Field(
        default_factory=list,
        description="per-author routed results, aligned to the instance's pp_names",
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

    @staticmethod
    def _routed_name_fields(parsed) -> dict:
        """given/surname/middle for a routed answer (all None when `parsed` is None)."""
        return {
            "given_name": parsed.given_name if parsed else None,
            "surname": parsed.surname if parsed else None,
            "middle_name": parsed.middle_name if parsed and parsed.middle_name else None,
        }

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

    # ---- predict_batch: timo-served entrypoint for sinonym_v1 (flat name->Prediction) ----

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
                format_pattern=pattern.copy(deep=True),
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

    # ---- name-order routing (call `route`) --------------------------------
    # The routing core: `RoutingPredictor.predict_batch` (the sinonym_routing_v1 timo variant)
    # calls `route` per RoutingInstance; also callable directly by importing Predictor.
    # Needs per-paper grouping (pp_names + vys_pool_names), which the flat sinonym_v1
    # predict_batch(List[Instance]) contract can't express — hence the separate variant.

    def route_pp_vys(
        self,
        pp_names: list[str],
        vys_pool_names: list[str],
    ) -> list[RoutedPrediction]:
        """Run the pp-vys-abstain router for one paper's authors.

        - `pp_names`: the paper's author names (the PP batch); output is aligned to this order.
        - `vys_pool_names`: the venue-source-year author pool (the VYS batch), with **the paper's
          own authors as the FIRST `len(pp_names)` entries** (same strings, same order as
          `pp_names`), followed by the other venue authors. Validated:
          `vys_pool_names[:len(pp_names)] == pp_names`, else ValueError.

        The full pool sets the VYS batch order-vote (parsing is order-independent, so only the
        set of names matters); the paper's authors are the leading slice, so their VYS parses are
        `vys_pool_names[:len(pp_names)]`. Returns one RoutedPrediction per pp author: the FINAL
        routed parse plus the PP and VYS candidate parses and the router
        decision/reason/input_order_candidate. Routing emits the PP parse for `pp`, the VYS parse
        for `vys`, the `input_order_candidate` side for `abstain`, nothing for `not_person`.

        Example — build `vys_pool_names` as the paper's authors FIRST, then the *other* venue
        authors (do NOT re-include the paper's authors again; that would double-count them in the
        vote):

            pp_names = ["Yue Lin", "Wei Wang"]                 # this paper's 2 authors
            other_venue_authors = ["Jun Zhao", "Hui Li", ...]  # rest of the venue-source-year pool
            vys_pool_names = pp_names + other_venue_authors     # paper authors first
            predictor.route_pp_vys(pp_names, vys_pool_names)

        Returns 2 RoutedPredictions (one per pp author), aligned to `pp_names`.
        """
        from sinonym.coretypes import BatchParseResult
        from sinonym.pipeline.name_order_routing import route_pp_vys_abstain_batches

        if not pp_names:
            return []
        n = len(pp_names)
        if len(vys_pool_names) < n:
            message = (
                f"vys_pool_names (len {len(vys_pool_names)}) must contain at least the paper's "
                f"{n} authors — the paper is a subset of the venue pool"
            )
            raise ValueError(message)
        if list(vys_pool_names[:n]) != list(pp_names):
            message = "vys_pool_names must start with the paper's authors: vys_pool_names[:len(pp_names)] == pp_names"
            raise ValueError(message)

        pp_batch = self._detector.analyze_name_batch(pp_names)
        pool = self._detector.analyze_name_batch(vys_pool_names)
        # The paper's authors are the leading slice of the pool (pool.names[:n] == pp_names, checked
        # above), so take the VYS batch context straight from the pool's own leading slice.
        vys_batch = BatchParseResult(
            names=list(pool.names[:n]),
            results=list(pool.results[:n]),
            format_pattern=pool.format_pattern,
            individual_analyses=list(pool.individual_analyses[:n]),
            improvements=[i for i in pool.improvements if i < n],
            name_order_evidence=list(pool.name_order_evidence[:n]),
        )
        rows = route_pp_vys_abstain_batches(pp_batch, vys_batch)

        # format_pattern is batch-wide (same for every row); build each once and share it —
        # the candidate Predictions are output DTOs and never mutate it.
        pp_fp = self._to_format_pattern(pp_batch.format_pattern)
        vys_fp = self._to_format_pattern(vys_batch.format_pattern)
        out: list[RoutedPrediction] = []
        for i, row in enumerate(rows):
            pred = row["router_prediction"]
            ioc = row.get("input_order_candidate", "unknown")
            pp_res = pp_batch.results[i]
            vys_res = vys_batch.results[i]
            if pred == "pp":
                chosen = pp_res
            elif pred == "vys":
                chosen = vys_res
            elif pred == "abstain":
                # abstain always carries input_order_candidate "pp" or "vys" (never "unknown");
                # emit that (given-first) side, and fail loudly if the invariant ever breaks.
                chosen = {"pp": pp_res, "vys": vys_res}.get(ioc)
                if chosen is None:
                    message = f"abstain with unexpected input_order_candidate={ioc!r} (expected 'pp'/'vys')"
                    raise ValueError(message)
            elif pred == "not_person":
                chosen = None
            else:
                message = f"pp-vys router returned unexpected router_prediction={pred!r}"
                raise ValueError(message)
            parsed = chosen.parsed if (chosen is not None and chosen.success) else None
            out.append(
                RoutedPrediction(
                    success=bool(chosen is not None and chosen.success),
                    **self._routed_name_fields(parsed),
                    router_prediction=pred,
                    router_reason=row.get("router_reason", ""),
                    input_order_candidate=ioc,
                    pp=self._to_prediction(pp_res, format_pattern=pp_fp.copy(deep=True)),
                    vys=self._to_prediction(vys_res, format_pattern=vys_fp.copy(deep=True)),
                ),
            )
        return out

    def route_pp(self, names: list[str]) -> list[PPRoutedPrediction]:
        """PP-only (pp-abstain) router — for when there is no VYS venue pool.

        Runs a single PP batch and applies the self-contained pp-abstain router, which decides
        per author between `pp` (trust the PP-batch reorder), `abstain` (keep the input-order
        parse), and `not_person`. Returns one PPRoutedPrediction per name (aligned to `names`):
        the final routed parse + decision/reason + the PP candidate parse.
        """
        from sinonym.pipeline.name_order_routing import (
            build_pp_abstain_rows,
            input_order_parsed,
            route_pp_abstain_rows,
        )

        if not names:
            return []

        pp_batch = self._detector.analyze_name_batch(names)
        rows = route_pp_abstain_rows(build_pp_abstain_rows(pp_batch, self._detector))
        # format_pattern is batch-wide (same for every row); build once and share it.
        pp_fp = self._to_format_pattern(pp_batch.format_pattern)

        out: list[PPRoutedPrediction] = []
        for i, row in enumerate(rows):
            pred = row["router_prediction"]
            res = pp_batch.results[i]
            if pred == "pp":
                parsed = res.parsed if res.success else None
            elif pred == "abstain":
                # abstain = "emit the preprocessed input-order parse": the as-typed reading
                # (trailing token = surname), independent of both the batch reorder and the
                # standalone parser's own order choice. Falls back to the batch parse only
                # when no as-typed reading exists (failed parse / single token).
                parsed = input_order_parsed(res) or (res.parsed if res.success else None)
            elif pred == "not_person":  # valid Route value; emit nothing (pp-abstain router doesn't produce it today)
                parsed = None
            else:
                message = f"pp-abstain router returned unexpected router_prediction={pred!r}"
                raise ValueError(message)
            out.append(
                PPRoutedPrediction(
                    success=bool(parsed is not None),
                    **self._routed_name_fields(parsed),
                    router_prediction=pred,
                    router_reason=row.get("router_reason", ""),
                    pp=self._to_prediction(res, format_pattern=pp_fp.copy(deep=True)),
                ),
            )
        return out

    def route(
        self,
        pp_names: list[str],
        vys_pool_names: list[str] | None = None,
    ) -> list[RoutedPrediction]:
        """Unified router: use PP+VYS routing when a venue pool is given, else PP-only fallback.

        - If `vys_pool_names` is falsy (None or empty), routes PP-only via the pp-abstain router
          (`route_pp`); returned `RoutedPrediction`s have `vys=None` and `input_order_candidate=None`.
        - Otherwise routes PP-vs-VYS via `route_pp_vys` (paper authors must be the leading slice of
          `vys_pool_names`; see that method).

        Always returns `list[RoutedPrediction]` (one per pp author, aligned to `pp_names`), so callers
        get a single response shape whether or not venue context is available.
        """
        if vys_pool_names:
            return self.route_pp_vys(pp_names, vys_pool_names)
        # PPRoutedPrediction is a field-subset of RoutedPrediction; widen each to the unified
        # shape by adding the PP-only sentinels (no venue pool → no vys / input_order_candidate).
        return [RoutedPrediction(**r.dict(), input_order_candidate=None, vys=None) for r in self.route_pp(pp_names)]


class RoutingPredictor(Predictor):
    """timo-served routing variant: one RoutingInstance (paper) -> one RoutedPaperPrediction.

    Wraps `Predictor.route`. Stays 1:1 with the timo instance/prediction contract — the paper's
    per-author routed results are nested in `RoutedPaperPrediction.authors` (aligned to pp_names),
    so paper boundaries are explicit and empty papers still emit one prediction.
    """

    def predict_batch(self, instances: list[RoutingInstance]) -> list[RoutedPaperPrediction]:  # type: ignore[override]
        return [RoutedPaperPrediction(authors=self.route(inst.pp_names, inst.vys_pool_names)) for inst in instances]
