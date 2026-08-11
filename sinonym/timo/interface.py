"""Single TIMO boundary for terminal, source-shaped author resolution."""

from pydantic import BaseSettings as _BaseSettings
from pydantic import Field as _Field

from sinonym.coretypes import BatchParseResult as _BatchParseResult
from sinonym.detector import ChineseNameDetector as _ChineseNameDetector
from sinonym.detector import ParallelMode as _ParallelMode
from sinonym.pipeline.name_order_routing import (
    build_pp_abstain_rows as _build_pp_abstain_rows,
)
from sinonym.pipeline.name_order_routing import (
    route_pp_abstain_rows as _route_pp_abstain_rows,
)
from sinonym.pipeline.name_order_routing import (
    route_pp_vys_abstain_batches as _route_pp_vys_abstain_batches,
)
from sinonym.services.batch_analysis import RelatedBatchParseResult as _RelatedBatchParseResult
from sinonym.timo._resolution import (
    Instance,
    ResolvedAuthorFields,
    SourceAuthorFields,
    _Model,
    _Resolver,
)

__all__ = [
    "Instance",
    "Prediction",
    "Predictor",
    "PredictorConfig",
    "ResolvedAuthorFields",
    "SourceAuthorFields",
]


class Prediction(_Model):
    """One paper's positionally aligned, directly writable author fields."""

    authors: list[ResolvedAuthorFields] = _Field(default_factory=list)


class PredictorConfig(_BaseSettings):
    """Batch execution settings for :class:`Predictor`."""

    parallel: _ParallelMode = "auto"
    mp_max_workers: int | None = None
    mp_chunk_size: int = 64
    mp_min_parallel_batches: int | None = None
    mp_start_method: str = "auto"

    class Config:
        env_prefix = "SINONYM_"


class Predictor:
    """Resolve each paper to one authoritative set of author fields."""

    def __init__(self, config: PredictorConfig, artifacts_dir: str):
        self._config = config
        self._artifacts_dir = artifacts_dir
        self._detector = _ChineseNameDetector()
        self._resolver = _Resolver(self._detector)

    def _resolve_pp_vys_batch(
        self,
        *,
        sources: list[SourceAuthorFields],
        pp_batch: _BatchParseResult,
        pool: _BatchParseResult,
    ) -> list[ResolvedAuthorFields]:
        rows = _route_pp_vys_abstain_batches(pp_batch, pool)
        if len(rows) != len(sources):
            message = "PP/VYS router output no longer aligns with source authors"
            raise RuntimeError(message)

        return [
            self._resolver.resolve_pp_vys_author(
                source=source,
                paper_authors=sources,
                raw_name=pp_batch.names[index],
                paper_names=pp_batch.names,
                focal_index=index,
                row=row,
                pp_result=pp_batch.results[index],
                vys_result=pool.results[index],
            )
            for index, (source, row) in enumerate(zip(sources, rows, strict=True))
        ]

    def _resolve_pp_batch(
        self,
        *,
        sources: list[SourceAuthorFields],
        pp_batch: _BatchParseResult,
    ) -> list[ResolvedAuthorFields]:
        rows = _route_pp_abstain_rows(_build_pp_abstain_rows(pp_batch, self._detector))
        if len(rows) != len(sources):
            message = "PP router output no longer aligns with source authors"
            raise RuntimeError(message)

        return [
            self._resolver.resolve_pp_author(
                source=source,
                paper_authors=sources,
                raw_name=pp_batch.names[index],
                paper_names=pp_batch.names,
                focal_index=index,
                row=row,
                result=pp_batch.results[index],
            )
            for index, (source, row) in enumerate(zip(sources, rows, strict=True))
        ]

    @staticmethod
    def _validate_batch_alignment(
        submitted_names: list[str],
        batch_result: _BatchParseResult,
        *,
        batch_index: int,
    ) -> None:
        """Reject a batch result that no longer matches its submitted slots."""
        if list(batch_result.names) != submitted_names:
            message = f"batch result {batch_index} names/order do not match the submitted batch"
            raise RuntimeError(message)

        expected = len(submitted_names)
        aligned_lengths = {
            "results": len(batch_result.results),
            "individual_analyses": len(batch_result.individual_analyses),
            "name_order_evidence": len(batch_result.name_order_evidence),
        }
        mismatched = {name: length for name, length in aligned_lengths.items() if length != expected}
        if mismatched:
            message = f"batch result {batch_index} has misaligned fields: expected {expected}, got {mismatched}"
            raise RuntimeError(message)
        if any(index < 0 or index >= expected for index in batch_result.improvements):
            message = f"batch result {batch_index} has an out-of-range improvement index"
            raise RuntimeError(message)

    def _validate_related_batch_results(
        self,
        requests: list[tuple[list[str], list[str] | None]],
        batch_results: list[_RelatedBatchParseResult],
    ) -> None:
        """Validate complete PP rows and focal VYS rows before resolution."""
        if len(batch_results) != len(requests):
            message = f"batch analysis returned the wrong number of batches: expected {len(requests)}, got {len(batch_results)}"
            raise RuntimeError(message)

        for request_index, ((pp_names, vys_pool_names), batch_result) in enumerate(
            zip(requests, batch_results, strict=True),
        ):
            self._validate_batch_alignment(pp_names, batch_result.pp_batch, batch_index=request_index * 2)
            if vys_pool_names is None:
                if batch_result.vys_batch is not None or batch_result.vys_context_names is not None:
                    message = "PP-only analysis unexpectedly returned a VYS result"
                    raise RuntimeError(message)
                continue
            if batch_result.vys_batch is None or batch_result.vys_context_names != tuple(vys_pool_names):
                message = f"batch result {request_index * 2 + 1} names/order do not match the submitted batch"
                raise RuntimeError(message)
            self._validate_batch_alignment(pp_names, batch_result.vys_batch, batch_index=request_index * 2 + 1)

    def predict_batch(self, instances: list[Instance]) -> list[Prediction]:
        """Resolve one prediction per paper, preserving paper and author order."""
        predictions: list[Prediction | None] = [None] * len(instances)
        requests: list[tuple[list[str], list[str] | None]] = []
        plans: list[tuple[int, int]] = []

        for instance_index, instance in enumerate(instances):
            pp_names = [author.full_name() for author in instance.pp_authors]
            if not pp_names:
                predictions[instance_index] = Prediction(authors=[])
                continue

            vys_pool_names = [*pp_names, *instance.vys_other_names] if instance.vys_other_names else None
            plans.append((instance_index, len(requests)))
            requests.append((pp_names, vys_pool_names))

        batch_results = self._detector._analyze_related_batch_requests(  # noqa: SLF001
            requests,
            parallel=self._config.parallel,
            min_parallel_batches=self._config.mp_min_parallel_batches,
            max_workers=self._config.mp_max_workers,
            chunk_size=self._config.mp_chunk_size,
            mp_start_method=self._config.mp_start_method,
        )
        self._validate_related_batch_results(requests, batch_results)

        for instance_index, request_index in plans:
            instance = instances[instance_index]
            batch_result = batch_results[request_index]
            if requests[request_index][1] is None:
                authors = self._resolve_pp_batch(
                    sources=instance.pp_authors,
                    pp_batch=batch_result.pp_batch,
                )
            else:
                if batch_result.vys_batch is None:
                    message = "PP/VYS routing plan is missing its VYS batch result"
                    raise RuntimeError(message)
                authors = self._resolve_pp_vys_batch(
                    sources=instance.pp_authors,
                    pp_batch=batch_result.pp_batch,
                    pool=batch_result.vys_batch,
                )
            predictions[instance_index] = Prediction(authors=authors)

        if any(prediction is None for prediction in predictions):
            message = "prediction plan did not fill every instance slot"
            raise RuntimeError(message)
        return [prediction for prediction in predictions if prediction is not None]
