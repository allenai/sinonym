"""Internal decision types for terminal TIMO name resolution.

The routed API exposes one operational author-field result.  These types keep
the evidence path that produced that result separate from the action taken on
the name, and make the legal combinations a closed, inspectable table.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from sinonym.coretypes.results import CanonicalName


class ResolutionProvenance(str, Enum):
    """Policy/evidence path that produced the resolved author fields."""

    PP = "pp"
    VYS = "vys"
    SCALAR = "scalar"
    SOURCE = "source"


class ResolutionAction(str, Enum):
    """How the writer must handle the resolved author slot."""

    ASSIGN = "assign"
    PRESERVE_INPUT = "preserve_input"
    SUPPRESS = "suppress"


class EastAsianEvidenceReason(str, Enum):
    """Closed set of evidence outcomes emitted by East Asian name routing."""

    JAPANESE_ITERATION_MARK_ONE_SIDED_EXCLUSIVE = "japanese_iteration_mark_one_sided_exclusive"
    JAPANESE_ITERATION_MARK_DUAL_EXCLUSIVE = "japanese_iteration_mark_dual_exclusive"
    IDENTITY_BACKED_EXACT_FULL_SURFACE = "identity_backed_exact_full_surface"
    KOREAN_WESTERN_SUFFIX_CONFLICT = "korean_western_suffix_conflict"
    KOREAN_NATIVE_THREE_SYLLABLE = "korean_native_three_syllable"
    JAPANESE_NATIVE_DICTIONARY = "japanese_native_dictionary"
    JAPANESE_NATIVE_SPACED_DICTIONARY = "japanese_native_spaced_dictionary"
    VIETNAMESE_GIVEN_FIRST_EXACT_SURFACE = "vietnamese_given_first_exact_surface"
    VIETNAMESE_UNICODE_SURNAME_FIRST = "vietnamese_unicode_surname_first"
    KOREAN_COMPACT_GIVEN_UNIQUE_SPLIT = "korean_compact_given_unique_split"
    KOREAN_ROMANIZED_STRICT = "korean_romanized_strict"
    JAPANESE_ROMANIZED_DIRECTIONAL_DICTIONARY = "japanese_romanized_directional_dictionary"


class ResolutionReason(str, Enum):
    """Closed set of measurable terminal resolution reasons."""

    MIXED_SCRIPT_SAFETY_SUPPRESSION = "mixed_script_safety_suppression"
    ROUTED_CJK_SAFETY_SUPPRESSION = "routed_cjk_safety_suppression"
    HANDLED_EVIDENCE_FAILURE = "handled_evidence_failure"
    HARD_SCALAR_MATERIALIZATION_FAILED = "hard_scalar_materialization_failed"
    SCALAR_KNOWN_COMPOUND_SURNAME_PRESERVE_INPUT = "scalar_known_compound_surname_preserve_input"
    SCALAR_CLEAN_SOURCE_SURNAME_REPARTITION_ASSIGNMENT = "scalar_clean_source_surname_repartition_assignment"
    STRUCTURED_SURNAME_INITIAL_TAIL_ASSIGNMENT = "structured_surname_initial_tail_assignment"

    JAPANESE_ITERATION_MARK_ASSIGNMENT = "japanese_iteration_mark_assignment"
    IDENTITY_BACKED_EXACT_ASSIGNMENT = "identity_backed_exact_assignment"
    REVIEWED_EXACT_SOURCE_ASSIGNMENT = "reviewed_exact_source_assignment"
    REVIEWED_SOURCE_PATTERN_ASSIGNMENT = "reviewed_source_pattern_assignment"
    KOREAN_WESTERN_CONFLICT_PRESERVE_INPUT = "korean_western_conflict_preserve_input"
    JAPANESE_GIVEN_FIRST_REORDER_VETO_PRESERVE_INPUT = "japanese_given_first_reorder_veto_preserve_input"
    VIETNAMESE_GIVEN_FIRST_REORDER_VETO_PRESERVE_INPUT = "vietnamese_given_first_reorder_veto_preserve_input"
    INITIALS_COMMA_REORDER_VETO_PRESERVE_INPUT = "initials_comma_reorder_veto_preserve_input"
    REVIEWED_EXACT_SOURCE_REORDER_VETO_PRESERVE_INPUT = "reviewed_exact_source_reorder_veto_preserve_input"

    CONTEXT_SUPPORTED_REORDER_VETO_PRESERVE_INPUT = "context_supported_reorder_veto_preserve_input"

    PP_SELECTED = "pp_selected"
    VYS_SELECTED = "vys_selected"
    PP_VYS_ABSTAIN_PP_INPUT = "pp_vys_abstain_pp_input"
    PP_VYS_ABSTAIN_VYS_INPUT = "pp_vys_abstain_vys_input"
    PP_ONLY_ABSTAIN_INPUT = "pp_only_abstain_input"
    PP_ONLY_ABSTAIN_REVIEWED_COMPOUND_SURNAME = "pp_only_abstain_reviewed_compound_surname"
    BATCH_ABSTAIN_MATERIALIZATION_FAILED = "batch_abstain_materialization_failed"

    SCALAR_BASELINE = "scalar_baseline"

    REVIEWED_NON_PERSON_PATTERN = "reviewed_non_person_pattern"
    NON_PERSON_SOURCE_PASSTHROUGH = "non_person_source_passthrough"
    NO_USABLE_SEMANTIC_RESULT = "no_usable_semantic_result"


@dataclass(frozen=True, slots=True)
class ResolutionDecisionSpec:
    """The only legal provenance and action for one reason."""

    provenance: ResolutionProvenance
    action: ResolutionAction


_RESOLUTION_REASON_GROUPS = {
    (ResolutionProvenance.SOURCE, ResolutionAction.PRESERVE_INPUT): (
        ResolutionReason.MIXED_SCRIPT_SAFETY_SUPPRESSION,
        ResolutionReason.ROUTED_CJK_SAFETY_SUPPRESSION,
        ResolutionReason.HANDLED_EVIDENCE_FAILURE,
        ResolutionReason.HARD_SCALAR_MATERIALIZATION_FAILED,
        ResolutionReason.SCALAR_KNOWN_COMPOUND_SURNAME_PRESERVE_INPUT,
        ResolutionReason.INITIALS_COMMA_REORDER_VETO_PRESERVE_INPUT,
        ResolutionReason.REVIEWED_EXACT_SOURCE_REORDER_VETO_PRESERVE_INPUT,
        ResolutionReason.PP_ONLY_ABSTAIN_REVIEWED_COMPOUND_SURNAME,
        ResolutionReason.BATCH_ABSTAIN_MATERIALIZATION_FAILED,
        ResolutionReason.NON_PERSON_SOURCE_PASSTHROUGH,
        ResolutionReason.NO_USABLE_SEMANTIC_RESULT,
    ),
    (ResolutionProvenance.SOURCE, ResolutionAction.ASSIGN): (
        ResolutionReason.SCALAR_CLEAN_SOURCE_SURNAME_REPARTITION_ASSIGNMENT,
        ResolutionReason.STRUCTURED_SURNAME_INITIAL_TAIL_ASSIGNMENT,
        ResolutionReason.REVIEWED_EXACT_SOURCE_ASSIGNMENT,
        ResolutionReason.REVIEWED_SOURCE_PATTERN_ASSIGNMENT,
    ),
    (ResolutionProvenance.SOURCE, ResolutionAction.SUPPRESS): (ResolutionReason.REVIEWED_NON_PERSON_PATTERN,),
    (ResolutionProvenance.SCALAR, ResolutionAction.PRESERVE_INPUT): (
        ResolutionReason.KOREAN_WESTERN_CONFLICT_PRESERVE_INPUT,
        ResolutionReason.JAPANESE_GIVEN_FIRST_REORDER_VETO_PRESERVE_INPUT,
        ResolutionReason.VIETNAMESE_GIVEN_FIRST_REORDER_VETO_PRESERVE_INPUT,
        ResolutionReason.CONTEXT_SUPPORTED_REORDER_VETO_PRESERVE_INPUT,
    ),
    (ResolutionProvenance.SCALAR, ResolutionAction.ASSIGN): (
        ResolutionReason.JAPANESE_ITERATION_MARK_ASSIGNMENT,
        ResolutionReason.IDENTITY_BACKED_EXACT_ASSIGNMENT,
        ResolutionReason.SCALAR_BASELINE,
    ),
    (ResolutionProvenance.PP, ResolutionAction.PRESERVE_INPUT): (
        ResolutionReason.PP_VYS_ABSTAIN_PP_INPUT,
        ResolutionReason.PP_ONLY_ABSTAIN_INPUT,
    ),
    (ResolutionProvenance.PP, ResolutionAction.ASSIGN): (ResolutionReason.PP_SELECTED,),
    (ResolutionProvenance.VYS, ResolutionAction.PRESERVE_INPUT): (ResolutionReason.PP_VYS_ABSTAIN_VYS_INPUT,),
    (ResolutionProvenance.VYS, ResolutionAction.ASSIGN): (ResolutionReason.VYS_SELECTED,),
}
_RESOLUTION_DISPOSITIONS = {
    reason: disposition for disposition, reasons in _RESOLUTION_REASON_GROUPS.items() for reason in reasons
}
RESOLUTION_DECISION_TABLE = MappingProxyType(
    {reason: ResolutionDecisionSpec(*_RESOLUTION_DISPOSITIONS[reason]) for reason in ResolutionReason},
)
del _RESOLUTION_DISPOSITIONS, _RESOLUTION_REASON_GROUPS


def resolution_decision_spec(reason: ResolutionReason) -> ResolutionDecisionSpec:
    """Return the single legal decision specification for ``reason``."""
    return RESOLUTION_DECISION_TABLE[reason]


EAST_ASIAN_EVIDENCE_RESOLUTION_REASONS = MappingProxyType(
    dict.fromkeys(EastAsianEvidenceReason)
    | {
        EastAsianEvidenceReason.JAPANESE_ITERATION_MARK_ONE_SIDED_EXCLUSIVE: (
            ResolutionReason.JAPANESE_ITERATION_MARK_ASSIGNMENT
        ),
        EastAsianEvidenceReason.JAPANESE_ITERATION_MARK_DUAL_EXCLUSIVE: (ResolutionReason.JAPANESE_ITERATION_MARK_ASSIGNMENT),
        EastAsianEvidenceReason.IDENTITY_BACKED_EXACT_FULL_SURFACE: ResolutionReason.IDENTITY_BACKED_EXACT_ASSIGNMENT,
        EastAsianEvidenceReason.KOREAN_WESTERN_SUFFIX_CONFLICT: (ResolutionReason.KOREAN_WESTERN_CONFLICT_PRESERVE_INPUT),
    },
)


def east_asian_evidence_resolution_reason(
    evidence_reason: EastAsianEvidenceReason,
) -> ResolutionReason | None:
    """Return the terminal hard resolution, if any, for one evidence outcome."""
    return EAST_ASIAN_EVIDENCE_RESOLUTION_REASONS[evidence_reason]


@dataclass(frozen=True, slots=True)
class ApplyAssignment:
    """The materialized result of a proven hard scalar assignment."""

    canonical_name: CanonicalName
    evidence_reason: EastAsianEvidenceReason

    def __post_init__(self) -> None:
        """Enforce the closed evidence set: the mapped reason must assign."""
        reason = EAST_ASIAN_EVIDENCE_RESOLUTION_REASONS.get(self.evidence_reason)
        if reason is None or resolution_decision_spec(reason).action is not ResolutionAction.ASSIGN:
            message = f"unsupported hard scalar assignment: {self.evidence_reason!r}"
            raise ValueError(message)

    @property
    def reason(self) -> ResolutionReason:
        """The terminal resolution reason determined by the evidence."""
        return EAST_ASIAN_EVIDENCE_RESOLUTION_REASONS[self.evidence_reason]


@dataclass(frozen=True, slots=True)
class PreserveBaseline:
    """Keep the current path's baseline after the proven Korean conflict veto."""

    canonical_name: CanonicalName
    evidence_reason: EastAsianEvidenceReason = EastAsianEvidenceReason.KOREAN_WESTERN_SUFFIX_CONFLICT

    def __post_init__(self) -> None:
        """Enforce the closed evidence set: the mapped reason must preserve."""
        reason = EAST_ASIAN_EVIDENCE_RESOLUTION_REASONS.get(self.evidence_reason)
        if reason is None or resolution_decision_spec(reason).action is not ResolutionAction.PRESERVE_INPUT:
            message = "PreserveBaseline is reserved for the Korean/Western conflict hard decision"
            raise ValueError(message)

    @property
    def reason(self) -> ResolutionReason:
        """The terminal resolution reason determined by the evidence."""
        return EAST_ASIAN_EVIDENCE_RESOLUTION_REASONS[self.evidence_reason]


HardScalarConstraint: TypeAlias = ApplyAssignment | PreserveBaseline


class EvidenceFailure(RuntimeError):  # noqa: N818 - explicit domain term
    """Typed evidence failure that must propagate unless explicitly handled.

    A caller that deliberately handles this failure must emit
    ``handled_evidence_failure``.  Invariant violations and programming errors
    must not be wrapped in this type or converted into source fallback.
    """


class HardScalarMaterializationFailure(RuntimeError):  # noqa: N818 - explicit domain term
    """A fired hard scalar rule could not produce valid person components."""
