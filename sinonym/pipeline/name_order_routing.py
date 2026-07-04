"""Name-order routing rules for PP/VYS/abstain pipeline contexts.

These rules combine two or more `sinonym.analyze_name_batch()` runs. They
intentionally stay out of the single-name detector API and default batch parser
path.

Route semantics:
- `pp`: emit the PP parse.
- `vys`: emit the VYS parse.
- `abstain`: emit the preprocessed input-order parse.
- `not_person`: terminal non-person decision; do not emit a person parse.

Materializing `abstain`:
- PP/VYS regime: emit the `input_order_candidate` side's parse. Router invariant:
  an abstain row always has `input_order_candidate` in {"pp", "vys"} — the only
  abstain-emitting rule requires a defined side — so consumers may treat
  abstain + "unknown" as a contract violation and raise.
- PP-only regime: there is no second run to pick from; emit the as-typed reading
  via `input_order_parsed(result)` (trailing token is the surname, everything
  else keeps its position), except spaced Han surname-first rows, where source
  spacing already marks the surname boundary and the PP parse is the input-order
  reading. Do NOT re-parse the name standalone: the single-name detector
  re-decides order, which defeats the point of abstaining and couples the routed
  output to parser-version behavior.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from sinonym.coretypes import BatchParseResult, NameFormat, NameOrderEvidence, ParsedName, ParseResult

Route = Literal["pp", "vys", "abstain", "not_person"]
Row = Mapping[str, object]
MutableRow = dict[str, object]
__all__ = [
    "MutableRow",
    "Route",
    "Row",
    "build_pp_abstain_rows",
    "build_pp_vys_abstain_rows",
    "input_order_parsed",
    "pp_abstain_parsed",
    "pp_abstain_router",
    "pp_vys_abstain_router",
    "route_pp_abstain_rows",
    "route_pp_vys_abstain_batches",
    "route_pp_vys_abstain_rows",
]

FORMAT_VALUES = ("surname_first", "given_first", "mixed")
ORDER_FORMAT_VALUES = ("surname_first", "given_first")
PREDICTION_VALUES = ("pp", "vys", "abstain", "not_person")
OLD_PREDICTION_VALUES = ("pp", "vys", "not_person")
PP_VYS_REASON_VALUES = (
    "endpoint_frequency_strongly_favors_pp",
    "endpoint_frequency_strongly_favors_vys",
    "strong_pp_paper_context",
    "strong_vys_batch_context",
    "weak_or_conflicting_evidence",
)
PP_NEW_REASON_VALUES = ("endpoint_frequency_strongly_favors_pp", "strong_pp_paper_context")
VYS_NEW_REASON_VALUES = ("endpoint_frequency_strongly_favors_vys", "strong_vys_batch_context")

PP_VYS_ABSTAIN_REQUIRED_COLUMNS = (
    "name",
    "pp_success",
    "pp_result",
    "pp_selected_format",
    "pp_selected_surname_position",
    "pp_selected_surname_frequency",
    "pp_batch_dominant_format",
    "pp_batch_threshold_met",
    "pp_batch_total_count",
    "pp_batch_confidence",
    "pp_batch_vote_margin",
    "pp_selected_surname_frequency_ratio",
    "vys_success",
    "vys_result",
    "vys_selected_format",
    "vys_selected_surname_position",
    "vys_selected_surname_frequency",
    "vys_batch_dominant_format",
    "vys_batch_threshold_met",
    "vys_batch_confidence",
    "vys_batch_vote_margin",
)

PP_ABSTAIN_REQUIRED_COLUMNS = (
    "pp_success",
    "pp_result_token_count",
    "selected_format",
    "batch_total_count",
    "selected_surname_frequency",
    "has_cjk",
    "has_latin",
    "cjk_has_space",
    "raw_tokens",
)

PP_ABSTAIN_WEAK_SURNAME_FREQUENCY_MAX = 500.0
PP_ABSTAIN_CLEAN_BILINGUAL_SURNAME_FREQUENCY_MIN = 2500.0
PP_ABSTAIN_TWO_TOKEN_RESULT_COUNT = 2
PP_ABSTAIN_MIXED_LONG_RAW_TOKEN_MIN = 3
PP_ABSTAIN_CLEAN_BILINGUAL_RAW_TOKEN_COUNT = 2
PP_VYS_REAL_PAPER_VOTES_MIN = 3
PP_VYS_LOW_BATCH_MAX = 2
PP_VYS_VERY_LOW_BATCH_MAX = 1
PP_VYS_LOW_RATIO_MAX = 5.0
PP_VYS_ALLOWED_RATIO_LOW_MAX = 1.0
PP_VYS_ALLOWED_RATIO_HIGH_MIN = 5.0
PP_VYS_ALLOWED_RATIO_HIGH_MAX = 20.0
PP_VYS_HIGH_CONFIDENCE_MIN = 0.85
PP_VYS_LARGER_PAPER_VOTES_MIN = 6
MIN_THREE_TOKEN_NAME_PRIOR_TOKENS = 3
PP_VYS_SELECTED_CLEAN_ENDPOINT_VYS_RATIO_MAX = 0.10
PP_VYS_SELECTED_CLEAN_ENDPOINT_PP_RATIO_MIN = 20.0
PP_VYS_SELECTED_CLEAN_STRONG_PP_CONFIDENCE_MIN = 0.70
PP_VYS_SELECTED_CLEAN_STRONG_VYS_CONFIDENCE_MIN = 0.85
PP_VYS_OLD_ENDPOINT_VYS_RATIO_MIN = 5.0
PP_VYS_OLD_ENDPOINT_VYS_MARGIN_MIN = 0.55
PP_VYS_OLD_SMALL_PP_CONTEXT_MAX = 3
PP_VYS_OLD_SMALL_PP_VYS_MARGIN_MIN = 0.65
PP_VYS_OLD_WEAK_PP_MARGIN_MAX = 0.20
PP_VYS_OLD_WEAK_PP_VYS_MARGIN_MIN = 0.35
PP_VYS_GARBAGE_RESULT_TOKEN_MAX = 4
MIN_INPUT_ORDER_DISPLAY_TOKENS = 2
ROUTING_TOKEN_RE = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", flags=re.UNICODE)

NAME_PRIOR_COMMON_CHINESE_SURNAMES = frozenset(
    {
        "bai",
        "bi",
        "cai",
        "cao",
        "chen",
        "cheng",
        "chong",
        "chou",
        "dai",
        "deng",
        "ding",
        "dong",
        "du",
        "duan",
        "fan",
        "fang",
        "feng",
        "fu",
        "gao",
        "gu",
        "guo",
        "han",
        "he",
        "hou",
        "hu",
        "huang",
        "jia",
        "jiang",
        "jiao",
        "jin",
        "kan",
        "kong",
        "lee",
        "lei",
        "li",
        "liang",
        "liao",
        "lin",
        "liu",
        "long",
        "lu",
        "luo",
        "ma",
        "meng",
        "pan",
        "peng",
        "qian",
        "qin",
        "qiu",
        "ren",
        "shao",
        "sha",
        "shen",
        "shi",
        "song",
        "su",
        "sun",
        "tan",
        "tang",
        "tian",
        "tong",
        "wang",
        "wei",
        "wu",
        "xiao",
        "xie",
        "xu",
        "xue",
        "yan",
        "yang",
        "yao",
        "ye",
        "yin",
        "yu",
        "yuan",
        "zeng",
        "zhang",
        "zhao",
        "zheng",
        "zhou",
        "zhu",
    },
)

NAME_PRIOR_KOREAN_SURNAMES = frozenset(
    {
        "ahn",
        "an",
        "bae",
        "baek",
        "cha",
        "chang",
        "cho",
        "choi",
        "chung",
        "go",
        "han",
        "hong",
        "hwang",
        "jang",
        "jeon",
        "jeong",
        "jo",
        "jung",
        "kang",
        "kim",
        "ko",
        "kwon",
        "lee",
        "lim",
        "moon",
        "oh",
        "paik",
        "pak",
        "park",
        "ryu",
        "seo",
        "shin",
        "sim",
        "song",
        "suh",
        "yang",
        "yim",
        "yoo",
        "yoon",
        "yu",
        "yun",
    },
)

NAME_PRIOR_KOREAN_GIVEN_SYLLABLES = frozenset(
    {
        "ae",
        "bin",
        "bo",
        "chan",
        "cheol",
        "chul",
        "dae",
        "dong",
        "eun",
        "gi",
        "ha",
        "hae",
        "han",
        "hee",
        "ho",
        "hong",
        "hoon",
        "hun",
        "hwan",
        "hye",
        "hyun",
        "hyung",
        "il",
        "jae",
        "jin",
        "jong",
        "joon",
        "jun",
        "jung",
        "ki",
        "kyu",
        "kyung",
        "kyeong",
        "min",
        "sang",
        "seok",
        "seong",
        "seung",
        "sik",
        "soo",
        "su",
        "suk",
        "sun",
        "sung",
        "tae",
        "uk",
        "woo",
        "won",
        "yeon",
        "yong",
        "young",
        "yun",
    },
)

NAME_PRIOR_CANTONESE_SOUTHEAST_ASIAN_SURNAMES = frozenset(
    {
        "chan",
        "chang",
        "chew",
        "chia",
        "chong",
        "chung",
        "fong",
        "ho",
        "hong",
        "khoo",
        "lai",
        "lee",
        "leong",
        "leung",
        "lim",
        "lin",
        "low",
        "ng",
        "ong",
        "pak",
        "pang",
        "tan",
        "teoh",
        "tiong",
        "wong",
        "yap",
        "yeo",
        "yong",
    },
)

NAME_PRIOR_CANTONESE_GIVEN_SYLLABLES = frozenset(
    {
        "boon",
        "chih",
        "ching",
        "chiu",
        "guan",
        "hee",
        "hong",
        "kai",
        "ke",
        "kwan",
        "ling",
        "man",
        "mei",
        "meng",
        "mun",
        "pei",
        "peng",
        "ping",
        "seng",
        "siau",
        "sin",
        "sook",
        "sze",
        "wai",
        "wan",
        "wei",
        "wen",
        "yee",
        "yong",
        "yu",
        "yue",
    },
)


@dataclass(frozen=True)
class PPVysFeatures:
    """Parsed feature values for one PP/VYS/abstain routing row."""

    ratio: float
    pp_batch_total_count: float
    pp_batch_confidence: float
    old_prediction: str
    new_prediction: str
    new_reason: str
    pp_selected_format: str
    vys_selected_format: str
    input_order_candidate: str
    name_tokens: tuple[str, ...]
    old_new_vys: bool
    old_pp_new_vys: bool
    old_vys_new_abstain: bool
    old_vys_new_pp: bool


def build_pp_abstain_rows(batch_result: BatchParseResult, detector_context: Any) -> list[MutableRow]:
    """Build `pp-abstain` router rows from a batch parse result.

    `NameOrderEvidence` contains parser-observable fields, while the router
    also needs source-script and Japanese-classifier context. The detector
    context supplies normalization and Japanese probability helpers for those
    fields.
    """
    rows: list[MutableRow] = []
    for result, evidence in _iter_result_evidence(batch_result):
        normalized_input = _normalize_for_routing(detector_context, evidence.raw_name)
        script_representation = _script_representation_for_routing(detector_context, normalized_input, evidence)
        rows.append(
            {
                "pp_success": result.success,
                "pp_result_token_count": _parse_result_token_count(result),
                "selected_format": _format_value(evidence.selected_format),
                "batch_total_count": batch_result.format_pattern.total_count,
                "selected_surname_frequency": _float_or_zero(evidence.selected_surname_frequency),
                "has_cjk": _has_cjk(script_representation),
                "has_latin": _has_latin(script_representation),
                "cjk_has_space": script_representation == "han_only" and len(normalized_input.tokens) > 1,
                "raw_tokens": evidence.raw_token_count,
            },
        )
    return rows


def build_pp_vys_abstain_rows(
    pp_batch_result: BatchParseResult,
    vys_batch_result: BatchParseResult,
) -> list[MutableRow]:
    """Build evidence-only `pp-vys-abstain` router rows from aligned batch results."""
    _require_aligned_batch_results(pp_batch_result, vys_batch_result)

    rows: list[MutableRow] = []
    for index in range(len(pp_batch_result.results)):
        pp_result = pp_batch_result.results[index]
        vys_result = vys_batch_result.results[index]
        pp_evidence = pp_batch_result.name_order_evidence[index]
        vys_evidence = vys_batch_result.name_order_evidence[index]

        rows.append(
            {
                "name": pp_evidence.raw_name or pp_batch_result.names[index],
                "pp_success": pp_result.success,
                "pp_result": _result_text(pp_result),
                "pp_selected_format": _format_value(pp_evidence.selected_format),
                "pp_selected_surname_position": pp_evidence.selected_surname_position,
                "pp_selected_surname_frequency": _float_or_zero(pp_evidence.selected_surname_frequency),
                "pp_batch_dominant_format": _format_value(pp_batch_result.format_pattern.dominant_format),
                "pp_batch_threshold_met": pp_batch_result.format_pattern.threshold_met,
                "pp_batch_total_count": pp_batch_result.format_pattern.total_count,
                "pp_batch_confidence": _format_pattern_decision_confidence(pp_batch_result.format_pattern),
                "pp_batch_vote_margin": pp_batch_result.format_pattern.vote_margin,
                "pp_selected_surname_frequency_ratio": pp_evidence.selected_over_alternate_surname_frequency_ratio,
                "vys_success": vys_result.success,
                "vys_result": _result_text(vys_result),
                "vys_selected_format": _format_value(vys_evidence.selected_format),
                "vys_selected_surname_position": vys_evidence.selected_surname_position,
                "vys_selected_surname_frequency": _float_or_zero(vys_evidence.selected_surname_frequency),
                "vys_batch_dominant_format": _format_value(vys_batch_result.format_pattern.dominant_format),
                "vys_batch_threshold_met": vys_batch_result.format_pattern.threshold_met,
                "vys_batch_total_count": vys_batch_result.format_pattern.total_count,
                "vys_batch_confidence": _format_pattern_decision_confidence(vys_batch_result.format_pattern),
                "vys_batch_vote_margin": vys_batch_result.format_pattern.vote_margin,
            },
        )
    return rows


def route_pp_vys_abstain_batches(
    pp_batch_result: BatchParseResult,
    vys_batch_result: BatchParseResult,
) -> list[MutableRow]:
    """Route aligned PP and VYS batch results with the selected PP/VYS/abstain policy."""
    return route_pp_vys_abstain_rows(build_pp_vys_abstain_rows(pp_batch_result, vys_batch_result))


def route_pp_vys_abstain_rows(rows: Iterable[Row]) -> list[MutableRow]:
    """Route PP/VYS disagreements, defaulting to PP except known VYS/abstain slices.

    Required input columns are listed in `PP_VYS_ABSTAIN_REQUIRED_COLUMNS`.
    The returned rows add audit fields derived from evidence:
    - `old_prediction`
    - `old_reason`
    - `new_prediction`
    - `new_reason`

    They also add:
    - `input_order_candidate`
    - `router_prediction`
    - `router_reason`

    `router_prediction` can be `pp`, `vys`, `abstain`, or terminal
    `not_person`. The terminal route is emitted before deeper parse-evidence
    validation because non-person rows do not have meaningful PP/VYS route
    features.
    """
    return [_route_pp_vys_abstain_row(row) for row in rows]


def _route_pp_vys_abstain_row(row: Row) -> MutableRow:
    _require_columns(row, PP_VYS_ABSTAIN_REQUIRED_COLUMNS)
    routing_row = _pp_vys_policy_row(row)
    terminal_route = _not_person_route(routing_row)
    if terminal_route is not None:
        return terminal_route

    features = _pp_vys_features(routing_row)
    route, reason = _base_pp_vys_route(features)
    override = _effective_plus_override(features)
    if override is not None:
        route, reason = override
    name_prior_override = _name_prior_override(features)
    if name_prior_override is not None:
        route, reason = name_prior_override

    routed = dict(routing_row)
    routed["input_order_candidate"] = features.input_order_candidate
    routed["router_prediction"] = route
    routed["router_reason"] = reason
    return routed


def _pp_vys_policy_row(row: Row) -> MutableRow:
    """Return a row with old/new policy audit fields derived from evidence."""
    old_prediction, old_reason = _old_pp_vys_decision_from_row(row)
    if old_prediction == "not_person":
        new_prediction, new_reason = "not_person", "weak_or_conflicting_evidence"
    else:
        new_prediction, new_reason = _selected_clean_pp_vys_decision_from_row(row)
    _require_consistent_new_prediction_reason(new_prediction, new_reason)

    routed = dict(row)
    routed["old_prediction"] = old_prediction
    routed["old_reason"] = old_reason
    routed["new_prediction"] = new_prediction
    routed["new_reason"] = new_reason
    return routed


def _pp_vys_features(row: Row) -> PPVysFeatures:
    ratio = _optional_float(row, "pp_selected_surname_frequency_ratio", default=math.nan)
    pp_batch_total_count = _required_float(row, "pp_batch_total_count")
    pp_batch_confidence = _required_float(row, "pp_batch_confidence")
    old_prediction = _required_string(row, "old_prediction", allowed=("pp", "vys"))
    new_prediction = _required_string(row, "new_prediction", allowed=PREDICTION_VALUES)
    new_reason = _required_string(row, "new_reason", allowed=PP_VYS_REASON_VALUES)
    _require_consistent_new_prediction_reason(new_prediction, new_reason)
    pp_selected_format = _required_string(row, "pp_selected_format", allowed=ORDER_FORMAT_VALUES)
    vys_selected_format = _required_string(row, "vys_selected_format", allowed=ORDER_FORMAT_VALUES)
    old_new_vys = old_prediction == "vys" and new_prediction == "vys"
    old_pp_new_vys = old_prediction == "pp" and new_prediction == "vys"
    old_vys_new_abstain = old_prediction == "vys" and new_prediction == "abstain"
    old_vys_new_pp = old_prediction == "vys" and new_prediction == "pp"

    return PPVysFeatures(
        ratio=ratio,
        pp_batch_total_count=pp_batch_total_count,
        pp_batch_confidence=pp_batch_confidence,
        old_prediction=old_prediction,
        new_prediction=new_prediction,
        new_reason=new_reason,
        pp_selected_format=pp_selected_format,
        vys_selected_format=vys_selected_format,
        input_order_candidate=_input_order_candidate(row),
        name_tokens=_name_tokens(_required_string(row, "name")),
        old_new_vys=old_new_vys,
        old_pp_new_vys=old_pp_new_vys,
        old_vys_new_abstain=old_vys_new_abstain,
        old_vys_new_pp=old_vys_new_pp,
    )


def _not_person_route(row: Row) -> MutableRow | None:
    """Return the terminal non-person route without interpreting parse evidence."""
    _required_string(row, "old_prediction", allowed=OLD_PREDICTION_VALUES)
    new_prediction = _required_string(row, "new_prediction", allowed=PREDICTION_VALUES)
    if new_prediction != "not_person":
        return None

    new_reason = _required_string(row, "new_reason", allowed=PP_VYS_REASON_VALUES)
    _require_consistent_new_prediction_reason(new_prediction, new_reason)

    routed = dict(row)
    routed["input_order_candidate"] = "unknown"
    routed["router_prediction"] = "not_person"
    routed["router_reason"] = "not_person"
    return routed


def _require_consistent_new_prediction_reason(new_prediction: str, new_reason: str) -> None:
    """Reject PP/VYS routing rows whose prediction and reason point different ways."""
    if new_reason in PP_NEW_REASON_VALUES and new_prediction != "pp":
        message = "new_reason must agree with new_prediction"
        raise ValueError(message)
    if new_reason in VYS_NEW_REASON_VALUES and new_prediction != "vys":
        message = "new_reason must agree with new_prediction"
        raise ValueError(message)


def _base_pp_vys_route(features: PPVysFeatures) -> tuple[Route, str]:
    route: Route = "pp"
    reason = "default_pp"

    base_overrides: tuple[tuple[bool, Route, str], ...] = (
        (features.old_new_vys, "vys", "old_new_vys"),
        (features.old_pp_new_vys, "vys", "old_pp_new_vys"),
        (features.old_vys_new_abstain, "vys", "old_vys_new_abstain"),
        (features.old_vys_new_pp, "vys", "old_vys_new_pp_default_vys"),
        (_endpoint_pp_with_real_paper_votes(features), "pp", "endpoint_pp_with_real_paper_votes"),
        (_strong_pp_allowed_ratio(features), "pp", "strong_pp_allowed_ratio"),
        (_reliable_input_order_abstain(features), "abstain", "reliable_input_order_abstain"),
    )
    for condition, candidate_route, candidate_reason in base_overrides:
        if condition:
            route = candidate_route
            reason = candidate_reason
    return route, reason


def _effective_plus_override(features: PPVysFeatures) -> tuple[Route, str] | None:
    selected: tuple[Route, str] | None = None
    overrides: tuple[tuple[bool, Route, str], ...] = (
        (_endpoint_pp_high_conf_two_vote(features), "pp", "endpoint_pp_high_conf_two_vote"),
        (_pp_context_over_vys_batch_for_larger_papers(features), "pp", "larger_pp_paper_overrides_vys_batch"),
    )
    for condition, route, reason in overrides:
        if condition:
            selected = (route, reason)
    return selected


def _endpoint_pp_with_real_paper_votes(features: PPVysFeatures) -> bool:
    return (
        features.old_vys_new_pp
        and features.new_reason == "endpoint_frequency_strongly_favors_pp"
        and features.pp_batch_total_count >= PP_VYS_REAL_PAPER_VOTES_MIN
    )


def _strong_pp_allowed_ratio(features: PPVysFeatures) -> bool:
    return (
        features.old_vys_new_pp
        and features.new_reason == "strong_pp_paper_context"
        and (
            features.ratio <= PP_VYS_ALLOWED_RATIO_LOW_MAX
            or PP_VYS_ALLOWED_RATIO_HIGH_MIN < features.ratio <= PP_VYS_ALLOWED_RATIO_HIGH_MAX
        )
    )


def _reliable_input_order_abstain(features: PPVysFeatures) -> bool:
    input_pp_old_new_pp = (
        features.input_order_candidate == "pp" and features.old_prediction == "pp" and features.new_prediction == "pp"
    )
    endpoint_vys_small_pp_context = (
        features.input_order_candidate == "vys"
        and features.new_reason == "endpoint_frequency_strongly_favors_vys"
        and features.pp_batch_total_count <= PP_VYS_LOW_BATCH_MAX
    )
    weak_vys_small_pp_context = (
        features.input_order_candidate == "vys"
        and features.new_reason == "weak_or_conflicting_evidence"
        and features.pp_batch_total_count <= PP_VYS_VERY_LOW_BATCH_MAX
        and features.ratio <= PP_VYS_LOW_RATIO_MAX
    )
    strong_vys_small_pp_context = (
        features.input_order_candidate == "vys"
        and features.new_reason == "strong_vys_batch_context"
        and features.pp_batch_total_count <= PP_VYS_LOW_BATCH_MAX
        and features.ratio <= PP_VYS_LOW_RATIO_MAX
    )
    return input_pp_old_new_pp or endpoint_vys_small_pp_context or weak_vys_small_pp_context or strong_vys_small_pp_context


def _endpoint_pp_high_conf_two_vote(features: PPVysFeatures) -> bool:
    return (
        features.old_vys_new_pp
        and features.new_reason == "endpoint_frequency_strongly_favors_pp"
        and features.pp_batch_total_count == PP_VYS_LOW_BATCH_MAX
        and features.pp_batch_confidence >= PP_VYS_HIGH_CONFIDENCE_MIN
    )


def _pp_context_over_vys_batch_for_larger_papers(features: PPVysFeatures) -> bool:
    return (
        features.old_pp_new_vys
        and features.new_reason == "strong_vys_batch_context"
        and features.pp_batch_total_count >= PP_VYS_LARGER_PAPER_VOTES_MIN
    )


def _name_prior_override(features: PPVysFeatures) -> tuple[Route, str] | None:
    overrides: tuple[tuple[Route | None, str], ...] = (
        (
            _repeated_tail_given_surname_first_route(features),
            "name_prior_repeated_tail_given_surname_first",
        ),
        (
            _korean_given_first_three_token_route(features),
            "name_prior_korean_given_first_three_token",
        ),
        (
            _cantonese_given_first_route(features),
            "name_prior_cantonese_given_first",
        ),
    )
    for route, reason in overrides:
        if route is not None:
            return route, reason
    return None


def _repeated_tail_given_surname_first_route(features: PPVysFeatures) -> Route | None:
    if _has_repeated_tail_chinese_surname(features.name_tokens):
        return _desired_format_route(features, "surname_first")
    return None


def _has_repeated_tail_chinese_surname(tokens: tuple[str, ...]) -> bool:
    """Return whether a name has a Chinese surname plus reduplicated given tail."""
    return (
        len(tokens) >= MIN_THREE_TOKEN_NAME_PRIOR_TOKENS
        and tokens[-1] == tokens[-2]
        and tokens[0] in NAME_PRIOR_COMMON_CHINESE_SURNAMES
    )


def _korean_given_first_three_token_route(features: PPVysFeatures) -> Route | None:
    tokens = features.name_tokens
    if _has_repeated_tail_chinese_surname(tokens):
        return None
    if (
        len(tokens) >= MIN_THREE_TOKEN_NAME_PRIOR_TOKENS
        and tokens[-1] in NAME_PRIOR_KOREAN_SURNAMES
        and all(len(token) == 1 or token in NAME_PRIOR_KOREAN_GIVEN_SYLLABLES for token in tokens[:-1])
    ):
        return _desired_format_route(features, "given_first")
    return None


def _cantonese_given_first_route(features: PPVysFeatures) -> Route | None:
    tokens = features.name_tokens
    if _has_repeated_tail_chinese_surname(tokens):
        return None
    if (
        len(tokens) >= MIN_THREE_TOKEN_NAME_PRIOR_TOKENS
        and tokens[-1] in NAME_PRIOR_CANTONESE_SOUTHEAST_ASIAN_SURNAMES
        and all(token in NAME_PRIOR_CANTONESE_GIVEN_SYLLABLES for token in tokens[:-1])
    ):
        return _desired_format_route(features, "given_first")
    return None


def _desired_format_route(features: PPVysFeatures, desired_format: str) -> Route | None:
    pp_has_desired_format = features.pp_selected_format == desired_format
    vys_has_desired_format = features.vys_selected_format == desired_format
    if pp_has_desired_format and not vys_has_desired_format:
        return "pp"
    if vys_has_desired_format and not pp_has_desired_format:
        return "vys"
    return None


def route_pp_abstain_rows(rows: Iterable[Row]) -> list[MutableRow]:
    """Route no-VYS rows to PP or input-order abstain.

    Required input columns are listed in `PP_ABSTAIN_REQUIRED_COLUMNS`.
    The returned rows add:
    - `router_prediction`
    - `router_reason`
    """
    output: list[MutableRow] = []
    for row in rows:
        _require_columns(row, PP_ABSTAIN_REQUIRED_COLUMNS)
        routed = dict(row)
        if _pp_abstain_failed_parse(row):
            routed["router_prediction"] = "not_person"
            routed["router_reason"] = "not_person"
            output.append(routed)
            continue

        predicates = _pp_abstain_predicates(row)

        route: Route = "abstain"
        reason = "default_abstain"
        if predicates["accept_surname_first"]:
            route = "pp"
            reason = "surname_first_two_token"
        if predicates["clean_bilingual_given_first"]:
            route = "pp"
            reason = "clean_bilingual_given_first"
        if route == "abstain":
            reason = _first_true_reason(
                predicates,
                (
                    "spaced_cjk_zero_batch_surname_first",
                    "weak_zero_batch",
                    "zero_batch_mixed_long",
                ),
                default="default_abstain",
            )

        routed["router_prediction"] = route
        routed["router_reason"] = reason
        output.append(routed)
    return output


def pp_vys_abstain_router(dataframe: Any) -> Any:
    """Pandas wrapper for `route_pp_vys_abstain_rows`.

    This keeps notebook/pipeline usage close to the original scratch rule while
    letting the tested implementation stay dependency-light.
    """
    _require_frame_columns(dataframe, PP_VYS_ABSTAIN_REQUIRED_COLUMNS)
    rows = route_pp_vys_abstain_rows(dataframe.to_dict("records"))
    return _with_routing_columns(
        dataframe,
        rows,
        (
            "old_prediction",
            "old_reason",
            "new_prediction",
            "new_reason",
            "input_order_candidate",
            "router_prediction",
            "router_reason",
        ),
    )


def pp_abstain_router(dataframe: Any) -> Any:
    """Pandas wrapper for `route_pp_abstain_rows`."""
    _require_frame_columns(dataframe, PP_ABSTAIN_REQUIRED_COLUMNS)
    rows = route_pp_abstain_rows(dataframe.to_dict("records"))
    return _with_routing_columns(dataframe, rows, ("router_prediction", "router_reason"))


def _iter_result_evidence(batch_result: BatchParseResult):
    if len(batch_result.results) != len(batch_result.name_order_evidence):
        message = "batch_result must include one NameOrderEvidence row per ParseResult"
        raise ValueError(message)
    if len(batch_result.names) != len(batch_result.results):
        message = "batch_result names and results must have the same length"
        raise ValueError(message)
    return zip(batch_result.results, batch_result.name_order_evidence, strict=True)


def _require_aligned_batch_results(pp_batch_result: BatchParseResult, vys_batch_result: BatchParseResult) -> None:
    if pp_batch_result.names != vys_batch_result.names:
        message = "pp and vys batch results must have aligned names"
        raise ValueError(message)
    if len(pp_batch_result.results) != len(pp_batch_result.names):
        message = "pp batch result must include one ParseResult per name"
        raise ValueError(message)
    if len(vys_batch_result.results) != len(vys_batch_result.names):
        message = "vys batch result must include one ParseResult per name"
        raise ValueError(message)
    if len(pp_batch_result.name_order_evidence) != len(pp_batch_result.names):
        message = "pp batch result must include one NameOrderEvidence row per name"
        raise ValueError(message)
    if len(vys_batch_result.name_order_evidence) != len(vys_batch_result.names):
        message = "vys batch result must include one NameOrderEvidence row per name"
        raise ValueError(message)


def _old_pp_vys_decision_from_row(row: Row) -> tuple[str, str]:
    """Return the earlier PP/VYS backoff decision used by the selected hybrid."""
    pp_success = _required_bool(row, "pp_success")
    vys_success = _required_bool(row, "vys_success")
    pp_text = _optional_string(row, "pp_result")
    vys_text = _optional_string(row, "vys_result")
    if (
        not pp_success
        or not vys_success
        or _routing_token_count(pp_text) > PP_VYS_GARBAGE_RESULT_TOKEN_MAX
        or _routing_token_count(vys_text) > PP_VYS_GARBAGE_RESULT_TOKEN_MAX
    ):
        return "not_person", "garbage_or_failed_parse"

    if pp_text == vys_text:
        return "pp", "pp_vys_same"

    pp_margin = _required_float(row, "pp_batch_vote_margin")
    pp_ct = _required_float(row, "pp_batch_total_count")
    vys_margin = _required_float(row, "vys_batch_vote_margin")

    pp_freq = _optional_float(row, "pp_selected_surname_frequency", default=0.0)
    vys_freq = _optional_float(row, "vys_selected_surname_frequency", default=0.0)
    if pp_freq > 0:
        vys_over_pp_freq_ratio = vys_freq / pp_freq
    elif vys_freq > 0:
        # PP surname is unseen while VYS surname is attested: maximal VYS evidence.
        vys_over_pp_freq_ratio = math.inf
    else:
        vys_over_pp_freq_ratio = 0.0
    pp_position = _required_string(row, "pp_selected_surname_position", allowed=("first", "last", "internal", "unknown"))
    vys_position = _required_string(row, "vys_selected_surname_position", allowed=("first", "last", "internal", "unknown"))
    endpoint_disagrees = pp_position in {"first", "last"} and vys_position in {"first", "last"} and pp_position != vys_position
    vys_batch_dominant_format = _required_string(row, "vys_batch_dominant_format", allowed=FORMAT_VALUES)
    use_vys = (
        (
            endpoint_disagrees
            and vys_over_pp_freq_ratio >= PP_VYS_OLD_ENDPOINT_VYS_RATIO_MIN
            and vys_margin >= PP_VYS_OLD_ENDPOINT_VYS_MARGIN_MIN
        )
        or (
            pp_ct <= PP_VYS_OLD_SMALL_PP_CONTEXT_MAX
            and vys_batch_dominant_format == "given_first"
            and vys_margin >= PP_VYS_OLD_SMALL_PP_VYS_MARGIN_MIN
        )
        or (pp_margin <= PP_VYS_OLD_WEAK_PP_MARGIN_MAX and vys_margin >= PP_VYS_OLD_WEAK_PP_VYS_MARGIN_MIN)
    )
    if use_vys:
        return "vys", "vys_backoff_rule"
    return "pp", "default_pp"


def _selected_clean_pp_vys_decision_from_row(row: Row) -> tuple[str, str]:
    """Return the selected clean PP/VYS/abstain decision before hybrid overrides."""
    ratio = _optional_float(row, "pp_selected_surname_frequency_ratio", default=math.nan)
    if ratio <= PP_VYS_SELECTED_CLEAN_ENDPOINT_VYS_RATIO_MAX:
        return "vys", "endpoint_frequency_strongly_favors_vys"
    if ratio >= PP_VYS_SELECTED_CLEAN_ENDPOINT_PP_RATIO_MIN:
        return "pp", "endpoint_frequency_strongly_favors_pp"
    if _strong_pp_paper_context_from_row(row):
        return "pp", "strong_pp_paper_context"
    if _strong_vys_batch_context_from_row(row):
        return "vys", "strong_vys_batch_context"
    return "abstain", "weak_or_conflicting_evidence"


def _strong_pp_paper_context_from_row(row: Row) -> bool:
    return (
        _required_bool(row, "pp_batch_threshold_met")
        and _required_string(row, "pp_batch_dominant_format", allowed=FORMAT_VALUES)
        == _required_string(row, "pp_selected_format", allowed=FORMAT_VALUES)
        and _required_float(row, "pp_batch_total_count") >= PP_VYS_REAL_PAPER_VOTES_MIN
        and _required_float(row, "pp_batch_confidence") >= PP_VYS_SELECTED_CLEAN_STRONG_PP_CONFIDENCE_MIN
    )


def _strong_vys_batch_context_from_row(row: Row) -> bool:
    return (
        _required_bool(row, "vys_batch_threshold_met")
        and _required_string(row, "vys_batch_dominant_format", allowed=FORMAT_VALUES)
        == _required_string(row, "vys_selected_format", allowed=FORMAT_VALUES)
        and _required_float(row, "vys_batch_confidence") >= PP_VYS_SELECTED_CLEAN_STRONG_VYS_CONFIDENCE_MIN
    )


def _format_pattern_decision_confidence(pattern: Any) -> float:
    confidence = getattr(pattern, "decision_confidence", None)
    if confidence is None:
        confidence = getattr(pattern, "confidence", 0.0)
    return float(confidence)


def _input_order_display(parsed: ParsedName) -> list[tuple[str, str]]:
    """Rebuild the input-order (role, token) sequence from a parse's positional labels.

    `ParsedName.order` is a positional label sequence in which a role may repeat
    (e.g. ``["middle", "given", "middle", "surname"]`` for ``J. Ming K. Zhang``).
    Tokens are drawn from each role's list in order: the final occurrence of a role
    drains its remaining tokens and every earlier occurrence consumes exactly one.
    This keeps a single-occurrence role that owns several tokens intact (e.g.
    ``["surname", "given"]`` with two given tokens) while never emitting a token
    twice when a label repeats.
    """
    tokens_by_role = {
        "surname": list(parsed.surname_tokens),
        "given": list(parsed.given_tokens),
        "middle": list(parsed.middle_tokens),
    }
    last_occurrence = {role: index for index, role in enumerate(parsed.order)}
    consumed = dict.fromkeys(tokens_by_role, 0)
    display: list[tuple[str, str]] = []
    for index, role in enumerate(parsed.order):
        tokens = tokens_by_role.get(role)
        if not tokens or consumed[role] >= len(tokens):
            continue
        if index == last_occurrence[role]:
            display.extend((role, token) for token in tokens[consumed[role]:])
            consumed[role] = len(tokens)
        else:
            display.append((role, tokens[consumed[role]]))
            consumed[role] += 1
    return display


def input_order_parsed(result: ParseResult) -> ParsedName | None:
    """Materialize the preprocessed input-order (as-typed) parse for an abstain route.

    The input-order reading interprets the preprocessed tokens as already being in
    the library's Given-Surname output order: the trailing token is the surname and
    every preceding token keeps its position (initials stay middle tokens, the rest
    form the given name). This is the single-batch counterpart of the PP/VYS
    ``input_order_candidate``: it never re-decides name order, so it is stable
    across parser versions and batch compositions.

    Returns None for failed parses and single-token names.
    """
    original = result.parsed_original_order if result.success else None
    if original is None or not original.surname_tokens:
        return None

    display = _input_order_display(original)
    if len(display) < MIN_INPUT_ORDER_DISPLAY_TOKENS or display[-1][0] == "middle":
        return None

    surname_token_count = len(original.surname_tokens)
    if all(role == "surname" for role, _token in display[-surname_token_count:]):
        # The original reading already puts the surname last: the normalized parse
        # (including compound-surname and hyphenation handling) is the input-order parse.
        parsed = result.parsed
    else:
        surname = display[-1][1]
        middle_tokens = [token for role, token in display[:-1] if role == "middle"]
        given_tokens = [token for role, token in display[:-1] if role != "middle"]
        parsed = (
            ParsedName(
                surname=surname,
                given_name="-".join(given_tokens),
                surname_tokens=[surname],
                given_tokens=given_tokens,
                middle_name=" ".join(middle_tokens),
                middle_tokens=middle_tokens,
                order=["given", "middle", "surname"],
            )
            if given_tokens
            else None
        )
    return parsed


def pp_abstain_parsed(result: ParseResult, route_row: Row) -> ParsedName | None:
    """Return the final parsed person components for a PP-only abstain row.

    Latin-only abstain keeps the as-typed given-first reading. Spaced Han rows
    guarded by `spaced_cjk_zero_batch_surname_first` already have a source
    surname boundary, so the PP parse is the stable input-order reading.
    """
    if _pp_abstain_uses_pp_parse_for_abstain(route_row):
        return result.parsed if result.success else None
    return input_order_parsed(result) or (result.parsed if result.success else None)


def _result_text(result: ParseResult) -> str:
    return str(result.result) if result.success else ""


def _routing_token_count(text: str) -> int:
    return len(ROUTING_TOKEN_RE.findall(text or ""))


def _parse_result_token_count(result: ParseResult) -> int:
    """Count text tokens of the rendered result string.

    This is the convention the pp-abstain rules were tuned on: a hyphenated
    given name renders as one token ("Chang-Qing Zhang" -> 2), and a failed
    parse counts as 1.
    """
    if not result.success:
        return 1
    return _routing_token_count(_result_text(result))


def _format_value(value: NameFormat | str) -> str:
    if isinstance(value, NameFormat):
        return value.value
    return str(value)


def _float_or_zero(value: float | None) -> float:
    if value is None:
        return 0.0
    return float(value)


def _normalize_for_routing(detector_context: Any, raw_name: str) -> Any:
    normalizer = getattr(detector_context, "_normalizer", None)
    if normalizer is None or not hasattr(normalizer, "apply"):
        message = "detector_context must expose a normalizer with apply()"
        raise ValueError(message)
    return normalizer.apply(raw_name)


def _script_representation_for_routing(
    detector_context: Any,
    normalized_input: Any,
    evidence: NameOrderEvidence,
) -> str:
    normalizer = getattr(detector_context, "_normalizer", None)
    if normalizer is not None and hasattr(normalizer, "classify_script_representation"):
        return normalizer.classify_script_representation(normalized_input)
    return evidence.script_representation


def _has_cjk(script_representation: str) -> bool:
    return script_representation in {"han_only", "bilingual_aligned", "mixed_script"}


def _has_latin(script_representation: str) -> bool:
    return script_representation in {"latin_only", "bilingual_aligned", "mixed_script"}


def _input_order_candidate(row: Row) -> str:
    pp_preserves_input_order = (
        _required_string(
            row,
            "pp_selected_format",
            allowed=ORDER_FORMAT_VALUES,
        )
        == "given_first"
    )
    vys_preserves_input_order = (
        _required_string(
            row,
            "vys_selected_format",
            allowed=ORDER_FORMAT_VALUES,
        )
        == "given_first"
    )
    if pp_preserves_input_order and not vys_preserves_input_order:
        return "pp"
    if vys_preserves_input_order and not pp_preserves_input_order:
        return "vys"
    return "unknown"


def _name_tokens(name: str) -> tuple[str, ...]:
    return tuple(token.lower() for token in re.findall(r"[A-Za-z]+", name))


def _pp_abstain_failed_parse(row: Row) -> bool:
    """Return whether the PP parse failed, routing the row to a terminal non-person decision.

    `pp_success` is a required pp-abstain column (see `PP_ABSTAIN_REQUIRED_COLUMNS`),
    so a failed parse always routes `not_person`, matching production. This is the
    real path for the fixture's failed-parse rows (encoded as `pp_result_token_count
    == 1` with `selected_format == "mixed"`).
    """
    return not _required_bool(row, "pp_success")


def _pp_abstain_uses_pp_parse_for_abstain(row: Row) -> bool:
    return row.get("router_reason") == "spaced_cjk_zero_batch_surname_first"


def _pp_abstain_predicates(row: Row) -> dict[str, bool]:
    pp_result_token_count = _required_float(row, "pp_result_token_count")
    batch_total_count = _required_float(row, "batch_total_count")
    selected_surname_frequency = _required_float(row, "selected_surname_frequency")
    raw_tokens = _required_float(row, "raw_tokens")
    has_cjk = _required_bool(row, "has_cjk")
    has_latin = _required_bool(row, "has_latin")
    cjk_has_space = _required_bool(row, "cjk_has_space")
    selected_format = _required_string(row, "selected_format", allowed=FORMAT_VALUES)

    surname_first_two_token = pp_result_token_count == PP_ABSTAIN_TWO_TOKEN_RESULT_COUNT and selected_format == "surname_first"
    weak_zero_batch = batch_total_count == 0 and selected_surname_frequency < PP_ABSTAIN_WEAK_SURNAME_FREQUENCY_MAX
    zero_batch_mixed_long = batch_total_count == 0 and has_cjk and has_latin and raw_tokens >= PP_ABSTAIN_MIXED_LONG_RAW_TOKEN_MIN
    spaced_cjk_zero_batch_surname_first = (
        surname_first_two_token and batch_total_count == 0 and has_cjk and not has_latin and cjk_has_space
    )
    accept_surname_first = surname_first_two_token and not zero_batch_mixed_long and not spaced_cjk_zero_batch_surname_first
    clean_bilingual_given_first = (
        pp_result_token_count == PP_ABSTAIN_TWO_TOKEN_RESULT_COUNT
        and selected_format == "given_first"
        and has_cjk
        and has_latin
        and raw_tokens == PP_ABSTAIN_CLEAN_BILINGUAL_RAW_TOKEN_COUNT
        and selected_surname_frequency >= PP_ABSTAIN_CLEAN_BILINGUAL_SURNAME_FREQUENCY_MIN
    )
    return {
        "surname_first_two_token": surname_first_two_token,
        "weak_zero_batch": weak_zero_batch,
        "zero_batch_mixed_long": zero_batch_mixed_long,
        "spaced_cjk_zero_batch_surname_first": spaced_cjk_zero_batch_surname_first,
        "accept_surname_first": accept_surname_first,
        "clean_bilingual_given_first": clean_bilingual_given_first,
    }


def _first_true_reason(predicates: Mapping[str, bool], names: tuple[str, ...], *, default: str) -> str:
    for name in names:
        if predicates[name]:
            return name
    return default


def _require_columns(row: Row, required_columns: tuple[str, ...]) -> None:
    missing = [column for column in required_columns if column not in row]
    if missing:
        message = f"missing required columns: {', '.join(missing)}"
        raise ValueError(message)


def _require_frame_columns(dataframe: Any, required_columns: tuple[str, ...]) -> None:
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        message = f"missing required columns: {', '.join(missing)}"
        raise ValueError(message)


def _with_routing_columns(dataframe: Any, rows: list[MutableRow], columns: tuple[str, ...]) -> Any:
    output = dataframe.copy()
    for column in columns:
        output[column] = [row[column] for row in rows]
    return output


def _is_nan(value: object) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _required_string(row: Row, column: str, *, allowed: tuple[str, ...] | None = None) -> str:
    value = row[column]
    if value is None or _is_nan(value):
        message = f"{column} is required"
        raise ValueError(message)
    text = str(value)
    if text == "":
        message = f"{column} is required"
        raise ValueError(message)
    if allowed is not None and text not in allowed:
        message = f"{column} must be one of {', '.join(allowed)}"
        raise ValueError(message)
    return text


def _optional_string(row: Row, column: str) -> str:
    value = row[column]
    if value is None:
        return ""
    if _is_nan(value):
        return ""
    return str(value)


def _required_float(row: Row, column: str) -> float:
    value = row[column]
    if value is None or _is_nan(value):
        message = f"{column} must be a finite number"
        raise ValueError(message)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        message = f"{column} must be a finite number"
        raise ValueError(message) from None
    if not math.isfinite(parsed):
        message = f"{column} must be a finite number"
        raise ValueError(message)
    return parsed


def _optional_float(row: Row, column: str, *, default: float) -> float:
    value = row[column]
    if value is None or value == "" or _is_nan(value):
        return default
    return _required_float(row, column)


def _required_bool(row: Row, column: str) -> bool:
    value = row[column]
    if isinstance(value, bool):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if value is None or _is_nan(value):
        message = f"{column} must be a boolean"
        raise ValueError(message)
    if isinstance(value, int | float) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"1", "true", "t", "yes", "y"}:
            return True
        if normalized in {"0", "false", "f", "no", "n"}:
            return False
    message = f"{column} must be a boolean"
    raise ValueError(message)
