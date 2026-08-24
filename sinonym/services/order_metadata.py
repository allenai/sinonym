"""Helpers for structured name-order metadata."""

from __future__ import annotations

from sinonym.coretypes import NameFormat


def original_component_order(
    selected_format: NameFormat,
    given_tokens: list[str],
    middle_tokens: list[str],
) -> list[str]:
    """Return source component order from the selected parse and middle initials."""
    given_order = _given_middle_order(given_tokens, middle_tokens)
    if selected_format == NameFormat.GIVEN_FIRST:
        return [*given_order, "surname"]
    return ["surname", *given_order]


def _given_middle_order(given_tokens: list[str], middle_tokens: list[str]) -> list[str]:
    """Return source order for the given-name span after initials are peeled."""
    if not middle_tokens:
        return ["given"]

    labels = _given_middle_labels(given_tokens)
    if "middle" not in labels:
        compact_labels = _compact_given_middle_labels(given_tokens, middle_tokens)
        labels = compact_labels or [*labels, "middle"]
        return _compact_component_labels(labels)
    return labels or ["given"]


def _given_middle_labels(given_tokens: list[str]) -> list[str]:
    """Return source labels after applying the formatter's mixed-span rule."""
    given_parts = _given_name_parts(given_tokens)
    part_lengths = [_token_letter_length(part) for part in given_parts]
    has_spelled_component = any(length > 1 for length in part_lengths)
    if not has_spelled_component:
        return ["given"] * len(given_parts)
    return ["middle" if length == 1 else "given" for length in part_lengths]


def _given_name_parts(given_tokens: list[str]) -> list[str]:
    """Return source tokens while retaining explicit hyphen binding."""
    parts: list[str] = []
    for token in given_tokens:
        clean_token = token.strip("-")
        if not clean_token:
            continue
        parts.append(clean_token)
    return parts


def _compact_given_middle_labels(given_tokens: list[str], middle_tokens: list[str]) -> list[str]:
    """Infer edge-position middle initials that came from compact given tokens."""
    source = "".join(_letters_only(token).lower() for token in given_tokens)
    middle = "".join(_letters_only(token).lower() for token in middle_tokens)
    if not source or not middle or len(source) <= len(middle):
        return []

    if source.startswith(middle):
        return ["middle", "given"]
    if source.endswith(middle):
        return ["given", "middle"]
    return []


def _compact_component_labels(labels: list[str]) -> list[str]:
    """Collapse adjacent duplicate component labels."""
    order: list[str] = []
    for label in labels:
        if not order or order[-1] != label:
            order.append(label)
    return order or ["given"]


def _token_letter_length(token: str) -> int:
    """Return the alphabetic length used for initial detection."""
    return len(_letters_only(token.replace("-", "")))


def _letters_only(token: str) -> str:
    """Return alphabetic characters from a token."""
    return "".join(char for char in token if char.isalpha())
