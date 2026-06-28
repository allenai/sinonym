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

    return _compact_component_labels(_given_middle_labels(given_tokens))


def _given_middle_labels(given_tokens: list[str]) -> list[str]:
    """Return per-token labels after applying the formatter's initial-peeling rules."""
    labels = ["given"] * len(given_tokens)
    if len(given_tokens) > 1:
        part_lengths = [_token_letter_length(token) for token in given_tokens]

        leading_count = _leading_initial_count(part_lengths)
        if leading_count > 0 and any(length > 1 for length in part_lengths[leading_count:]):
            for index in range(leading_count):
                labels[index] = "middle"

        remaining_lengths = part_lengths[leading_count:]
        trailing_count = _trailing_initial_count(remaining_lengths)
        if trailing_count > 0 and any(length > 1 for length in remaining_lengths[:-trailing_count]):
            start = len(given_tokens) - trailing_count
            for index in range(start, len(given_tokens)):
                labels[index] = "middle"
    return labels


def _leading_initial_count(part_lengths: list[int]) -> int:
    """Return the leading single-letter run length."""
    count = 0
    for length in part_lengths:
        if length != 1:
            break
        count += 1
    return count


def _trailing_initial_count(part_lengths: list[int]) -> int:
    """Return the trailing single-letter run length."""
    count = 0
    for length in reversed(part_lengths):
        if length != 1:
            break
        count += 1
    return count


def _compact_component_labels(labels: list[str]) -> list[str]:
    """Collapse adjacent duplicate component labels."""
    order: list[str] = []
    for label in labels:
        if not order or order[-1] != label:
            order.append(label)
    return order or ["given"]


def _token_letter_length(token: str) -> int:
    """Return the alphabetic length used for initial detection."""
    return len("".join(char for char in token.replace("-", "") if char.isalpha()))
